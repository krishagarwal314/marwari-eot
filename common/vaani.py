"""Vaani -> 16 kHz FLAC recordings, one output parquet per source shard (Drive-friendly).

Output: <drive>/data/raw/vaani/shards/<key>.parquet with one row per kept recording:
  rec_id (sha1 of the original audio bytes, 16 hex chars), group (marwari|related),
  source_file, all scalar Vaani metadata columns, duration_s, flac (bytes).
Done markers: <drive>/data/raw/vaani/_done/<key>.json (row/language counts for the report).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

import numpy as np

from . import audio as A


def shard_key(repo_path: str) -> str:
    return repo_path.replace("/", "__").removesuffix(".parquet")


def list_folder_files(cfg: dict, folder: str, cache_json: str) -> list[str]:
    """All parquet files under ``folder`` in the pinned Vaani revision (cached on Drive)."""
    if os.path.exists(cache_json):
        with open(cache_json) as f:
            files = json.load(f)
    else:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(cfg["hf"]["vaani"], repo_type="dataset",
                                        revision=cfg["hf"]["vaani_revision"])
        with open(cache_json + ".tmp", "w") as f:
            json.dump(files, f)
        os.replace(cache_json + ".tmp", cache_json)
    prefix = folder.rstrip("/") + "/"
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def download(cfg: dict, repo_path: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(cfg["hf"]["vaani"], repo_path, repo_type="dataset",
                           revision=cfg["hf"]["vaani_revision"], local_dir=local_dir)


def group_of(language, cfg: dict) -> str | None:
    lang = str(language or "").strip().lower()
    if lang in cfg["vaani"]["marwari_languages"]:
        return "marwari"
    if lang in cfg["vaani"]["related_languages"]:
        return "related"
    return None


def _scalar_meta(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k == "audio":
            continue
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = json.dumps(list(v), ensure_ascii=False)
        # bytes / dict columns (e.g. embedded images) are dropped
    return out


def process_shard(local_parquet: str, repo_path: str, cfg: dict, sr: int = A.SR,
                  batch_rows: int = 32) -> tuple[list[dict], dict]:
    """Decode, filter by language group, resample to 16 kHz mono, FLAC-encode."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(local_parquet)
    rows, langs, kept, errors = [], Counter(), Counter(), 0
    for batch in pf.iter_batches(batch_size=batch_rows):
        for r in batch.to_pylist():
            langs[str(r.get("language"))] += 1
            g = group_of(r.get("language"), cfg)
            if g is None:
                continue
            a = r.get("audio") or {}
            raw = a.get("bytes")
            if raw is None:
                errors += 1
                continue
            try:
                wav = A.decode_bytes(raw, sr)
            except Exception:
                errors += 1
                continue
            meta = _scalar_meta(r)
            meta.update(
                rec_id=hashlib.sha1(raw).hexdigest()[:16],
                group=g,
                source_file=repo_path,
                src_audio_path=a.get("path"),
                duration_s=float(len(wav) / sr),
                flac=A.encode(wav, "FLAC", sr),
            )
            rows.append(meta)
            kept[g] += 1
    stats = {"rows_in": int(sum(langs.values())), "kept": dict(kept), "decode_errors": errors,
             "languages": dict(langs.most_common())}
    return rows, stats


def write_rows(rows: list[dict], out_path: str) -> None:
    import pandas as pd

    tmp = out_path + ".tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, out_path)


def build_index(shard_dir: str) -> "pd.DataFrame":
    """All recordings without audio; duplicates (same rec_id from several folders) keep the first
    occurrence in config folder order (the caller sorts shards by that order)."""
    import pandas as pd
    import pyarrow.parquet as pq

    frames = []
    for f in sorted(os.listdir(shard_dir)):
        if not f.endswith(".parquet"):
            continue
        p = os.path.join(shard_dir, f)
        cols = [c for c in pq.ParquetFile(p).schema_arrow.names if c != "flac"]
        df = pq.read_table(p, columns=cols).to_pandas()
        df["shard_file"] = f
        df["row_in_shard"] = np.arange(len(df))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(index: "pd.DataFrame") -> dict:
    out = {}
    for g, d in index.groupby("group"):
        out[g] = {
            "recordings": int(len(d)),
            "hours": round(float(d["duration_s"].sum() / 3600), 2),
            "speakers": int(d["speakerID"].nunique()) if "speakerID" in d else None,
            "districts": d["district"].value_counts().to_dict() if "district" in d else None,
            "gender": d["gender"].value_counts().to_dict() if "gender" in d else None,
            "languages": d["language"].value_counts().to_dict() if "language" in d else None,
            "with_transcript": int(d["isTranscriptionAvailable"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            if "isTranscriptionAvailable" in d else None,
            "median_duration_s": round(float(d["duration_s"].median()), 2),
        }
    return out


def run_shard(cfg: dict, repo_path: str, shard_dir: str, done_dir: str, dl_dir: str) -> dict:
    """Download -> process -> write -> done marker -> delete download. Safe to call in a subprocess."""
    import time

    key = shard_key(repo_path)
    done_path = os.path.join(done_dir, key + ".json")
    if os.path.exists(done_path):
        with open(done_path) as f:
            return json.load(f)
    t0 = time.time()
    local = download(cfg, repo_path, dl_dir)
    try:
        rows, stats = process_shard(local, repo_path, cfg)
    finally:
        os.remove(local)
    if rows:
        write_rows(rows, os.path.join(shard_dir, key + ".parquet"))
    stats.update(repo_path=repo_path, key=key, seconds=round(time.time() - t0, 1),
                 hours_kept=round(sum(r["duration_s"] for r in rows) / 3600, 3))
    with open(done_path + ".tmp", "w") as f:
        json.dump(stats, f)
    os.replace(done_path + ".tmp", done_path)
    return stats
