"""Fast track: Vaani shard -> sampled Marwari recordings -> clips, in one pass on local disk.

Raw audio never goes to Drive (free Drive is 15 GB). Only the 8 s clips are saved, one parquet
per Vaani shard, and each write is checked by size before the done marker is written.
Speaker split is by a seeded hash of the speaker id (10% test, 10% dev, 80% train), so no
pre-pass over all speakers is needed and a speaker always lands in the same split.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import traceback

import numpy as np

from . import audio as A
from . import clips as C
from . import vaani as V

_VAD = None


def speaker_of(row: dict) -> str | None:
    s = row.get("speakerID")
    if s is not None and str(s).strip() and str(s) != "nan":
        return str(s)
    hit = re.search(r"_(\d{5,})_", str((row.get("audio") or {}).get("path") or ""))
    return f"PATH_{hit.group(1)}" if hit else None  # Jaipur rows have no speakerID


def _copy_checked(src: str, dst: str, tries: int = 5) -> None:
    size = os.path.getsize(src)
    for i in range(tries):
        try:
            shutil.copyfile(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
            if os.path.getsize(dst) == size:
                return
            raise OSError(f"size mismatch on {dst}")
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def run_shard(cfg: dict, repo_path: str, out_dir: str, done_dir: str, work_root: str) -> dict:
    import pandas as pd
    import pyarrow.parquet as pq
    import torch

    global _VAD
    key = V.shard_key(repo_path)
    done_path = os.path.join(done_dir, key + ".json")
    if os.path.exists(done_path):
        with open(done_path) as f:
            return json.load(f)
    t0 = time.time()
    seed, frac = cfg["seed"], cfg["clips"]["rec_sample_frac"]
    window, tel_frac = cfg["audio"]["window_s"], cfg["clips"]["telephony_frac_train"]
    torch.set_num_threads(1)
    if _VAD is None:
        from silero_vad import load_silero_vad

        _VAD = load_silero_vad()
    work = os.path.join(work_root, key)
    os.makedirs(work, exist_ok=True)
    stage = "download"
    try:
        local = V.download(cfg, repo_path, work)
        stage = "process"
        rows, n_rows, n_used, n_nospk, n_nospeech = [], 0, 0, 0, 0
        for batch in pq.ParquetFile(local).iter_batches(batch_size=32):
            for r in batch.to_pylist():
                n_rows += 1
                if V.group_of(r.get("language"), cfg) != "marwari":
                    continue
                raw = (r.get("audio") or {}).get("bytes")
                if raw is None:
                    continue
                rec_id = hashlib.sha1(raw).hexdigest()[:16]
                if C.unit_hash(seed, "rec", rec_id) >= frac:
                    continue
                spk = speaker_of(r)
                if spk is None:
                    n_nospk += 1
                    continue
                n_used += 1
                split = C.split_for_new_speaker(spk, cfg, seed)
                wav = A.decode_bytes(raw)
                segs = C.vad_segments(wav, _VAD, cfg)
                rng = np.random.default_rng(int(C.unit_hash(seed, rec_id) * 2 ** 32))
                cands = C.candidates(segs, len(wav), cfg, rng)
                if not cands:
                    n_nospeech += 1
                for c in cands:
                    cid = C.clip_id(rec_id, c["cut"])
                    rows.append({
                        "clip_id": cid, "rec_id": rec_id, "speakerID": spk, "split": split,
                        "district": r.get("district"), "gender": r.get("gender"), "language": r.get("language"),
                        "utterance_seq": r.get("UtteranceSequenceID"), "kind": c["kind"],
                        "position_label": c["position_label"], "cut_s": c["cut"] / C.SR,
                        "speech_end_s": c["speech_end"] / C.SR, "pause_ms": c["pause_ms"],
                        "trail_real_ms": c["trail_real_ms"], "rec_duration_s": len(wav) / C.SR,
                        "transcript": r.get("transcript") or None,
                        "telephony": bool(split == "train" and C.unit_hash(seed, "tel", cid) < tel_frac),
                        "flac": A.encode(C.cut_window(wav, c["cut"], window)),
                    })
        os.remove(local)
        stage = "write"
        if rows:
            tmp = os.path.join(work, "clips.parquet")
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            _copy_checked(tmp, os.path.join(out_dir, key + ".parquet"))
        stats = {"key": key, "rows_in": n_rows, "recordings_used": n_used, "no_speaker": n_nospk,
                 "no_speech": n_nospeech, "clips": len(rows), "seconds": round(time.time() - t0, 1)}
        marker = os.path.join(work, "done.json")
        with open(marker, "w") as f:
            json.dump(stats, f)
        _copy_checked(marker, done_path)
        return stats
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()
        raise RuntimeError(f"stage={stage}: {e!r} | {' / '.join(tb[-3:])}") from None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def clip_index(out_dir: str) -> "pd.DataFrame":
    import pandas as pd
    import pyarrow.parquet as pq

    frames = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".parquet"):
            p = os.path.join(out_dir, f)
            cols = [c for c in pq.ParquetFile(p).schema_arrow.names if c != "flac"]
            d = pq.read_table(p, columns=cols).to_pandas()
            d["clip_shard"] = f
            frames.append(d)
    return V.unify_object_columns(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
