"""Turn-boundary candidates from VAD segments, speaker splits, clip ids.

Label convention: position_label 1 = COMPLETE (cut after the final speech), 0 = INCOMPLETE
(cut inside an internal pause). Every cut keeps ``trailing_ms`` of real audio after the speech end
(zero-padded only if the recording stops sooner), matching what Smart Turn sees in production
after the VAD's stop delay.
"""
from __future__ import annotations

import hashlib

import numpy as np

SR = 16000


def clip_id(rec_id: str, cut_sample: int) -> str:
    return hashlib.sha1(f"{rec_id}:{cut_sample}".encode()).hexdigest()[:16]


def unit_hash(*parts) -> float:
    """Deterministic number in [0, 1) from strings (stable across runs and machines)."""
    h = hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()
    return int(h[:15], 16) / 16 ** 15


def vad_segments(wav: np.ndarray, model, cfg: dict) -> list[dict]:
    """Silero speech segments in samples: [{"start", "end"}, ...]."""
    import torch
    from silero_vad import get_speech_timestamps

    c = cfg["clips"]
    return get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)), model, sampling_rate=SR,
        threshold=c["vad_threshold"], min_silence_duration_ms=c["min_silence_ms"],
        speech_pad_ms=c["speech_pad_ms"], return_seconds=False,
    )


def candidates(segs: list[dict], n_samples: int, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """Cut points for one recording: one COMPLETE + up to k INCOMPLETE."""
    if not segs:
        return []
    c = cfg["clips"]
    trail = int(cfg["audio"]["trailing_silence_ms"] * SR / 1000)
    out = [{"kind": "complete", "position_label": 1, "speech_end": int(segs[-1]["end"]),
            "cut": int(segs[-1]["end"]) + trail, "pause_ms": None}]
    pauses = []
    for a, b in zip(segs[:-1], segs[1:]):
        gap_ms = (b["start"] - a["end"]) * 1000 / SR
        if c["pause_min_ms"] <= gap_ms <= c["pause_max_ms"] and a["end"] >= c["min_context_s"] * SR:
            pauses.append({"kind": "incomplete", "position_label": 0, "speech_end": int(a["end"]),
                           "cut": int(a["end"]) + trail, "pause_ms": float(gap_ms)})
    k = min(c["max_incomplete_per_rec"], len(pauses))
    if k:
        for i in sorted(rng.choice(len(pauses), size=k, replace=False)):
            out.append(pauses[i])
    for o in out:
        o["trail_real_ms"] = float(max(0, min(o["cut"], n_samples) - o["speech_end"]) * 1000 / SR)
    return out


def cut_window(wav: np.ndarray, cut: int, window_s: int) -> np.ndarray:
    """Audio up to ``cut`` (zero-padded on the right if the recording is shorter), last window_s."""
    x = wav[: min(cut, len(wav))]
    if cut > len(wav):
        x = np.concatenate([x, np.zeros(cut - len(wav), dtype=np.float32)])
    return x[-window_s * SR:].astype(np.float32)


def assign_speaker_splits(speakers: "pd.DataFrame", cfg: dict, seed: int) -> "pd.DataFrame":
    """speakers: one row per speakerID with the stratify_by columns.

    Within each stratum, speakers are ordered by a seeded hash; the first test_frac go to test,
    the next dev_frac to dev, the rest to train (each stratum with >= 3 speakers gives at least
    one test and one dev speaker). Returns speakerID, split, plus stratify columns.
    """
    s = cfg["splits"]
    df = speakers.copy()
    df["_h"] = [unit_hash(seed, sp) for sp in df["speakerID"]]
    df["split"] = "train"
    for _, g in df.groupby(s["stratify_by"], dropna=False):
        g = g.sort_values("_h")
        n = len(g)
        n_test = int(round(n * s["test_frac"]))
        n_dev = int(round(n * s["dev_frac"]))
        if n >= 3:
            n_test, n_dev = max(1, n_test), max(1, n_dev)
        df.loc[g.index[:n_test], "split"] = "test"
        df.loc[g.index[n_test:n_test + n_dev], "split"] = "dev"
    return df.drop(columns="_h")


def split_for_new_speaker(speaker_id: str, cfg: dict, seed: int) -> str:
    """Speakers that appear after the split file was frozen: hash threshold, same fractions."""
    u = unit_hash(seed, speaker_id)
    s = cfg["splits"]
    if u < s["test_frac"]:
        return "test"
    if u < s["test_frac"] + s["dev_frac"]:
        return "dev"
    return "train"


_VAD = None


def build_clips_for_shard(cfg: dict, raw_shard_path: str, keep_rec_ids: list[str], split_of: dict,
                          out_path: str, done_path: str, seed: int) -> dict:
    """VAD + candidates for every kept recording in one raw Vaani shard; writes one clip parquet.

    Safe to call in a subprocess (each process loads its own Silero model, single-threaded).
    """
    import json
    import os
    import time

    import pandas as pd
    import pyarrow.parquet as pq
    import torch

    from . import audio as A

    global _VAD
    if os.path.exists(done_path):
        with open(done_path) as f:
            return json.load(f)
    t0 = time.time()
    torch.set_num_threads(1)
    if _VAD is None:
        from silero_vad import load_silero_vad

        _VAD = load_silero_vad()
    window = cfg["audio"]["window_s"]
    tel_frac = cfg["clips"]["telephony_frac_train"]
    keep = set(keep_rec_ids)

    rows, n_rec, n_no_speech = [], 0, 0
    pf = pq.ParquetFile(raw_shard_path)
    for batch in pf.iter_batches(batch_size=32):
        for r in batch.to_pylist():
            if r["rec_id"] not in keep:
                continue
            n_rec += 1
            wav = A.decode_bytes(r["flac"])
            segs = vad_segments(wav, _VAD, cfg)
            rng = np.random.default_rng(int(unit_hash(seed, r["rec_id"]) * 2 ** 32))
            cands = candidates(segs, len(wav), cfg, rng)
            if not cands:
                n_no_speech += 1
            spk_id = str(r["speakerID"])
            split = split_of[spk_id]
            for c in cands:
                cid = clip_id(r["rec_id"], c["cut"])
                rows.append({
                    "clip_id": cid, "rec_id": r["rec_id"], "speakerID": spk_id,
                    "district": r.get("district"), "state": r.get("state"), "gender": r.get("gender"),
                    "language": r.get("language"), "split": split, "kind": c["kind"],
                    "position_label": c["position_label"], "cut_s": c["cut"] / SR,
                    "speech_end_s": c["speech_end"] / SR, "pause_ms": c["pause_ms"],
                    "trail_real_ms": c["trail_real_ms"], "rec_duration_s": len(wav) / SR,
                    "n_speech_segments": len(segs),
                    "has_transcript": str(r.get("isTranscriptionAvailable")).lower() in ("true", "1", "yes"),
                    "transcript": r.get("transcript"),
                    "telephony": bool(split == "train" and unit_hash(seed, "tel", cid) < tel_frac),
                    "flac": A.encode(cut_window(wav, c["cut"], window)),
                })
    if rows:
        tmp = out_path + ".tmp"
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
    stats = {"raw_shard": os.path.basename(raw_shard_path), "recordings": n_rec, "no_speech": n_no_speech,
             "clips": len(rows), "seconds": round(time.time() - t0, 1)}
    with open(done_path + ".tmp", "w") as f:
        json.dump(stats, f)
    os.replace(done_path + ".tmp", done_path)
    return stats
