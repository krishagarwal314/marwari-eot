"""Whisper log-mel features exactly as Smart Turn computes them, plus sharded storage.

The extractor call mirrors pipecat-ai/smart-turn inference.py / train.py (commit 4786657e):
truncate/left-pad to the last 8 s, then WhisperFeatureExtractor(chunk_length=8) with
padding="max_length", max_length=8*16000, truncation=True, do_normalize=True -> (80, 800).
"""
from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from typing import Iterable, Iterator, Sequence

import numpy as np

from .audio import SR, truncate_audio_to_last_n_seconds

WINDOW_S = 8


@lru_cache(maxsize=1)
def _extractor():
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor(chunk_length=WINDOW_S)


def extract(audio: np.ndarray) -> np.ndarray:
    """One clip (16 kHz mono float32) -> float32 (80, 800)."""
    return extract_batch([audio])[0]


def extract_batch(audios: Sequence[np.ndarray]) -> np.ndarray:
    """List of clips -> float32 (B, 80, 800). Same numbers as calling Smart Turn per clip."""
    windows = [truncate_audio_to_last_n_seconds(np.asarray(a, np.float32), n_seconds=WINDOW_S)
               for a in audios]
    out = _extractor()(
        windows,
        sampling_rate=SR,
        return_tensors="np",
        padding="max_length",
        max_length=WINDOW_S * SR,
        truncation=True,
        do_normalize=True,
    )
    return out.input_features.astype(np.float32)


# --- sharded feature storage ---------------------------------------------------------------------
# Layout per dataset directory:
#   shard_00000.npy        (N, 80, 800) features in cfg dtype
#   shard_00000.meta.json  list of per-clip dicts (id, label, language, speaker, ...)
#   shard_00000.done       written last; a shard without it is incomplete and gets redone

def shard_paths(root: str, idx: int) -> dict[str, str]:
    base = os.path.join(root, f"shard_{idx:05d}")
    return {"npy": base + ".npy", "meta": base + ".meta.json", "done": base + ".done"}


def shard_done(root: str, idx: int) -> bool:
    return os.path.exists(shard_paths(root, idx)["done"])


def write_shard(root: str, idx: int, feats: np.ndarray, meta: list[dict], dtype: str) -> None:
    os.makedirs(root, exist_ok=True)
    p = shard_paths(root, idx)
    tmp = p["npy"] + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, feats.astype(dtype))
    os.replace(tmp, p["npy"])
    with open(p["meta"] + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(p["meta"] + ".tmp", p["meta"])
    with open(p["done"], "w") as f:
        f.write(str(len(meta)))


def done_shards(root: str) -> list[int]:
    if not os.path.isdir(root):
        return []
    return sorted(int(f[6:11]) for f in os.listdir(root) if f.startswith("shard_") and f.endswith(".done"))


def iter_shards(root: str) -> Iterator[tuple[np.ndarray, list[dict]]]:
    for idx in done_shards(root):
        p = shard_paths(root, idx)
        with open(p["meta"]) as f:
            meta = json.load(f)
        yield np.load(p["npy"], mmap_mode="r"), meta


def copy_to_local(src_root: str, dst_root: str) -> str:
    """Copy finished shards from Drive to local disk (skips ones already copied)."""
    os.makedirs(dst_root, exist_ok=True)
    for idx in done_shards(src_root):
        s, d = shard_paths(src_root, idx), shard_paths(dst_root, idx)
        if os.path.exists(d["done"]):
            continue
        for k in ("npy", "meta"):
            shutil.copyfile(s[k], d[k] + ".tmp")
            os.replace(d[k] + ".tmp", d[k])
        shutil.copyfile(s["done"], d["done"])
    return dst_root


def batched(items: Iterable, n: int) -> Iterator[list]:
    buf = []
    for x in items:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
