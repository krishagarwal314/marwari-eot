"""Audio-LLM labelling (Gemini) with a resumable on-Drive cache, and the label decision rule.

Cache: one JSONL file per clip shard, one line per answered clip:
  {"clip_id", "label", "confidence", "model", "ts"}
Failed requests are not cached, so the next run retries them.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

LABELS = ("COMPLETE", "INCOMPLETE", "UNSURE")
_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "enum": list(LABELS)},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["label", "confidence"],
}


def read_cache(path: str) -> dict[str, dict]:
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:  # torn last line after a disconnect
                    continue
                out[d["clip_id"]] = d
    return out


class RateLimiter:
    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        time.sleep(max(0.0, t - now))


class GeminiLabeler:
    def __init__(self, cfg: dict, api_key: str):
        from google import genai

        self.cfg = cfg["labelling"]
        self.client = genai.Client(api_key=api_key)
        self.limiter = RateLimiter(self.cfg["requests_per_minute"])

    def available_models(self) -> list[str]:
        return [m.name.removeprefix("models/") for m in self.client.models.list()]

    def label(self, wav_bytes: bytes) -> dict:
        from google.genai import types

        kw = dict(temperature=0, response_mime_type="application/json", response_schema=_SCHEMA)
        if self.cfg.get("thinking_budget") is not None:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=self.cfg["thinking_budget"])
        last = None
        for attempt in range(self.cfg["max_retries"]):
            self.limiter.wait()
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg["model"],
                    contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), self.cfg["prompt"]],
                    config=types.GenerateContentConfig(**kw),
                )
                d = json.loads(resp.text)
                lab = str(d["label"]).upper()
                if lab not in LABELS:
                    raise ValueError(f"bad label {lab}")
                return {"label": lab, "confidence": float(d["confidence"])}
            except Exception as e:  # rate limits, 5xx, malformed JSON: back off and retry
                last = e
                code = getattr(e, "code", None)
                if code is not None and 400 <= int(code) < 500 and int(code) != 429:
                    break  # bad request / auth: retrying will not help
                time.sleep(min(60, 2 ** attempt) + random.random())
        raise RuntimeError(f"Gemini failed: {last!r}")

    def label_many(self, items, cache_path: str, progress_every: int = 200) -> dict:
        """items: iterable of (clip_id, wav_bytes). Appends answers to cache_path as they arrive."""
        done = read_cache(cache_path)
        todo = [(cid, b) for cid, b in items if cid not in done]
        n_ok = n_err = 0
        errors = []
        with ThreadPoolExecutor(self.cfg["workers"]) as ex, open(cache_path, "a") as f:
            futs = {ex.submit(self.label, b): cid for cid, b in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                cid = futs[fut]
                try:
                    r = fut.result()
                    rec = {"clip_id": cid, **r, "model": self.cfg["model"], "ts": time.time()}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done[cid] = rec
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    errors.append(repr(e)[:300])
                if i % progress_every == 0:
                    os.fsync(f.fileno())
                    print(f"  {i}/{len(todo)} ok={n_ok} err={n_err}")
        return {"answered_now": n_ok, "errors": n_err, "error_examples": errors[:3], "cached_total": len(done)}


def decide(position_label: int, llm_label: str | None, llm_conf: float | None, threshold: float) -> tuple:
    """(final_label or None, reason). Keep when the LLM agrees with the position rule, or when it
    disagrees but is confident (then the LLM wins). Drop UNSURE and unconfident disagreements."""
    if llm_label is None:
        return None, "unlabelled"
    if llm_label == "UNSURE":
        return None, "llm_unsure"
    llm_bin = 1 if llm_label == "COMPLETE" else 0
    if llm_bin == position_label:
        return llm_bin, "agree"
    if llm_conf is not None and llm_conf >= threshold:
        return llm_bin, "llm_confident_override"
    return None, "disagree_unconfident"
