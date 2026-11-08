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
LANGS = ("MARWARI", "HINDI", "MIXED", "OTHER")
_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "enum": list(LABELS)},
        "confidence": {"type": "NUMBER"},
        "language": {"type": "STRING", "enum": list(LANGS)},
    },
    "required": ["label", "confidence", "language"],
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
                lang = str(d.get("language", "")).upper()
                return {"label": lab, "confidence": float(d["confidence"]),
                        "language": lang if lang in LANGS else None}
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


def decide_with_text(position_label: int, llm_label: str | None, llm_conf: float | None,
                     text_label: int | None, threshold: float) -> tuple:
    """Like ``decide`` but uses the transcript cue (02b) when there is one: then the clip is kept
    only if Gemini and the text cue agree, whatever the position rule said."""
    if text_label is None or (isinstance(text_label, float) and text_label != text_label):  # None/NaN
        return decide(position_label, llm_label, llm_conf, threshold)
    if llm_label is None:
        return None, "unlabelled"
    if llm_label == "UNSURE":
        return None, "llm_unsure"
    llm_bin = 1 if llm_label == "COMPLETE" else 0
    if llm_bin == int(text_label):
        return llm_bin, "llm_text_agree"
    return None, "llm_text_disagree"


class QuotaExhausted(RuntimeError):
    """The Gemini daily quota is used up; re-run later (answers so far are cached)."""


_BATCH_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "clip": {"type": "INTEGER"},
            "label": {"type": "STRING", "enum": list(LABELS)},
            "confidence": {"type": "NUMBER"},
            "language": {"type": "STRING", "enum": list(LANGS)},
        },
        "required": ["clip", "label", "confidence", "language"],
    },
}


def _is_daily_quota(e: Exception) -> bool:
    t = repr(e).lower()
    return "perday" in t or "per day" in t or "requests_per_day" in t or "daily" in t


class BatchGeminiLabeler(GeminiLabeler):
    """Several clips per request (fewer requests -> free tier lasts much longer)."""

    def label_batch(self, wavs: list[bytes]) -> list[dict]:
        from google.genai import types

        n = len(wavs)
        contents = []
        for i, w in enumerate(wavs, 1):
            contents += [f"Clip {i}:", types.Part.from_bytes(data=w, mime_type="audio/wav")]
        contents.append(
            f"There are {n} separate clips above. Judge each one on its own. {self.cfg['prompt']} "
            f"Return a JSON array with exactly {n} objects, one per clip, using clip numbers 1 to {n}."
        )
        kw = dict(temperature=0, response_mime_type="application/json", response_schema=_BATCH_SCHEMA)
        if self.cfg.get("thinking_budget") is not None:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=self.cfg["thinking_budget"])
        last = None
        for attempt in range(self.cfg["max_retries"]):
            self.limiter.wait()
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg["model"], contents=contents, config=types.GenerateContentConfig(**kw))
                arr = json.loads(resp.text)
                by_clip = {int(d["clip"]): d for d in arr}
                if sorted(by_clip) != list(range(1, n + 1)):
                    raise ValueError(f"got clips {sorted(by_clip)[:5]}... expected 1..{n}")
                out = []
                for i in range(1, n + 1):
                    d = by_clip[i]
                    lab, lang = str(d["label"]).upper(), str(d.get("language", "")).upper()
                    out.append({"label": lab if lab in LABELS else "UNSURE", "confidence": float(d["confidence"]),
                                "language": lang if lang in LANGS else None})
                return out
            except Exception as e:
                if _is_daily_quota(e):
                    raise QuotaExhausted(repr(e)[:300]) from None
                last = e
                code = getattr(e, "code", None)
                if code is not None and 400 <= int(code) < 500 and int(code) != 429:
                    break
                time.sleep(min(90, 5 * 2 ** attempt) + random.random())
        raise RuntimeError(f"Gemini batch failed: {last!r}")

    def label_many_batched(self, items, cache_path: str, batch_size: int) -> dict:
        """items: list of (clip_id, wav_bytes). Sequential batches; each answer cached at once."""
        done = read_cache(cache_path)
        todo = [(c, b) for c, b in items if c not in done]
        n_ok = n_err = 0
        with open(cache_path, "a") as f:
            for i in range(0, len(todo), batch_size):
                chunk = todo[i:i + batch_size]
                try:
                    res = self.label_batch([b for _, b in chunk])
                except QuotaExhausted:
                    raise
                except Exception as e:
                    n_err += len(chunk)
                    print("   batch failed:", repr(e)[:200])
                    continue
                for (cid, _), r in zip(chunk, res):
                    f.write(json.dumps({"clip_id": cid, **r, "model": self.cfg["model"], "ts": time.time()}) + "\n")
                n_ok += len(chunk)
                f.flush()
                os.fsync(f.fileno())
        return {"answered_now": n_ok, "failed": n_err, "cached_total": len(done) + n_ok}
