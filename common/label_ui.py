"""Colab listening/labelling UI (ipywidgets). Every click is appended to a CSV on Drive at once.

Keyboard: click in the small "keys" box once, then press  c = COMPLETE, i = INCOMPLETE,
u = UNSURE, r = replay, b = back (undo the last answer).
"""
from __future__ import annotations

import csv
import os
import time
from typing import Callable

import numpy as np

FIELDS = ["clip_id", "annotator", "label", "ts"]


def load_answers(csv_path: str, annotator: str | None = None) -> dict[str, str]:
    """Latest answer per clip (later rows override earlier ones, so 'back' + re-answer works)."""
    out = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                if annotator is None or r["annotator"] == annotator:
                    if r["label"] == "UNDO":
                        out.pop(r["clip_id"], None)
                    else:
                        out[r["clip_id"]] = r["label"]
    return out


class LabelUI:
    def __init__(self, clip_ids: list[str], get_audio: Callable[[str], np.ndarray], csv_path: str,
                 annotator: str, info: Callable[[str], str] | None = None, sr: int = 16000):
        self.ids, self.get_audio, self.csv_path = list(clip_ids), get_audio, csv_path
        self.annotator, self.info, self.sr = annotator, info, sr
        self.answers = load_answers(csv_path, annotator)
        self.history: list[str] = []
        self.pos = self._next_unlabelled(0)
        self._build()

    def _next_unlabelled(self, start: int) -> int:
        for i in range(start, len(self.ids)):
            if self.ids[i] not in self.answers:
                return i
        return len(self.ids)

    def _append(self, clip_id: str, label: str) -> None:
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({"clip_id": clip_id, "annotator": self.annotator, "label": label, "ts": time.time()})
            f.flush()
            os.fsync(f.fileno())

    def _build(self):
        import ipywidgets as W
        from IPython.display import display

        self.W = W
        self.header = W.HTML()
        self.player = W.Output()
        btn = lambda d, s: W.Button(description=d, button_style=s, layout=W.Layout(width="150px"))
        self.b_c, self.b_i, self.b_u = btn("COMPLETE (c)", "success"), btn("INCOMPLETE (i)", "danger"), btn("UNSURE (u)", "warning")
        self.b_r, self.b_b = btn("replay (r)", ""), btn("back (b)", "")
        self.keys = W.Text(placeholder="click here, then use keys", layout=W.Layout(width="220px"))
        self.b_c.on_click(lambda _: self.answer("COMPLETE"))
        self.b_i.on_click(lambda _: self.answer("INCOMPLETE"))
        self.b_u.on_click(lambda _: self.answer("UNSURE"))
        self.b_r.on_click(lambda _: self.show())
        self.b_b.on_click(lambda _: self.back())
        self.keys.observe(self._on_key, names="value")
        display(W.VBox([self.header, self.player,
                        W.HBox([self.b_c, self.b_i, self.b_u]), W.HBox([self.b_r, self.b_b, self.keys])]))
        self.show()

    def _on_key(self, change):
        v = (change["new"] or "")[-1:].lower()
        if not v:
            return
        self.keys.value = ""
        {"c": lambda: self.answer("COMPLETE"), "i": lambda: self.answer("INCOMPLETE"),
         "u": lambda: self.answer("UNSURE"), "r": self.show, "b": self.back}.get(v, lambda: None)()

    def show(self):
        from IPython.display import Audio, display

        done = len([i for i in self.ids if i in self.answers])
        if self.pos >= len(self.ids):
            self.header.value = f"<b>All {len(self.ids)} clips labelled. Saved to {self.csv_path}</b>"
            self.player.clear_output()
            return
        cid = self.ids[self.pos]
        extra = f" | {self.info(cid)}" if self.info else ""
        self.header.value = (f"<b>{done}/{len(self.ids)} done</b> | clip {self.pos + 1}: <code>{cid}</code>{extra}"
                             "<br>Did the speaker <b>finish their thought</b> at the end of the clip?")
        with self.player:
            self.player.clear_output(wait=True)
            display(Audio(self.get_audio(cid), rate=self.sr, autoplay=True))

    def answer(self, label: str):
        if self.pos >= len(self.ids):
            return
        cid = self.ids[self.pos]
        self._append(cid, label)
        self.answers[cid] = label
        self.history.append(cid)
        self.pos = self._next_unlabelled(self.pos + 1)
        self.show()

    def back(self):
        if not self.history:
            return
        cid = self.history.pop()
        self._append(cid, "UNDO")
        self.answers.pop(cid, None)
        self.pos = self.ids.index(cid)
        self.show()
