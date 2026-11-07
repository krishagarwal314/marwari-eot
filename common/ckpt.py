"""Crash-safe checkpointing and metric logging for Colab sessions that die without warning.

Run directory on Drive: checkpoints/<run_name>/
  step_00000500.pt ...   last ``keep_last`` periodic checkpoints
  last.pt                newest checkpoint (what resume loads)
  best.pt                best by dev balanced accuracy
  metrics.csv            fallback log (W&B may be unavailable)
  wandb_id.txt           so the W&B run resumes instead of starting a new one
Every file is written to ``<name>.tmp`` and then renamed, so a disconnect never leaves a
half-written last.pt.
"""
from __future__ import annotations

import csv
import os
import random
import re
import shutil

import numpy as np
import torch


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, run_dir: str, keep_last: int = 3):
        self.run_dir = run_dir
        self.keep_last = keep_last
        os.makedirs(run_dir, exist_ok=True)

    @property
    def last_path(self) -> str:
        return os.path.join(self.run_dir, "last.pt")

    @property
    def best_path(self) -> str:
        return os.path.join(self.run_dir, "best.pt")

    def save(self, *, model, optimizer, scheduler, scaler, epoch: int, global_step: int,
             best_score: float, is_best: bool = False, extra: dict | None = None) -> str:
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_score": best_score,
            "rng": rng_state(),
            "extra": extra or {},
        }
        step_path = os.path.join(self.run_dir, f"step_{global_step:08d}.pt")
        _atomic_save(state, step_path)
        shutil.copyfile(step_path, self.last_path + ".tmp")
        os.replace(self.last_path + ".tmp", self.last_path)
        if is_best:
            shutil.copyfile(step_path, self.best_path + ".tmp")
            os.replace(self.best_path + ".tmp", self.best_path)
        self._prune()
        return step_path

    def _prune(self) -> None:
        steps = sorted(f for f in os.listdir(self.run_dir) if re.fullmatch(r"step_\d{8}\.pt", f))
        for f in steps[: -self.keep_last]:
            os.remove(os.path.join(self.run_dir, f))
        for f in os.listdir(self.run_dir):  # leftovers from an interrupted save
            if f.endswith(".tmp"):
                os.remove(os.path.join(self.run_dir, f))

    def has_checkpoint(self) -> bool:
        return os.path.exists(self.last_path)

    def resume(self, *, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> dict | None:
        """Load last.pt into the given objects. Returns the checkpoint dict, or None if fresh start."""
        if not self.has_checkpoint():
            return None
        ck = torch.load(self.last_path, map_location=map_location, weights_only=False)
        model.load_state_dict(ck["model"])
        if optimizer is not None:
            optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and ck["scheduler"] is not None:
            scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        return ck


class CSVLog:
    """Append-only metrics CSV on Drive. Handles new columns appearing later in a run."""

    def __init__(self, path: str):
        self.path = path

    def log(self, row: dict) -> None:
        rows, fields = [], list(row.keys())
        if os.path.exists(self.path):
            with open(self.path, newline="") as f:
                r = csv.DictReader(f)
                header = list(r.fieldnames or [])
                if set(row) <= set(header):  # common case: plain append
                    with open(self.path, "a", newline="") as fa:
                        csv.DictWriter(fa, fieldnames=header, restval="").writerow(row)
                    return
                rows = list(r)
                fields = header + [k for k in row if k not in header]
        rows.append(row)
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.path)


def wandb_init(cfg: dict, run_dir: str, run_name: str, config: dict):
    """Resumable W&B run (id persisted in the run dir). Returns None if W&B is not configured."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    id_path = os.path.join(run_dir, "wandb_id.txt")
    if os.path.exists(id_path):
        run_id = open(id_path).read().strip()
    else:
        run_id = wandb.util.generate_id()
        with open(id_path, "w") as f:
            f.write(run_id)
    return wandb.init(project=cfg["logging"]["wandb_project"], entity=cfg["logging"]["wandb_entity"],
                      name=run_name, id=run_id, resume="allow", config=config)
