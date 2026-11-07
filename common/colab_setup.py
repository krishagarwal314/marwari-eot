"""Shared first cell for every notebook: Drive, config, secrets, GPU settings, seeds.

    from common.colab_setup import setup
    rt = setup()          # rt.cfg, rt.path("features"), rt.device, rt.amp_dtype, ...
"""
from __future__ import annotations

import os
import random
import subprocess
from dataclasses import dataclass, field

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def load_config(path: str | None = None) -> dict:
    with open(path or os.path.join(REPO_ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Runtime:
    cfg: dict
    root: str
    device: str
    gpu_name: str | None
    amp_dtype: object            # torch.bfloat16 / torch.float16 / None (CPU)
    use_grad_scaler: bool
    num_workers: int
    missing_secrets: list = field(default_factory=list)

    def path(self, key: str, *parts: str) -> str:
        """Path under a Drive sub-directory from config ``dirs``. Creates the parent directory."""
        p = os.path.join(self.root, self.cfg["dirs"][key], *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def local(self, *parts: str) -> str:
        p = os.path.join(self.cfg["project"]["local_cache"], *parts)
        os.makedirs(p, exist_ok=True)
        return p


def _mount_drive() -> None:
    if in_colab() and not os.path.ismount("/content/drive"):
        from google.colab import drive

        drive.mount("/content/drive")


def _load_secrets(cfg: dict) -> list[str]:
    """Copy Colab Secrets into env vars. Returns names of missing required secrets."""
    names = cfg["secrets"]["required"] + cfg["secrets"]["optional"]
    if in_colab():
        from google.colab import userdata

        for n in names:
            if os.environ.get(n):
                continue
            try:
                v = userdata.get(n)
            except Exception:  # SecretNotFoundError / NotebookAccessError
                v = None
            if v:
                os.environ[n] = v
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    return [n for n in cfg["secrets"]["required"] if not os.environ.get(n)]


def _gpu_settings():
    import torch

    if not torch.cuda.is_available():
        return "cpu", None, None, False
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    except FileNotFoundError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    name = torch.cuda.get_device_name(0)
    # Ampere+ (A100, L4, H100: capability >= 8) -> bf16; T4/V100 -> fp16 + GradScaler.
    if torch.cuda.get_device_capability(0)[0] >= 8:
        return "cuda", name, torch.bfloat16, False
    return "cuda", name, torch.float16, True


def setup(config_path: str | None = None, mount: bool = True, verbose: bool = True) -> Runtime:
    cfg = load_config(config_path)
    if mount:
        _mount_drive()
    root = cfg["project"]["drive_root"] if in_colab() else os.environ.get(
        "MARWARI_EOT_ROOT", os.path.join(REPO_ROOT, "_local_root"))
    os.makedirs(root, exist_ok=True)
    for d in cfg["dirs"].values():
        os.makedirs(os.path.join(root, d), exist_ok=True)

    missing = _load_secrets(cfg)
    device, gpu, amp_dtype, scaler = _gpu_settings()
    seed_everything(cfg["seed"])
    nw = cfg["runtime"]["num_workers"] or os.cpu_count() or 2

    rt = Runtime(cfg, root, device, gpu, amp_dtype, scaler, nw, missing)
    if verbose:
        print(f"root={root}\ndevice={device} gpu={gpu} amp={amp_dtype} grad_scaler={scaler} "
              f"workers={nw}")
        if missing:
            print(f"MISSING required secrets: {missing} (add them in Colab > Secrets)")
    return rt


def ensure_smart_turn_repo(cfg: dict) -> str:
    """Clone pipecat-ai/smart-turn at the pinned commit (for reference / parity checks)."""
    d, commit = cfg["project"]["smart_turn_dir"], cfg["project"]["smart_turn_commit"]
    if not os.path.isdir(os.path.join(d, ".git")):
        subprocess.run(["git", "clone", "-q", cfg["project"]["smart_turn_repo_url"], d], check=True)
    subprocess.run(["git", "-C", d, "checkout", "-q", commit], check=True)
    return d


def gpu_utilization() -> float | None:
    """Current GPU utilisation in percent (nvidia-smi), or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def find_max_batch(step_fn, start: int, minimum: int = 1) -> int:
    """Largest batch size (start, start/2, ...) for which ``step_fn(bs)`` runs without CUDA OOM.

    ``step_fn`` should run one full forward+backward+optimizer step at that batch size.
    """
    import torch

    bs = start
    while bs >= minimum:
        try:
            step_fn(bs)
            torch.cuda.synchronize()
            return bs
        except torch.cuda.OutOfMemoryError:
            bs //= 2
        finally:
            torch.cuda.empty_cache()
    raise RuntimeError("Not even the minimum batch fits on this GPU")


def plan_batch(max_fit: int, headroom: float, effective: int) -> tuple[int, int]:
    """(micro_batch, grad_accum) so micro_batch*grad_accum == effective, micro <= headroom*max_fit."""
    cap = max(1, int(max_fit * headroom))
    accum = -(-effective // cap)  # ceil
    while effective % accum:
        accum += 1
    return effective // accum, accum
