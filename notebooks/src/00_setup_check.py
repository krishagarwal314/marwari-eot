# %% [markdown]
# # 00 · Setup check
# Verifies everything Phase 0 needs: Drive, GPU and speed settings, Colab Secrets (HF, Gemini, W&B),
# access to the Smart Turn v3.2 datasets and to Vaani, and whether the released Smart Turn v3.2
# weights can be fine-tuned in PyTorch (**DECISION** evidence).
# Safe to re-run; it only reads data and writes `reports/phase0_env_check.json` on Drive.
#
# Runtime: any GPU runtime. Nothing heavy runs here (a few thousand streamed clips at most).

# %%
# Bootstrap. The repo is private: put a GitHub token (repo read access) in Colab Secrets as
# GH_TOKEN. Set REPO_URL = None to copy the code from Drive instead (MyDrive/marwari-eot/code).
REPO_URL = "https://github.com/krishagarwal314/marwari-eot.git"
REPO_DIR = "/content/marwari-eot"
DRIVE_CODE = "/content/drive/MyDrive/marwari-eot/code"

import os, subprocess, sys
if REPO_URL:
    url = REPO_URL
    try:
        from google.colab import userdata
        tok = userdata.get("GH_TOKEN")
        url = REPO_URL.replace("https://", f"https://{tok}@")
    except Exception:
        pass
    if not os.path.isdir(f"{REPO_DIR}/.git"):
        subprocess.run(["git", "clone", "-q", url, REPO_DIR], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull", "-q", "--ff-only"], check=False)
else:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run(["rsync", "-a", "--delete", f"{DRIVE_CODE}/", f"{REPO_DIR}/"], check=True)
sys.path.insert(0, REPO_DIR)
%pip install -q -r {REPO_DIR}/requirements.txt

# %%
from common.colab_setup import setup
rt = setup()
cfg = rt.cfg
checks = {"gpu": rt.gpu_name, "amp_dtype": str(rt.amp_dtype), "grad_scaler": rt.use_grad_scaler,
          "num_workers": rt.num_workers, "missing_secrets": rt.missing_secrets}

# %% [markdown]
# ## Secrets
# Each one is tested with a small API call, not just looked up.

# %%
import os
from huggingface_hub import HfApi

try:
    who = HfApi().whoami(token=os.environ.get("HF_TOKEN"))
    checks["hf_user"] = who["name"]
    print("HF token OK:", who["name"])
except Exception as e:
    checks["hf_user"] = None
    print("HF token FAILED:", e)

try:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    flash = [m.name for m in client.models.list() if "flash" in m.name]
    checks["gemini_flash_models"] = flash
    print("Gemini key OK. Flash models:", flash)
except Exception as e:
    checks["gemini_flash_models"] = None
    print("Gemini key FAILED:", e)

checks["wandb"] = bool(os.environ.get("WANDB_API_KEY"))
print("W&B key present:", checks["wandb"], "(optional; CSV logging on Drive is used either way)")

# %% [markdown]
# ## GPU: largest batch that fits (Whisper-tiny Smart Turn, one train step)

# %%
import torch
from common.colab_setup import find_max_batch, plan_batch
from models.smart_turn import build_from_whisper

if rt.device == "cuda":
    model = build_from_whisper(cfg["hf"]["whisper"]["tiny"]).to("cuda").train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def step(bs):
        x = torch.randn(bs, cfg["audio"]["n_mels"], cfg["audio"]["n_frames"], device="cuda")
        y = torch.randint(0, 2, (bs,), device="cuda")
        with torch.autocast("cuda", dtype=rt.amp_dtype):
            loss = model(x, labels=y)["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    max_bs = find_max_batch(step, cfg["runtime"]["batch_probe_start"])
    micro, accum = plan_batch(max_bs, cfg["runtime"]["batch_headroom"], cfg["runtime"]["effective_batch"])
    checks.update(max_batch_tiny=max_bs, micro_batch_tiny=micro, grad_accum_tiny=accum)
    print(f"max batch {max_bs} -> micro {micro} x accum {accum} = {micro*accum}")
    del model, opt
    torch.cuda.empty_cache()
else:
    print("No GPU: switch the runtime to GPU (A100 or L4 preferred).")

# %% [markdown]
# ## Smart Turn v3.2 datasets: load and print the real schema

# %%
from datasets import load_dataset, Audio
from common import audio as A

for key in ["data_train", "data_test"]:
    name, rev = cfg["hf"][key], cfg["hf"][key + "_revision"]
    ds = load_dataset(name, split="train", streaming=True, revision=rev).cast_column("audio", Audio(decode=False))
    print(f"\n== {name}@{rev[:8]}\n{ds.features}")
    for i, row in enumerate(ds.take(3)):
        wav = A.decode_hf_audio(row["audio"])
        print({k: v for k, v in row.items() if k != "audio"}, f"| {len(wav)/16000:.2f}s")
    checks[f"{key}_columns"] = list(ds.features)

# %% [markdown]
# ## Vaani: gated access, configs, schema
# Only checks access and prints what is there. Phase 1 code is written against this output.

# %%
from datasets import get_dataset_config_names

try:
    vaani_cfgs = get_dataset_config_names(cfg["hf"]["vaani"])
    RAJ_NAMES = ("marwari", "marwadi", "rajasthan", "mewari", "dhundhari", "jaipuri", "shekhawati", "harauti", "bagri", "wagdi")
    raj = [c for c in vaani_cfgs if any(n in c.lower() for n in RAJ_NAMES)]
    checks["vaani_n_configs"], checks["vaani_rajasthan_configs"] = len(vaani_cfgs), raj
    print(len(vaani_cfgs), "configs. Rajasthan:", raj or "(none matched; first 50 below)")
    if not raj:
        print(vaani_cfgs[:50])
    probe = raj[0] if raj else vaani_cfgs[0]
    vds = load_dataset(cfg["hf"]["vaani"], probe, split="train", streaming=True)
    if "audio" in vds.features:
        vds = vds.cast_column("audio", Audio(decode=False))
    print(f"\n== schema of config {probe}\n{vds.features}")
    row = next(iter(vds))
    print({k: v for k, v in row.items() if k != "audio"})
    checks["vaani_columns"] = list(vds.features)
except Exception as e:
    checks["vaani_error"] = repr(e)
    print("Vaani FAILED (have you accepted the dataset terms on huggingface.co?):", e)

# %% [markdown]
# ## DECISION evidence: can Smart Turn v3.2 be fine-tuned in PyTorch?
# 1. List the files in the model repo (PyTorch/safetensors weights?).
# 2. Copy the fp32 ONNX weights into the PyTorch `SmartTurnV3Model` (strict load) and compare it
#    with onnxruntime on real test clips. Also check that storing features as fp16 does not change
#    predictions.

# %%
import numpy as np
from huggingface_hub import list_repo_files
from common import features as F, metrics as M
from models.load_pretrained import load_smart_turn_v32, max_diff_vs_onnx, download_onnx

files = list_repo_files(cfg["hf"]["smart_turn_model_repo"], revision=cfg["hf"]["smart_turn_model_revision"])
torch_files = [f for f in files if f.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt"))]
checks["smart_turn_repo_files"], checks["smart_turn_torch_weights"] = files, torch_files
print("repo files:", files, "\nPyTorch weight files:", torch_files or "NONE (ONNX only)")

st_model, onnx_path = load_smart_turn_v32(cfg)
print("strict load OK; params:", sum(p.numel() for p in st_model.parameters()))
checks["parity_randn_maxdiff"] = max_diff_vs_onnx(st_model, onnx_path, np.random.default_rng(0).standard_normal((4, 80, 800)).astype(np.float32))

# %%
N_PARITY = 2000
test = load_dataset(cfg["hf"]["data_test"], split="train", streaming=True, revision=cfg["hf"]["data_test_revision"])
test = test.cast_column("audio", Audio(decode=False)).shuffle(seed=cfg["seed"], buffer_size=5000)

sess = M.onnx_cpu_session(onnx_path, threads=os.cpu_count())
st_model = st_model.to(rt.device).eval()
y, p_onnx, p_torch, p_fp16, langs = [], [], [], [], []
for batch in F.batched(test.take(N_PARITY), 100):
    X = F.extract_batch([A.decode_hf_audio(r["audio"]) for r in batch])
    y += [int(r["endpoint_bool"]) for r in batch]
    langs += [r["language"] for r in batch]
    p_onnx.append(sess.run(None, {"input_features": X})[0].ravel())
    with torch.no_grad():
        for arr, out in [(X, p_torch), (X.astype(np.float16).astype(np.float32), p_fp16)]:
            out.append(st_model(torch.from_numpy(arr).to(rt.device))["logits"].float().cpu().numpy().ravel())
y = np.array(y); p_onnx, p_torch, p_fp16 = map(np.concatenate, (p_onnx, p_torch, p_fp16))

res = {name: M.binary_metrics(y, p) for name, p in [("onnx_fp32", p_onnx), ("torch_fp32", p_torch), ("torch_fp16_features", p_fp16)]}
checks["parity_real"] = {
    "n": int(len(y)),
    "max_abs_diff_torch_vs_onnx": float(np.abs(p_torch - p_onnx).max()),
    "pred_flips_torch_vs_onnx": int(((p_torch > .5) != (p_onnx > .5)).sum()),
    "max_abs_diff_fp16_features": float(np.abs(p_fp16 - p_torch).max()),
    "pred_flips_fp16_features": int(((p_fp16 > .5) != (p_torch > .5)).sum()),
    "accuracy": {k: v["accuracy"] for k, v in res.items()},
    "balanced_accuracy": {k: v["balanced_accuracy"] for k, v in res.items()},
}
print(checks["parity_real"])
print("Published v3.2-gpu accuracy on the full 31,527-clip test set: 93.71% (this is a random 2,000-clip subset)")

# %%
import json, datetime
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
checks["timestamp"] = stamp
with open(rt.path("reports", "phase0_env_check.json"), "w") as f:
    json.dump(checks, f, indent=2, default=str)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
with open(rt.path("logs", f"pip_freeze_{stamp}.txt"), "w") as f:
    f.write(freeze)
print("saved reports/phase0_env_check.json and logs/pip_freeze_*.txt")
print(json.dumps({k: v for k, v in checks.items() if k != "smart_turn_repo_files"}, indent=1, default=str))
