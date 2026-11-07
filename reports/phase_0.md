# Phase 0: repo layout, environment, weights check

Status: code written; **notebook 00 has not yet been run on Colab.** It produces
`reports/phase0_env_check.json` on Drive. Paste its output here once it has run.

## What exists

| Piece | File | State |
|---|---|---|
| Config (paths, pinned HF revisions, seeds, batch/ckpt settings) | `config.yaml` | done |
| Smart Turn v3 architecture, copied unchanged from upstream | `models/smart_turn.py` | done |
| Released v3.2 weights → PyTorch | `models/load_pretrained.py` | done, parity checked (below) |
| Drive, secrets, GPU / bf16-vs-fp16, tf32, batch probe | `common/colab_setup.py` | done, needs Colab run |
| Upstream 8 s window + feature call, sharded storage with done markers | `common/features.py` | done |
| Decoding without torchcodec, telephony sim (8 kHz, μ-law/AMR/Opus, band-pass, SNR) | `common/audio.py` | done; AMR/Opus need ffmpeg codecs (checked in 02) |
| Metrics (acc, bal-acc, AUC, false-completion, false-wait, bootstrap CI by speaker, kappa, latency) | `common/metrics.py` | done, unit-checked |
| Atomic checkpoints, last-3 + best, RNG states, CSV log, resumable W&B | `common/ckpt.py` | done |
| Setup check | `notebooks/00_setup_check.ipynb` | done, needs Colab run |

Notebooks 01–10 are not written yet. Each one is written in its own phase, after the previous
notebook has printed the real schema it will depend on.

## Facts checked against the source (7 Nov 2026)

* `pipecat-ai/smart-turn-v3` @ `f766f81d` contains **ONNX files only**: v3.0, v3.1-cpu/gpu,
  v3.2-cpu/gpu. No safetensors or PyTorch weights. "cpu" = static int8 (QDQ), "gpu" = fp32.
* The v3.2 fp32 ONNX: input `input_features` (B, 80, 800), output `logits` (B, 1). Despite the
  name, the output is a **sigmoid probability**, not a logit. Opset 18, 8.0 M parameters, Whisper-tiny.
* `smart-turn-data-v3.2-train`: 270,946 clips. `-test`: 31,527 clips across 23 languages.
  Columns: `audio, id, language, endpoint_bool, midfiller, endfiller, synthetic, spoken_text, dataset`.
  There is **no speaker column**, so replay data cannot be split by speaker (fine: replay is only
  used for regression checks, and its test set is their fixed split).
* Published v3.2-gpu accuracy: 93.71% overall. Per language it ranges from Vietnamese 82.47% to
  Korean 97.64%; Hindi 92.76%, Marathi 88.11%. This is the stock baseline for Phase 5.
* Vaani is gated (`auto`), files are laid out as `audio/<State>/<District>/...parquet`.
  The schema will be printed in 00/01 before any code depends on it.

## Parity: converted PyTorch v3.2 vs official ONNX

The fp32 ONNX keeps parameter names for most tensors. The encoder and pooling Linear weights were
exported unnamed and transposed. Each one is named by following its MatMul to the bias `Add`.
`k_proj` has no bias, so it is the unnamed MatMul right after `q_proj`. The weights load with
`strict=True`.

| Input | max \|p_torch − p_onnx\| |
|---|---|
| 4 random tensors | 6e-8 |
| 8 synthetic clips through the real feature extractor | 2.7e-6 |
| 2,000 real test clips, accuracy vs published | **pending: runs in notebook 00 on Colab** |

## DECISION 1: which weights to start Phase 4 from

| Option | What it is | Pros | Cons |
|---|---|---|---|
| **(a′) Copy ONNX weights into the upstream PyTorch class** (done above) | Fine-tune the real v3.2 | Starts from the exact shipped model; exports back to the same ONNX contract; cheap | Cannot cover E5 (no released Whisper-base Smart Turn) |
| (a) onnx2torch graph conversion | As the brief described | — | Produces a graph module, not the upstream class. Harder to freeze layers or set two learning rates. Not needed now that (a′) works |
| (b) Re-run the Smart Turn recipe from Whisper-tiny/base on v3.2-train + Marwari | Train from scratch | Clean, replay built in | ~4 epochs over 271k clips per seed × 3 seeds per config. E0 may not reach 93.7% because their exact training run cannot be fully reproduced |

**Recommendation:** use (a′) for E0–E4 and E6. E0 becomes "v3.2 + replay-only fine-tuning", which
checks that the training loop does no harm. Use (b) only for E5 (Whisper-base), since no base
checkpoint exists. In that case E5 has to be compared against a tiny model trained the same way
from scratch, not against fine-tuned v3.2.

## Things Mehar needs to know now

1. **Drive storage.** Precomputed fp16 features take 128 KB per clip. The full replay train set is
   about 35 GB (fp32 would be 69 GB), plus about 4 GB for the replay test set. Free Drive (15 GB)
   is not enough. Either get a 100 GB+ Google One plan, or I extract features for a fixed replay
   subset instead (for example 60k clips, stratified by language ≈ 8 GB). Say which you prefer before Phase 4.
2. The code is at https://github.com/krishagarwal314/marwari-eot (private). For Colab to clone it,
   add a GitHub token with read access to this repo as the Colab Secret `GH_TOKEN`.
3. Run `notebooks/00_setup_check.ipynb` on an A100 or L4 runtime and send me the JSON it prints.

**Waiting for: the DECISION 1 choice, the storage choice, and the output of notebook 00.**
