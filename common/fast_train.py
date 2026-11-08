"""Fast track: features, replay sampling, fine-tuning with replay, evaluation, ONNX export.

Everything big lives on local disk / in RAM (features are recomputed if the session dies);
only checkpoints, the final ONNX files and the results go to Drive.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import audio as A
from . import features as F


# --- features --------------------------------------------------------------------------------

def _feat_chunk(args):
    wav_list = args
    return F.extract_batch(wav_list).astype(np.float16)


def features_parallel(wavs: list[np.ndarray], workers: int, chunk: int = 64) -> np.ndarray:
    """(N, 80, 800) fp16, same computation as Smart Turn, in parallel over CPU cores."""
    chunks = [wavs[i:i + chunk] for i in range(0, len(wavs), chunk)]
    if not chunks:
        return np.zeros((0, 80, 800), np.float16)
    with ProcessPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(_feat_chunk, chunks)))


def telephony_safe(wav: np.ndarray, seed_float: float) -> np.ndarray:
    """Telephony augmentation; falls back to mu-law if ffmpeg lacks AMR/Opus."""
    rng = np.random.default_rng(int(seed_float * 2 ** 32))
    try:
        y, _ = A.telephony(wav, rng)
    except Exception:
        y, _ = A.telephony(wav, rng, codec="mulaw")
    return y


# --- replay data (Smart Turn v3.2) -----------------------------------------------------------

def sample_smart_turn(cfg: dict, which: str, per_lang: int, local_dir: str, max_files: int,
                      seed: int, workers: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Language-balanced sample from pipecat-ai/smart-turn-data-v3.2-{train,test}.

    Reads parquet files one at a time (downloaded to local disk, deleted after use) until every
    language has ``per_lang`` clips or ``max_files`` files were read. Features are computed per
    file so raw audio never piles up in RAM. Returns (features fp16, labels, languages, ids).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo, rev = cfg["hf"][f"data_{which}"], cfg["hf"][f"data_{which}_revision"]
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
                   if f.startswith("data/") and f.endswith(".parquet"))
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))][:max_files]
    count: dict[str, int] = {}
    feats, labels, langs, ids = [], [], [], []
    for fi, f in enumerate(files):
        path = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=local_dir)
        wavs = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["id", "audio", "language", "endpoint_bool"]):
            for r in batch.to_pylist():
                lang = r["language"]
                if count.get(lang, 0) >= per_lang:
                    continue
                wavs.append(A.decode_hf_audio(r["audio"]))
                labels.append(int(bool(r["endpoint_bool"])))
                langs.append(lang)
                ids.append(r["id"])
                count[lang] = count.get(lang, 0) + 1
        os.remove(path)
        feats.append(features_parallel(wavs, workers))
        del wavs
        full = sum(v >= per_lang for v in count.values())
        print(f"  {which} file {fi + 1}/{len(files)}: {len(labels)} clips, {full}/{len(count)} languages full")
        if count and full == len(count) and fi >= 1:
            break
    return np.concatenate(feats), np.asarray(labels, np.int64), langs, ids


# --- training --------------------------------------------------------------------------------

def predict(model, X: np.ndarray, device: str, amp_dtype, batch: int = 256) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(device)
            with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
                p = model(x)["logits"]
            out.append(p.float().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


def train(model, mar_X, mar_y, rep_X, rep_y, rep_lang, dev_X, dev_y, rt, tcfg: dict, ckpt, log) -> dict:
    """Mixed-batch fine-tuning: each batch is share m Marwari, 1-m replay (languages sampled evenly).
    AdamW with separate LRs for encoder/head, warm-up + cosine, early stopping on Marwari dev
    balanced accuracy. Resumes from ckpt (CheckpointManager) if a checkpoint exists."""
    import torch

    from models.smart_turn import encoder_layer_groups
    from . import metrics as M

    dev = rt.device
    B, m = tcfg["batch_size"], tcfg["marwari_share"]
    nm = max(1, round(B * m)) if m > 0 else 0
    nr = B - nm
    steps = int(math.ceil(tcfg["epochs"] * len(mar_y) / max(1, nm)))
    groups = encoder_layer_groups(model)
    opt = torch.optim.AdamW([{"params": groups["encoder"], "lr": tcfg["lr_encoder"]},
                             {"params": groups["head"], "lr": tcfg["lr_head"]}],
                            weight_decay=tcfg["weight_decay"])
    warm = max(1, int(steps * tcfg["warmup_frac"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=rt.use_grad_scaler)
    model.to(dev)

    lang_idx = {}
    for i, l in enumerate(rep_lang):
        lang_idx.setdefault(l, []).append(i)
    lang_keys = sorted(lang_idx)
    lang_arr = {k: np.array(v) for k, v in lang_idx.items()}

    rng = np.random.default_rng(tcfg["seed"])
    start, best, bad = 0, -1.0, 0
    ck = ckpt.resume(model=model, optimizer=opt, scheduler=sched, scaler=scaler)
    if ck is not None:
        start, best = ck["global_step"], ck["best_score"]
        bad = ck["extra"].get("bad", 0)
        rng.bit_generator.state = ck["extra"]["np_rng"]
        print(f"resumed at step {start}, best dev bal-acc {best:.4f}")
        if ck["extra"].get("stopped"):
            return {"steps": start, "best_dev_bacc": best, "resumed_finished": True}

    mar_y_arr, rep_y_arr = np.asarray(mar_y), np.asarray(rep_y)
    t0 = time.time()
    stopped = False
    s = start
    for step in range(start, steps):
        model.train()
        im = rng.integers(0, len(mar_y_arr), nm)
        if nr:
            langs = rng.choice(len(lang_keys), nr)
            ir = np.array([rng.choice(lang_arr[lang_keys[k]]) for k in langs])
            x = np.concatenate([mar_X[im], rep_X[ir]])
            y = np.concatenate([mar_y_arr[im], rep_y_arr[ir]])
        else:
            x, y = mar_X[im], mar_y_arr[im]
        xt = torch.from_numpy(x.astype(np.float32)).to(dev, non_blocking=True)
        yt = torch.from_numpy(y).to(dev)
        with torch.autocast("cuda", dtype=rt.amp_dtype, enabled=rt.amp_dtype is not None):
            loss = model(xt, labels=yt)["loss"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        s = step + 1
        if s % tcfg["eval_every"] == 0 or s == steps:
            p = predict(model, dev_X, dev, rt.amp_dtype)
            bacc = M.binary_metrics(dev_y, p)["balanced_accuracy"]
            improved = bacc > best
            if improved:
                best, bad = bacc, 0
            else:
                bad += 1
            stopped = bad >= tcfg["patience"]
            ckpt.save(model=model, optimizer=opt, scheduler=sched, scaler=scaler, epoch=0, global_step=s,
                      best_score=best, is_best=improved,
                      extra={"np_rng": rng.bit_generator.state, "bad": bad, "stopped": stopped})
            row = {"step": s, "loss": float(loss), "dev_bacc": round(bacc, 4), "best": round(best, 4),
                   "lr_enc": opt.param_groups[0]["lr"], "min": round((time.time() - t0) / 60, 1)}
            log.log(row)
            print(row)
            if stopped:
                print(f"early stop: no dev improvement for {tcfg['patience']} evals")
                break
    return {"steps": s, "best_dev_bacc": best, "planned_steps": steps}


# --- ONNX ------------------------------------------------------------------------------------

def export_onnx(model, path: str, opset: int = 18) -> str:
    """Same contract as Smart Turn: input 'input_features' (B,80,800) -> output 'logits' (B,1) probs."""
    import torch

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_features):
            p = self.inner(input_features)["logits"]
            return p.reshape(p.shape[0], 1)

    w = Wrap(model).eval().cpu().float()
    x = torch.randn(2, 80, 800)
    try:
        from torch.export import Dim

        torch.onnx.export(w, (x,), path, input_names=["input_features"], output_names=["logits"],
                          opset_version=opset, dynamic_shapes={"input_features": {0: Dim.DYNAMIC}},
                          external_data=False, dynamo=True)
    except Exception as e:
        print("dynamo export failed, using legacy exporter:", repr(e)[:200])
        torch.onnx.export(w, (x,), path, input_names=["input_features"], output_names=["logits"],
                          opset_version=17, dynamic_axes={"input_features": {0: "batch"}, "logits": {0: "batch"}},
                          dynamo=False)
    return path


def onnx_predict(path: str, X: np.ndarray, threads: int = 0, batch: int = 64) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    out = [sess.run(None, {"input_features": X[i:i + batch].astype(np.float32)})[0].reshape(-1)
           for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros(0)


def quantize_int8(fp32_path: str, out_path: str) -> str:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # The dynamo exporter writes intermediate shape annotations (value_info) that onnx shape
    # inference later contradicts ("Inferred shape and existing shape differ"). They are optional,
    # so drop them before quantising.
    m = onnx.load(fp32_path)
    del m.graph.value_info[:]
    clean = fp32_path.replace(".onnx", ".novi.onnx")
    onnx.save(m, clean)
    try:
        quantize_dynamic(clean, out_path, weight_type=QuantType.QInt8)
    finally:
        os.remove(clean)
    return out_path
