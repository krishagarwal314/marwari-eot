# %% [markdown]
# # 04 (fast) · Baseline -> fine-tune -> regression check -> ONNX
# One notebook for the rest of the fast track. **GPU runtime (T4 is fine).**
# 1. Features for the Gemini-labelled Marwari clips (train clips flagged for telephony get the
#    8 kHz phone simulation) and a language-balanced replay sample of Smart Turn's own data.
# 2. **Baseline:** stock Smart Turn v3.2 on Marwari dev/test and on the replay test sample.
# 3. **Fine-tune** (half Marwari, half replay per batch), early stopping on Marwari dev.
# 4. **Compare** fine-tuned vs stock: Marwari test (with 95% CI by speaker) and every language.
# 5. **Export** fp32 + int8 ONNX (same input/output as Smart Turn) and re-test the int8 file.
#
# If the session dies: run it again. Training resumes from its last checkpoint on Drive;
# features are recomputed (~10-20 min).

# %% [bootstrap]

# %%
import os, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from common.colab_setup import setup
from common import audio as A, metrics as M, fast_train as FT
from common.ckpt import CheckpointManager, CSVLog
from models.load_pretrained import load_smart_turn_v32

rt = setup()
cfg, T = rt.cfg, rt.cfg["fast_train"]
assert rt.device == "cuda", "Switch to a GPU runtime (Runtime > Change runtime type > T4 GPU)."
CACHE = rt.local("fast_feats")
CLIPS = os.path.join(rt.root, cfg["dirs"]["clips"], "fast_v1")
lab = pd.read_parquet(rt.path("labels", "fast_labels_v1.parquet"))
ds = lab[lab.in_dataset].copy()
ds["final_label"] = ds.final_label.astype(int)
print(ds.groupby("split").final_label.agg(n="size", complete_frac="mean").round(3))

# %% [markdown]
# ## 1. Features

# %%
mar_path = os.path.join(CACHE, "marwari.npz")
if os.path.exists(mar_path):
    z = np.load(mar_path, allow_pickle=True)
    MX, mids = z["X"], list(z["ids"])
else:
    t0 = time.time()
    feats, mids = [], []
    tel = dict(zip(ds.clip_id, ds.telephony))
    for shard, g in ds.groupby("clip_shard"):
        t = pq.read_table(os.path.join(CLIPS, shard), columns=["clip_id", "flac"]).to_pandas()
        t = t[t.clip_id.isin(set(g.clip_id))].drop_duplicates("clip_id")
        wavs = []
        for cid, fl in zip(t.clip_id, t.flac):
            w = A.decode_bytes(fl)
            if tel.get(cid):
                w = FT.telephony_safe(w, float(int(cid[:8], 16)) / 16 ** 8)
            wavs.append(w)
        feats.append(FT.features_parallel(wavs, rt.num_workers))
        mids += list(t.clip_id)
    MX = np.concatenate(feats)
    np.savez(mar_path, X=MX, ids=np.array(mids))
    print(f"Marwari features {MX.shape} in {(time.time()-t0)/60:.1f} min")
pos = {c: i for i, c in enumerate(mids)}
def subset(split):
    d = ds[(ds.split == split) & ds.clip_id.isin(pos)]
    return MX[[pos[c] for c in d.clip_id]], d.final_label.to_numpy(), d
trX, trY, trD = subset("train")
dvX, dvY, dvD = subset("dev")
teX, teY, teD = subset("test")
print("train", trX.shape, "dev", dvX.shape, "test", teX.shape)

# %%
rep_path = os.path.join(CACHE, "replay.npz")
if os.path.exists(rep_path):
    z = np.load(rep_path, allow_pickle=True)
    rX, rY, rL, qX, qY, qL = z["rX"], z["rY"], list(z["rL"]), z["qX"], z["qY"], list(z["qL"])
else:
    dl = rt.local("st_download")
    rX, rY, rL, _ = FT.sample_smart_turn(cfg, "train", T["replay_train_per_lang"], dl, T["replay_max_files"], cfg["seed"], rt.num_workers)
    qX, qY, qL, _ = FT.sample_smart_turn(cfg, "test", T["replay_test_per_lang"], dl, 10, cfg["seed"], rt.num_workers)
    np.savez(rep_path, rX=rX, rY=rY, rL=np.array(rL), qX=qX, qY=qY, qL=np.array(qL))
# 10% of the replay sample (fixed) is replay dev; the rest is replay train
rng = np.random.default_rng(cfg["seed"])
is_dev = rng.random(len(rY)) < 0.1
print("replay train", (~is_dev).sum(), "| replay dev", is_dev.sum(), "| replay test", len(qY),
      "| languages", len(set(qL)))

# %% [markdown]
# ## 2. Baseline: stock Smart Turn v3.2

# %%
def evaluate(model, tag):
    res = {}
    p = FT.predict(model, teX, rt.device, rt.amp_dtype)
    res["marwari_test"] = M.report(teY, p, n_boot=1000, groups=teD.speakerID.to_numpy())
    res["marwari_dev"] = M.binary_metrics(dvY, FT.predict(model, dvX, rt.device, rt.amp_dtype))
    pq_ = FT.predict(model, qX, rt.device, rt.amp_dtype)
    res["replay_test_overall"] = M.binary_metrics(qY, pq_)
    res["replay_test_by_lang"] = {k: v["accuracy"] for k, v in M.per_group(qY, pq_, np.array(qL)).items()}
    print(f"[{tag}] Marwari test bal-acc {res['marwari_test']['balanced_accuracy']:.3f} "
          f"(95% CI {res['marwari_test']['balanced_accuracy_ci_lo']:.3f}-{res['marwari_test']['balanced_accuracy_ci_hi']:.3f}), "
          f"replay acc {res['replay_test_overall']['accuracy']:.3f}")
    return res, p

stock, stock_onnx = load_smart_turn_v32(cfg)
stock = stock.to(rt.device)
base, base_p = evaluate(stock, "stock v3.2")
del stock; torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Fine-tune

# %%
run_dir = os.path.dirname(rt.path("checkpoints", T["run_name"], "x"))
ck = CheckpointManager(run_dir, keep_last=cfg["checkpoint"]["keep_last"])
log = CSVLog(os.path.join(run_dir, "metrics.csv"))
model, _ = load_smart_turn_v32(cfg)
torch.manual_seed(T["seed"])
info = FT.train(model, trX, trY, rX[~is_dev], rY[~is_dev], [l for l, d in zip(rL, is_dev) if not d],
                dvX, dvY, rt, T, ck, log)
print(info)
best = torch.load(ck.best_path, map_location="cpu", weights_only=False)
model.load_state_dict(best["model"]); model.to(rt.device)
p_rdev = FT.predict(model, rX[is_dev], rt.device, rt.amp_dtype)
print("replay dev acc after fine-tune:", round(M.binary_metrics(rY[is_dev], p_rdev)["accuracy"], 3))

# %% [markdown]
# ## 4. Fine-tuned vs stock

# %%
ft, ft_p = evaluate(model, "fine-tuned")
langs = sorted(base["replay_test_by_lang"])
rows = [{"set": "MARWARI test (bal-acc)", "stock": base["marwari_test"]["balanced_accuracy"],
         "fine_tuned": ft["marwari_test"]["balanced_accuracy"]}]
rows += [{"set": l, "stock": base["replay_test_by_lang"][l], "fine_tuned": ft["replay_test_by_lang"][l]} for l in langs]
tab = pd.DataFrame(rows)
tab["change_pts"] = ((tab.fine_tuned - tab.stock) * 100).round(1)
print(tab.round(3).to_string(index=False))
gain = (ft["marwari_test"]["balanced_accuracy"] - base["marwari_test"]["balanced_accuracy"]) * 100
worst = tab[tab.set != "MARWARI test (bal-acc)"].change_pts.min()
print(f"\nMarwari gain: {gain:+.1f} points | worst other-language change: {worst:+.1f} points "
      f"(each language has ~{T['replay_test_per_lang']} test clips, so ±2-3 points is noise)")

# %% [markdown]
# ## 5. Export ONNX (fp32 + int8) and re-test int8

# %%
out_dir = os.path.dirname(rt.path("models", T["run_name"], "x"))
os.makedirs(out_dir, exist_ok=True)
fp32 = FT.export_onnx(model, os.path.join(out_dir, "smart-turn-v3.2-marwari-fp32.onnx"))
int8 = FT.quantize_int8(fp32, os.path.join(out_dir, "smart-turn-v3.2-marwari-int8.onnx"))
onnx_res = {}
for name, path in [("fp32", fp32), ("int8", int8)]:
    p_mt = FT.onnx_predict(path, teX)
    p_rt = FT.onnx_predict(path, qX)
    sess = M.onnx_cpu_session(path, threads=1)
    lat = M.latency_ms(lambda: sess.run(None, {"input_features": teX[:1].astype(np.float32)}), n=100, warmup=10)
    onnx_res[name] = {"marwari_test_bacc": M.binary_metrics(teY, p_mt)["balanced_accuracy"],
                      "replay_test_acc": M.binary_metrics(qY, p_rt)["accuracy"],
                      "max_diff_vs_torch_marwari": float(np.abs(p_mt - ft_p).max()), **lat,
                      "size_mb": round(os.path.getsize(path) / 1e6, 1)}
print(json.dumps(onnx_res, indent=1))

# %%
results = {"stock": base, "fine_tuned": ft, "training": info, "onnx": onnx_res,
           "marwari_gain_points": round(gain, 2), "worst_language_change_points": float(worst),
           "n": {"marwari_train": int(len(trY)), "marwari_dev": int(len(dvY)), "marwari_test": int(len(teY)),
                 "marwari_test_speakers": int(teD.speakerID.nunique()), "replay_train": int((~is_dev).sum()),
                 "replay_test": int(len(qY))},
           "labels": "Gemini Flash Lite (silver), not human-verified"}
with open(rt.path("reports", "fast_results.json"), "w") as f:
    json.dump(results, f, indent=1, default=str)
with open(rt.path("reports", "fast_results.md"), "w") as f:
    f.write(tab.round(3).to_string(index=False))
print(json.dumps({k: results[k] for k in ["marwari_gain_points", "worst_language_change_points", "n", "onnx"]}, indent=1, default=str))
print("\nMarwari test, stock:", {k: round(v, 3) for k, v in base["marwari_test"].items() if isinstance(v, float)})
print("Marwari test, fine-tuned:", {k: round(v, 3) for k, v in ft["marwari_test"].items() if isinstance(v, float)})
