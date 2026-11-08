# %% [markdown]
# # 03 (fast) · Gemini labels for the fast-track clips
# Gemini hears each clip (20 per request) and answers COMPLETE / INCOMPLETE / UNSURE, a confidence,
# and the language. Final label: keep when Gemini agrees with the position rule, or disagrees with
# confidence >= `labelling.confident_threshold`; drop UNSURE; drop clips Gemini calls Hindi.
#
# Order: **dev, then test, then train**, so the evaluation sets are labelled first. Every answer is
# cached on Drive at once. If the free daily quota runs out, the notebook stops cleanly: run it
# again later (or the next day) and it continues. Needs GEMINI_API_KEY. No GPU needed.

# %% [bootstrap]

# %%
import os, json
import pandas as pd, pyarrow.parquet as pq
from common.colab_setup import setup
from common import audio as A, clips as C
from common.labeling import BatchGeminiLabeler, QuotaExhausted, read_cache, decide

rt = setup()
cfg = rt.cfg
L = cfg["labelling"]
SEED = cfg["seed"]
CLIPS = os.path.join(rt.root, cfg["dirs"]["clips"], "fast_v1")
clips = pd.read_parquet(os.path.join(rt.root, cfg["dirs"]["clips"], "fast_v1_index.parquet"))
CACHE = os.path.join(rt.root, cfg["dirs"]["labels"], "gemini_fast_v1")
os.makedirs(CACHE, exist_ok=True)
assert os.environ.get("GEMINI_API_KEY"), "Add GEMINI_API_KEY in Colab Secrets, then re-run."
lab = BatchGeminiLabeler(cfg, os.environ["GEMINI_API_KEY"])
models = lab.available_models()
print("flash models:", [m for m in models if "flash" in m])
assert L["model"] in models, f"set labelling.model in config.yaml to one of the models above"

# %%
order = {"dev": 0, "test": 1, "train": 2}
sel = clips.assign(_o=clips.split.map(order), _h=[C.unit_hash(SEED, "label", c) for c in clips.clip_id])
sel = sel.sort_values(["_o", "_h"])
if L["max_clips"]:
    sel = sel.head(L["max_clips"])
done_ids = set()
for f in os.listdir(CACHE):
    done_ids |= set(read_cache(os.path.join(CACHE, f)))
print(f"{len(sel)} clips selected, {len(done_ids & set(sel.clip_id))} already labelled;",
      sel.split.value_counts().to_dict())

# %%
# Label in priority order. Clips are read shard by shard, but a split is finished before the next.
B = L["batch_size"]
try:
    for split in ["dev", "test", "train"]:
        part = sel[(sel.split == split) & ~sel.clip_id.isin(done_ids)]
        print(f"\n== {split}: {len(part)} to label")
        for shard, g in part.groupby("clip_shard", sort=False):
            t = pq.read_table(os.path.join(CLIPS, shard), columns=["clip_id", "flac"]).to_pandas()
            t = t[t.clip_id.isin(set(g.clip_id))]
            items = [(c, A.encode(A.decode_bytes(fl), "WAV")) for c, fl in zip(t.clip_id, t.flac)]
            r = lab.label_many_batched(items, os.path.join(CACHE, shard.removesuffix(".parquet") + ".jsonl"), B)
            print(f"  {shard[-30:]}: {r}")
except QuotaExhausted as e:
    print("\nDAILY QUOTA REACHED. Everything so far is saved. Run this notebook again later to continue.\n", e)

# %% [markdown]
# ## Final labels (works on whatever is labelled so far)

# %%
ans = {}
for f in os.listdir(CACHE):
    ans.update(read_cache(os.path.join(CACHE, f)))
df = clips.copy()
for k in ["label", "confidence", "language"]:
    df[f"llm_{k}"] = df.clip_id.map(lambda i: ans.get(i, {}).get(k))
dec = [decide(p, l, c, L["confident_threshold"]) for p, l, c in zip(df.position_label, df.llm_label, df.llm_confidence)]
df["final_label"] = [d[0] for d in dec]
df["reason"] = [d[1] for d in dec]
df["is_hindi"] = df.llm_language == "HINDI"
df["kept"] = df.final_label.notna() & ~(df.is_hindi & L["drop_hindi"])

# Balance train and dev to target_complete_frac (drop the surplus class, fixed hash order).
target = cfg["clips"]["target_complete_frac"]
df["in_dataset"] = False
for split in ["train", "dev", "test"]:
    k = df[(df.split == split) & df.kept].copy()
    if k.empty:
        continue
    if split == "test":   # silver test set: every kept clip, unbalanced (balanced accuracy is reported)
        df.loc[k.index, "in_dataset"] = True
        continue
    k["_h"] = [C.unit_hash(SEED, "bal", c) for c in k.clip_id]
    n1, n0 = int((k.final_label == 1).sum()), int((k.final_label == 0).sum())
    keep1, keep0 = min(n1, int(n0 * target / (1 - target))), min(n0, int(n1 * (1 - target) / target))
    chosen = pd.concat([k[k.final_label == 1].nsmallest(keep1, "_h"), k[k.final_label == 0].nsmallest(keep0, "_h")])
    df.loc[chosen.index, "in_dataset"] = True

df.drop(columns=["transcript"]).to_parquet(rt.path("labels", "fast_labels_v1.parquet"), index=False)
lb = df[df.llm_label.notna()]
report = {
    "labelled": int(len(lb)), "of": int(len(df)),
    "labelled_by_split": lb.split.value_counts().to_dict(),
    "gemini_vs_position": pd.crosstab(lb.kind, lb.llm_label).to_dict(),
    "reasons": lb.reason.value_counts().to_dict(),
    "gemini_language": lb.llm_language.value_counts(dropna=False).to_dict(),
    "dataset_by_split": df[df.in_dataset].groupby("split").final_label.agg(n="size", complete_frac="mean").round(3).to_dict(),
}
with open(rt.path("reports", "fast_labels.json"), "w") as f:
    json.dump(report, f, indent=1, default=str)
print(json.dumps(report, indent=1, default=str))
