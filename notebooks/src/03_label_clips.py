# %% [markdown]
# # 03 · Label clips with Gemini, then the listening check
# Position-rule labels are often wrong: a pause can be a real sentence end. Gemini listens to each
# clip and answers COMPLETE / INCOMPLETE / UNSURE with a confidence. Final label:
# * keep if Gemini **agrees** with the position rule;
# * keep with Gemini's label if it disagrees **and** confidence >= `labelling.confident_threshold`;
# * drop UNSURE and unconfident disagreements.
#
# * if the clip has a transcript cue from 02b, keep it only when Gemini **and** the text cue agree;
# * clips Gemini or the transcript marks as Hindi are left out of the Marwari sets (`drop_hindi`).
#
# Every answer is cached on Drive as it arrives (one JSONL per clip shard). Re-running skips
# clips that already have an answer. Then a 200-clip blind listening check by a native speaker.
# **DECISION:** continue only if listener agreement >= 90%.
#
# Needs `GEMINI_API_KEY` in Colab Secrets. No GPU needed.

# %% [bootstrap]

# %%
import os, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from common.colab_setup import setup
from common import audio as A, clips as C
from common.labeling import GeminiLabeler, decide_with_text

rt = setup()
cfg = rt.cfg
L = cfg["labelling"]
SEED = cfg["seed"]
CLIPS_DIR = os.path.join(rt.root, cfg["dirs"]["clips"], cfg["clips"]["version"])
clips = pd.read_parquet(os.path.join(rt.root, cfg["dirs"]["clips"], f"{cfg['clips']['version']}_index.parquet"))
CACHE = os.path.join(rt.root, cfg["dirs"]["labels"], f"gemini_{L['version']}")
os.makedirs(CACHE, exist_ok=True)
assert os.environ.get("GEMINI_API_KEY"), "Add GEMINI_API_KEY in Colab Secrets and re-run."

labeler = GeminiLabeler(cfg, os.environ["GEMINI_API_KEY"])
models = labeler.available_models()
print("flash models available:", [x for x in models if "flash" in x])
assert L["model"] in models, f"{L['model']} not available; set labelling.model in config.yaml to one of the above"

# %% [markdown]
# ## Which clips to label
# Order: dev, then train, then test. `labelling.max_clips` caps the total for a trial run;
# the cap picks clips in a fixed hash order, so raising it later only adds clips.

# %%
sel = clips[clips.split.isin(L["label_splits"])].copy()
sel["_order"] = sel.split.map({s: i for i, s in enumerate(["dev", "train", "test"])})
sel["_h"] = [C.unit_hash(SEED, "label", c) for c in sel.clip_id]
sel = sel.sort_values(["_order", "_h"])
if L["max_clips"]:
    sel = sel.head(L["max_clips"])
print(len(sel), "clips selected;", sel.split.value_counts().to_dict())

def items_for_shard(shard, ids):
    t = pq.read_table(os.path.join(CLIPS_DIR, shard), columns=["clip_id", "flac"]).to_pandas()
    t = t[t.clip_id.isin(ids)]
    for cid, fl in zip(t.clip_id, t.flac):
        yield cid, A.encode(A.decode_bytes(fl), "WAV")

# %% [markdown]
# ## Trial: 20 clips
# Check that the answers look sensible before labelling everything.

# %%
first_shard = sel.clip_shard.iloc[0]
trial_ids = set(sel[sel.clip_shard == first_shard].clip_id.head(20))
cache_path = os.path.join(CACHE, first_shard.removesuffix(".parquet") + ".jsonl")
print(labeler.label_many(items_for_shard(first_shard, trial_ids), cache_path))
from common.labeling import read_cache
c = read_cache(cache_path)
print(pd.DataFrame([{**c[i], **sel.set_index("clip_id").loc[i, ["kind", "pause_ms"]].to_dict()} for i in trial_ids if i in c]))

# %% [markdown]
# ## Label everything selected (resumable)

# %%
for shard, g in sel.groupby("clip_shard", sort=False):
    cache_path = os.path.join(CACHE, shard.removesuffix(".parquet") + ".jsonl")
    done = read_cache(cache_path)
    ids = set(g.clip_id) - set(done)
    if not ids:
        continue
    print(f"{shard}: {len(ids)} to label")
    print("  ", labeler.label_many(items_for_shard(shard, ids), cache_path))

# %% [markdown]
# ## Final labels, balancing, report

# %%
ans = {}
for f in os.listdir(CACHE):
    if f.endswith(".jsonl"):
        ans.update(read_cache(os.path.join(CACHE, f)))
lab = clips.copy()
lab["llm_label"] = lab.clip_id.map(lambda i: ans.get(i, {}).get("label"))
lab["llm_conf"] = lab.clip_id.map(lambda i: ans.get(i, {}).get("confidence"))
lab["llm_language"] = lab.clip_id.map(lambda i: ans.get(i, {}).get("language"))

TEXT = rt.path("labels", f"text_features_{cfg['clips']['version']}.parquet")
if os.path.exists(TEXT):
    tf = pd.read_parquet(TEXT)[["clip_id", "text_label", "text_reason", "text_before_cut", "text_language", "cut_off"]]
    lab = lab.merge(tf, on="clip_id", how="left")
else:
    print("No text features (run 02b to use transcripts); using Gemini + position rule only.")
    for c in ["text_label", "text_reason", "text_before_cut", "text_language", "cut_off"]:
        lab[c] = None

dec = [decide_with_text(p, l, c, t, L["confident_threshold"])
       for p, l, c, t in zip(lab.position_label, lab.llm_label, lab.llm_conf, lab.text_label)]
lab["final_label"] = [d[0] for d in dec]
lab["reason"] = [d[1] for d in dec]
lab["is_hindi"] = (lab.llm_language == "HINDI") | (lab.text_language == "hindi")
lab["kept"] = lab.final_label.notna() & ~(lab.is_hindi & L["drop_hindi"])

# Balance train/dev to target_complete_frac by dropping the surplus class (fixed hash order).
# Test clips are not balanced here: Phase 2 draws its own 55/45 set from them.
target = cfg["clips"]["target_complete_frac"]
lab["in_dataset"] = False
for split in ["train", "dev"]:
    k = lab[(lab.split == split) & lab.kept].copy()
    if k.empty:
        continue
    k["_h"] = [C.unit_hash(SEED, "bal", c) for c in k.clip_id]
    n1, n0 = int((k.final_label == 1).sum()), int((k.final_label == 0).sum())
    keep1 = min(n1, int(n0 * target / (1 - target)))
    keep0 = min(n0, int(n1 * (1 - target) / target))
    chosen = pd.concat([k[k.final_label == 1].nsmallest(keep1, "_h"), k[k.final_label == 0].nsmallest(keep0, "_h")])
    lab.loc[chosen.index, "in_dataset"] = True
lab.loc[(lab.split == "test") & lab.kept, "in_dataset"] = False  # test pool only, never training

out = rt.path("labels", f"labels_{L['version']}.parquet")
lab.drop(columns=["transcript"]).to_parquet(out, index=False)
labelled = lab[lab.llm_label.notna()]
report = {
    "clips_labelled": int(len(labelled)),
    "llm_vs_position": pd.crosstab(labelled.kind, labelled.llm_label).to_dict(),
    "reasons": labelled.reason.value_counts().to_dict(),
    "llm_language": labelled.llm_language.value_counts(dropna=False).to_dict(),
    "dropped_as_hindi": int((labelled.final_label.notna() & labelled.is_hindi).sum()),
    "text_vs_llm_on_transcribed": pd.crosstab(labelled.text_label.fillna(-1), labelled.llm_label).to_dict(),
    "position_rule_agreement_with_llm_excl_unsure": float(
        (labelled[labelled.llm_label != "UNSURE"].position_label ==
         (labelled[labelled.llm_label != "UNSURE"].llm_label == "COMPLETE").astype(int)).mean()),
    "in_dataset_by_split": lab[lab.in_dataset].groupby("split").final_label.agg(["size", "mean"]).round(3).to_dict(),
    "test_pool_kept": int(((lab.split == "test") & lab.kept).sum()),
}
with open(rt.path("reports", "phase1_labels.json"), "w") as f:
    json.dump(report, f, indent=1, default=str)
print(json.dumps(report, indent=1, default=str))

# %% [markdown]
# ## Listening check
# 200 random clips from the final train+dev set. The listener does not see any label. For clips
# from transcribed recordings, the transcript **up to the cut** is shown (never the words after it).
# Type your name in ANNOTATOR, run the cell, and answer each clip. Every click is saved to Drive
# immediately, so you can stop and continue later by re-running this cell.

# %%
ANNOTATOR = "mehar"   # each listener uses their own name

from common.label_ui import LabelUI
CHECK_IDS = rt.path("labels", f"listening_check_{L['version']}_ids.csv")
if not os.path.exists(CHECK_IDS):   # frozen once written
    pool = lab[lab.in_dataset].copy()
    pool["_h"] = [C.unit_hash(SEED, "check", c) for c in pool.clip_id]
    pool.nsmallest(L["check_sample_size"], "_h")[["clip_id", "clip_shard"]].to_csv(CHECK_IDS, index=False)
check = pd.read_csv(CHECK_IDS)
audio_cache = {}
for shard, g in check.groupby("clip_shard"):
    t = pq.read_table(os.path.join(CLIPS_DIR, shard), columns=["clip_id", "flac"]).to_pandas()
    for cid, fl in zip(t.clip_id, t.flac):
        if cid in set(g.clip_id):
            audio_cache[cid] = A.decode_bytes(fl)
before = lab.set_index("clip_id").text_before_cut
def info(cid):
    t = before.get(cid)
    return f"text up to the cut: …{t[-120:]}" if isinstance(t, str) and t else "(no transcript)"
ui = LabelUI(list(check.clip_id), audio_cache.__getitem__,
             rt.path("labels", f"listening_check_{L['version']}.csv"), ANNOTATOR, info=info)

# %% [markdown]
# ## Agreement (DECISION: continue only if >= 90%)

# %%
from common.label_ui import load_answers
human = load_answers(rt.path("labels", f"listening_check_{L['version']}.csv"), ANNOTATOR)
h = check[["clip_id"]].copy()
h["human"] = h.clip_id.map(human)
h = h.merge(lab[["clip_id", "final_label", "position_label", "llm_label", "kind"]], on="clip_id")
answered = h[h.human.notna()]
sure = answered[answered.human != "UNSURE"]
hb = (sure.human == "COMPLETE").astype(int)
res = {
    "answered": int(len(answered)), "of": int(len(h)),
    "human_unsure": int((answered.human == "UNSURE").sum()),
    "agreement_final_label": float((hb == sure.final_label).mean()) if len(sure) else None,
    "agreement_position_rule": float((hb == sure.position_label).mean()) if len(sure) else None,
    "agreement_by_final_label": sure.assign(ok=(hb == sure.final_label)).groupby("final_label").ok.mean().round(3).to_dict(),
    "required": L["agreement_required"],
}
res["decision"] = ("PASS: continue to Phase 2" if res["agreement_final_label"] is not None and
                   res["agreement_final_label"] >= L["agreement_required"] else "NOT YET / FAIL")
with open(rt.path("reports", f"phase1_listening_check_{ANNOTATOR}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
