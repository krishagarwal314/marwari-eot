# %% [markdown]
# # 02 · Build clips (turn-boundary candidates)
# 1. Freeze the **speaker split** (80/10/10, stratified by district x gender). Test speakers are
#    held back for Phase 2 and never trained on.
# 2. Run Silero VAD on every Marwari recording.
#    * **COMPLETE candidate:** cut 200 ms after the last speech.
#    * **INCOMPLETE candidates:** cut 200 ms into an internal pause of 200–800 ms (up to 2 per recording).
# 3. Keep the last 8 s of each clip (Smart Turn left-pads shorter clips at feature time).
#    Mark 50% of *train* clips for telephony augmentation. The augmentation itself is applied
#    in 06, after labelling, so Gemini and the listeners hear clean audio.
#
# These are *position-rule* labels only; 03 checks them with Gemini.
# Note: transcripts are per recording and not word-aligned, so pauses after fillers
# ("toh", "matlab", ...) cannot be targeted yet; every pause in range is a candidate.
#
# **Re-runnable:** one done marker per raw shard. CPU-bound.

# %% [bootstrap]

# %%
import os, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from common.colab_setup import setup
from common import clips as C

rt = setup()
cfg = rt.cfg
SEED = cfg["seed"]
RAW = os.path.join(rt.root, cfg["dirs"]["raw"], "vaani")
V_CLIPS = cfg["clips"]["version"]
OUT = os.path.join(rt.root, cfg["dirs"]["clips"], V_CLIPS)
DONE = os.path.join(OUT, "_done")
os.makedirs(DONE, exist_ok=True)
SPLIT_FILE = rt.path("splits", f"speakers_{cfg['splits']['version']}.parquet")

idx = pd.read_parquet(os.path.join(RAW, "recordings_index.parquet"))
m = idx[idx.group == "marwari"].copy()

# Some districts (Jaipur) have no speakerID. The Vaani audio path holds the speaker number
# ("..._Churu_111812_..." for RACH_111812). Check that rule on rows that do have a speakerID,
# and use it only if it agrees almost always.
import re
def spk_num_from_path(p):
    hit = re.search(r"_(\d{5,})_", str(p or ""))
    return hit.group(1) if hit else None
has = m[m.speakerID.notna()]
nums = has.src_audio_path.map(spk_num_from_path)
match = (nums.notna() & [str(n) in str(s) for n, s in zip(nums, has.speakerID)]).mean()
print(f"path rule matches the real speakerID on {match:.1%} of {len(has)} rows")
miss = m.speakerID.isna()
print(miss.sum(), "recordings without speakerID, by district:", m[miss].district.value_counts().to_dict())
if match >= 0.98:
    m.loc[miss, "speakerID"] = m.loc[miss, "src_audio_path"].map(
        lambda p: f"PATH_{spk_num_from_path(p)}" if spk_num_from_path(p) else None)
    print("filled from path:", int(miss.sum() - m.speakerID.isna().sum()))
else:
    print("path rule not reliable: recordings without speakerID are dropped")
print(m.speakerID.isna().sum(), "still without speaker (dropped: cannot split by speaker)")
m = m[m.speakerID.notna()]
m["speakerID"] = m["speakerID"].astype(str)

# %% [markdown]
# ## Speaker split (frozen after the first run)

# %%
def mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else "unknown"

spk = m.groupby("speakerID").agg(district=("district", mode), gender=("gender", mode),
                                 hours=("duration_s", lambda s: s.sum() / 3600)).reset_index()
if os.path.exists(SPLIT_FILE):
    frozen = pd.read_parquet(SPLIT_FILE)
    new = spk[~spk.speakerID.isin(frozen.speakerID)].copy()
    if len(new):
        new["split"] = [C.split_for_new_speaker(s, cfg, SEED) for s in new.speakerID]
        frozen = pd.concat([frozen, new], ignore_index=True)
        frozen.to_parquet(SPLIT_FILE, index=False)
        print(f"{len(new)} new speakers added to the frozen split by hash")
    splits = frozen
else:
    splits = C.assign_speaker_splits(spk, cfg, SEED)
    splits.to_parquet(SPLIT_FILE, index=False)
    print("split written:", SPLIT_FILE)
split_of = dict(zip(splits.speakerID, splits.split))
assert splits.speakerID.is_unique
print(splits.split.value_counts().to_string())
print(splits.groupby("split").hours.sum().round(1).to_string())
print(pd.crosstab([splits.district, splits.gender], splits.split).to_string())

# %% [markdown]
# ## Look at one recording first
# Prints the VAD segments and cut points for one recording, and plays the clips.

# %%
from IPython.display import Audio, display
from silero_vad import load_silero_vad
from common import audio as A
import pyarrow.parquet as pq

r0 = m.iloc[0]
row = pq.read_table(os.path.join(RAW, "shards", r0.shard_file)).slice(int(r0.row_in_shard), 1).to_pylist()[0]
wav = A.decode_bytes(row["flac"])
segs = C.vad_segments(wav, load_silero_vad(), cfg)
print(f"{len(wav)/16000:.1f}s, segments (s):", [(round(s['start']/16000, 2), round(s['end']/16000, 2)) for s in segs])
for c in C.candidates(segs, len(wav), cfg, np.random.default_rng(0)):
    print(c)
    display(Audio(C.cut_window(wav, c["cut"], cfg["audio"]["window_s"]), rate=16000))

# %% [markdown]
# ## All recordings (parallel, resumable)

# %%
WORKERS = max(1, rt.num_workers)
use = m
if cfg["clips"]["max_recordings"]:
    use = m.assign(_h=[C.unit_hash(SEED, "rec", r) for r in m.rec_id]).nsmallest(cfg["clips"]["max_recordings"], "_h")
    print(f"using {len(use)} of {len(m)} recordings ({use.duration_s.sum()/3600:.1f} h); by split:",
          use.speakerID.map(split_of).value_counts().to_dict())
by_shard = {s: dict(zip(g.rec_id, g.speakerID)) for s, g in use.groupby("shard_file")}
todo = [s for s in by_shard if not os.path.exists(os.path.join(DONE, s.removesuffix(".parquet") + ".json"))]
print(f"{len(by_shard)} raw shards, {len(todo)} to do, {WORKERS} workers")
t0 = time.time()
with ProcessPoolExecutor(WORKERS) as ex:
    futs = {}
    for s in todo:
        key = s.removesuffix(".parquet")
        futs[ex.submit(C.build_clips_for_shard, cfg, os.path.join(RAW, "shards", s), by_shard[s],
                       split_of, os.path.join(OUT, key + ".parquet"),
                       os.path.join(DONE, key + ".json"), SEED)] = s
    for fut in as_completed(futs):
        try:
            st = fut.result()
            print(f"  {st['raw_shard']}: {st['recordings']} rec -> {st['clips']} clips "
                  f"(no speech {st['no_speech']}) {st['seconds']}s | {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"  FAILED {futs[fut]}: {e!r} (will retry on the next run)")

# %% [markdown]
# ## Clip index and report

# %%
frames = []
for f in sorted(os.listdir(OUT)):
    if f.endswith(".parquet"):
        p = os.path.join(OUT, f)
        cols = [c for c in pq.ParquetFile(p).schema_arrow.names if c != "flac"]
        d = pq.read_table(p, columns=cols).to_pandas()
        d["clip_shard"] = f
        frames.append(d)
clips = pd.concat(frames, ignore_index=True)
assert clips.clip_id.is_unique
# a speaker must never appear in two splits
assert clips.groupby("speakerID").split.nunique().max() == 1
clips.to_parquet(os.path.join(rt.root, cfg["dirs"]["clips"], f"{V_CLIPS}_index.parquet"), index=False)

stats = {
    "clips": int(len(clips)),
    "by_split_kind": pd.crosstab(clips.split, clips.kind).to_dict(),
    "complete_frac_by_split": clips.groupby("split").position_label.mean().round(3).to_dict(),
    "speakers_by_split": clips.groupby("split").speakerID.nunique().to_dict(),
    "pause_ms_quantiles": clips.pause_ms.dropna().quantile([.1, .5, .9]).round(0).to_dict(),
    "complete_trail_real_ms_lt_200": int(((clips.kind == "complete") & (clips.trail_real_ms < 200)).sum()),
    "telephony_marked_train": int(clips.telephony.sum()),
    "recordings_without_speech": int(sum(json.load(open(os.path.join(DONE, f)))["no_speech"] for f in os.listdir(DONE))),
}
with open(rt.path("reports", "phase1_clips_stats.json"), "w") as f:
    json.dump(stats, f, indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
