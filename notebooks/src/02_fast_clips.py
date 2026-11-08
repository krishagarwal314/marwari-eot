# %% [markdown]
# # 02 (fast) · Vaani Marwari -> clips, in one pass
# Replaces 01 + 02 for the fast track. Each Marwari shard is downloaded to Colab's local disk; a
# fixed random 1-in-8 of its recordings are cut into turn-boundary clips (VAD: one COMPLETE cut
# after the last speech, up to two INCOMPLETE cuts in 200-800 ms pauses); only the 8 s clips are
# saved to Drive. Speakers are split 80/10/10 by a seeded hash. Re-runnable: one done marker per
# shard. CPU is enough; more CPU cores = faster.

# %% [bootstrap]

# %%
import os, json, time, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from common.colab_setup import setup
from common import fast as F, vaani as V

rt = setup()
cfg = rt.cfg
OUT = os.path.join(rt.root, cfg["dirs"]["clips"], "fast_v1")
DONE = os.path.join(OUT, "_done")
os.makedirs(DONE, exist_ok=True)
WORK = rt.local("fast_work")
WORKERS = max(1, min(rt.num_workers, 6))
drive = shutil.disk_usage("/content/drive")
print(f"Drive free: {drive.free/1e9:.1f} GB of {drive.total/1e9:.1f} GB | workers {WORKERS}")
if drive.free < 6e9:
    print("WARNING: less than 6 GB free on Drive. Delete MyDrive/marwari-eot/data/raw and empty the Drive trash first.")

files = V.list_folder_files(cfg, "audio/Marwari", os.path.join(rt.local("meta"), "vaani_files.json"))
todo = [p for p in files if not os.path.exists(os.path.join(DONE, V.shard_key(p) + ".json"))]
print(f"{len(files)} Marwari shards, {len(todo)} to do; keeping {cfg['clips']['rec_sample_frac']:.3f} of recordings")

# %%
t0 = time.time()
with ProcessPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(F.run_shard, cfg, p, OUT, DONE, WORK): p for p in todo}
    for fut in as_completed(futs):
        try:
            s = fut.result()
            print(f"  {s['key'][-22:]}: {s['recordings_used']} rec -> {s['clips']} clips "
                  f"(no spk {s['no_speaker']}, no speech {s['no_speech']}) {s['seconds']}s | {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"  FAILED {futs[fut]}: {e} (re-run this cell to retry)")

# %%
clips = F.clip_index(OUT)
assert clips.clip_id.is_unique
assert clips.groupby("speakerID").split.nunique().max() == 1, "a speaker is in two splits"
clips.to_parquet(os.path.join(rt.root, cfg["dirs"]["clips"], "fast_v1_index.parquet"), index=False)
stats = {
    "shards_done": len(os.listdir(DONE)), "of": len(files),
    "clips": int(len(clips)), "recordings": int(clips.rec_id.nunique()),
    "speakers_by_split": clips.groupby("split").speakerID.nunique().to_dict(),
    "clips_by_split_kind": pd.crosstab(clips.split, clips.kind).to_dict(),
    "districts_by_split": pd.crosstab(clips.district, clips.split).to_dict(),
    "complete_ends_with_lt_200ms_real_silence": int(((clips.kind == "complete") & (clips.trail_real_ms < 200)).sum()),
    "with_transcript": int(clips.transcript.notna().sum()),
    "drive_free_gb": round(shutil.disk_usage("/content/drive").free / 1e9, 1),
}
with open(rt.path("reports", "fast_clips_stats.json"), "w") as f:
    json.dump(stats, f, indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
