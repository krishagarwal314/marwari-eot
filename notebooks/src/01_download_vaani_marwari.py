# %% [markdown]
# # 01 · Download Vaani Marwari
# Downloads Vaani one parquet shard at a time to local disk, keeps the Marwari rows (and related
# dialect rows as a separate group), resamples them to 16 kHz mono FLAC, and writes one compact
# parquet per source shard to Drive. Then it deletes the download.
#
# **Re-runnable:** every finished source shard has a done marker on Drive and is skipped next
# time. If the session dies, just run all cells again.
#
# Order of folders comes from `config.yaml -> vaani.folders`. `audio/Marwari` (~18 GB) comes
# first. You can stop after it and move on to 02; the other folders only add more data.
# CPU-bound: any runtime works; more CPU cores = faster.

# %% [bootstrap]

# %%
import os, json, time, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from common.colab_setup import setup
from common import vaani as V

rt = setup()
cfg = rt.cfg
RAW = os.path.join(rt.root, cfg["dirs"]["raw"], "vaani")
SHARDS, DONE = os.path.join(RAW, "shards"), os.path.join(RAW, "_done")
DL = rt.local("vaani_download")
for d in (SHARDS, DONE):
    os.makedirs(d, exist_ok=True)
FILE_LIST = os.path.join(RAW, f"vaani_files_{cfg['hf']['vaani_revision'][:8]}.json")
WORKERS = max(1, min(rt.num_workers, 6))   # each worker holds ~1 shard (~0.5 GB) in flight
print("workers:", WORKERS, "| local disk free GB:", round(shutil.disk_usage("/content").free / 1e9, 1))

# %%
folders = [f for f in cfg["vaani"]["folders"] if f not in cfg["vaani"]["skip_folders"]]
plan = {f: V.list_folder_files(cfg, f, FILE_LIST) for f in folders}
for f, files in plan.items():
    n_done = sum(os.path.exists(os.path.join(DONE, V.shard_key(p) + ".json")) for p in files)
    print(f"{f:28s} {len(files):4d} shards, {n_done} done")

# %% [markdown]
# ## Safety check on the first Marwari shard
# Before downloading 18 GB, process one shard and look at the `language` values. If no rows are
# kept, the language names in `config.yaml` do not match the data: the cell stops and shows the
# values it found.

# %%
first = plan[folders[0]][0]
s = V.run_shard(cfg, first, SHARDS, DONE, DL)
print(json.dumps(s, indent=1, ensure_ascii=False))
assert sum(s["kept"].values()) > 0, (
    "No rows kept. Look at 'languages' above and update vaani.marwari_languages in config.yaml.")
ex = pd.read_parquet(os.path.join(SHARDS, s["key"] + ".parquet")).drop(columns=["flac"]).head(3)
print(ex.T)

# %% [markdown]
# ## All shards (parallel, resumable)

# %%
t0 = time.time()
for folder, files in plan.items():
    todo = [p for p in files if not os.path.exists(os.path.join(DONE, V.shard_key(p) + ".json"))]
    print(f"\n== {folder}: {len(todo)} to do")
    if not todo:
        continue
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(V.run_shard, cfg, p, SHARDS, DONE, DL): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                st = fut.result()
                print(f"  {p.split('/', 1)[1]:60s} kept={st['kept']} h={st['hours_kept']} "
                      f"err={st['decode_errors']} {st['seconds']}s | total {(time.time()-t0)/60:.1f} min")
            except Exception as e:
                print(f"  FAILED {p}: {e!r} (will retry on the next run)")

# %% [markdown]
# ## Index, de-duplication and report
# The same recording can appear in a language folder and a district folder. Duplicates (same
# audio bytes) keep the copy from the earliest folder in config order.

# %%
idx = V.build_index(SHARDS)
rank = {f: i for i, f in enumerate(cfg["vaani"]["folders"])}
idx["folder"] = idx["source_file"].map(lambda p: next(f for f in cfg["vaani"]["folders"] if p.startswith(f + "/")))
idx["folder_rank"] = idx["folder"].map(rank)
n_before = len(idx)
idx = idx.sort_values(["folder_rank", "shard_file", "row_in_shard"]).drop_duplicates("rec_id", keep="first")
print(f"{n_before} rows -> {len(idx)} unique recordings ({n_before - len(idx)} duplicates dropped)")
idx.to_parquet(os.path.join(RAW, "recordings_index.parquet"), index=False)

lang_counts = {}
for f in os.listdir(DONE):
    for k, v in json.load(open(os.path.join(DONE, f)))["languages"].items():
        lang_counts[k] = lang_counts.get(k, 0) + v
summary = V.summarize(idx)
summary["all_language_values_seen"] = dict(sorted(lang_counts.items(), key=lambda kv: -kv[1]))
summary["folders_done"] = {f: sum(os.path.exists(os.path.join(DONE, V.shard_key(p) + ".json")) for p in fs)
                           for f, fs in plan.items()}
summary["folders_total"] = {f: len(fs) for f, fs in plan.items()}
with open(rt.path("reports", "phase1_vaani_stats.json"), "w") as f:
    json.dump(summary, f, indent=1, ensure_ascii=False, default=str)
print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))

# %%
# Per-district table for the Marwari group (speakers, hours, gender split)
m = idx[idx.group == "marwari"]
tab = m.groupby("district").agg(recordings=("rec_id", "size"), speakers=("speakerID", "nunique"),
                                hours=("duration_s", lambda s: round(s.sum() / 3600, 2)))
tab = tab.join(pd.crosstab(m.district, m.gender)).sort_values("hours", ascending=False)
print(tab.to_string())
print("\nTOTAL", len(m), "recordings,", m.speakerID.nunique(), "speakers,", round(m.duration_s.sum() / 3600, 1), "hours")
