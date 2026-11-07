# %% [markdown]
# # 02b · Transcripts: word timings and text cues
# About 6% of the Marwari recordings (≈9 h, Nagaur + Churu) have transcripts. For each one:
# * **cut-off check:** a transcript ending in `--` means the speaker was cut off, so that
#   recording's end is *not* a finished thought;
# * **language check:** Marwari vs Hindi marker words. Some "Marwari" recordings are really Hindi;
# * **word timings:** MMS-1b-all (Hindi adapter, Devanagari) + CTC alignment. These tell us the
#   word right before every cut, which gives a **text-based label** independent of Gemini:
#   a cut after "और / तो / के / में / एक …" or inside a word is INCOMPLETE; a recording end after
#   a finished verb ("है", "छै", "दिखे" …) is COMPLETE.
#
# Run after 02. **GPU runtime** (T4 is fine). Re-runnable: one done marker per clip shard.

# %% [bootstrap]

# %%
import os, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from common.colab_setup import setup
from common import audio as A, transcript as T
from common.align import MMSAligner, context_at

rt = setup()
cfg = rt.cfg
RAW = os.path.join(rt.root, cfg["dirs"]["raw"], "vaani")
V_CLIPS = cfg["clips"]["version"]
clips = pd.read_parquet(os.path.join(rt.root, cfg["dirs"]["clips"], f"{V_CLIPS}_index.parquet"))
idx = pd.read_parquet(os.path.join(RAW, "recordings_index.parquet"))
OUT = os.path.join(rt.root, cfg["dirs"]["labels"], f"text_{V_CLIPS}")
os.makedirs(os.path.join(OUT, "_done"), exist_ok=True)

# %% [markdown]
# ## Recording-level: cut-off and language (text only, fast)

# %%
rec = idx[(idx.group == "marwari") & idx.rec_id.isin(clips.rec_id)].copy()
rec["has_transcript"] = rec.transcript.fillna("").str.strip().ne("")
tr = rec[rec.has_transcript].copy()
parsed = tr.transcript.map(T.parse)
tr["words"] = parsed.map(lambda p: p["words"])
tr["cut_off"] = parsed.map(lambda p: p["cut_off"])
tr["ends_danda"] = parsed.map(lambda p: p["ends_danda"])
tr["pause_tags"] = parsed.map(lambda p: p["pause_tags"])
lang = tr.words.map(T.language_score)
for k in ["mw_markers", "hi_markers", "text_language"]:
    tr[k] = lang.map(lambda d: d[k])
rec_stats = {
    "marwari_recordings_with_clips": int(len(rec)),
    "with_transcript": int(len(tr)),
    "transcribed_hours": round(float(tr.duration_s.sum() / 3600), 2),
    "transcribed_by_district": tr.district.value_counts().to_dict(),
    "cut_off_recordings": int(tr.cut_off.sum()),
    "ends_with_danda": int(tr.ends_danda.sum()),
    "with_pause_tags": int((tr.pause_tags > 0).sum()),
    "text_language": tr.text_language.value_counts().to_dict(),
    "text_language_by_speaker_majority": tr.groupby("speakerID").text_language.agg(lambda s: s.mode().iloc[0]).value_counts().to_dict(),
}
print(json.dumps(rec_stats, indent=1, ensure_ascii=False))
tr.drop(columns=["words"]).to_parquet(os.path.join(OUT, "recordings_text.parquet"), index=False)

# %%
# A few examples of each language verdict, to judge whether the marker lists are sensible
for v in ["marwari", "mixed", "hindi"]:
    print(f"\n== {v}")
    for t in tr[tr.text_language == v].transcript.head(4):
        print(" ", T.parse(t)["clean"][:160])

# %% [markdown]
# ## Word timings + text cue for every clip of a transcribed recording (GPU)

# %%
aligner = MMSAligner(cfg["text"]["align_model"], cfg["text"]["align_lang"], rt.device)
tol = int(cfg["text"]["align_tolerance_ms"] * 16)
words_of = dict(zip(tr.rec_id, tr.words))
cut_off_of = dict(zip(tr.rec_id, tr.cut_off))
tc = clips[clips.rec_id.isin(words_of)]
print(len(tc), "clips from", tc.rec_id.nunique(), "transcribed recordings")

t0 = time.time()
# Clip shards are named after the raw Vaani shard their recordings come from (see 02).
for shard, g in tc.groupby("clip_shard"):
    key = shard.removesuffix(".parquet")
    done = os.path.join(OUT, "_done", key + ".json")
    if os.path.exists(done):
        continue
    tab = pq.read_table(os.path.join(RAW, "shards", shard), columns=["rec_id", "flac"]).to_pandas()
    tab = tab[tab.rec_id.isin(set(g.rec_id))].drop_duplicates("rec_id")
    rows = []
    for rid, fl in zip(tab.rec_id, tab.flac):
        wav = A.decode_bytes(fl)
        words = words_of[rid]
        try:
            spans = aligner.align(wav, words)
        except Exception as e:
            print("  align failed", rid, repr(e)[:120])
            spans = None
        n_aligned = sum(s is not None for s in spans) if spans else 0
        for c in g[g.rec_id == rid].itertuples():
            row = {"clip_id": c.clip_id, "aligned": spans is not None,
                   "aligned_word_frac": n_aligned / max(1, len(words))}
            if spans is not None:
                ctx = context_at(spans, words, int(round(c.speech_end_s * 16000)), tol)
                lab, why = T.text_rule(ctx["prev_word"], ctx["cut_inside_word"], c.kind == "complete", cut_off_of[rid])
                row.update(ctx, prev_word_category=T.word_category(ctx["prev_word"]), text_label=lab, text_reason=why)
            rows.append(row)
    pd.DataFrame(rows).to_parquet(os.path.join(OUT, key + ".parquet"), index=False)
    with open(done, "w") as f:
        json.dump({"clips": len(rows)}, f)
    print(f"  {shard}: {len(rows)} clips | {(time.time()-t0)/60:.1f} min")

# %% [markdown]
# ## Join and report
# Sanity check: for COMPLETE candidates the word before the cut should almost always be the last
# word of the transcript. If that rate is low, the alignment is unreliable.

# %%
tf = pd.concat([pd.read_parquet(os.path.join(OUT, f)) for f in os.listdir(OUT)
                if f.endswith(".parquet") and f != "recordings_text.parquet"], ignore_index=True)
tf = tf.merge(clips[["clip_id", "rec_id", "kind", "position_label", "split"]], on="clip_id")
tf = tf.merge(tr[["rec_id", "cut_off", "text_language", "mw_markers", "hi_markers"]], on="rec_id")
last_word = {r: w[-1] if w else None for r, w in words_of.items()}
comp = tf[(tf.kind == "complete") & tf.aligned]
report = {
    "clips_with_text": int(len(tf)),
    "aligned": int(tf.aligned.sum()),
    "complete_prev_word_is_last_word": float((comp.prev_word == comp.rec_id.map(last_word)).mean()),
    "text_label_vs_position": pd.crosstab(tf.kind, tf.text_label.fillna(-1).astype(int)).to_dict(),
    "text_reason": tf.text_reason.value_counts().to_dict(),
    "prev_word_category_incomplete": tf[tf.kind == "incomplete"].prev_word_category.value_counts().to_dict(),
    "top_prev_words_incomplete": tf[tf.kind == "incomplete"].prev_word.value_counts().head(25).to_dict(),
}
tf.to_parquet(rt.path("labels", f"text_features_{V_CLIPS}.parquet"), index=False)
with open(rt.path("reports", "phase1_text_stats.json"), "w") as f:
    json.dump({"recordings": rec_stats, "clips": report}, f, indent=1, ensure_ascii=False, default=str)
print(json.dumps(report, indent=1, ensure_ascii=False, default=str))

# %%
# Listen to a few incomplete cuts with the text before the cut
from IPython.display import Audio, display
ex = tf[(tf.kind == "incomplete") & tf.aligned].sample(5, random_state=0)
cl = clips.set_index("clip_id")
for r in ex.itertuples():
    t = pq.read_table(os.path.join(rt.root, cfg["dirs"]["clips"], V_CLIPS, cl.loc[r.clip_id, "clip_shard"]),
                      columns=["clip_id", "flac"]).to_pandas().set_index("clip_id")
    print(f"…{r.text_before_cut[-80:]}  ||  next: {r.next_word}  -> text_label={r.text_label} ({r.text_reason})")
    display(Audio(A.decode_bytes(t.loc[r.clip_id, "flac"]), rate=16000))
