# Fast-track results: Smart Turn v3.2 on Marwari (9 Nov 2026)

**Status: preliminary.** One open check (see "Open check") can still lower the Marwari number.

## Setup

* **Data:** Vaani Marwari, a random 1 in 8 recordings (~10k recordings, Nagaur / Churu / Jaisalmer / Jaipur).
  Clips are cut with Silero VAD:
  * COMPLETE = end of the recording;
  * INCOMPLETE = a 200–800 ms pause inside the recording.

  The last 200 ms of **every** clip is replaced with the clip's own background noise (see "Shortcut found and removed").
* **Labels (silver):** Gemini 3.5 / 3.1 Flash Lite, 20 clips per request. A clip is kept only when Gemini **agrees**
  with the position rule (60% of clips). Clips Gemini judged to be Hindi (13%) are removed. Not verified by a native
  speaker.
* **Splits by speaker** (seeded hash): train 5,144 clips, dev 302, test 557 clips from 73 held-out speakers.
* **Training:** converted Smart Turn v3.2 weights (exact ONNX parity). Each batch is half Marwari, half Smart Turn v3.2
  train replay (23 languages, sampled evenly). AdamW (lr 2e-5 encoder / 1e-4 head), 643 steps on a T4, one seed.
* **Regression set:** 150 clips per language from `smart-turn-data-v3.2-test` (3,450 clips).

## Results

| Marwari test (557 clips) | Stock v3.2 | Fine-tuned (fp32 ONNX) |
|---|---|---|
| Balanced accuracy (95% CI, speaker bootstrap) | 55.3% (50.1–59.9) | **90.9% (87.8–93.9)** |
| ROC-AUC | 0.61 | 0.97 |
| False completion (agent interrupts) | 20.6% | 8.1% |
| False wait (agent waits too long) | 68.8% | 10.2% |

| Other languages (150 clips each) | Stock | Fine-tuned |
|---|---|---|
| Overall accuracy | 93.7% | 92.7% |
| Largest drops | | Spanish −4.0, Russian / Vietnamese / Chinese −2.7, several −2.0 |
| Gains | | Bengali +2.0, French +2.0, Ukrainian +1.3 |

±2–3 points per language is within sampling noise at 150 clips. The overall −1.0 point is a small but real cost.

### ONNX

| File | Marwari bal-acc | Replay acc | p50 / p90 latency (1 CPU thread) | Size |
|---|---|---|---|---|
| fp32 | 90.9% | 92.7% | 75 / 90 ms | 32.4 MB |
| int8 (dynamic) | **73.7%** | 90.8% | 87 / 136 ms | 8.9 MB |

**Ship the fp32 file.** Dynamic int8 loses 17 points on Marwari and is not faster on this CPU.

## Shortcut found and removed

The first run scored 98.6%, which was a shortcut. Vaani recordings stop right at the end of speech, so 283 of the 285
COMPLETE test clips ended in zero-padding (digital silence) and no INCOMPLETE clip did. The rule
"ends in zero-padding ⇒ COMPLETE" alone scored 99.6%.

After replacing every clip's last 200 ms with that clip's own background noise, the tail's loudness carries no label
information (probe AUC 0.506).

## Open check

The loudness of the **last 200 ms of speech before the cut** still separates the classes (AUC 0.82). This is either:

* real prosody (speakers trail off at the end of a thought), or
* a recording artifact (the Vaani app chopping the final word).

The stock model's 55% (vs 95% on Hindi) is consistent with either. Listening to test clips decides it. If the endings
are chopped, the COMPLETE cut should move earlier and the model should be retrained.

## Limitations

* Silver labels from a Lite LLM, kept only where they agree with the position rule. Ambiguous cases are excluded, so
  real-world accuracy will be lower.
* Picture-description monologues, not conversations. No live-call test yet.
* One seed. Seed variance is unknown.
* Vaani data is CC-BY-NC 4.0, so this is research use only.
