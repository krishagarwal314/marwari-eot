# marwari-eot

End-of-turn detection for Marwari. The goal: take Pipecat's Smart Turn v3.2 (a Whisper-tiny audio model that
decides whether a speaker has finished their turn), measure it on Marwari, and fine-tune it without hurting the 23
languages it already supports.

This README covers two rounds of experiments. Read both before quoting a number.

## Round 1

### Data

Marwari speech comes from [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani) (ARTPARK / IISc): 149 hours,
469 speakers. The districts are Nagaur, Churu, Jaisalmer and Jaipur. Each recording is one person describing an
image, usually 5–10 seconds long.

A random 1 in 8 recordings was used, about 10k in total. Each recording was run through Silero VAD and cut into clips:

- **complete**: everything up to the final word of the recording
- **incomplete**: everything up to an internal pause of 200–800 ms, at most two per recording

Each clip keeps the last 8 seconds of audio, which is the window Smart Turn uses, plus 200 ms after the speech. Those
200 ms are filled with background noise taken from the same clip. Half of the training clips also go through an
8 kHz phone-line simulation.

### Labels

Every clip was sent to Gemini Flash Lite, 20 clips per request, with the question: *has the speaker finished their
thought?* Gemini also says which language it hears.

A clip was kept only when Gemini's answer matched the clip's position (complete or incomplete). Clips Gemini heard as
Hindi were dropped, which was 13% of them.

Speakers were split 70/10/20 into train, dev and test, so no speaker appears in two splits. That gives 5,144 train
clips, 302 dev clips and 557 test clips. The test clips come from 73 speakers.

### Training

The starting point is Smart Turn v3.2 itself. Its weights are only released as ONNX, so they were copied into the
PyTorch model class from the [Smart Turn repo](https://github.com/pipecat-ai/smart-turn). The PyTorch copy matches
the ONNX file to within 1e-5 on 2,000 test clips.

Every training batch is half Marwari and half replay data from `smart-turn-data-v3.2-train`, with the 23 languages
sampled evenly. The replay half is there to stop the model forgetting the other languages.

Optimiser settings: AdamW, learning rate 2e-5 for the encoder and 1e-4 for the head, warm-up then cosine decay.
Early stopping is on dev balanced accuracy. One run on a Colab T4 takes about 2 minutes.

### Results

Marwari test set, 557 clips. The 95% confidence intervals come from a bootstrap over speakers.

| | Stock v3.2 | Fine-tuned |
|---|---|---|
| Balanced accuracy | 55.3% (50.1–59.9) | **90.9%** (87.8–93.9) |
| ROC-AUC | 0.61 | 0.97 |
| Interrupts the speaker (false completion) | 20.6% | 8.1% |
| Waits too long (false wait) | 68.8% | 10.2% |

Other languages, scored on 150 clips per language from `smart-turn-data-v3.2-test`:

| | Stock | Fine-tuned |
|---|---|---|
| Accuracy, all 23 languages | 93.7% | 92.7% |
| Largest drop | | Spanish −4.0 |

Exported model files:

| File | Marwari balanced acc. | p50 latency (1 CPU thread) | Size |
|---|---|---|---|
| fp32 ONNX | 90.9% | 75 ms | 32 MB |
| int8 ONNX (dynamic) | 73.7% | 87 ms | 9 MB |

Dynamic int8 quantisation costs 17 points and is not faster, so the fp32 file is the one to use.

 from it.
