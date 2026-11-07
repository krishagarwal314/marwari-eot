# marwari-eot

Measure how well end-of-turn (EOT) models detect turn ends in Marwari speech, then fine-tune
Pipecat Smart Turn v3 so Marwari accuracy rises while its 23 existing languages stay within noise.

Everything runs in Colab. Code lives in this repo; data, features, checkpoints and reports live on
Google Drive under `MyDrive/marwari-eot/`.

```
config.yaml          all paths, revisions, seeds, ratios, batch settings
common/              colab_setup.py  audio.py  features.py  metrics.py  ckpt.py
models/              smart_turn.py (upstream architecture)  load_pretrained.py (ONNX -> PyTorch)
notebooks/src/       notebook sources (# %% cells) -> python tools/build_notebooks.py -> notebooks/*.ipynb
reports/             phase_N.md written at the end of each phase
```

Label convention everywhere: `1 = COMPLETE` (turn ended), `0 = INCOMPLETE`.

Upstream code reused unchanged (BSD 2-Clause, (c) Daily): the `SmartTurnV3Model` architecture,
`truncate_audio_to_last_n_seconds`, and the Whisper feature-extractor call, from
pipecat-ai/smart-turn @ `4786657e`.
