"""Word timings for Vaani transcripts: MMS-1b-all (Hindi adapter, Devanagari CTC) + CTC Viterbi.

The Hindi adapter's vocabulary covers the Marwari transcripts' characters (ळ is mapped to ल;
NFC text keeps nukta letters like ड़ as ड + ़, both in the vocabulary). Unknown characters are
dropped. Words are aligned without separator tokens: a word spans from its first character's
first frame to its last character's last frame.
"""
from __future__ import annotations

import unicodedata

import numpy as np

FRAME = 320          # wav2vec2 stride in samples at 16 kHz (20 ms)
CHAR_MAP = {"ळ": "ल"}


def ctc_viterbi(logp: np.ndarray, tokens: list[int], blank: int = 0) -> np.ndarray | None:
    """Best CTC path for ``tokens`` through log-probs (T, V).

    Returns an int array (T,) with the token index (0..L-1) emitted at each frame or -1 for blank,
    or None if the audio is too short for the text.
    """
    T, L = logp.shape[0], len(tokens)
    if L == 0:
        return None
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = tokens
    S = len(ext)
    # transitions allowed from s-2 when ext[s] is a token that differs from ext[s-2]
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[3::2] = ext[3::2] != ext[1:-2:2]
    NEG = -1e30
    dp = np.full(S, NEG)
    dp[0] = logp[0, blank]
    dp[1] = logp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, T):
        stay = dp
        prev1 = np.concatenate([[NEG], dp[:-1]])
        prev2 = np.where(skip_ok, np.concatenate([[NEG, NEG], dp[:-2]]), NEG)
        cand = np.stack([stay, prev1, prev2])
        arg = cand.argmax(0)
        dp = cand[arg, np.arange(S)] + logp[t, ext]
        back[t] = arg
    if max(dp[-1], dp[-2]) <= NEG / 2:
        return None
    s = S - 1 if dp[-1] >= dp[-2] else S - 2
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = (s - 1) // 2 if s % 2 == 1 else -1
        s -= back[t, s]
    return path


def text_to_tokens(words: list[str], vocab: dict[str, int]) -> tuple[list[int], list[int]]:
    """Character ids and, for each id, the index of the word it belongs to."""
    ids, owner = [], []
    for wi, w in enumerate(words):
        for ch in unicodedata.normalize("NFC", w):
            ch = CHAR_MAP.get(ch, ch)
            if ch in vocab:
                ids.append(vocab[ch])
                owner.append(wi)
    return ids, owner


def word_spans(path: np.ndarray, owner: list[int], n_words: int) -> list[tuple[int, int] | None]:
    """(start_sample, end_sample) per word, None for words with no aligned characters."""
    spans: list[list[int] | None] = [None] * n_words
    for t, k in enumerate(path):
        if k < 0:
            continue
        w = owner[k]
        if spans[w] is None:
            spans[w] = [t, t + 1]
        else:
            spans[w][1] = t + 1
    return [(s[0] * FRAME, s[1] * FRAME) if s else None for s in spans]


class MMSAligner:
    def __init__(self, model_id: str = "facebook/mms-1b-all", lang: str = "hin", device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id, target_lang=lang)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=lang, ignore_mismatched_sizes=True)
        self.model = self.model.to(device).eval()
        if device == "cuda":
            self.model = self.model.half()
        self.device = device
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id

    def align(self, wav: np.ndarray, words: list[str]) -> list[tuple[int, int] | None] | None:
        ids, owner = text_to_tokens(words, self.vocab)
        if not ids:
            return None
        x = self.proc(wav, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with self.torch.no_grad():
            logits = self.model(x.to(self.model.dtype)).logits[0].float()
        logp = self.torch.log_softmax(logits, -1).cpu().numpy()
        path = ctc_viterbi(logp, ids, self.blank)
        if path is None:
            return None
        return word_spans(path, owner, len(words))


def context_at(spans: list, words: list[str], speech_end: int, tol: int = 960) -> dict:
    """Words around a cut at ``speech_end`` (samples). ``tol`` (60 ms) absorbs VAD/aligner jitter."""
    prev_i = next_i = None
    inside = False
    for i, sp in enumerate(spans):
        if sp is None:
            continue
        s, e = sp
        if e <= speech_end + tol:
            prev_i = i
        elif s >= speech_end - tol and next_i is None:
            next_i = i
        if s < speech_end - tol and e > speech_end + tol:
            inside = True
    return {
        "prev_word": words[prev_i] if prev_i is not None else None,
        "next_word": words[next_i] if next_i is not None else None,
        "text_before_cut": " ".join(words[: prev_i + 1]) if prev_i is not None else "",
        "cut_inside_word": inside,
        "words_after_cut": len(words) - (prev_i + 1 if prev_i is not None else 0),
    }
