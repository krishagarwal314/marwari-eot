"""Vaani transcript parsing and text cues for turn ends.

Vaani transcript conventions (seen in the Marwari subset):
  <noise>...</noise>, <horn>, <bird_squawk>, ...   background events (dropped)
  [breathing]                                       speaker sounds (dropped)
  {building}                                        English gloss after a loanword (dropped)
  <pause>                                           transcriber-marked pause (counted, dropped)
  "।" at the end                                    utterance finished
  "--" at the end                                   speaker cut off mid-sentence
"""
from __future__ import annotations

import re
import unicodedata

_GLOSS = re.compile(r"\{[^}]*\}")
_TAG = re.compile(r"<[^>]*>|\[[^\]]*\]")
_PAUSE = re.compile(r"<pause>", re.I)
_PUNCT = re.compile(r"[।॥,.!?;:\"'“”‘’()…—]")

# Words that cannot end a thought: connectives, postpositions, relatives, articles.
# A cut right after one of these is INCOMPLETE regardless of how long the pause is.
CONNECTIVE = {
    # and / or / but / then / so / that / because
    "और", "अर", "अर्", "ओर", "तो", "तौ", "कि", "की", "कै", "पण", "पर", "लेकिन", "मगर", "या", "फिर", "फेर",
    "जद", "जणा", "क्योंकि", "इसलिए", "जिससे", "ताकि", "मतलब", "यानी", "अगर", "जो", "जको", "जके", "जकी",
    "जिसमें", "जिसकी", "जिसका", "जिसके", "जिस", "जिसे", "जहां", "जठै",
    # postpositions (Hindi + Marwari)
    "का", "के", "को", "में", "से", "ने", "पे", "तक", "लिए", "वास्ते", "सूं", "सू", "मांय", "माय", "माइने",
    "मायने", "मैंने", "माइन", "मे", "ऊपर", "नीचे", "पास", "साथ", "बिच", "बीच", "आगे", "पीछे",
    # determiners / numerals that start a noun phrase
    "एक", "दो", "तीन", "यह", "ये", "वह", "वो", "इ", "ई", "बा", "बो", "बी", "आ", "उ", "इस", "उस", "इसमें",
}
# Finite verb forms that end a clause in Hindi/Marwari (the clause is grammatically complete,
# which makes COMPLETE plausible but not certain: the speaker may still add another clause).
CLAUSE_FINAL = {"है", "हैं", "हो", "हूँ", "हूं", "था", "थी", "थे", "थो", "छै", "छे", "छो", "छा", "ही",
                "दिखे", "दिके", "लागे", "मिले", "रोवे", "गया", "गयी", "गई", "ग्यो", "गयो"}

# Marwari vs Hindi markers (word endings / words). Counted on cleaned text.
_MW_ENDINGS = ("योडो", "योडा", "योडी", "ेड़ो", "ेड़ा", "ेड़ी", "ियोडो", "रियो", "रिया", "रहिया", "ड़ो")
_MW_WORDS = {"माइने", "मायने", "माइन", "निचाने", "लुगाई", "लुगाइया", "सगली", "जको", "जके", "जकी", "छोरा",
             "छोरो", "छोरी", "बा", "बो", "बी", "इ", "आ", "सामे", "भाईडा", "लागरी", "दिके", "दिखे", "भांत", "कने",
             "छै", "छे", "म्हारो", "थारो", "कोनी", "घणो", "घणा", "अठै", "बठै"}
_HI_WORDS = {"दिखाई", "व्यक्ति", "जिसमें", "जिसकी", "जिसका", "जिसके", "इसके", "इसमें", "हुए", "हुई", "हुआ",
             "रहा", "रही", "रहे", "गया", "यह", "हमें"}


def normalize(word: str) -> str:
    return unicodedata.normalize("NFC", word)


def parse(transcript: str | None) -> dict:
    """Clean text plus flags. ``cut_off`` = the speaker was cut off mid-sentence at the end."""
    if not transcript or not str(transcript).strip():
        return {"clean": "", "words": [], "cut_off": None, "ends_danda": None, "pause_tags": 0}
    t = str(transcript)
    pause_tags = len(_PAUSE.findall(t))
    t = _TAG.sub(" ", _GLOSS.sub(" ", t))
    t = re.sub(r"\s+", " ", t).strip()
    cut_off = t.endswith("--") or t.endswith("-")
    ends_danda = t.endswith("।") or t.endswith("॥")
    words = [normalize(w) for w in _PUNCT.sub(" ", t.replace("--", " ")).split()]
    words = [w.strip("-") for w in words if w.strip("-")]
    return {"clean": " ".join(words), "words": words, "cut_off": cut_off, "ends_danda": ends_danda,
            "pause_tags": pause_tags}


def language_score(words: list[str]) -> dict:
    """Counts of Marwari vs Hindi markers and a coarse verdict (marwari / hindi / mixed / unknown)."""
    mw = sum(1 for w in words if w in _MW_WORDS or w.endswith(_MW_ENDINGS))
    hi = sum(1 for w in words if w in _HI_WORDS)
    if mw + hi == 0:
        verdict = "unknown"
    elif mw >= 2 * hi and mw >= 1:
        verdict = "marwari"
    elif hi >= 2 * mw and hi >= 2:
        verdict = "hindi"
    else:
        verdict = "mixed"
    return {"mw_markers": mw, "hi_markers": hi, "text_language": verdict}


def word_category(word: str | None) -> str:
    """'connective' (cannot end a thought), 'clause_final' (finite verb), or 'other'."""
    if not word:
        return "none"
    w = normalize(word)
    if w in CONNECTIVE:
        return "connective"
    if w in CLAUSE_FINAL:
        return "clause_final"
    return "other"


def text_rule(prev_word: str | None, cut_inside_word: bool, is_recording_end: bool,
              cut_off: bool | None) -> tuple[int | None, str]:
    """Text-based label for a cut: (1 COMPLETE / 0 INCOMPLETE / None no opinion, reason)."""
    if cut_inside_word:
        return 0, "cut_inside_word"
    if is_recording_end and cut_off:
        return 0, "transcript_cut_off"
    cat = word_category(prev_word)
    if cat == "connective":
        return 0, "after_connective"
    if cat == "clause_final":
        return (1, "clause_final_at_end") if is_recording_end else (None, "clause_final_mid")
    return (None, "after_other_word")
