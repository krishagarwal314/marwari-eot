"""Audio I/O, the Smart Turn 8 s window, and telephony simulation.

Decoding is done here with soundfile/librosa from raw bytes (load HF datasets with
``Audio(decode=False)``), so we do not depend on torchcodec matching Colab's torch build.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile

import numpy as np

SR = 16000


# --- copied from pipecat-ai/smart-turn audio_utils.py (commit 4786657e, BSD-2) ---------------
def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode="constant", constant_values=0)
    return audio_array
# ---------------------------------------------------------------------------------------------


def to_mono_16k(audio: np.ndarray, sr: int, target_sr: int = SR) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # soundfile gives (n, channels)
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def decode_bytes(data: bytes, target_sr: int = SR) -> np.ndarray:
    """Decode any soundfile-readable bytes (wav/flac/ogg/mp3) to mono float32 at target_sr."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return to_mono_16k(audio, sr, target_sr)


def decode_hf_audio(value: dict, target_sr: int = SR) -> np.ndarray:
    """Decode an HF ``Audio(decode=False)`` cell: {"bytes": ..., "path": ...}."""
    if value.get("bytes") is not None:
        return decode_bytes(value["bytes"], target_sr)
    import soundfile as sf

    audio, sr = sf.read(value["path"], dtype="float32", always_2d=False)
    return to_mono_16k(audio, sr, target_sr)


def add_trailing_silence(audio: np.ndarray, ms: int, sr: int = SR) -> np.ndarray:
    return np.concatenate([audio, np.zeros(int(sr * ms / 1000), dtype=np.float32)])


# --- telephony simulation ----------------------------------------------------------------------

def _mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu) / mu * 2 - 1  # 8-bit quantisation
    return (np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu).astype(np.float32)


def _ffmpeg_codec(x8k: np.ndarray, codec: str) -> np.ndarray:
    """Round-trip 8 kHz audio through AMR-NB or Opus with ffmpeg. Raises if ffmpeg lacks the codec."""
    import soundfile as sf

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")
    enc = {"amr": ["-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-ar", "8000", "-ac", "1"],
           "opus": ["-c:a", "libopus", "-b:a", "16k", "-ar", "8000", "-ac", "1"]}[codec]
    ext = {"amr": "amr", "opus": "ogg"}[codec]
    with tempfile.TemporaryDirectory() as d:
        src, mid, out = f"{d}/in.wav", f"{d}/mid.{ext}", f"{d}/out.wav"
        sf.write(src, x8k, 8000)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, *enc, mid], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mid, "-ar", "8000", "-ac", "1", out],
                       check=True)
        y, _ = sf.read(out, dtype="float32")
    n = len(x8k)  # codecs add/remove a few ms of delay; keep the original length (end-aligned)
    return y[-n:] if len(y) >= n else np.pad(y, (n - len(y), 0))


def _bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def add_noise_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    p_sig = np.mean(x ** 2) + 1e-10
    p_noise = np.mean(noise ** 2) + 1e-10
    return (x + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))).astype(np.float32)


def telephony(x: np.ndarray, rng: np.random.Generator, codec: str | None = None,
              noise: np.ndarray | None = None, snr_range=(10.0, 30.0),
              band=(300.0, 3400.0), sr: int = SR) -> tuple[np.ndarray, dict]:
    """8 kHz down/up, random codec (mulaw/amr/opus), 300-3400 Hz band-pass, optional noise.

    ``noise`` should be a real background-noise recording at ``sr``; if None, white noise is
    used (and recorded as such in the returned info dict).
    Returns (audio at sr, info) so every augmentation is reproducible and logged.
    """
    import librosa

    codec = codec or rng.choice(["mulaw", "amr", "opus"])
    y = librosa.resample(x, orig_sr=sr, target_sr=8000)
    if codec == "mulaw":
        y = _mu_law(y)
    else:
        y = _ffmpeg_codec(y, codec)
    y = _bandpass(y, 8000, band[0], min(band[1], 3999.0))
    y = librosa.resample(y, orig_sr=8000, target_sr=sr).astype(np.float32)
    y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    snr = float(rng.uniform(*snr_range))
    noise_kind = "file" if noise is not None else "white"
    if noise is None:
        noise = rng.standard_normal(len(y)).astype(np.float32)
    y = add_noise_snr(y, noise, snr)
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak
    return y, {"codec": str(codec), "snr_db": snr, "noise": noise_kind}
