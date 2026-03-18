"""
dataset_generator.py
====================
Synthetic Morse Code Dataset Generator
Deep Learning–Enhanced Probabilistic Morse Code Decoding

Generates labeled (dot / dash / silence) audio segments as Mel-spectrogram
arrays together with their ground-truth class labels. Everything is built on
NumPy + SciPy so no network-dependent packages are required.

Label encoding
--------------
  0 → silence
  1 → dot
  2 → dash
"""

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.io import wavfile
import os
import random
import math

# ─────────────────────────────────────────────
# Morse dictionary
# ─────────────────────────────────────────────
MORSE_DICT = {
    'A':'.-',   'B':'-...', 'C':'-.-.', 'D':'-..',  'E':'.',
    'F':'..-.', 'G':'--.',  'H':'....', 'I':'..',   'J':'.---',
    'K':'-.-',  'L':'.-..', 'M':'--',   'N':'-.',   'O':'---',
    'P':'.--.', 'Q':'--.-', 'R':'.-.',  'S':'...',  'T':'-',
    'U':'..-',  'V':'...-', 'W':'.--',  'X':'-..-', 'Y':'-.--',
    'Z':'--..', '0':'-----','1':'.----','2':'..---','3':'...--',
    '4':'....-','5':'.....','6':'-....','7':'--...','8':'---..',
    '9':'----.'
}

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FS          = 8000          # sample rate (Hz) – low for speed
N_FFT       = 256
HOP         = 64
N_MELS      = 32
LABEL_MAP   = {'silence': 0, 'dot': 1, 'dash': 2}
CLASS_NAMES = ['silence', 'dot', 'dash']


# ══════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════

def _sine_tone(freq: float, duration: float, fs: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine-wave tone."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at a given SNR (dB)."""
    if snr_db >= 60:          # virtually noiseless
        return signal
    sig_power = np.mean(signal ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(signal)) * math.sqrt(noise_power)
    return (signal + noise).astype(np.float32)


def _apply_timing_jitter(duration: float, jitter: float) -> float:
    """Return duration perturbed by ±jitter fraction."""
    return duration * (1.0 + random.uniform(-jitter, jitter))


def _mel_filterbank(fs: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Construct a simple triangular Mel filterbank (n_mels × n_fft//2+1)."""
    f_min, f_max = 0.0, fs / 2.0
    mel_min = 2595 * math.log10(1 + f_min / 700)
    mel_max = 2595 * math.log10(1 + f_max / 700)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins     = np.floor((n_fft + 1) * freq_pts / fs).astype(int)
    fb       = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, ctr):
            if ctr != lo:
                fb[m-1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi != ctr:
                fb[m-1, k] = (hi - k) / (hi - ctr)
    return fb


_FILTERBANK = _mel_filterbank(FS, N_FFT, N_MELS)


def compute_log_mel_spectrogram(audio: np.ndarray,
                                 fs: int = FS,
                                 n_fft: int = N_FFT,
                                 hop: int = HOP,
                                 n_mels: int = N_MELS) -> np.ndarray:
    """
    Return a (n_mels, T) log-Mel spectrogram for *audio*.
    Uses scipy.signal.spectrogram under the hood.
    """
    fb = _FILTERBANK if (fs == FS and n_fft == N_FFT and n_mels == N_MELS) \
         else _mel_filterbank(fs, n_fft, n_mels)

    f, t, Sxx = scipy_spectrogram(audio, fs=fs, nperseg=n_fft,
                                   noverlap=n_fft - hop, window='hann')
    mel = fb @ Sxx                            # (n_mels, T)
    log_mel = np.log(mel + 1e-9)
    return log_mel.astype(np.float32)


# ══════════════════════════════════════════════
# Segment generator – single symbol
# ══════════════════════════════════════════════

def _generate_segment(label: str,
                       dot_dur: float,
                       freq: float,
                       snr_db: float,
                       jitter: float,
                       distort: bool) -> np.ndarray:
    """
    Return a raw audio segment for one symbol (dot / dash / silence).
    """
    if label == 'dot':
        dur = _apply_timing_jitter(dot_dur, jitter)
        raw = _sine_tone(freq, dur, FS)
    elif label == 'dash':
        dur = _apply_timing_jitter(dot_dur * 3, jitter)
        raw = _sine_tone(freq, dur, FS)
    else:   # silence  – one dot-unit
        dur = _apply_timing_jitter(dot_dur, jitter)
        raw = np.zeros(int(FS * dur), dtype=np.float32)

    raw = _add_noise(raw, snr_db)

    if distort and label != 'silence':
        # simple amplitude modulation distortion
        t = np.arange(len(raw)) / FS
        raw = raw * (1 + 0.3 * np.sin(2 * np.pi * 3.0 * t))

    return raw


# ══════════════════════════════════════════════
# Spectrogram patch from segment
# ══════════════════════════════════════════════

_PATCH_FRAMES = 16   # fixed width (time frames) for each training patch


def _audio_to_patch(audio: np.ndarray) -> np.ndarray:
    """
    Convert a variable-length audio segment into a fixed (N_MELS, _PATCH_FRAMES)
    spectrogram patch via linear interpolation along the time axis.
    """
    spec = compute_log_mel_spectrogram(audio)           # (n_mels, T)
    T = spec.shape[1]
    if T == 0:
        return np.zeros((N_MELS, _PATCH_FRAMES), dtype=np.float32)
    if T == _PATCH_FRAMES:
        return spec
    # Resize time axis
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, _PATCH_FRAMES)
    patch = np.zeros((N_MELS, _PATCH_FRAMES), dtype=np.float32)
    for m in range(N_MELS):
        patch[m] = np.interp(x_new, x_old, spec[m])
    return patch


# ══════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════

def generate_dataset(n_samples: int = 3000,
                     snr_range: tuple = (5, 30),
                     dot_dur_range: tuple = (0.05, 0.15),
                     freq_range: tuple = (600, 1000),
                     jitter_range: tuple = (0.0, 0.15),
                     distort_prob: float = 0.2,
                     seed: int = 42) -> dict:
    """
    Generate a balanced synthetic Morse dataset.

    Returns
    -------
    dict with keys:
        'X'       : np.ndarray  shape (N, N_MELS, PATCH_FRAMES)
        'y'       : np.ndarray  shape (N,)   integer labels 0/1/2
        'labels'  : list of str  class names
        'n_per_class': dict
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    classes = ['silence', 'dot', 'dash']
    n_per_class = n_samples // len(classes)
    extra = n_samples - n_per_class * len(classes)

    X_list, y_list = [], []

    for ci, cls in enumerate(classes):
        count = n_per_class + (1 if ci < extra else 0)
        for _ in range(count):
            snr    = rng.uniform(*snr_range)
            dot_d  = rng.uniform(*dot_dur_range)
            freq   = rng.uniform(*freq_range)
            jitter = rng.uniform(*jitter_range)
            distort = rng.random() < distort_prob

            audio = _generate_segment(cls, dot_d, freq, snr, jitter, distort)
            patch = _audio_to_patch(audio)

            # Normalise patch to zero mean, unit std
            mu, sigma = patch.mean(), patch.std() + 1e-8
            patch = (patch - mu) / sigma

            X_list.append(patch)
            y_list.append(LABEL_MAP[cls])

    X = np.stack(X_list, axis=0)          # (N, N_MELS, PATCH_FRAMES)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    print(f"[DatasetGenerator] Generated {len(y)} samples  "
          f"(silence={np.sum(y==0)}, dot={np.sum(y==1)}, dash={np.sum(y==2)})")

    return {
        'X': X,
        'y': y,
        'labels': CLASS_NAMES,
        'n_per_class': {c: int(np.sum(y == i)) for i, c in enumerate(CLASS_NAMES)},
        'patch_shape': (N_MELS, _PATCH_FRAMES),
    }


def save_dataset(dataset: dict, path: str = 'morse_dataset.npz'):
    np.savez_compressed(path, X=dataset['X'], y=dataset['y'])
    print(f"[DatasetGenerator] Saved → {path}")


def load_dataset(path: str = 'morse_dataset.npz') -> dict:
    data = np.load(path)
    return {
        'X': data['X'],
        'y': data['y'],
        'labels': CLASS_NAMES,
        'patch_shape': (N_MELS, _PATCH_FRAMES),
    }


def generate_morse_audio_labeled(text: str,
                                   dot_dur: float = 0.1,
                                   freq: float = 800.0,
                                   snr_db: float = 20.0,
                                   jitter: float = 0.05) -> tuple:
    """
    Generate a full Morse audio waveform for *text* and return
    (audio_array, segment_list) where each element of segment_list is
    (start_sample, end_sample, label_str).
    """
    audio_parts = []
    segments    = []
    cursor      = 0

    def _append(raw, label):
        nonlocal cursor
        start = cursor
        audio_parts.append(raw)
        cursor += len(raw)
        segments.append((start, cursor, label))

    silence_unit = np.zeros(int(FS * dot_dur), dtype=np.float32)

    for char in text.upper():
        if char == ' ':
            # 7-unit word gap
            gap = np.zeros(int(FS * dot_dur * 7), dtype=np.float32)
            _append(gap, 'silence')
            continue

        if char not in MORSE_DICT:
            continue

        for si, sym in enumerate(MORSE_DICT[char]):
            label = 'dot' if sym == '.' else 'dash'
            seg_audio = _generate_segment(label, dot_dur, freq, snr_db, jitter, False)
            _append(seg_audio, label)

            if si < len(MORSE_DICT[char]) - 1:
                _append(silence_unit.copy(), 'silence')

        # 3-unit letter gap
        letter_gap = np.zeros(int(FS * dot_dur * 3), dtype=np.float32)
        _append(letter_gap, 'silence')

    audio = np.concatenate(audio_parts) if audio_parts else np.array([], dtype=np.float32)
    return audio, segments


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Morse Dataset Generator")
    print("=" * 60)
    ds = generate_dataset(n_samples=3000)
    save_dataset(ds)
    print("\nSample shapes:")
    print(f"  X : {ds['X'].shape}")
    print(f"  y : {ds['y'].shape}")
    print(f"  Class distribution: {ds['n_per_class']}")
