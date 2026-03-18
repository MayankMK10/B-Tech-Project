"""
morse_decoder.py
================
Full Inference Pipeline — Deep Learning Morse Decoder
Deep Learning–Enhanced Probabilistic Morse Code Decoding

Pipeline
--------
  Audio / Video Input
      ↓
  Log-Mel Spectrogram (dataset_generator.compute_log_mel_spectrogram)
      ↓
  Sliding-window patch extraction
      ↓
  CNN Symbol Classifier  (MorseCNN.predict_proba)
      ↓
  Sequence smoothing (simple LSTM-style temporal filter)
      ↓
  Run-length Morse sequence builder
      ↓
  Morse → Character mapping (MORSE_DICT)
      ↓
  Raw character string

The decoded character string is then passed to language_model.py for
sentence reconstruction.
"""

import numpy as np
import os
import cv2
import math
from scipy.io import wavfile
from dataset_generator import (
    compute_log_mel_spectrogram,
    N_MELS, _PATCH_FRAMES, FS,
    LABEL_MAP, CLASS_NAMES
)
from cnn_training import MorseCNN, MODEL_PATH

# ─────────────────────────────────────────────
# Morse dictionary (reverse: code → character)
# ─────────────────────────────────────────────
MORSE_DICT = {
    '.-':'A',   '-...':'B', '-.-.':'C', '-..':'D',  '.':'E',
    '..-.':'F', '--.':'G',  '....':'H', '..':'I',   '.---':'J',
    '-.-':'K',  '.-..':'L', '--':'M',   '-.':'N',   '---':'O',
    '.--.':'P', '--.-':'Q', '.-.':'R',  '...':'S',  '-':'T',
    '..-':'U',  '...-':'V', '.--':'W',  '-..-':'X', '-.--':'Y',
    '--..':'Z', '-----':'0','.----':'1','..---':'2','...--':'3',
    '....-':'4','.....':'5','-....':'6','--...':'7','---..' :'8',
    '----.':'9'
}

# ─────────────────────────────────────────────
# Label indices
# ─────────────────────────────────────────────
SILENCE_IDX = 0
DOT_IDX     = 1
DASH_IDX    = 2


# ══════════════════════════════════════════════
# Sliding-window patch extractor
# ══════════════════════════════════════════════

def extract_patches(spec: np.ndarray,
                    patch_frames: int = _PATCH_FRAMES,
                    hop_frames:   int = 4) -> np.ndarray:
    """
    Slide a window of width *patch_frames* over the spectrogram columns.

    Parameters
    ----------
    spec        : (N_MELS, T)  log-Mel spectrogram
    patch_frames: width of each patch (must match training)
    hop_frames  : step size in frames

    Returns
    -------
    patches : (P, N_MELS, patch_frames)  normalised patches
    """
    n_mels, T = spec.shape
    if T < patch_frames:
        # Pad if too short
        pad  = patch_frames - T
        spec = np.pad(spec, ((0, 0), (0, pad)), mode='edge')
        T    = patch_frames

    starts  = list(range(0, T - patch_frames + 1, hop_frames))
    patches = []
    for s in starts:
        patch = spec[:, s: s + patch_frames].astype(np.float32)
        mu, sigma = patch.mean(), patch.std() + 1e-8
        patches.append((patch - mu) / sigma)

    return np.stack(patches, axis=0)       # (P, N_MELS, patch_frames)


# ══════════════════════════════════════════════
# Simple LSTM-style temporal smoother (NumPy)
# ══════════════════════════════════════════════

class SimpleTemporalSmoother:
    """
    One-layer LSTM-like recurrent smoother that operates on a sequence of
    probability vectors (P, 3). Implemented in pure NumPy.
    Uses exponential moving average — lightweight but effective.
    """

    def __init__(self, alpha: float = 0.4):
        """alpha ∈ (0,1]: smaller → more smoothing."""
        self.alpha = alpha

    def smooth(self, probs: np.ndarray) -> np.ndarray:
        """
        probs : (P, 3) raw softmax outputs
        Returns smoothed (P, 3) probabilities.
        """
        smoothed = np.zeros_like(probs)
        smoothed[0] = probs[0]
        for t in range(1, len(probs)):
            smoothed[t] = self.alpha * probs[t] + (1 - self.alpha) * smoothed[t - 1]
        # Re-normalise rows
        row_sums = smoothed.sum(axis=1, keepdims=True) + 1e-12
        return smoothed / row_sums


# ══════════════════════════════════════════════
# Run-length → Morse string converter
# ══════════════════════════════════════════════

def labels_to_morse(labels: np.ndarray,
                    silence_thresh: int = 6,
                    letter_thresh:  int = 14) -> str:
    """
    Convert a 1-D array of frame-level labels (0=silence, 1=dot, 2=dash)
    into a Morse code string using run-length encoding.

    Parameters
    ----------
    silence_thresh : consecutive silence frames → inter-symbol gap
    letter_thresh  : consecutive silence frames → inter-letter gap
    """
    # ── Run-length encoding ──
    runs = []
    i    = 0
    while i < len(labels):
        lbl   = labels[i]
        start = i
        while i < len(labels) and labels[i] == lbl:
            i += 1
        runs.append((lbl, i - start))

    morse = ''
    for lbl, length in runs:
        if lbl == DOT_IDX:
            morse += '.'
        elif lbl == DASH_IDX:
            morse += '-'
        elif lbl == SILENCE_IDX:
            if length >= letter_thresh:
                morse += ' '   # letter boundary

    return morse.strip()


# ══════════════════════════════════════════════
# Morse → Character
# ══════════════════════════════════════════════

def morse_to_text(morse: str) -> str:
    letters = morse.strip().split(' ')
    text    = ''
    for code in letters:
        code = code.strip()
        if code in MORSE_DICT:
            text += MORSE_DICT[code]
        elif code == '/':
            text += ' '
    return text


# ══════════════════════════════════════════════
# Audio loading (scipy fallback, no librosa)
# ══════════════════════════════════════════════

def _load_audio(path: str) -> tuple:
    """Load WAV audio → (float32 array, sample_rate)."""
    sr, data = wavfile.read(path)
    # Convert to float32 mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    # Normalise
    max_val = np.abs(data).max()
    if max_val > 0:
        data /= max_val
    return data, sr


def _resample_simple(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Nearest-neighbour resample (simple & dependency-free)."""
    if orig_sr == target_sr:
        return audio
    ratio  = target_sr / orig_sr
    n_out  = int(len(audio) * ratio)
    idx    = np.round(np.linspace(0, len(audio) - 1, n_out)).astype(int)
    return audio[idx]


# ══════════════════════════════════════════════
# Threshold-based decoder (legacy baseline)
# ══════════════════════════════════════════════

def decode_audio_threshold(path: str) -> str:
    """Original energy-threshold decoder kept as baseline."""
    audio, sr = _load_audio(path)
    energy  = np.abs(audio)
    window  = max(1, int(0.01 * sr))
    smooth  = np.convolve(energy, np.ones(window) / window, mode='same')
    thresh  = np.mean(smooth) * 2
    binary  = smooth > thresh

    segments, state, count = [], binary[0], 0
    for v in binary:
        if v == state:
            count += 1
        else:
            segments.append((state, count / sr))
            state, count = v, 1
    segments.append((state, count / sr))

    beep = [d for s, d in segments if s]
    if not beep:
        return ''
    unit       = min(beep)
    dot_thresh = unit * 2
    letter_gap = unit * 3

    morse = ''
    for s, d in segments:
        if s:
            morse += '.' if d < dot_thresh else '-'
        else:
            if d > letter_gap:
                morse += ' '
    return morse


def decode_video_threshold(path: str) -> str:
    """Original brightness-threshold decoder kept as baseline."""
    cap    = cv2.VideoCapture(path)
    bright = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright.append(np.mean(gray))
    cap.release()

    bright = np.array(bright)
    thresh = np.mean(bright)
    binary = bright > thresh

    segments, state, count = [], binary[0], 0
    for v in binary:
        if v == state:
            count += 1
        else:
            segments.append((state, count))
            state, count = v, 1
    segments.append((state, count))

    on   = [c for s, c in segments if s]
    if not on:
        return ''
    unit       = min(on)
    dot_thresh = unit * 2
    letter_gap = unit * 3

    morse = ''
    for s, d in segments:
        if s:
            morse += '.' if d < dot_thresh else '-'
        else:
            if d > letter_gap:
                morse += ' '
    return morse


# ══════════════════════════════════════════════
# Deep Learning Decoder
# ══════════════════════════════════════════════

class DeepMorseDecoder:
    """
    Full deep learning inference pipeline.

    Parameters
    ----------
    model_path : path to morse_cnn_model.npz
    alpha      : temporal smoothing factor
    """

    def __init__(self, model_path: str = MODEL_PATH, alpha: float = 0.4):
        self.cnn      = MorseCNN()
        self.smoother = SimpleTemporalSmoother(alpha=alpha)
        self._loaded  = False

        if os.path.exists(model_path):
            self.cnn.load(model_path)
            self._loaded = True
        else:
            print(f"[Decoder] WARNING: model not found at {model_path}. "
                  "Run cnn_training.py first.")

    # ── Audio input ──────────────────────────

    def decode_audio(self, path: str, hop_frames: int = 4) -> tuple:
        """
        Decode audio file → (morse_string, raw_text, confidence_array)
        """
        audio, sr = _load_audio(path)
        if sr != FS:
            audio = _resample_simple(audio, sr, FS)

        spec    = compute_log_mel_spectrogram(audio)
        patches = extract_patches(spec, hop_frames=hop_frames)

        if len(patches) == 0:
            return '', '', np.array([])

        X     = patches.reshape(len(patches), -1).astype(np.float32)
        probs = self.cnn.predict_proba(X)           # (P, 3)
        probs = self.smoother.smooth(probs)
        preds = probs.argmax(axis=1)

        morse = labels_to_morse(preds)
        text  = morse_to_text(morse)
        return morse, text, probs

    # ── Video input ──────────────────────────

    def decode_video(self, path: str) -> tuple:
        """
        Extract brightness signal from video → decode as if audio.
        Returns (morse_string, raw_text).
        """
        cap    = cv2.VideoCapture(path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        bright = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright.append(float(np.mean(gray)))
        cap.release()

        if not bright:
            return '', ''

        bright = np.array(bright, dtype=np.float32)

        # Normalise to [-1, 1]
        b_min, b_max = bright.min(), bright.max()
        if b_max > b_min:
            bright = 2 * (bright - b_min) / (b_max - b_min) - 1
        else:
            return '', ''

        # Resample to audio sample rate (treat each frame as a sample cluster)
        samples_per_frame = max(1, int(FS / fps))
        audio = np.repeat(bright, samples_per_frame).astype(np.float32)

        spec    = compute_log_mel_spectrogram(audio)
        patches = extract_patches(spec)

        if len(patches) == 0:
            return '', ''

        X     = patches.reshape(len(patches), -1).astype(np.float32)
        probs = self.cnn.predict_proba(X)
        probs = self.smoother.smooth(probs)
        preds = probs.argmax(axis=1)

        morse = labels_to_morse(preds)
        text  = morse_to_text(morse)
        return morse, text


# ── CLI usage ────────────────────────────────

if __name__ == '__main__':
    import sys

    decoder = DeepMorseDecoder()

    print("1  Audio file (WAV)")
    print("2  Video file (MP4)")
    choice = input("Select: ")

    if choice == '1':
        fpath = input("WAV path [morse.wav]: ").strip() or 'morse.wav'
        morse, text, conf = decoder.decode_audio(fpath)
        print(f"\nMorse  : {morse}")
        print(f"Text   : {text}")
        print(f"Mean confidence: {conf.max(axis=1).mean()*100:.1f}%")
    elif choice == '2':
        fpath = input("MP4 path [morse_video.mp4]: ").strip() or 'morse_video.mp4'
        morse, text = decoder.decode_video(fpath)
        print(f"\nMorse  : {morse}")
        print(f"Text   : {text}")
    else:
        print("Invalid choice.")
