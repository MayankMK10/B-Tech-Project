"""
main_pipeline.py
================
Master Orchestrator — Deep Learning Morse Decoding System
Deep Learning–Enhanced Probabilistic Morse Code Decoding

Execution Modes
---------------
  1  Full pipeline (train + evaluate + decode demo)
  2  Train CNN only
  3  Decode audio file
  4  Decode video file
  5  Run evaluation suite only
  6  Language model demo

Entry-point for the complete research system.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Local modules ────────────────────────────
from dataset_generator  import generate_dataset, save_dataset, load_dataset, generate_morse_audio_labeled
from cnn_training        import run_training, MorseCNN, MODEL_PATH, plot_training_history
from morse_decoder       import DeepMorseDecoder, decode_audio_threshold, morse_to_text
from language_model      import UnigramLM, BigramLM, NeuralCharLM, segment_text, compare_lm
from evaluation          import run_full_evaluation

FIGURE_DIR   = 'figures'
DATASET_PATH = 'morse_dataset.npz'
os.makedirs(FIGURE_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# Banner
# ══════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  Deep Learning–Enhanced Probabilistic Morse Code Decoding   ║
║  Research System  v2.0                                       ║
╚══════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════
# Mode 1 – Full pipeline
# ══════════════════════════════════════════════

def run_full_pipeline(demo_text: str = 'SOS'):
    print(BANNER)
    print("▶  Mode: Full Pipeline\n")

    # ── Step 1: Dataset ──────────────────────
    print("━" * 55)
    print("  STEP 1 / 5  —  Dataset Generation")
    print("━" * 55)
    if os.path.exists(DATASET_PATH):
        print(f"  Loading existing dataset: {DATASET_PATH}")
        ds = load_dataset(DATASET_PATH)
    else:
        print("  Generating synthetic dataset (3 000 samples) …")
        ds = generate_dataset(n_samples=3000)
        save_dataset(ds, DATASET_PATH)
    print(f"  Dataset shape: X={ds['X'].shape}  y={ds['y'].shape}\n")

    # ── Step 2: CNN Training ─────────────────
    print("━" * 55)
    print("  STEP 2 / 5  —  CNN Training")
    print("━" * 55)
    model = run_training(dataset_path=DATASET_PATH)

    # ── Step 3: Language Models ──────────────
    print("\n" + "━" * 55)
    print("  STEP 3 / 5  —  Language Model Initialisation")
    print("━" * 55)
    unigram = UnigramLM()
    bigram  = BigramLM(unigram)
    neural  = NeuralCharLM()

    # ── Step 4: Decoding Demo ────────────────
    print("\n" + "━" * 55)
    print(f"  STEP 4 / 5  —  Decoding Demo  (text='{demo_text}')")
    print("━" * 55)
    _demo_decode(demo_text, model, unigram, bigram, neural)

    # ── Step 5: Evaluation ───────────────────
    print("\n" + "━" * 55)
    print("  STEP 5 / 5  —  Evaluation Suite")
    print("━" * 55)
    run_full_evaluation()

    print("\n✔  Full pipeline completed.")
    print(f"  Figures saved to ./{FIGURE_DIR}/")


# ══════════════════════════════════════════════
# Decoding demo helper
# ══════════════════════════════════════════════

def _demo_decode(text: str,
                  model: MorseCNN,
                  unigram: UnigramLM,
                  bigram: BigramLM,
                  neural: NeuralCharLM):
    """
    Generate a synthetic Morse audio for *text*, run both the threshold
    decoder and the deep CNN decoder, then compare language model outputs.
    """
    from dataset_generator import generate_morse_audio_labeled, FS
    from scipy.io import wavfile
    from morse_decoder import (
        compute_log_mel_spectrogram, extract_patches,
        labels_to_morse, SimpleTemporalSmoother
    )

    demo_wav = '_demo_temp.wav'

    # Generate audio
    audio, segments = generate_morse_audio_labeled(text, dot_dur=0.1, snr_db=20)
    wavfile.write(demo_wav, FS, (audio * 32767).astype(np.int16))
    print(f"  Demo audio: {len(audio)/FS:.2f}s  ({len(segments)} segments)")

    # ── Threshold decoder ──
    morse_thresh = decode_audio_threshold(demo_wav)
    text_thresh  = morse_to_text(morse_thresh)
    print(f"\n  [Threshold]  Morse : {morse_thresh}")
    print(f"  [Threshold]  Text  : {text_thresh}")

    # ── CNN decoder ──
    from dataset_generator import compute_log_mel_spectrogram, _PATCH_FRAMES
    spec    = compute_log_mel_spectrogram(audio)
    patches = extract_patches(spec)

    if len(patches) > 0:
        X     = patches.reshape(len(patches), -1).astype(np.float32)
        probs = model.predict_proba(X)
        sm    = SimpleTemporalSmoother(alpha=0.4)
        probs = sm.smooth(probs)
        preds = probs.argmax(axis=1)

        morse_cnn = labels_to_morse(preds)
        text_cnn  = morse_to_text(morse_cnn)
        mean_conf = probs.max(axis=1).mean() * 100
    else:
        morse_cnn = ''
        text_cnn  = ''
        mean_conf = 0

    print(f"\n  [CNN]        Morse : {morse_cnn}")
    print(f"  [CNN]        Text  : {text_cnn}")
    print(f"  [CNN]        Confidence: {mean_conf:.1f}%")

    # ── Language model reconstruction ──
    raw = text_cnn if text_cnn else text_thresh
    if raw:
        print(f"\n  [Language Models] Input: '{raw}'")
        results = compare_lm(raw, unigram, bigram, neural)
        for name, res in results.items():
            sent = res.get('sentence', '')
            print(f"    {name:<20}: {sent}")

    # Clean up temp file
    if os.path.exists(demo_wav):
        os.remove(demo_wav)

    # ── Spectrogram figure ──
    _plot_demo_spectrogram(audio, segments, text)


def _plot_demo_spectrogram(audio: np.ndarray, segments: list, title_text: str):
    """Annotated spectrogram with segment boundaries."""
    from dataset_generator import compute_log_mel_spectrogram, FS, _PATCH_FRAMES

    spec = compute_log_mel_spectrogram(audio)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis',
              interpolation='nearest',
              extent=[0, len(audio)/FS, 0, spec.shape[0]])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Mel Bin')
    ax.set_title(f'Annotated Morse Spectrogram — "{title_text}"',
                 fontsize=13, fontweight='bold')

    colours = {'dot': 'lime', 'dash': 'orange', 'silence': 'red'}
    for start, end, label in segments[:40]:
        t_s = start / FS
        t_e = end   / FS
        c   = colours.get(label, 'white')
        ax.axvspan(t_s, t_e, alpha=0.25, color=c)

    import matplotlib.patches as _mp
    patches = [_mp.Patch(color=c, label=l, alpha=0.5) for l, c in colours.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, 'demo_spectrogram.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Demo spectrogram → {out}")


# ══════════════════════════════════════════════
# Mode 3 / 4 – Decode existing file
# ══════════════════════════════════════════════

def decode_file(path: str, is_audio: bool, unigram: UnigramLM, bigram: BigramLM):
    model   = MorseCNN()
    decoder = DeepMorseDecoder()

    if is_audio:
        morse, raw_text, probs = decoder.decode_audio(path)
        print(f"\nMorse Code : {morse}")
        print(f"Raw Text   : {raw_text}")
        if len(probs):
            print(f"Confidence : {probs.max(axis=1).mean()*100:.1f}%")
    else:
        morse, raw_text = decoder.decode_video(path)
        print(f"\nMorse Code : {morse}")
        print(f"Raw Text   : {raw_text}")

    if raw_text:
        seg, conf = segment_text(raw_text, unigram, bigram)
        print(f"\nFinal Sentence : {' '.join(seg)}")
        print(f"LM Confidence  : {conf*100:.1f}%")


# ══════════════════════════════════════════════
# Menu
# ══════════════════════════════════════════════

def interactive_menu():
    print(BANNER)
    print("  Select mode:")
    print("  1  Full pipeline  (train + evaluate + demo)")
    print("  2  Train CNN only")
    print("  3  Decode audio file")
    print("  4  Decode video file")
    print("  5  Run evaluation suite only")
    print("  6  Language model demo")
    print()
    return input("  Choice [1]: ").strip() or '1'


def main():
    choice = interactive_menu()

    if choice == '1':
        demo = input("  Demo decode text [SOS]: ").strip().upper() or 'SOS'
        run_full_pipeline(demo_text=demo)

    elif choice == '2':
        run_training(dataset_path=DATASET_PATH)

    elif choice in ('3', '4'):
        is_audio = (choice == '3')
        default  = 'morse.wav' if is_audio else 'morse_video.mp4'
        ext      = 'WAV' if is_audio else 'MP4'
        path     = input(f"  {ext} path [{default}]: ").strip() or default
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            sys.exit(1)
        uni = UnigramLM()
        bi  = BigramLM(uni)
        decode_file(path, is_audio, uni, bi)

    elif choice == '5':
        run_full_evaluation()

    elif choice == '6':
        text = input("  Enter continuous text (no spaces) [HELLOMORSE]: ").strip().upper() or 'HELLOMORSE'
        uni    = UnigramLM()
        bi     = BigramLM(uni)
        neural = NeuralCharLM()
        results = compare_lm(text, uni, bi, neural)
        print(f"\nInput: {text}\n")
        for name, res in results.items():
            print(f"  [{name}]  {res.get('sentence','')}")
    else:
        print("  Invalid choice.")


if __name__ == '__main__':
    main()