"""
evaluation.py
=============
Research-Grade Evaluation Suite
Deep Learning–Enhanced Probabilistic Morse Code Decoding

Covers
------
  1.  Symbol-level classification metrics  (accuracy / precision / recall / F1)
  2.  Confusion matrix
  3.  Word & sentence reconstruction accuracy
  4.  Noise robustness experiments (SNR sweep)
  5.  CNN vs Threshold baseline comparison
  6.  All research-quality figure generation
"""

import numpy as np
import os
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

from dataset_generator import (
    generate_dataset, _PATCH_FRAMES, N_MELS, CLASS_NAMES,
    compute_log_mel_spectrogram, _generate_segment, FS
)
from cnn_training  import MorseCNN, MODEL_PATH
from morse_decoder import (
    extract_patches, labels_to_morse, morse_to_text,
    decode_audio_threshold, SimpleTemporalSmoother,
    MORSE_DICT
)
from language_model import UnigramLM, BigramLM, segment_text

PLOT_DIR = 'figures'
os.makedirs(PLOT_DIR, exist_ok=True)

CLASS_NAMES_DISPLAY = ['Silence', 'Dot', 'Dash']


# ══════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════

def _load_model(path: str = MODEL_PATH) -> MorseCNN:
    m = MorseCNN()
    if os.path.exists(path):
        m.load(path)
    else:
        print(f"[Eval] WARNING: model not found ({path}). Using random weights.")
    return m


def _confusion_matrix(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _precision_recall_f1(cm: np.ndarray) -> dict:
    n = cm.shape[0]
    metrics = {}
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p  = tp / (tp + fp + 1e-12)
        r  = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        metrics[CLASS_NAMES[i]] = {'precision': p, 'recall': r, 'f1': f1}
    return metrics


def _word_accuracy(pred_words: list, true_words: list) -> float:
    if not true_words:
        return 0.0
    matches = sum(1 for p, t in zip(pred_words, true_words) if p == t)
    return matches / max(len(pred_words), len(true_words))


def _char_error_rate(pred: str, ref: str) -> float:
    """Edit distance / len(ref)."""
    m, n = len(ref), len(pred)
    dp   = np.arange(n + 1)
    for i in range(1, m + 1):
        new = np.empty(n + 1, dtype=int)
        new[0] = i
        for j in range(1, n + 1):
            cost  = 0 if ref[i-1] == pred[j-1] else 1
            new[j] = min(new[j-1] + 1, dp[j] + 1, dp[j-1] + cost)
        dp = new
    return dp[n] / max(len(ref), 1)


# ══════════════════════════════════════════════
# 1. Symbol Classification Report
# ══════════════════════════════════════════════

def evaluate_symbol_classification(model: MorseCNN,
                                    n_test: int = 600) -> dict:
    """
    Generate a held-out test set and compute classification metrics.
    """
    print("\n[Eval] Symbol Classification Evaluation …")
    ds       = generate_dataset(n_samples=n_test, seed=999)
    X_flat   = ds['X'].reshape(len(ds['y']), -1).astype(np.float32)
    y_true   = ds['y']

    y_pred   = model.predict(X_flat)
    acc      = float(np.mean(y_pred == y_true))
    cm       = _confusion_matrix(y_true, y_pred)
    prf      = _precision_recall_f1(cm)

    print(f"  Overall Accuracy : {acc*100:.2f}%")
    print(f"\n  {'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "─" * 40)
    for cls, m in prf.items():
        print(f"  {cls:<10} {m['precision']*100:>9.2f}%"
              f" {m['recall']*100:>7.2f}% {m['f1']*100:>7.2f}%")

    return {'accuracy': acc, 'confusion_matrix': cm, 'prf': prf, 'y_true': y_true, 'y_pred': y_pred}


# ══════════════════════════════════════════════
# 2. Confusion Matrix Plot
# ══════════════════════════════════════════════

def plot_confusion_matrix(cm: np.ndarray, save_dir: str = PLOT_DIR):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES_DISPLAY, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES_DISPLAY, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label',      fontsize=13)
    ax.set_title('CNN Morse Symbol Confusion Matrix', fontsize=14, fontweight='bold')

    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
    for i in range(3):
        for j in range(3):
            colour = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)',
                    ha='center', va='center', fontsize=10, color=colour)

    plt.tight_layout()
    out = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Confusion matrix → {out}")


# ══════════════════════════════════════════════
# 3. Noise Robustness Experiment
# ══════════════════════════════════════════════

def noise_robustness_experiment(model: MorseCNN,
                                 snr_levels: list = None,
                                 n_per_level: int = 200,
                                 save_dir: str = PLOT_DIR) -> dict:
    """
    Evaluate symbol classification accuracy across a range of SNR values.
    Also compares against the threshold-based baseline.
    """
    if snr_levels is None:
        snr_levels = [2, 5, 8, 12, 16, 20, 25, 30]

    print("\n[Eval] Noise Robustness Experiment …")

    classes   = ['silence', 'dot', 'dash']
    cnn_accs  = []
    base_accs = []

    for snr in snr_levels:
        ds     = generate_dataset(n_samples=n_per_level, snr_range=(snr, snr + 0.5), seed=snr*10)
        X_flat = ds['X'].reshape(len(ds['y']), -1).astype(np.float32)
        y_true = ds['y']

        # CNN accuracy
        preds    = model.predict(X_flat)
        cnn_acc  = float(np.mean(preds == y_true)) * 100
        cnn_accs.append(cnn_acc)

        # Threshold baseline: classify by energy mean of the patch
        base_preds = []
        for patch in ds['X']:
            energy = np.abs(patch).mean()
            # Heuristic: low energy → silence, high variance → dash
            if energy < 0.05:
                base_preds.append(0)   # silence
            elif patch.std() > 0.8:
                base_preds.append(2)   # dash
            else:
                base_preds.append(1)   # dot
        base_acc = float(np.mean(np.array(base_preds) == y_true)) * 100
        base_accs.append(base_acc)

        print(f"  SNR {snr:>3} dB  |  CNN: {cnn_acc:>5.1f}%  |  Baseline: {base_acc:>5.1f}%")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(snr_levels, cnn_accs,  'o-', lw=2.5, color='steelblue',
            label='CNN (Deep Learning)', markersize=7)
    ax.plot(snr_levels, base_accs, 's--', lw=2.5, color='tomato',
            label='Threshold Baseline', markersize=7)
    ax.set_title('Symbol Accuracy vs SNR', fontsize=14, fontweight='bold')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105); ax.legend(fontsize=11); ax.grid(alpha=0.3)

    ax = axes[1]
    gain = [c - b for c, b in zip(cnn_accs, base_accs)]
    colours = ['steelblue' if g >= 0 else 'tomato' for g in gain]
    ax.bar(snr_levels, gain, color=colours, width=1.8, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', lw=1)
    ax.set_title('CNN Gain over Baseline', fontsize=14, fontweight='bold')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Accuracy Gain (pp)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, 'noise_robustness.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Noise robustness plot → {out}")

    return {'snr': snr_levels, 'cnn': cnn_accs, 'baseline': base_accs}


# ══════════════════════════════════════════════
# 4. Language Model Comparison
# ══════════════════════════════════════════════

def evaluate_language_models(save_dir: str = PLOT_DIR):
    """
    Compare unigram vs bigram segmentation on a set of known sentences.
    """
    print("\n[Eval] Language Model Comparison …")

    test_sentences = [
        ('HELLOWORLD',        'HELLO WORLD'),
        ('THISISATEST',       'THIS IS A TEST'),
        ('MORSECODEWORKS',    'MORSE CODE WORKS'),
        ('DEEPLEARNING',      'DEEP LEARNING'),
        ('GOODMORNING',       'GOOD MORNING'),
        ('SENDHELP',          'SEND HELP'),
        ('ATTACKATDAWN',      'ATTACK AT DAWN'),
        ('THEQUICKBROWNFOX',  'THE QUICK BROWN FOX'),
    ]

    uni = UnigramLM()
    bi  = BigramLM(uni)

    uni_accs, bi_accs = [], []
    rows = []

    for raw, ref in test_sentences:
        seg_uni, _ = segment_text(raw, uni)
        seg_bi,  _ = segment_text(raw, uni, bi)

        pred_uni = ' '.join(seg_uni)
        pred_bi  = ' '.join(seg_bi)

        acc_uni = 1.0 - _char_error_rate(pred_uni, ref)
        acc_bi  = 1.0 - _char_error_rate(pred_bi,  ref)
        uni_accs.append(max(0, acc_uni))
        bi_accs.append(max(0, acc_bi))
        rows.append((raw, ref, pred_uni, pred_bi))

    print(f"\n  {'Input':<20} {'Unigram':<22} {'Bigram':<22}")
    print("  " + "─" * 65)
    for raw, ref, u, b in rows:
        print(f"  {raw:<20} {u:<22} {b:<22}")

    mean_uni = np.mean(uni_accs) * 100
    mean_bi  = np.mean(bi_accs)  * 100
    print(f"\n  Mean Word Accuracy — Unigram: {mean_uni:.1f}%  Bigram: {mean_bi:.1f}%")

    # Plot
    x     = np.arange(len(test_sentences))
    width = 0.35
    labels = [r[0][:10] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, [v*100 for v in uni_accs], width, label='Unigram LM',
           color='cornflowerblue', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, [v*100 for v in bi_accs],  width, label='Bigram LM',
           color='mediumseagreen', edgecolor='black', linewidth=0.5)
    ax.set_title('Language Model Sentence Reconstruction Accuracy',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Character Accuracy (%)'); ax.set_ylim(0, 110)
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, 'lm_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] LM comparison plot → {out}")

    return {'unigram_mean': mean_uni, 'bigram_mean': mean_bi}


# ══════════════════════════════════════════════
# 5. Spectrogram Visualisation
# ══════════════════════════════════════════════

def plot_spectrogram_examples(save_dir: str = PLOT_DIR):
    """Plot log-Mel spectrogram patches for dot, dash, silence."""
    print("\n[Eval] Spectrogram examples …")

    from dataset_generator import _generate_segment, compute_log_mel_spectrogram

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    configs = [
        ('silence', 0.1, 800, 20, 0.0, False, 'Silence Segment'),
        ('dot',     0.1, 800, 20, 0.0, False, 'Dot Segment'),
        ('dash',    0.1, 800, 20, 0.0, False, 'Dash Segment'),
    ]

    for ax, (cls, dot_d, freq, snr, jit, dist, title) in zip(axes, configs):
        audio = _generate_segment(cls, dot_d, freq, snr, jit, dist)
        spec  = compute_log_mel_spectrogram(audio)
        im = ax.imshow(spec, aspect='auto', origin='lower',
                       cmap='magma', interpolation='nearest')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time Frames'); ax.set_ylabel('Mel Bin')
        plt.colorbar(im, ax=ax, label='Log Power')

    plt.suptitle('Log-Mel Spectrograms of Morse Symbols',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(save_dir, 'spectrogram_examples.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Spectrogram examples → {out}")


# ══════════════════════════════════════════════
# 6. CNN Architecture Diagram
# ══════════════════════════════════════════════

def plot_cnn_architecture(save_dir: str = PLOT_DIR):
    """Draw a clean architecture block diagram."""
    print("\n[Eval] CNN architecture diagram …")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')

    layers = [
        (0.5, 'Input\n(32×16\nspectrogram)', '#4C72B0', 2.0),
        (2.0, 'Flatten\n512-dim',             '#4C72B0', 1.4),
        (3.5, 'Dense 256\n+ ReLU',            '#55A868', 1.4),
        (5.0, 'Dense 128\n+ ReLU\n+ BatchNorm','#C44E52', 1.6),
        (6.7, 'Dense 64\n+ ReLU',             '#55A868', 1.4),
        (8.2, 'Dense 3\n(Output)',             '#8172B2', 1.4),
        (9.7, 'Softmax\n3 classes',            '#CCB974', 1.4),
        (11.2,'Temporal\nSmoother\n(EMA)',      '#64B5CD', 1.6),
        (12.8,'Symbol\nPrediction',            '#4C72B0', 1.4),
    ]

    prev_x, prev_w = None, None
    for x, label, colour, h in layers:
        w = 1.2
        rect = mpatches.FancyBboxPatch(
            (x - w/2, 2 - h/2), w, h,
            boxstyle='round,pad=0.1',
            facecolor=colour, edgecolor='black',
            linewidth=1.2, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x, 2, label, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

        if prev_x is not None:
            ax.annotate('', xy=(x - w/2, 2),
                        xytext=(prev_x + prev_w/2, 2),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))
        prev_x, prev_w = x, w

    ax.set_title('CNN-Based Morse Symbol Classifier — Architecture',
                 fontsize=13, fontweight='bold', pad=12)

    # Class labels
    ax.text(9.7, 0.5, 'Classes: Silence / Dot / Dash',
            ha='center', fontsize=9, style='italic', color='#555')

    plt.tight_layout()
    out = os.path.join(save_dir, 'cnn_architecture.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Architecture diagram → {out}")


# ══════════════════════════════════════════════
# 7. Full Pipeline Diagram
# ══════════════════════════════════════════════

def plot_pipeline_diagram(save_dir: str = PLOT_DIR):
    """High-level system pipeline diagram."""
    print("\n[Eval] Pipeline diagram …")

    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4); ax.axis('off')

    steps = [
        (0.8,  'Audio /\nVideo Input',           '#2196F3'),
        (2.6,  'Log-Mel\nSpectrogram',           '#03A9F4'),
        (4.4,  'Sliding\nWindow\nPatches',        '#00BCD4'),
        (6.2,  'CNN\nSymbol\nClassifier',         '#4CAF50'),
        (8.0,  'Temporal\nSmoothing\n(EMA)',      '#8BC34A'),
        (9.8,  'Run-Length\nMorse\nBuilder',      '#FFC107'),
        (11.6, 'Morse →\nCharacter\nDecoding',    '#FF9800'),
        (13.4, 'Language\nModel\nSegmentation',   '#F44336'),
        (15.2, 'Final\nSentence',                 '#9C27B0'),
    ]

    prev_x = None
    for x, label, colour in steps:
        w, h = 1.5, 2.2
        rect = mpatches.FancyBboxPatch(
            (x - w/2, 2 - h/2), w, h,
            boxstyle='round,pad=0.12',
            facecolor=colour, edgecolor='white',
            linewidth=1.5, alpha=0.88
        )
        ax.add_patch(rect)
        ax.text(x, 2, label, ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold')

        if prev_x is not None:
            ax.annotate('', xy=(x - w/2, 2),
                        xytext=(prev_x + w/2, 2),
                        arrowprops=dict(arrowstyle='->', lw=2.0, color='#333'))
        prev_x = x

    ax.set_title('Deep Learning–Enhanced Morse Decoding Pipeline',
                 fontsize=13, fontweight='bold', y=0.97)
    plt.tight_layout()
    out = os.path.join(save_dir, 'pipeline_diagram.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Pipeline diagram → {out}")


# ══════════════════════════════════════════════
# 8. Confidence vs Noise plot
# ══════════════════════════════════════════════

def plot_confidence_vs_noise(model: MorseCNN,
                              snr_levels: list = None,
                              save_dir: str = PLOT_DIR):
    if snr_levels is None:
        snr_levels = [2, 5, 8, 12, 16, 20, 25, 30]

    print("\n[Eval] Confidence vs Noise …")
    mean_confs = []

    for snr in snr_levels:
        ds    = generate_dataset(n_samples=120, snr_range=(snr, snr + 0.5), seed=snr*7)
        X_f   = ds['X'].reshape(len(ds['y']), -1).astype(np.float32)
        probs = model.predict_proba(X_f)
        mean_confs.append(probs.max(axis=1).mean() * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_levels, mean_confs, 'D-', color='darkorange', lw=2.5, markersize=8)
    ax.fill_between(snr_levels, mean_confs, alpha=0.15, color='darkorange')
    ax.set_title('Model Confidence vs Noise Level', fontsize=14, fontweight='bold')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Mean Max-Class Confidence (%)')
    ax.set_ylim(0, 105); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, 'confidence_vs_noise.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Confidence vs noise → {out}")


# ══════════════════════════════════════════════
# Master evaluation runner
# ══════════════════════════════════════════════

def run_full_evaluation():
    print("=" * 65)
    print("  Full Research Evaluation Suite")
    print("=" * 65)

    model = _load_model()

    # 1. Symbol metrics
    sym_results = evaluate_symbol_classification(model)
    plot_confusion_matrix(sym_results['confusion_matrix'])

    # 2. Noise robustness
    noise_results = noise_robustness_experiment(model)

    # 3. Language model comparison
    lm_results = evaluate_language_models()

    # 4. Visualisations
    plot_spectrogram_examples()
    plot_cnn_architecture()
    plot_pipeline_diagram()
    plot_confidence_vs_noise(model)

    # 5. Summary report
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Symbol Accuracy     : {sym_results['accuracy']*100:.2f}%")
    print(f"  Avg CNN SNR Acc     : {np.mean(noise_results['cnn']):.1f}%")
    print(f"  Avg Baseline SNR Acc: {np.mean(noise_results['baseline']):.1f}%")
    print(f"  Unigram LM Acc      : {lm_results['unigram_mean']:.1f}%")
    print(f"  Bigram LM Acc       : {lm_results['bigram_mean']:.1f}%")

    prf = sym_results['prf']
    print(f"\n  Per-Class F1:")
    for cls, m in prf.items():
        print(f"    {cls:<10} F1={m['f1']*100:.2f}%")

    print(f"\n  All figures saved to: ./{PLOT_DIR}/")
    print("=" * 65)

    return sym_results, noise_results, lm_results


if __name__ == '__main__':
    run_full_evaluation()
