"""
cnn_training.py
===============
CNN-Based Morse Symbol Classifier — Pure NumPy Implementation
Deep Learning–Enhanced Probabilistic Morse Code Decoding

Architecture
------------
  Input  : (N_MELS=32, PATCH_FRAMES=16) log-Mel spectrogram patch
           flattened → 512-dim vector

  Layer 1 (Conv-like, implemented as weight matrix)
           512 → 256  + ReLU + MaxPool  →  128
  Layer 2
           128 →  64  + ReLU + BatchNorm
  Layer 3  (Dense)
            64 →  32  + ReLU
  Output   32 →   3   + Softmax  (silence / dot / dash)

Training uses mini-batch gradient descent with cross-entropy loss.
The trained weights are serialised with numpy.savez → morse_cnn_model.npz
(also written as morse_cnn_model.h5 placeholder for compatibility).

Usage
-----
  python cnn_training.py
"""

import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset_generator import (
    generate_dataset, load_dataset, save_dataset,
    CLASS_NAMES, N_MELS, _PATCH_FRAMES
)

# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
LEARNING_RATE = 0.01
EPOCHS        = 40
BATCH_SIZE    = 64
HIDDEN1       = 256
HIDDEN2       = 128
HIDDEN3       = 64
N_CLASSES     = 3
INPUT_DIM     = N_MELS * _PATCH_FRAMES          # 32 × 16 = 512
MODEL_PATH    = 'morse_cnn_model.npz'
PLOT_DIR      = 'figures'


# ══════════════════════════════════════════════
# Activation functions & helpers
# ══════════════════════════════════════════════

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def cross_entropy_loss(probs, y):
    """Mean cross-entropy loss."""
    n = len(y)
    log_p = np.log(probs[np.arange(n), y] + 1e-12)
    return -log_p.mean()

def batch_norm_forward(x, gamma, beta, eps=1e-8):
    mu    = x.mean(axis=0)
    var   = x.var(axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta, mu, var, x_hat

def batch_norm_backward(d_out, x_hat, gamma, var, eps=1e-8):
    N = d_out.shape[0]
    d_gamma = (d_out * x_hat).sum(axis=0)
    d_beta  = d_out.sum(axis=0)
    d_xhat  = d_out * gamma
    d_var   = (-0.5 * d_xhat * x_hat / (var + eps)).sum(axis=0)
    d_mu    = (-d_xhat / np.sqrt(var + eps)).sum(axis=0)
    d_x     = (d_xhat / np.sqrt(var + eps)
               + 2 * d_var * x_hat / N
               + d_mu / N)
    return d_x, d_gamma, d_beta


# ══════════════════════════════════════════════
# Network class
# ══════════════════════════════════════════════

class MorseCNN:
    """
    Fully-connected network (mimicking a flattened CNN output pipeline).
    Layers: Linear → ReLU → Linear → ReLU+BN → Linear → ReLU → Linear → Softmax
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        s   = lambda fan_in: np.sqrt(2.0 / fan_in)   # He init

        self.W1 = rng.standard_normal((INPUT_DIM, HIDDEN1)).astype(np.float32) * s(INPUT_DIM)
        self.b1 = np.zeros(HIDDEN1, dtype=np.float32)

        self.W2 = rng.standard_normal((HIDDEN1, HIDDEN2)).astype(np.float32) * s(HIDDEN1)
        self.b2 = np.zeros(HIDDEN2, dtype=np.float32)

        # BatchNorm parameters for layer 2
        self.gamma2 = np.ones(HIDDEN2,  dtype=np.float32)
        self.beta2  = np.zeros(HIDDEN2, dtype=np.float32)

        self.W3 = rng.standard_normal((HIDDEN2, HIDDEN3)).astype(np.float32) * s(HIDDEN2)
        self.b3 = np.zeros(HIDDEN3, dtype=np.float32)

        self.W4 = rng.standard_normal((HIDDEN3, N_CLASSES)).astype(np.float32) * s(HIDDEN3)
        self.b4 = np.zeros(N_CLASSES, dtype=np.float32)

        # Cache for backprop
        self._cache = {}

    # ── Forward pass ──────────────────────────

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """X: (B, INPUT_DIM). Returns softmax probabilities (B, 3)."""
        # Layer 1
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)

        # Layer 2 + BatchNorm
        z2 = a1 @ self.W2 + self.b2
        if training:
            a2_bn, mu2, var2, xhat2 = batch_norm_forward(z2, self.gamma2, self.beta2)
        else:
            mu2 = var2 = None
            xhat2 = (z2 - z2.mean(axis=0)) / (z2.std(axis=0) + 1e-8)
            a2_bn = self.gamma2 * xhat2 + self.beta2
        a2 = relu(a2_bn)

        # Layer 3
        z3 = a2 @ self.W3 + self.b3
        a3 = relu(z3)

        # Output
        z4   = a3 @ self.W4 + self.b4
        prob = softmax(z4)

        if training:
            self._cache = dict(X=X, z1=z1, a1=a1, z2=z2,
                               xhat2=xhat2, mu2=mu2, var2=var2,
                               a2_bn=a2_bn, a2=a2,
                               z3=z3, a3=a3, z4=z4)
        return prob

    # ── Backward pass ─────────────────────────

    def backward(self, prob: np.ndarray, y: np.ndarray) -> dict:
        """Compute gradients; return dict."""
        c = self._cache
        B = len(y)

        # Output gradient (softmax + cross-entropy combined)
        d4 = prob.copy()
        d4[np.arange(B), y] -= 1
        d4 /= B

        dW4 = c['a3'].T @ d4
        db4 = d4.sum(axis=0)

        # Layer 3
        da3 = d4 @ self.W4.T * relu_grad(c['z3'])
        dW3 = c['a2'].T @ da3
        db3 = da3.sum(axis=0)

        # Layer 2 + BatchNorm
        da2 = da3 @ self.W3.T * relu_grad(c['a2_bn'])
        da2_pre, dgamma2, dbeta2 = batch_norm_backward(
            da2, c['xhat2'], self.gamma2, c['var2'])
        dz2 = da2_pre
        dW2 = c['a1'].T @ dz2
        db2 = dz2.sum(axis=0)

        # Layer 1
        da1 = dz2 @ self.W2.T * relu_grad(c['z1'])
        dW1 = c['X'].T @ da1
        db1 = da1.sum(axis=0)

        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2,
                    gamma2=dgamma2, beta2=dbeta2,
                    W3=dW3, b3=db3, W4=dW4, b4=db4)

    # ── Gradient-descent update ────────────────

    def update(self, grads: dict, lr: float):
        for k, v in grads.items():
            param = getattr(self, k)
            setattr(self, k, param - lr * np.clip(v, -5, 5))

    # ── Predict ───────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions."""
        prob = self.forward(X, training=False)
        return prob.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)

    # ── Save / Load ───────────────────────────

    def save(self, path: str = MODEL_PATH):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 gamma2=self.gamma2, beta2=self.beta2,
                 W3=self.W3, b3=self.b3,
                 W4=self.W4, b4=self.b4)
        # Also write a stub h5-named file for compatibility
        h5_path = path.replace('.npz', '.h5')
        with open(h5_path, 'w') as fh:
            fh.write(f"# Morse CNN weights stored in {path}\n")
        print(f"[CNN] Model saved → {path}  (alias: {h5_path})")

    def load(self, path: str = MODEL_PATH):
        data = np.load(path)
        for k in data.files:
            setattr(self, k, data[k])
        print(f"[CNN] Model loaded ← {path}")


# ══════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════

def train(model: MorseCNN,
          X_train, y_train,
          X_val,   y_val,
          epochs:  int   = EPOCHS,
          batch:   int   = BATCH_SIZE,
          lr:      float = LEARNING_RATE) -> dict:

    history = {'train_loss': [], 'val_loss': [],
                'train_acc':  [], 'val_acc':  []}

    n = len(y_train)
    rng = np.random.default_rng(0)

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'Train Acc':>10}  {'Val Acc':>9}  {'Time':>6}")
    print("─" * 65)

    for ep in range(1, epochs + 1):
        t0 = time.time()

        # ── Shuffle ──
        idx = rng.permutation(n)
        X_s, y_s = X_train[idx], y_train[idx]

        # ── Mini-batches ──
        ep_loss = []
        for start in range(0, n, batch):
            Xb = X_s[start:start+batch]
            yb = y_s[start:start+batch]
            prob = model.forward(Xb, training=True)
            loss = cross_entropy_loss(prob, yb)
            grads = model.backward(prob, yb)
            model.update(grads, lr)
            ep_loss.append(loss)

        # ── Metrics ──
        tr_loss = float(np.mean(ep_loss))
        tr_acc  = float(np.mean(model.predict(X_train) == y_train))

        vp      = model.forward(X_val, training=False)
        vl      = float(cross_entropy_loss(vp, y_val))
        va      = float(np.mean(vp.argmax(1) == y_val))

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va)

        elapsed = time.time() - t0
        print(f"{ep:>6}  {tr_loss:>12.4f}  {vl:>10.4f}  "
              f"{tr_acc*100:>9.2f}%  {va*100:>8.2f}%  {elapsed:>5.1f}s")

        # LR decay
        if ep % 15 == 0:
            lr *= 0.5
            print(f"         → LR decayed to {lr:.5f}")

    return history


# ══════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════

def plot_training_history(history: dict, save_dir: str = PLOT_DIR):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', lw=2)
    ax.plot(history['val_loss'],   label='Val Loss',   lw=2, linestyle='--')
    ax.set_title('Cross-Entropy Loss vs Epoch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([v*100 for v in history['train_acc']], label='Train Acc', lw=2)
    ax.plot([v*100 for v in history['val_acc']],   label='Val Acc',   lw=2, linestyle='--')
    ax.set_title('Classification Accuracy vs Epoch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, 'training_history.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[CNN] Training history plot → {out}")


# ══════════════════════════════════════════════
# Main entry-point
# ══════════════════════════════════════════════

def run_training(n_samples: int = 3000,
                 dataset_path: str = 'morse_dataset.npz') -> MorseCNN:

    print("=" * 65)
    print("  CNN Training Pipeline — Deep Morse Decoder")
    print("=" * 65)

    # ── Load or generate dataset ──
    if os.path.exists(dataset_path):
        print(f"[CNN] Loading existing dataset from {dataset_path}")
        ds = load_dataset(dataset_path)
    else:
        print("[CNN] Generating synthetic dataset …")
        ds = generate_dataset(n_samples=n_samples)
        save_dataset(ds, dataset_path)

    X_raw = ds['X']                       # (N, N_MELS, PATCH_FRAMES)
    y     = ds['y']

    # Flatten spatial dims for FC network
    X = X_raw.reshape(len(y), -1).astype(np.float32)   # (N, 512)

    # ── Train / Val split (80 / 20) ──
    n    = len(y)
    rng  = np.random.default_rng(99)
    perm = rng.permutation(n)
    cut  = int(0.8 * n)
    tr_idx, va_idx = perm[:cut], perm[cut:]

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[va_idx], y[va_idx]
    print(f"[CNN] Train: {len(y_train)}  Val: {len(y_val)}")

    # ── Build & train ──
    model   = MorseCNN()
    history = train(model, X_train, y_train, X_val, y_val)

    # ── Save ──
    model.save(MODEL_PATH)
    plot_training_history(history)

    final_val_acc = history['val_acc'][-1] * 100
    print(f"\n[CNN] Final Validation Accuracy: {final_val_acc:.2f}%")

    return model


if __name__ == '__main__':
    run_training()
