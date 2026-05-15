"""Shared analysis code for the malaria cell CNN evaluation.

Why these tools (one sentence each):
- Stratified evaluation on the full public NIH set, because the repo records no
  train/test split and this is the only labeled data the models were built for.
- Nonparametric bootstrap percentile intervals (Efron & Tibshirani, 1993) for
  metrics, because accuracy-style metrics have no closed-form standard error that
  respects their bounded, correlated structure.
- Test-time augmentation over 8 geometric views (Ayhan & Berens, 2018), because it
  estimates input-dependent (aleatoric) uncertainty without retraining or changing
  the model.
- Expected calibration error with equal-width bins (Guo et al., 2017), because a
  screening tool's stated confidence must mean something in the long run.
"""

import os
import numpy as np
from PIL import Image

FIGDIR = "figures"

# ---------------------------------------------------------------- inference

class ORTModel:
    """Thin onnxruntime wrapper with a Keras-like predict() interface.

    I use ONNX for the analysis itself (not just the demo) so the numbers in
    the README are computed with the exact artifact shipped in site/.
    """

    def __init__(self, path, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def predict(self, x, verbose=0):
        return self.sess.run(None, {self.in_name: x.astype(np.float32)})[0]


# ---------------------------------------------------------------- data

def find_images(root):
    """Return sorted (path, label) list. Label 1 = Parasitized, 0 = Uninfected."""
    out = []
    for cls, lab in (("Parasitized", 1), ("Uninfected", 0)):
        d = os.path.join(root, cls)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".png"):
                out.append((os.path.join(d, f), lab))
    return out


def load_batch(paths, size, order="rgb", scale=1.0):
    """Read images with PIL, resize to size x size, return float32 NHWC array."""
    arr = np.empty((len(paths), size, size, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
        if order == "bgr":
            a = a[:, :, ::-1]
        arr[i] = a * scale
    return arr


def batch_predict(model, items, size, order, scale, batch=1024, verbose=False):
    """Predict P(parasitized) for every (path, label). Returns probs, labels."""
    probs, labels = [], []
    n = len(items)
    for s in range(0, n, batch):
        chunk = items[s:s + batch]
        x = load_batch([p for p, _ in chunk], size, order, scale)
        pr = model.predict(x, verbose=0)
        # auto-detect which softmax column is "parasitized"
        probs.append(pr)
        labels.extend(l for _, l in chunk)
        if verbose and s % (batch * 4) == 0:
            print(f"  {s}/{n}", flush=True)
    probs = np.concatenate(probs, axis=0)
    labels = np.array(labels, dtype=int)
    # column check on a small sample: parasitized column must correlate with labels
    col1_corr = np.corrcoef(probs[:2000, 1], labels[:2000])[0, 1]
    pcol = 1 if col1_corr > 0 else 0
    return probs[:, pcol].astype(np.float64), labels, pcol


# ---------------------------------------------------------------- metrics

def metrics_from(y, prob):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred)),          # recall of parasitized
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "brier": float(np.mean((prob - y) ** 2)),
    }


def bootstrap_ci(y, prob, n_boot=2000, seed=42):
    """Percentile bootstrap 95% CIs for the headline metrics and confusion counts."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc",
            "tn", "fp", "fn", "tp"]
    boot = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metrics_from(y[idx], prob[idx])
        for k in keys:
            boot[k][b] = m[k]
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in boot.items()}


# ---------------------------------------------------------------- calibration

def reliability(probs, y, n_bins=15):
    """Equal-width bins over [0,1]; returns bin centers, accuracy, confidence, counts."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    centers, acc, conf, counts = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        c = m.sum()
        counts.append(int(c))
        centers.append(float((edges[b] + edges[b + 1]) / 2))
        if c > 0:
            acc.append(float(y[m].mean()))
            conf.append(float(probs[m].mean()))
        else:
            acc.append(np.nan); conf.append(np.nan)
    return np.array(centers), np.array(acc), np.array(conf), np.array(counts)


def expected_calibration_error(probs, y, n_bins=15):
    centers, acc, conf, counts = reliability(probs, y, n_bins)
    n = len(y)
    ece = float(np.nansum(np.abs(acc - conf) * counts / n))
    return ece


# ---------------------------------------------------------------- TTA uncertainty

def tta_views(pil_img):
    """8 deterministic geometric views: 4 rotations x {original, horizontal flip}."""
    views = []
    for flip in (False, True):
        im = pil_img.transpose(Image.FLIP_LEFT_RIGHT) if flip else pil_img
        for angle in (0, 90, 180, 270):
            views.append(im.rotate(angle, expand=False))
    return views


def tta_predict(model, items, size, order, scale, batch=512, seed=7):
    """Mean, SD and 95% interval of P(parasitized) over the 8 TTA views."""
    rng = np.random.default_rng(seed)
    sub = [items[i] for i in rng.choice(len(items), size=min(2000, len(items)),
                                        replace=False)]
    means, sds, labels = [], [], []
    views_all, owner = [], []
    for j, (p, lab) in enumerate(sub):
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        for v in tta_views(im):
            a = np.asarray(v, dtype=np.float32)
            if order == "bgr":
                a = a[:, :, ::-1]
            views_all.append(a * scale)
            owner.append(j)
        labels.append(lab)
    views_all = np.stack(views_all).astype(np.float32)
    owner = np.array(owner)
    probs = []
    for s in range(0, len(views_all), batch):
        pr = model.predict(views_all[s:s + batch], verbose=0)
        probs.append(pr)
    probs = np.concatenate(probs, axis=0)
    col1_corr = np.corrcoef(probs[:2000, 1], np.repeat(labels, 8)[:2000])[0, 1]
    pcol = 1 if col1_corr > 0 else 0
    probs = probs[:, pcol].astype(np.float64)
    labels = np.array(labels, dtype=int)
    for j in range(len(sub)):
        w = probs[owner == j]
        means.append(float(w.mean()))
        sds.append(float(w.std(ddof=1)) if len(w) > 1 else 0.0)
    means = np.array(means); sds = np.array(sds)
    lo = np.clip(means - 1.96 * sds, 0, 1)
    hi = np.clip(means + 1.96 * sds, 0, 1)
    pred = (means >= 0.5).astype(int)
    return {
        "mean": means, "sd": sds, "lo": lo, "hi": hi,
        "labels": labels, "correct": (pred == labels),
        "paths": [p for p, _ in sub],
    }


# ---------------------------------------------------------------- error analysis helpers

def image_stats(paths, size=50):
    """Cheap morphology proxies: brightness, contrast, saturation, redness."""
    bright, contrast, sat, red = [], [], [], []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        bright.append(a.mean())
        contrast.append(a.std())
        mx, mn = a.max(axis=2), a.min(axis=2)
        sat.append(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
        red.append((a[:, :, 0] / (a.sum(axis=2) + 1e-6)).mean())
    return {"brightness": np.array(bright), "contrast": np.array(contrast),
            "saturation": np.array(sat), "redness": np.array(red)}


# ---------------------------------------------------------------- plotting

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def plot_confusion(m, ci, path, title):
    plt = _mpl()
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    cis = [[ci["tn"], ci["fp"]], [ci["fn"], ci["tp"]]]
    for i in range(2):
        for j in range(2):
            lo, hi = cis[i][j]
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}\n95% CI [{lo:,.0f}, {hi:,.0f}]",
                    ha="center", va="center", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nuninfected", "Predicted\ninfected"])
    ax.set_yticklabels(["Actual\nuninfected", "Actual\ninfected"])
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(results, path):
    from sklearn.metrics import roc_curve
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, (y, prob) in results.items():
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, both models")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_calibration(y, prob, path, title, ece):
    plt = _mpl()
    centers, acc, conf, counts = reliability(prob, y)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ok = ~np.isnan(acc)
    ax.plot(conf[ok], acc[ok], "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tta(tta, path):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    width = tta["hi"] - tta["lo"]
    axes[0].hist(width[tta["correct"]], bins=40, alpha=0.7, label="correct")
    axes[0].hist(width[~tta["correct"]], bins=40, alpha=0.7, label="wrong")
    axes[0].set_xlabel("95% interval width"); axes[0].set_ylabel("Images")
    axes[0].legend(frameon=False); axes[0].set_title("Interval width: right vs wrong")
    axes[1].hist(tta["sd"], bins=40)
    axes[1].set_xlabel("SD over 8 views"); axes[1].set_ylabel("Images")
    axes[1].set_title("TTA spread distribution")
    fig.suptitle("Test-time augmentation uncertainty (n=2000)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_error_montage(paths, probs, labels, path, title, k=12):
    plt = _mpl()
    k = min(k, len(paths))
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for ax, p, pr, lab in zip(axes.ravel(), paths[:k], probs[:k], labels[:k]):
        im = Image.open(p).convert("RGB")
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"true={'inf' if lab else 'uninf'} P={pr:.2f}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
