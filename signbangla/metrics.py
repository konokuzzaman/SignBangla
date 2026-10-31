"""Week 3 evaluation tools: accuracy, per-class precision/recall/F1, macro F1, confusion matrix, most-confused pairs.
Pure numpy/pandas (no scikit-learn needed). Use evaluate_and_save() after every training run."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .labels import N_CLASSES, short_names


def confusion_matrix(y_true, y_pred, n_classes=N_CLASSES):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)  # rows = true class, columns = predicted class
    return cm


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def evaluate(y_true, y_pred, n_classes=N_CLASSES, class_names=None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    used = (support + predicted) > 0  # classes that occur in y_true or y_pred (same rule as scikit-learn)
    names = class_names or short_names()
    per_class = pd.DataFrame(dict(class_idx=np.arange(n_classes), name=names, precision=precision,
                                  recall=recall, f1=f1, support=support))
    return dict(
        accuracy=float(tp.sum() / max(cm.sum(), 1)),
        macro_f1=float(f1[used].mean()),
        weighted_f1=float((f1 * support).sum() / max(support.sum(), 1)),
        macro_precision=float(precision[used].mean()),
        macro_recall=float(recall[used].mean()),
        n_samples=int(cm.sum()),
        confusion_matrix=cm,
        per_class=per_class,
    )


def most_confused_pairs(cm, class_names=None, top=10):
    """Largest off-diagonal cells: 'true word was predicted as other word'."""
    names = class_names or short_names()
    off = cm.copy()
    np.fill_diagonal(off, 0)
    rows = []
    for flat in np.argsort(off, axis=None)[::-1][:top]:
        t, p = np.unravel_index(flat, off.shape)
        if off[t, p] == 0:
            break
        rows.append(dict(true=names[t], predicted=names[p], count=int(off[t, p]),
                         share_of_true_class=float(off[t, p] / max(cm[t].sum(), 1))))
    return pd.DataFrame(rows, columns=["true", "predicted", "count", "share_of_true_class"])


def plot_confusion_matrix(cm, path, class_names=None, normalize=True, title="Confusion matrix"):
    names = class_names or short_names()
    m = cm.astype(float)
    if normalize:
        m = _safe_div(m, m.sum(1, keepdims=True))
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.28), max(8, n * 0.28)))
    im = ax.imshow(m, cmap="Greys", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_and_save(y_true, y_pred, out_dir, prefix="test", n_classes=N_CLASSES, class_names=None, top_pairs=10):
    """Compute all metrics and write metrics.json, per_class.csv, confusion_matrix.csv/.png, confused_pairs.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = evaluate(y_true, y_pred, n_classes, class_names)
    scalars = {k: v for k, v in r.items() if k not in ("confusion_matrix", "per_class")}
    with open(out_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(scalars, f, indent=1)
    r["per_class"].to_csv(out_dir / f"{prefix}_per_class.csv", index=False)
    pd.DataFrame(r["confusion_matrix"]).to_csv(out_dir / f"{prefix}_confusion_matrix.csv", index=False)
    plot_confusion_matrix(r["confusion_matrix"], out_dir / f"{prefix}_confusion_matrix.png", class_names,
                          title=f"{prefix} confusion matrix")
    pairs = most_confused_pairs(r["confusion_matrix"], class_names, top_pairs)
    pairs.to_csv(out_dir / f"{prefix}_confused_pairs.csv", index=False)
    r["confused_pairs"] = pairs
    return r
