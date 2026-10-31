"""Week 3: Dataset + DataLoader + augmentation for the processed (64, 67, 3) sequences.

Augmentations (training split only, each applied with probability augment.p):
  temporal crop + time warp (speed change), rotation, scaling, shifting, Gaussian noise.
  NO left/right flip. Absent body parts (exact zeros) stay exactly zero.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import resample_time
from .utils import subdir, work_dir

try:
    import torch
    from torch.utils.data import DataLoader, Dataset as _Base
    HAS_TORCH = True
except ImportError:  # the numpy parts still work without torch
    torch, DataLoader, _Base, HAS_TORCH = None, None, object, False


class Augmenter:
    def __init__(self, acfg):
        self.c = acfg

    def _temporal(self, seq, rng):
        c, T = self.c, len(seq)
        keep = 1 - rng.uniform(0, c["temporal_crop_max"])
        span = (T - 1) * keep
        start = rng.uniform(0, (T - 1) - span)
        w = c["time_warp"]
        ctrl = rng.uniform(1 - w, 1 + w, size=8)
        speed = np.interp(np.linspace(0, 7, T), np.arange(8), ctrl)
        cum = np.cumsum(speed)
        cum = (cum - cum[0]) / (cum[-1] - cum[0])
        return resample_time(seq, start + cum * span)

    def __call__(self, seq, rng):
        c, p = self.c, self.c["p"]
        seq = seq.copy()
        if rng.random() < p:
            seq = self._temporal(seq, rng)
        present = (np.abs(seq).sum(-1) > 0)[..., None]
        if rng.random() < p:
            a = np.deg2rad(rng.uniform(-c["rotation_deg"], c["rotation_deg"]))
            x, y = seq[..., 0].copy(), seq[..., 1].copy()
            seq[..., 0] = np.cos(a) * x - np.sin(a) * y
            seq[..., 1] = np.sin(a) * x + np.cos(a) * y
        if rng.random() < p:
            seq *= rng.uniform(*c["scale"])
        if rng.random() < p:
            seq[..., :2] += rng.uniform(-c["shift"], c["shift"], size=2)
        if c["noise_std"] > 0 and rng.random() < p:
            seq += rng.normal(0, c["noise_std"], seq.shape).astype(np.float32)
        return (seq * present).astype(np.float32)  # absent parts back to exactly 0


class SignDataset(_Base):
    """items: DataFrame with columns trial_id, class_idx. Returns (features, label)."""

    def __init__(self, items, processed_dir, augment=None, flatten=True, preload=False):
        self.items = items.reset_index(drop=True)
        self.dir = Path(processed_dir)
        self.augment, self.flatten = augment, flatten
        self.labels = self.items["class_idx"].to_numpy()
        self.cache = [self._load(i) for i in range(len(self.items))] if preload else None

    def _load(self, i):
        return np.load(self.dir / f"{self.items.at[i, 'trial_id']}.npy")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        seq = self.cache[i] if self.cache is not None else self._load(i)
        if self.augment is not None:
            seq = self.augment(seq, np.random.default_rng())
        if self.flatten:
            seq = seq.reshape(seq.shape[0], -1)
        y = int(self.labels[i])
        if HAS_TORCH:
            return torch.from_numpy(np.ascontiguousarray(seq, dtype=np.float32)), torch.tensor(y, dtype=torch.long)
        return seq.astype(np.float32), y


def class_weights(labels, n_classes=60):
    """Inverse-frequency weights (mean 1) - optional, for a weighted loss if classes are imbalanced."""
    counts = np.bincount(np.asarray(labels, int), minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return w / w[w > 0].mean()


def make_datasets(cfg):
    splits = pd.read_csv(work_dir(cfg) / "splits.csv")
    pdir, d = subdir(cfg, "processed"), cfg["dataloader"]
    out = {}
    for name in ("train", "val", "test"):
        aug = Augmenter(cfg["augment"]) if name == "train" else None
        out[name] = SignDataset(splits[splits["split"] == name], pdir, aug, d["flatten_features"], d["preload"])
    return out


def make_dataloaders(cfg):
    if not HAS_TORCH:
        raise ImportError("PyTorch is not installed. Install it from https://pytorch.org (pip install torch).")
    d = cfg["dataloader"]
    sets = make_datasets(cfg)
    return {name: DataLoader(ds, batch_size=d["batch_size"], shuffle=(name == "train"),
                             num_workers=d["num_workers"], drop_last=False) for name, ds in sets.items()}
