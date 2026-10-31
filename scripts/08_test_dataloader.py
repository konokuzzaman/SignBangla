"""WEEK 3 (Member 2): build the DataLoaders, check batch shapes, and save a picture of the augmentations.

    python scripts/08_test_dataloader.py
"""
import argparse

import _path  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from signbangla.dataset import HAS_TORCH, Augmenter, class_weights, make_dataloaders, make_datasets
from signbangla.utils import load_config, subdir
from signbangla.viz import draw_skeleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sets = make_datasets(cfg)
    for k, ds in sets.items():
        print(f"{k:5s}: {len(ds)} samples")
    if HAS_TORCH:
        loaders = make_dataloaders(cfg)
        x, y = next(iter(loaders["train"]))
        print(f"\ntrain batch: x {tuple(x.shape)} {x.dtype}, y {tuple(y.shape)} {y.dtype}  (batch, 64 frames, 201 features)")
        print(f"value range {x.min():.2f} .. {x.max():.2f}, NaN: {bool(np.isnan(x.numpy()).any())}")
        xv, yv = next(iter(loaders["val"]))
        print(f"val batch:   x {tuple(xv.shape)}")
    else:
        x, y = sets["train"][0]
        print(f"\n(PyTorch not installed - numpy check only) sample: x {x.shape}, label {y}")
    w = class_weights(sets["train"].labels)
    print(f"class weights (optional): min {w[w > 0].min():.2f}, max {w.max():.2f}")

    # augmentation preview: original + 5 augmented versions of one training sample, middle frame
    ds = sets["train"]
    seq = np.load(ds.dir / f"{ds.items.at[0, 'trial_id']}.npy")
    aug, rng = Augmenter(cfg["augment"]), np.random.default_rng(1)
    fig, axes = plt.subplots(1, 6, figsize=(14, 3))
    for i, ax in enumerate(axes):
        s = seq if i == 0 else aug(seq, rng)
        draw_skeleton(ax, s[32])
        ax.set_title("original" if i == 0 else f"augmented {i}", fontsize=8)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(2.5, -2.5)
    out = subdir(cfg, "quality_plots") / "augmentation_preview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\naugmentation preview saved: {out}")


if __name__ == "__main__":
    main()
