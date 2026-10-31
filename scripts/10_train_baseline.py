"""WEEK 4: Train the BiGRU baseline model on keypoint sequences.

Usage:
    python scripts/10_train_baseline.py
    python scripts/10_train_baseline.py --epochs 40 --hidden-dim 128 --device mps
    python scripts/10_train_baseline.py --out-dir work/results/bigru
"""
import argparse
import json
import random
import sys
from pathlib import Path

import _path  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from signbangla.dataset import class_weights, make_dataloaders, make_datasets
from signbangla.labels import N_CLASSES, class_names
from signbangla.metrics import evaluate, evaluate_and_save
from signbangla.models import BiGRUClassifier
from signbangla.utils import load_config, work_dir


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Test MPS availability
            try:
                _ = torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")
    return torch.device(device_arg)


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    y_true_list, y_pred_list = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=-1)
        y_true_list.extend(y.cpu().numpy().tolist())
        y_pred_list.extend(preds.cpu().numpy().tolist())
        total += len(y)

    metrics = evaluate(y_true_list, y_pred_list, n_classes=N_CLASSES)
    return total_loss / max(total, 1), metrics, np.array(y_true_list), np.array(y_pred_list)


def plot_training_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#0D9488", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#E11D48", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy & Macro-F1
    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#0D9488", linestyle="--")
    ax2.plot(epochs, history["val_acc"], label="Val Acc", color="#2563EB", lw=2)
    ax2.plot(epochs, history["val_f1"], label="Val Macro-F1", color="#7C3AED", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Accuracy & Macro-F1")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train BiGRU Baseline for Bengali Sign Language Recognition")
    ap.add_argument("--config", help="Path to config file")
    ap.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    ap.add_argument("--hidden-dim", type=int, default=128, help="BiGRU hidden state dimension")
    ap.add_argument("--num-layers", type=int, default=2, help="Number of BiGRU recurrent layers")
    ap.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    ap.add_argument("--no-attention", action="store_true", help="Disable temporal attention pooling")
    ap.add_argument("--weighted-loss", action="store_true", help="Use inverse frequency class weights")
    ap.add_argument("--device", default="auto", help="Execution device: auto, cpu, cuda, or mps")
    ap.add_argument("--out-dir", default="work/results/bigru", help="Directory to save checkpoints and metrics")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"============================================================")
    print(f"  SignBangla - BiGRU Baseline Training (Phase 2 / Week 4)   ")
    print(f"============================================================")
    print(f"Device: {device} | Random Seed: {args.seed}")
    print(f"Architecture: BiGRU (hidden_dim={args.hidden_dim}, layers={args.num_layers}, "
          f"attention={not args.no_attention}, dropout={args.dropout})")
    print(f"Output directory: {out_dir}\n")

    # Load DataLoaders
    loaders = make_dataloaders(cfg)
    datasets = make_datasets(cfg)
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    print(f"Dataset split sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Model input dimension: 201 (67 keypoints x 3 coordinates)
    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]

    model = BiGRUClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=N_CLASSES,
        dropout=args.dropout,
        use_attention=not args.no_attention,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")

    # Loss function & Class Weights
    if args.weighted_loss:
        weights = class_weights(train_ds.labels, n_classes=N_CLASSES)
        weight_tensor = torch.from_numpy(weights).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("Using weighted CrossEntropyLoss based on training class frequencies.")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = -1.0
    best_epoch = 0
    best_checkpoint_path = out_dir / "best_model.pt"

    print("Starting training...")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val Macro-F1':<12} | {'LR':<9}")
    print("-" * 78)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_metrics, _, _ = eval_epoch(model, loaders["val"], criterion, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "num_classes": N_CLASSES,
                    "use_attention": not args.no_attention,
                }
            }, best_checkpoint_path)

        flag = " *" if is_best else ""
        print(f"{epoch:<6d} | {train_loss:<10.4f} | {train_acc:<10.2%} | {val_loss:<10.4f} | {val_acc:<10.2%} | {val_f1:<12.4f} | {current_lr:<9.2e}{flag}")

    print("-" * 78)
    print(f"Training complete! Best Validation Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Saved best model checkpoint to: {best_checkpoint_path}")

    # Plot history
    plot_training_history(history, out_dir / "training_curves.png")
    print(f"Saved training curves to: {out_dir / 'training_curves.png'}")

    # Save training history JSON
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # -------------------------------------------------------------
    # Final Evaluation on Held-Out Test Set (Unseen Signers U4 & U8)
    # -------------------------------------------------------------
    print("\nLoading best model checkpoint for final evaluation on test set...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics, y_test_true, y_test_pred = eval_epoch(model, loaders["test"], criterion, device)
    r_test = evaluate_and_save(y_test_true, y_test_pred, out_dir, prefix="test")

    print("\n" + "=" * 60)
    print("           FINAL TEST SET EVALUATION RESULTS                ")
    print("=" * 60)
    print(f"  Test Loss:        {test_loss:.4f}")
    print(f"  Test Accuracy:    {r_test['accuracy']:.2%}  ({int(r_test['accuracy'] * len(y_test_true))}/{len(y_test_true)} correct)")
    print(f"  Test Macro-F1:    {r_test['macro_f1']:.4f}")
    print(f"  Test Weighted-F1: {r_test['weighted_f1']:.4f}")
    print(f"  Test Macro-Prec:  {r_test['macro_precision']:.4f}")
    print(f"  Test Macro-Rec:   {r_test['macro_recall']:.4f}")
    print("=" * 60)

    if not r_test["confused_pairs"].empty:
        print("\nTop Confused Pairs:")
        print(r_test["confused_pairs"].to_string(index=False))

    print(f"\nAll artifacts successfully saved in: {out_dir}")
    print("Generated files:")
    print(" - best_model.pt (PyTorch weights & metadata)")
    print(" - training_curves.png (Loss & Metric progression)")
    print(" - history.json (Raw epoch metrics)")
    print(" - test_metrics.json (Comprehensive test scalar metrics)")
    print(" - test_confusion_matrix.png & .csv (Row-normalized confusion matrix)")
    print(" - test_per_class.csv (Per-class precision, recall, F1, support)")
    print(" - test_confused_pairs.csv (Top confusion pairs)")


if __name__ == "__main__":
    main()
