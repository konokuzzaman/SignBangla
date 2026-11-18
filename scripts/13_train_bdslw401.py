"""Train BiGRU, BiLSTM, or BdSL-SPOTER on the 401-class BdSLW401 keypoint dataset.

Features:
- Full 401 classes (W001-W401) with 39,677 preprocessed trials (64, 67, 3).
- Signer-independent zero-leakage split:
    Train: 26,841 trials (14 signers)
    Val:    6,823 trials (2 signers: S01, S11)
    Test:   6,013 trials (2 signers: S04, S08)
- Real-time tracking and diagram generation with Weights & Biases (W&B).
- Safe offline fallback (prevents interactive terminal hangs when no API key is present).
- High-resolution local saving of training curves, confusion matrices, and model weights.

Usage:
    # Train BdSL-SPOTER Transformer
    python scripts/13_train_bdslw401.py --model-type spoter --epochs 35 --device mps

    # Train BiGRU Baseline
    python scripts/13_train_bdslw401.py --model-type bigru --epochs 35 --device mps

    # Train with online W&B logging using an API key
    python scripts/13_train_bdslw401.py --model-type spoter --wandb-mode online --wandb-key <YOUR_KEY>
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time

import _path  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from signbangla.dataset import Augmenter, SignDataset
from signbangla.metrics import evaluate, most_confused_pairs
from signbangla.models import (
    BdSLKinematicGraphTransformer,
    BdSLSpoterClassifier,
    BiGRUClassifier,
    BiLSTMClassifier,
)
from signbangla.utils import load_config

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False


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
            try:
                _ = torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")
    return torch.device(device_arg)


def init_wandb(args, model_name: str, total_params: int):
    """Safely initialize W&B without interactive blocking."""
    if not args.wandb or not HAS_WANDB:
        if args.wandb and not HAS_WANDB:
            print("[W&B Warning] wandb is not installed. Disabling W&B logging.")
        return None

    # Silence any interactive prompts
    os.environ["WANDB_SILENT"] = "true"

    api_key = args.wandb_key or os.environ.get("WANDB_API_KEY")
    mode = args.wandb_mode

    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        if mode == "auto":
            mode = "online"
    elif mode == "auto":
        # Check if already authenticated without prompting
        try:
            netrc_path = Path.home() / ".netrc"
            if netrc_path.exists() and "api.wandb.ai" in netrc_path.read_text():
                mode = "online"
            else:
                mode = "offline"
        except Exception:
            mode = "offline"

    run_name = args.wandb_run_name or f"{args.model_type}_bdslw401_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            mode=mode,
            dir=str(Path(args.out_dir).parent),
            config={
                "model_type": args.model_type,
                "architecture": model_name,
                "parameters": total_params,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "scheduler": "CosineAnnealingLR",
                "device": str(args.device),
                "num_classes": 401,
                "dataset": "BdSLW401",
                "seed": args.seed,
            },
        )
        is_offline = (mode == "offline" or getattr(run.settings, "mode", None) == "offline")
        if is_offline:
            print(f"[W&B Notice] WandB initialized in OFFLINE mode. Run dir: {run.dir}")
            print(f"             To sync run to your dashboard later: wandb sync {run.dir}")
        else:
            print(f"[W&B Notice] WandB streaming LIVE online. Run URL: {run.url}")
        return run, is_offline
    except Exception as e:
        print(f"[W&B Warning] wandb.init failed ({e}). Proceeding without W&B.")
        return None, True


def compute_velocity(x: torch.Tensor) -> torch.Tensor:
    """Concatenate 1st-order temporal difference (velocity) along joint coordinate dimension."""
    vel = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
    return torch.cat([x, vel], dim=-1)


def train_epoch(model, loader, criterion, optimizer, device, use_velocity: bool = False, max_grad_norm: float = 1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    top5_correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_velocity:
            x = compute_velocity(x)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_sz = len(y)
        total_loss += loss.item() * batch_sz
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()

        # Top-5 accuracy
        top5_preds = logits.topk(min(5, logits.size(-1)), dim=-1).indices
        top5_correct += (top5_preds == y.unsqueeze(1)).any(dim=-1).sum().item()

        total += batch_sz

    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        top5_correct / max(total, 1),
    )


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, n_classes: int = 401, use_velocity: bool = False):
    model.eval()
    total_loss, total = 0.0, 0
    top5_correct = 0
    y_true_list, y_pred_list = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_velocity:
            x = compute_velocity(x)

        logits = model(x)
        loss = criterion(logits, y)

        batch_sz = len(y)
        total_loss += loss.item() * batch_sz
        preds = logits.argmax(dim=-1)

        top5_preds = logits.topk(min(5, logits.size(-1)), dim=-1).indices
        top5_correct += (top5_preds == y.unsqueeze(1)).any(dim=-1).sum().item()

        y_true_list.extend(y.cpu().numpy().tolist())
        y_pred_list.extend(preds.cpu().numpy().tolist())
        total += batch_sz

    class_names = [f"W{i+1:03d}" for i in range(n_classes)]
    metrics = evaluate(y_true_list, y_pred_list, n_classes=n_classes, class_names=class_names)
    metrics["top5_accuracy"] = top5_correct / max(total, 1)

    return total_loss / max(total, 1), metrics, np.array(y_true_list), np.array(y_pred_list)


def plot_training_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#0D9488", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#E11D48", lw=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss (BdSLW401)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Accuracy & Macro-F1
    ax2.plot(epochs, history["train_acc"], label="Train Top-1 Acc", color="#0D9488", linestyle="--", lw=1.5)
    ax2.plot(epochs, history["val_acc"], label="Val Top-1 Acc", color="#2563EB", lw=2)
    ax2.plot(epochs, history["val_top5_acc"], label="Val Top-5 Acc", color="#10B981", lw=2)
    ax2.plot(epochs, history["val_f1"], label="Val Macro-F1", color="#7C3AED", lw=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Validation Accuracy & Macro-F1", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_confusion_matrix_401(cm, path, title="BdSLW401 Test Confusion Matrix"):
    """Render a clean 401x401 confusion matrix heatmap."""
    # Row normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    m = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(m, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title + " (Row-Normalized)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Class Index (0 - 400)", fontsize=11)
    ax.set_ylabel("True Class Index (0 - 400)", fontsize=11)

    # Ticks every 50 classes
    ticks = np.arange(0, 401, 50)
    tick_labels = [f"W{t+1:03d}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)

    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy / Recall Rate", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train BdSLW401 Models with Weights & Biases (W&B)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model-type", choices=["spoter", "graph_transformer", "bigru", "bilstm"], default="graph_transformer",
                        help="Model architecture: graph_transformer (Kinematic SOTA), spoter (Transformer), bigru, or bilstm")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 5e-4 for spoter, 1e-3 for recurrent)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay for AdamW (default: 1e-3 for spoter, 1e-4 for recurrent)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden state dimension / d_model")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers (default: 4 for spoter, 2 for recurrent)")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads for spoter")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability (default: 0.3 for spoter, 0.2 for recurrent)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Cross-entropy label smoothing (default: 0.1)")
    parser.add_argument("--use-velocity", dest="use_velocity", action="store_true", default=True, help="Include 1st-order joint velocity features (dim: 201 -> 402)")
    parser.add_argument("--no-velocity", dest="use_velocity", action="store_false", help="Disable velocity features")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs before cosine decay (default: 3)")
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, cuda, or mps")
    parser.add_argument("--data-dir", default="work/bdslw401/processed", help="Directory of preprocessed .npy files")
    parser.add_argument("--splits-file", default="work/bdslw401/splits.csv", help="Path to splits.csv")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # W&B arguments
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True, help="Enable W&B logging")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable W&B logging")
    parser.add_argument("--wandb-project", default="SignBangla-BdSLW401", help="W&B Project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B Run name")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline"], default="auto",
                        help="W&B mode: auto (uses online if key present, else offline), online, or offline")
    parser.add_argument("--wandb-key", default=None, help="W&B API key for direct authentication")

    args = parser.parse_args()

    # Defaults dependent on model type
    is_transformer = args.model_type in ("spoter", "graph_transformer")
    if args.lr is None:
        args.lr = 5e-4 if is_transformer else 1e-3
    if args.num_layers is None:
        args.num_layers = 4 if is_transformer else 2
    if args.dropout is None:
        args.dropout = 0.3 if is_transformer else 0.2
    if args.weight_decay is None:
        args.weight_decay = 1e-3 if is_transformer else 1e-4

    set_seed(args.seed)
    cfg = load_config(args.config) if Path(args.config).exists() else {"augment": {
        "p": 0.5, "rotation_deg": 10, "scale": [0.9, 1.1], "shift": 0.10,
        "temporal_crop_max": 0.10, "time_warp": 0.25, "noise_std": 0.01
    }}

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(f"work/bdslw401/results/{args.model_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    model_name_map = {
        "graph_transformer": "BdSL-Kinematic-Graph-Transformer (Phase 3 SOTA)",
        "spoter": "BdSL-SPOTER (Transformer)",
        "bigru": "BiGRU Baseline",
        "bilstm": "BiLSTM Baseline",
    }
    model_name = model_name_map[args.model_type]

    print("=" * 72)
    print(f"  SignBangla - {model_name} Training on BdSLW401 (401 Classes)")
    print("=" * 72)
    print(f"Device: {device} | Seed: {args.seed} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Output directory: {out_dir}")

    # Load splits metadata
    splits_path = Path(args.splits_file)
    pdir = Path(args.data_dir)
    if not splits_path.exists():
        sys.exit(f"Error: Splits file not found at {splits_path}")
    if not pdir.exists():
        sys.exit(f"Error: Processed data directory not found at {pdir}")

    splits_df = pd.read_csv(splits_path)
    train_df = splits_df[splits_df["split"] == "train"]
    val_df = splits_df[splits_df["split"] == "val"]
    test_df = splits_df[splits_df["split"] == "test"]

    print(f"Dataset split sizes: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    # Build Datasets and Loaders
    aug = Augmenter(cfg.get("augment", {}))
    train_ds = SignDataset(train_df, pdir, augment=aug, flatten=True)
    val_ds = SignDataset(val_df, pdir, augment=None, flatten=True)
    test_ds = SignDataset(test_df, pdir, augment=None, flatten=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    sample_x, _ = train_ds[0]
    raw_dim = sample_x.shape[-1]
    effective_use_velocity = args.use_velocity if args.model_type != "graph_transformer" else False
    input_dim = raw_dim * 2 if effective_use_velocity else raw_dim
    num_classes = 401

    if args.model_type == "graph_transformer":
        model = BdSLKinematicGraphTransformer(
            num_classes=num_classes,
            d_model=args.hidden_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "spoter":
        model = BdSLSpoterClassifier(
            input_dim=input_dim,
            d_model=args.hidden_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.hidden_dim * 4,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "bigru":
        model = BiGRUClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            dropout=args.dropout,
            use_attention=True,
        ).to(device)
    else:
        model = BiLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            dropout=args.dropout,
            use_attention=True,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kinematic_info = "4-Stream Kinematics (J, dJ, B, dB)" if args.model_type == "graph_transformer" else f"features: {input_dim}, velocity: {effective_use_velocity}"
    print(f"Model parameters: {total_params:,} ({kinematic_info})\n")

    # Initialize W&B
    wb_run, is_offline = init_wandb(args, model_name, total_params)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup_epochs > 0:
        scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-5)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "train_acc": [], "train_top5_acc": [],
        "val_loss": [], "val_acc": [], "val_top5_acc": [], "val_f1": []
    }
    best_val_f1 = -1.0
    best_epoch = 0
    best_checkpoint_path = out_dir / "best_model.pt"

    print("\nStarting training loop...")
    header = f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val Top-5':<10} | {'Val Macro-F1':<12} | {'Time':<7}"
    print(header)
    print("-" * len(header))

    t_train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        train_loss, train_acc, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, device, use_velocity=effective_use_velocity
        )
        val_loss, val_metrics, _, _ = eval_epoch(
            model, val_loader, criterion, device, n_classes=num_classes, use_velocity=effective_use_velocity
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        val_acc = val_metrics["accuracy"]
        val_top5 = val_metrics["top5_accuracy"]
        val_f1 = val_metrics["macro_f1"]
        val_weighted_f1 = val_metrics["weighted_f1"]
        epoch_time = time.time() - t_epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_top5_acc"].append(train_top5)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5_acc"].append(val_top5)
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
                    "model_type": args.model_type,
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "num_classes": num_classes,
                    "use_velocity": args.use_velocity,
                    "label_smoothing": args.label_smoothing,
                }
            }, best_checkpoint_path)

        flag = " *" if is_best else ""
        print(f"{epoch:<6d} | {train_loss:<10.4f} | {train_acc:<10.2%} | {val_loss:<10.4f} | {val_acc:<10.2%} | {val_top5:<10.2%} | {val_f1:<12.4f} | {epoch_time:<6.1f}s{flag}")

        # Log epoch metrics to W&B
        if wb_run is not None:
            wb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/top5_acc": train_top5,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/top5_acc": val_top5,
                "val/macro_f1": val_f1,
                "val/weighted_f1": val_weighted_f1,
                "lr": current_lr,
            })

    total_training_time = time.time() - t_train_start
    print("-" * len(header))
    print(f"Training completed in {total_training_time / 60:.1f} minutes.")
    print(f"Best Validation Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Saved best model checkpoint to: {best_checkpoint_path}")

    # Save training curves plot
    curves_plot_path = out_dir / "training_curves.png"
    plot_training_history(history, curves_plot_path)
    print(f"Saved training curves to: {curves_plot_path}")

    # Save history json
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # -------------------------------------------------------------
    # Final Evaluation on Held-Out Test Set (Signers S04 & S08)
    # -------------------------------------------------------------
    print("\nLoading best model checkpoint for final evaluation on test set (Signers 4 & 8)...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics, y_test_true, y_test_pred = eval_epoch(
        model, test_loader, criterion, device, n_classes=num_classes, use_velocity=effective_use_velocity
    )

    test_scalars = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_top5_accuracy": float(test_metrics["top5_accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_macro_precision": float(test_metrics["macro_precision"]),
        "test_macro_recall": float(test_metrics["macro_recall"]),
        "n_test_samples": int(test_metrics["n_samples"]),
        "best_epoch": int(best_epoch),
    }

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_scalars, f, indent=2)

    # Save per-class metrics
    per_class_df = test_metrics["per_class"]
    per_class_df.to_csv(out_dir / "test_per_class.csv", index=False)

    # Save Confusion Matrix & Plot
    cm = test_metrics["confusion_matrix"]
    pd.DataFrame(cm).to_csv(out_dir / "test_confusion_matrix.csv", index=False)
    cm_plot_path = out_dir / "test_confusion_matrix.png"
    plot_confusion_matrix_401(cm, cm_plot_path, title=f"{model_name} BdSLW401 Confusion Matrix")

    # Top confused pairs
    class_names = [f"W{i+1:03d}" for i in range(num_classes)]
    confused_pairs_df = most_confused_pairs(cm, class_names=class_names, top=20)
    confused_pairs_df.to_csv(out_dir / "test_confused_pairs.csv", index=False)

    print("\n" + "=" * 68)
    print("             FINAL TEST SET EVALUATION RESULTS (BdSLW401)")
    print("=" * 68)
    print(f"  Test Loss:            {test_loss:.4f}")
    print(f"  Test Top-1 Accuracy:  {test_metrics['accuracy']:.2%} ({int(test_metrics['accuracy'] * len(y_test_true))}/{len(y_test_true)} correct)")
    print(f"  Test Top-5 Accuracy:  {test_metrics['top5_accuracy']:.2%}")
    print(f"  Test Macro-F1:        {test_metrics['macro_f1']:.4f}")
    print(f"  Test Weighted-F1:     {test_metrics['weighted_f1']:.4f}")
    print(f"  Test Macro-Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Test Macro-Recall:    {test_metrics['macro_recall']:.4f}")
    print("=" * 68)

    if not confused_pairs_df.empty:
        print("\nTop 10 Confused Pairs:")
        print(confused_pairs_df.head(10).to_string(index=False))

    # Log summary and diagrams to W&B
    if wb_run is not None:
        wb_run.summary.update(test_scalars)
        log_payload = {
            "test/loss": test_loss,
            "test/accuracy": test_metrics["accuracy"],
            "test/top5_accuracy": test_metrics["top5_accuracy"],
            "test/macro_f1": test_metrics["macro_f1"],
            "test/weighted_f1": test_metrics["weighted_f1"],
            "charts/training_curves": wandb.Image(str(curves_plot_path), caption="Training & Validation Curves"),
            "charts/confusion_matrix": wandb.Image(str(cm_plot_path), caption="Test Confusion Matrix (Row-Normalized)"),
        }

        # Tables and Artifacts are only supported online by W&B backend
        if not is_offline:
            try:
                log_payload["tables/top_confused_pairs"] = wandb.Table(dataframe=confused_pairs_df)
            except Exception:
                pass

        wb_run.log(log_payload)

        if not is_offline:
            try:
                artifact = wandb.Artifact(f"{args.model_type}_bdslw401", type="model")
                artifact.add_file(str(best_checkpoint_path))
                artifact.add_file(str(out_dir / "test_metrics.json"))
                wb_run.log_artifact(artifact)
            except Exception as e:
                print(f"[W&B Warning] Could not upload model artifact: {e}")

        wb_run.finish()
        print(f"\n[W&B] Run successfully finished and logged.")

    print(f"\nAll artifacts successfully saved to: {out_dir}")
    print("Generated files:")
    print(" - best_model.pt (PyTorch model weights & config)")
    print(" - training_curves.png (Loss, Top-1/Top-5 Acc, Macro-F1 progression)")
    print(" - history.json (Per-epoch training and validation metrics)")
    print(" - test_metrics.json (Comprehensive held-out test evaluation scores)")
    print(" - test_confusion_matrix.png (Row-normalized 401x401 heatmap)")
    print(" - test_confusion_matrix.csv (Raw 401x401 confusion counts)")
    print(" - test_per_class.csv (Per-class precision, recall, F1, support)")
    print(" - test_confused_pairs.csv (Top 20 confused sign pairs)")


if __name__ == "__main__":
    main()
