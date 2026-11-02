"""Sync existing trained models (BiGRU and BiLSTM) to Weights & Biases without retraining.

Reads work/results/bigru and work/results/bilstm, logging full training histories,
test evaluation metrics, per-class tables, and interactive confusion matrices.

Usage:
    python scripts/sync_to_wandb.py
    python scripts/sync_to_wandb.py --project SignBangla
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
import wandb

from signbangla.labels import class_names


def sync_run(result_dir: Path, model_type: str, project: str):
    if not result_dir.exists():
        print(f"[!] Directory {result_dir} does not exist. Skipping.")
        return

    history_file = result_dir / "history.json"
    metrics_file = result_dir / "test_metrics.json"
    checkpoint_file = result_dir / "best_model.pt"
    per_class_file = result_dir / "test_per_class.csv"
    confused_file = result_dir / "test_confused_pairs.csv"
    cm_file = result_dir / "test_confusion_matrix.csv"

    if not metrics_file.exists():
        print(f"[!] {metrics_file} not found. Skipping {model_type}.")
        return

    test_metrics = json.loads(metrics_file.read_text())
    history = json.loads(history_file.read_text()) if history_file.exists() else {}

    run_name = f"{model_type}-baseline"
    print(f"\n============================================================")
    print(f"  Syncing {run_name} to Weights & Biases (Project: {project}) ")
    print(f"============================================================")

    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "model_type": model_type,
            "architecture": f"{model_type.upper()} + Temporal Attention",
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "num_classes": 60,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "seed": 42,
            "dataset": "BdSLW60 (60 Words, 18 Signers)",
            "protocol": "Signer-Independent (Test: U4, U8 | Val: U1, U11 | Train: 14 signers)",
        },
        reinit=True,
    )

    # 1. Log epoch-by-epoch training history
    if "train_loss" in history:
        n_epochs = len(history["train_loss"])
        print(f"[*] Logging {n_epochs} training epochs...")
        for ep in range(n_epochs):
            wandb.log({
                "epoch": ep + 1,
                "train/loss": history["train_loss"][ep],
                "train/acc": history["train_acc"][ep],
                "val/loss": history["val_loss"][ep],
                "val/acc": history["val_acc"][ep],
                "val/macro_f1": history["val_f1"][ep],
            }, step=ep + 1)

    # 2. Log final test metrics
    print(f"[*] Logging test metrics (Accuracy: {test_metrics.get('accuracy', 0):.2%})...")
    wandb.log({
        "test/loss": test_metrics.get("loss", 0.0),
        "test/accuracy": test_metrics.get("accuracy", 0.0),
        "test/macro_f1": test_metrics.get("macro_f1", 0.0),
        "test/weighted_f1": test_metrics.get("weighted_f1", 0.0),
        "test/macro_precision": test_metrics.get("macro_precision", 0.0),
        "test/macro_recall": test_metrics.get("macro_recall", 0.0),
    })
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            wandb.run.summary[f"test_{k}"] = v

    # 3. Tables: Per-Class & Confused Pairs
    if per_class_file.exists():
        df_pc = pd.read_csv(per_class_file)
        wandb.log({"test/per_class_table": wandb.Table(dataframe=df_pc)})

    if confused_file.exists():
        df_conf = pd.read_csv(confused_file)
        wandb.log({"test/confused_pairs_table": wandb.Table(dataframe=df_conf)})

    # 4. Images
    if (result_dir / "training_curves.png").exists():
        wandb.log({"plots/training_curves": wandb.Image(str(result_dir / "training_curves.png"))})
    if (result_dir / "test_confusion_matrix.png").exists():
        wandb.log({"plots/confusion_matrix": wandb.Image(str(result_dir / "test_confusion_matrix.png"))})

    # 5. Model artifact
    if checkpoint_file.exists():
        artifact = wandb.Artifact(f"{model_type}-model", type="model")
        artifact.add_file(str(checkpoint_file))
        wandb.log_artifact(artifact)

    wandb.finish()
    print(f"[✓] {run_name} successfully uploaded to W&B!")


def main():
    ap = argparse.ArgumentParser(description="Sync existing trained models to Weights & Biases")
    ap.add_argument("--project", default="SignBangla", help="W&B project name")
    ap.add_argument("--results-dir", default="work/results", help="Directory containing result folders")
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    sync_run(res_dir / "bigru", "bigru", args.project)
    sync_run(res_dir / "bilstm", "bilstm", args.project)

    print("\n============================================================")
    print(f"  All models successfully synced to W&B Project: '{args.project}'! ")
    print("============================================================\n")


if __name__ == "__main__":
    main()
