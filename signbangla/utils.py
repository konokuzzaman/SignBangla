"""Small helpers: config loading, folders, safe file writes."""
import json
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def dataset_root(cfg):
    p = Path(cfg["paths"]["dataset_root"]).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p


def work_dir(cfg):
    p = Path(cfg["paths"]["work_dir"]).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def subdir(cfg, name):
    p = work_dir(cfg) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_save_npy(path, arr):
    """Write to a temp file, then rename, so a crash never leaves a half-written .npy."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def progress(it, **kw):
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it
