"""MODULE 2: choose the 3 best trials of every word and save them at 30 fps -> work/sign_dictionary/.

    python scripts/20_build_sign_dictionary.py

Needs steps 3-5 (work/raw_keypoints, work/trials, work/quality_report.csv). Upside-down videos are never chosen.
Writes W<number>.npy per word, rest_pose.npy and dictionary.json; see signbangla/speech2sign/dictionary.py.
"""
import argparse

import _path  # noqa: F401
import pandas as pd

from signbangla.labels import N_CLASSES
from signbangla.speech2sign.dictionary import build_dictionary
from signbangla.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)

    info = build_dictionary(cfg)
    words = info["words"]
    trials = pd.DataFrame([dict(word=w, **t) for w, e in words.items() for t in e["trials"]])
    print(f"\nWords: {len(words)} / {N_CLASSES} | trials chosen: {len(trials)} | at {info['fps']} fps")
    print(f"Words with fewer than 3 trials: {[w for w, e in words.items() if len(e['trials']) < 3] or 'none'}")
    print(f"Hand of chosen trials: {trials['hand'].value_counts().to_dict()}")
    print(f"Source fps of chosen trials: {trials['source_fps'].round(-1).value_counts().to_dict()}")
    print(f"Frames per sign at {info['fps']} fps: min {trials['frames'].min()}, median {int(trials['frames'].median())}, "
          f"max {trials['frames'].max()}")
    print(f"Signers used: {sorted(trials['signer'].unique())}")
    print(f"Rest pose: {info['rest_info']}")
    print("\nSaved work/sign_dictionary/. Next: python scripts/20_m2_demo.py")


if __name__ == "__main__":
    main()
