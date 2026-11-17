"""MODULE 2: choose the 3 best trials of every word and save them at 30 fps -> work/sign_dictionary/.

    python scripts/20_build_sign_dictionary.py                   BdSLW60 -> work/sign_dictionary/
    python scripts/20_build_sign_dictionary.py --core            BdSLW60 + BdSLW401 + core vocabulary
                                                                 -> work/sign_dictionary_core/
    python scripts/20_build_sign_dictionary.py --core --use-unverified   also the INCLUDE (Indian SL) signs

Needs steps 3-5 (work/raw_keypoints, work/trials, work/quality_report.csv). Upside-down videos are never chosen.
Writes W<number>.npy per word, rest_pose.npy and dictionary.json; see signbangla/speech2sign/dictionary.py.
--core needs work/sign_dictionary first, and uses scripts/33 (BdSLW401), 41 (BdSL40, INCLUDE) and 42 (digits) where
their outputs exist; see signbangla/speech2sign/core_dictionary.py. It reports which core words have a sign.
"""
import argparse

import _path  # noqa: F401
import pandas as pd

from signbangla.labels import N_CLASSES
from signbangla.speech2sign import bdsl300 as b3
from signbangla.speech2sign.core_dictionary import build_core_dictionary, coverage
from signbangla.speech2sign.dictionary import build_dictionary
from signbangla.utils import ROOT, load_config, work_dir


def report_base(info):
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


def report_core(info, cfg):
    words = info["words"]
    by_source = pd.Series([e["source"] for e in words.values()]).value_counts().to_dict()
    print(f"\nDictionary: {len(words)} words {by_source}"
          f"{' (INCLUDE included)' if info['use_unverified_signs'] else ' (INCLUDE left out: use_unverified_signs is false)'}")
    cov = coverage(info)
    main = cov[cov["original_list"]]
    print(f"\nCore list ({len(main)} words): {int(main['has_sign'].sum())} have a sign "
          f"({int((main['verified'] == 'yes').sum())} verified for BdSL)")
    table = main.groupby("category", sort=False).agg(words=("id", "size"), with_sign=("has_sign", "sum"))
    print(table.to_string())
    signed = main[main["has_sign"]]
    print("\nBy source:", signed["source"].value_counts().to_dict())
    rows = b3.read_word_list(ROOT / cfg["speech2sign"]["core_vocab"]["word_list"])
    b300 = [r for r in rows if r.get("source") == b3.SOURCE]
    used = [k for k, e in words.items() if e["source"] == b3.SOURCE]
    print(f"BdSL-300: {sum(b3.is_true(r['verified']) for r in b300)} of {len(b300)} words verified in the word list; "
          f"{len(used)} used ({' '.join(used) or 'none'}). Unverified BdSL-300 words are never used.")
    extra = cov[~cov["original_list"] & cov["has_sign"]]
    print(f"Extra BdSL40 words with a sign: {' '.join(extra['bangla'])}")
    for cat, rows in main[~main["has_sign"]].groupby("category", sort=False):
        print(f"  still missing ({cat}): {' '.join(rows['bangla'])}")
    path = work_dir(cfg) / cfg["speech2sign"]["core_vocab"]["dictionary_dir"] / "core_coverage.csv"
    cov.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {path.parent} and core_coverage.csv. Try: python scripts/21_speech_to_sign_demo.py --text \"আমি অসুস্থ\"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--core", action="store_true", help="build work/sign_dictionary_core (BdSLW60 + 401 + core words)")
    ap.add_argument("--use-unverified", action="store_true", help="--core: include the INCLUDE (Indian SL) signs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.core:
        report_core(build_core_dictionary(cfg, use_unverified=True if args.use_unverified else None), cfg)
    else:
        report_base(build_dictionary(cfg))


if __name__ == "__main__":
    main()
