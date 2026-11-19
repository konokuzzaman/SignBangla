"""Speak the most common demo sentences and words once, so the app plays them from the cache without waiting.

    python scripts/62_app_precache_voice.py            (up to 100 texts)
    python scripts/62_app_precache_voice.py --list     (only print what would be spoken)

Texts, in this order, without repeats: data/app/test_sentences.csv, Module 3's 50 test sentences (expected Bangla), the
DU phrases, the Module 3 screen examples as built sentences, then the most frequent sign words of Ban-Sign-Sent-9K.
Audio: the app's voice (app.sign.voice, female) into work/app/speech_cache, the folder the app reads. Needs
requirements-speech.txt (edge-tts) and an internet connection; already cached texts are skipped.
"""
import argparse
import csv
import sys
import time
from collections import Counter

import _path  # noqa: F401

from signbangla.app.server import read_sentences
from signbangla.utils import ROOT, load_config, subdir, work_dir


def demo_texts(cfg, limit):
    out = []

    def add(texts):
        for t in texts:
            t = str(t).strip()
            if t and t not in out and len(out) < limit:
                out.append(t)

    app_tests = ROOT / cfg["app"]["test_sentences"]
    if app_tests.exists():
        add(read_sentences(app_tests))
    m3 = cfg["module3"]
    with open(ROOT / m3["test_sentences"], encoding="utf-8-sig", newline="") as f:
        add(r["expected_bangla"] for r in csv.DictReader(f))
    du = work_dir(cfg) / cfg["dynamic_du"]["dir"] / "class_table.csv"
    if du.exists():
        with open(du, encoding="utf-8-sig", newline="") as f:
            add(r["bangla"] for r in csv.DictReader(f))
    from signbangla.sign2text.builder import SentenceBuilder
    builder = SentenceBuilder.from_config(cfg, ROOT)
    add(builder.build(e.split())[0] for e in m3["screen"]["examples"])
    counts = Counter()
    with open(ROOT / cfg["bansign9k"]["sentences"], encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            counts.update(w.strip("।?") for w in r["sign"].split())
    add(w for w, _ in counts.most_common())
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    texts = demo_texts(cfg, args.limit)
    print(f"{len(texts)} texts")
    if args.list:
        print("\n".join(texts))
        return
    try:
        from signbangla.sign2speech.speech import edge_module, synthesize
        edge_module()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Speech output is not available: {e}")
    folder = subdir(cfg, cfg["app"]["sign"]["speech_cache"])
    done = failed = 0
    started = time.perf_counter()
    for i, t in enumerate(texts, 1):
        try:
            synthesize(t, folder, cfg["app"]["sign"]["voice"])
            done += 1
        except Exception as e:  # noqa: BLE001 - one failure must not stop the rest
            failed += 1
            print(f"  failed: {t}: {type(e).__name__}: {e}")
        if i % 10 == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    print(f"{done} cached, {failed} failed in {time.perf_counter() - started:.0f} s ({folder.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
