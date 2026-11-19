"""Context prediction, evaluated: hide one word of a test sentence at a time, predict it from the other words, and count
how often it comes back (top-1 and top-3), for the n-gram model and the pretrained fill-mask model.

    python scripts/61_app_prediction_eval.py              (both models; the fill-mask model must be downloaded once)
    python scripts/61_app_prediction_eval.py --models ngram

Sentences: data/app/test_sentences.csv (app.test_sentences). Without it, the Ban-Sign-Sent-9K test split is used: the
n-gram model is built from the train split only, so these sentences are unseen (but in the same style as its training
sentences, so expect high scores there). The source is written in every row.
Candidates: every word that has a sign (the app's dictionary), without না / নেই / question words, which prediction
never inserts; those words are not hidden either.
With the recogniser: where a hidden word is one of the 401 BdSLW401 words, one recorded clip of it (held-out signers,
work/bdslw401/eval_full) gives the recogniser's real top-5; the context then re-ranks them as in the app.
Writes work/app/prediction_eval.csv (one row per hidden word and model) and prediction_eval_summary.json.
"""
import argparse
import csv
import json
import math
import sys
import time
import unicodedata
from collections import defaultdict

import _path  # noqa: F401
import numpy as np

from signbangla.app.pipeline_speech2sign import SpeechToSign
from signbangla.app.predict import PROTECTED, Lemmatiser, MaskedLM, softmax
from signbangla.app.server import read_sentences
from signbangla.utils import ROOT, load_config, work_dir


def nfc(s):
    return unicodedata.normalize("NFC", s)


def test_sets(cfg, which):
    """-> [(source name, sentences)]: data/app/test_sentences.csv and / or the Ban-Sign-Sent-9K test split."""
    out = []
    path = ROOT / cfg["app"]["test_sentences"]
    if which in ("app", "both") and path.exists():
        out.append(("app_test_sentences", read_sentences(path)))
    if which in ("bansign9k", "both") or not out:
        b = cfg["bansign9k"]
        with open(ROOT / cfg["module3"]["corpus"]["splits"], encoding="utf-8-sig", newline="") as f:
            test = {r["sentence_id"] for r in csv.DictReader(f) if r["split"] == "test"}
        with open(ROOT / b["sentences"], encoding="utf-8-sig", newline="") as f:
            out.append(("bansign9k_test_split", [r["natural"] for r in csv.DictReader(f) if r["sentence_id"] in test]))
    return out


class RecordedClips:
    """The recogniser's real top-5 for a recorded clip of a word (BdSLW401 held-out signers)."""

    def __init__(self, cfg, recognizer):
        self.rec = recognizer
        base = work_dir(cfg) / "bdslw401" / "eval_full"
        self.dir = base / "processed"
        self.by_word = defaultdict(list)
        if recognizer is None or not (base / "manifest.csv").exists():
            return
        index = {nfc(w): i for i, (w, _e) in enumerate(recognizer.words)}
        self.index = index
        with open(base / "manifest.csv", encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "test" and r["saved"] == "True"]
        rows.sort(key=lambda r: (r["passes_filter"] != "True", r["trial_id"]))
        for r in rows:
            self.by_word[int(r["class_idx"])].append(r["file_name"])

    def top5(self, word):
        i = getattr(self, "index", {}).get(nfc(word))
        if i is None or not self.by_word.get(i):
            return None
        from signbangla.sign2text.recognizer import velocity
        torch = self.rec.torch
        seq = np.load(self.dir / self.by_word[i][0])
        x = seq.reshape(seq.shape[0], -1)
        x = velocity(x) if self.rec.use_velocity else x
        with torch.no_grad():
            p = torch.softmax(self.rec.model(torch.from_numpy(x[None].astype(np.float32)).to(self.rec.device)), -1)[0]
        p = p.cpu().numpy()
        best = np.argsort(p)[::-1][:5]
        return [(self.rec.words[k][0], float(p[k])) for k in best]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--models", default="ngram,mlm")
    ap.add_argument("--sentences", choices=["both", "app", "bansign9k"], default="both")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg["app"]["predict"]

    speech = SpeechToSign(cfg, load_asr=False)
    signs, _ = speech.signed_words()
    vocab = sorted(signs)
    lemmatise = Lemmatiser(speech.glosser.matcher)
    sets = test_sets(cfg, args.sentences)
    for source, sentences in sets:
        print(f"{len(sentences)} test sentences ({source})")
    print(f"{len(vocab)} candidate words with a sign")

    models = {}
    for name in args.models.split(","):
        started = time.perf_counter()
        if name == "ngram":
            models[name] = speech.predictor.lm
        elif name == "mlm":
            try:
                models[name] = MaskedLM(p["mlm_model"], p["device"])
            except Exception as e:  # noqa: BLE001
                print(f"  fill-mask model not available: {e}")
                continue
        print(f"  {name}: loaded in {time.perf_counter() - started:.1f} s")

    from signbangla.app.pipeline_sign2text import SignToText
    s2t = SignToText(cfg, ROOT, camera_index=None, speak=lambda t: "", monitor=False, predict=False)
    clips = RecordedClips(cfg, s2t.recognizer)

    rows, times = [], defaultdict(list)
    lam = p["lm_weight"]
    for source, sent in [(src, s) for src, ss in sets for s in ss]:
        words = lemmatise(sent)
        for i, target in enumerate(words):
            if target in PROTECTED or target not in signs:
                continue
            top5 = clips.top5(target)
            for name, lm in models.items():
                started = time.perf_counter()
                scores = lm.scores(words, i, vocab)
                times[name].append(time.perf_counter() - started)
                ranked = sorted(scores, key=lambda w: -scores[w])
                rank = ranked.index(target) + 1
                row = dict(source=source, sentence=sent, words=" ".join(words), position=i, target=target, model=name,
                           rank=rank, top1=ranked[0], top1_hit=rank == 1, top3_hit=rank <= 3,
                           recogniser_top5="", recogniser_top1_hit="", with_context_top1="", with_context_hit="")
                if top5:
                    lmp = softmax(lm.scores(words, i, [w for w, _ in top5]))
                    comb = {w: (1 - lam) * math.log(max(s, 1e-6)) + lam * math.log(max(lmp[w], 1e-9))
                            for w, s in top5}
                    best = max(comb, key=comb.get)
                    row.update(recogniser_top5=" ".join(f"{w}:{s:.2f}" for w, s in top5),
                               recogniser_top1_hit=top5[0][0] == target, with_context_top1=best,
                               with_context_hit=best == target)
                rows.append(row)

    out = work_dir(cfg) / p["eval"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]) if rows else ["source"])
        w.writeheader()
        w.writerows(rows)

    summary = dict(candidates=len(vocab), sets={src: len(ss) for src, ss in sets}, results={})
    print(f"\nHidden words: one at a time; candidates: all {len(vocab)} words with a sign")
    print(f"{'sentences':<22} {'model':<6} {'gaps':>5} {'top-1':>7} {'top-3':>7} {'ms/gap':>7} | {'with clip':>9} "
          f"{'recogniser':>10} {'+ context':>9}")
    for source, _ in sets:
        for name in models:
            r = [x for x in rows if x["model"] == name and x["source"] == source]
            c = [x for x in r if x["recogniser_top5"]]
            s = dict(gaps=len(r), top1=round(float(np.mean([x["top1_hit"] for x in r])), 3) if r else 0,
                     top3=round(float(np.mean([x["top3_hit"] for x in r])), 3) if r else 0,
                     ms_per_gap=round(1000 * float(np.mean(times[name])), 1) if times[name] else 0,
                     with_clip=len(c),
                     recogniser_top1=round(float(np.mean([x["recogniser_top1_hit"] for x in c])), 3) if c else None,
                     with_context_top1=round(float(np.mean([x["with_context_hit"] for x in c])), 3) if c else None)
            summary["results"][f"{source}/{name}"] = s
            print(f"{source:<22} {name:<6} {s['gaps']:>5} {s['top1']:>7.1%} {s['top3']:>7.1%} {s['ms_per_gap']:>7} | "
                  f"{s['with_clip']:>9} {(s['recogniser_top1'] or 0):>10.1%} {(s['with_context_top1'] or 0):>9.1%}")
    (out.parent / "prediction_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT).as_posix()} and prediction_eval_summary.json")


if __name__ == "__main__":
    main()
