"""MODULE 3: evaluate the rule-based sentence builder on data/module3/test_sentences.csv.

    python scripts/50_sentence_builder_eval.py
    python scripts/50_sentence_builder_eval.py --rebuild-lexicon     (build lexicon.csv again; keeps its verified values)

The first run writes the missing data files (lexicon.csv from the project's word lists; verbs.csv, predicates.csv and
test_sentences.csv from signbangla/sign2text/tables.py). Existing files are never overwritten.

Reports exact-match rate, chrF and BLEU (sacrebleu if it is installed, otherwise the small implementations below),
per category too. Writes to work/module3/eval/: results.csv (every row), failures.csv (with the trace), summary.json,
summary.md.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter

import _path  # noqa: F401

from signbangla.sign2text.builder import SentenceBuilder, lexicon_sources, module3_paths
from signbangla.sign2text.lexicon import read_csv, write_missing
from signbangla.utils import ROOT, load_config, subdir

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

PUNCT = ("।", "?")


def same(a, b):
    """Equal after Unicode normalisation and whitespace (punctuation counts)."""
    import unicodedata
    return " ".join(unicodedata.normalize("NFC", a).split()) == " ".join(unicodedata.normalize("NFC", b).split())


# ---------------------------------------------------------------- chrF (character n-grams 1-6, beta 2, no spaces)
def _char_ngrams(text, n):
    s = "".join(text.split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf_stats(hyp, ref, max_n=6):
    out = []
    for n in range(1, max_n + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        out.append((sum((h & r).values()), sum(h.values()), sum(r.values())))
    return out


def chrf_from_stats(stats, beta=2.0):
    """Average n-gram precision and recall over n, then the F-score (as sacrebleu's chrF)."""
    ps, rs = [], []
    for match, hyp, ref in stats:
        if hyp > 0 and ref > 0:
            ps.append(match / hyp)
            rs.append(match / ref)
    if not ps:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def corpus_chrf(hyps, refs):
    if sacrebleu:
        return sacrebleu.corpus_chrf(hyps, [refs]).score
    total = [[0, 0, 0] for _ in range(6)]
    for h, r in zip(hyps, refs):
        for i, (m, a, b) in enumerate(chrf_stats(h, r)):
            total[i][0] += m
            total[i][1] += a
            total[i][2] += b
    return chrf_from_stats(total)


# ---------------------------------------------------------------- BLEU-4 (words; । and ? are their own tokens)
def _tokens(text):
    for p in PUNCT:
        text = text.replace(p, f" {p} ")
    return text.split()


def corpus_bleu(hyps, refs):
    if sacrebleu:
        return sacrebleu.corpus_bleu(hyps, [refs]).score
    match, total = [0] * 4, [0] * 4
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        ht, rt = _tokens(h), _tokens(r)
        hyp_len, ref_len = hyp_len + len(ht), ref_len + len(rt)
        for n in range(1, 5):
            hc = Counter(tuple(ht[i:i + n]) for i in range(len(ht) - n + 1))
            rc = Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
            match[n - 1] += sum((hc & rc).values())
            total[n - 1] += sum(hc.values())
    if hyp_len == 0:
        return 0.0
    # zero counts are replaced by 1 / 2^k (sacrebleu's "exp" smoothing)
    logs, k = [], 1
    for m, t in zip(match, total):
        if t == 0:
            return 0.0
        if m == 0:
            k *= 2
            logs.append(math.log(1 / (k * t)))
        else:
            logs.append(math.log(m / t))
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100 * bp * math.exp(sum(logs) / 4)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--rebuild-lexicon", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = module3_paths(cfg)
    written = write_missing(paths, lexicon_sources(cfg), rebuild_lexicon=args.rebuild_lexicon)
    for p in written:
        print(f"Wrote {p.relative_to(ROOT)}")
    builder = SentenceBuilder.from_config(cfg, write=False)
    rows = read_csv(paths["test_sentences"])
    out_dir = subdir(cfg, cfg["module3"]["eval_dir"])

    results = []
    for r in rows:
        q = r["is_question"].strip().lower() in ("true", "yes", "1")
        sentence, trace, unknown = builder.build(r["sign_words"], is_question=q)
        expected = r["expected_bangla"].strip()
        results.append(dict(id=r["id"], category=r["category"], sign_words=r["sign_words"], expected=expected,
                            output=sentence, exact=same(sentence, expected),
                            chrf=round(chrf_from_stats(chrf_stats(sentence, expected)), 1),
                            unknown=" ".join(unknown), trace=" | ".join(trace)))

    def score(rs):
        hyps, refs = [x["output"] for x in rs], [x["expected"] for x in rs]
        return dict(n=len(rs), exact=sum(x["exact"] for x in rs), exact_rate=round(sum(x["exact"] for x in rs) / len(rs), 3),
                    chrf=round(corpus_chrf(hyps, refs), 1), bleu=round(corpus_bleu(hyps, refs), 1))

    summary = dict(all=score(results), metric_source="sacrebleu" if sacrebleu else "built-in chrF and BLEU",
                   per_category={c: score([x for x in results if x["category"] == c])
                                 for c in dict.fromkeys(x["category"] for x in results)})
    columns = list(results[0])
    for name, rs in (("results.csv", results), ("failures.csv", [x for x in results if not x["exact"]])):
        with open(out_dir / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, columns)
            w.writeheader()
            w.writerows(rs)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"Module 3 sentence builder: {summary['all']['exact']}/{summary['all']['n']} exact "
             f"({100 * summary['all']['exact_rate']:.0f}%), chrF {summary['all']['chrf']}, BLEU {summary['all']['bleu']} "
             f"({summary['metric_source']})", "", "| Category | Rows | Exact | chrF | BLEU |", "|---|---:|---:|---:|---:|"]
    for c, s in summary["per_category"].items():
        lines.append(f"| {c} | {s['n']} | {s['exact']}/{s['n']} | {s['chrf']} | {s['bleu']} |")
    a = summary["all"]
    lines.append(f"| **all** | {a['n']} | {a['exact']}/{a['n']} | {a['chrf']} | {a['bleu']} |")
    fails = [x for x in results if not x["exact"]]
    if fails:
        lines += ["", "Failures:"] + [f"- {x['id']} {x['sign_words']} -> {x['output']} (expected {x['expected']})"
                                       for x in fails]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir.relative_to(ROOT)}: results.csv, failures.csv, summary.json, summary.md")


if __name__ == "__main__":
    main()
