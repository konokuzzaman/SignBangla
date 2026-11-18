"""Ban-Sign-Sent-9K sentence list: download the xlsx, save it as CSV with a vocabulary table, and run the Module 3
sentence builder on every sign sentence.

    python scripts/46_bansign9k_sentences.py

Writes data/bansign9k/sentences.csv, vocabulary.csv, word_timings.csv and typos.csv, and work/module3/eval_bansign9k/
(results.csv, summary.json, summary.md). The builder rules are not changed here: summary.md compares its output with
the natural sentences (exact match, chrF, BLEU, per last sign word) and lists what fails. The xlsx is never changed;
typos are only listed.
"""
import argparse
import csv
import importlib.util
import json
import sys
import urllib.request
from collections import Counter, defaultdict

import _path  # noqa: F401

from signbangla.extra_sources import bansign9k as B
from signbangla.sign2text.builder import SentenceBuilder
from signbangla.utils import ROOT, load_config, subdir

_spec = importlib.util.spec_from_file_location("builder_eval", ROOT / "scripts" / "50_sentence_builder_eval.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)  # same(), chrF and BLEU as for the Module 3 test sentences


def download(cfg, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return False
    url = f"https://huggingface.co/datasets/{cfg['bansign9k']['hf_dataset']}/resolve/main/{dest.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "signbangla"})
    with urllib.request.urlopen(req, timeout=120) as r:
        tmp.write_bytes(r.read())
    tmp.replace(dest)
    return True


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(rows)


def score(rs, key="output"):
    hyps, refs = [x[key] for x in rs], [x["natural"] for x in rs]
    exact = sum(E.same(h, r) for h, r in zip(hyps, refs))
    return dict(n=len(rs), exact=exact, exact_rate=round(exact / len(rs), 3) if rs else 0,
                chrf=round(E.corpus_chrf(hyps, refs), 1), bleu=round(E.corpus_bleu(hyps, refs), 1))


def verb_patterns(sentences):
    """How each sign pattern is said in the natural sentence: (last sign word, আগে present, খাওয়ায়া/খাওয়া present,
    হবে before না) -> Counter of the natural sentence's verb and what follows it."""
    table = defaultdict(Counter)
    for s in sentences:
        w, nat = B.sign_words(s["sign"]), B.sign_words(s["natural"])
        verbs = [i for i, x in enumerate(nat) if x.startswith(("খা", "খে"))]
        tail = " ".join(nat[verbs[-1]:]) if verbs else "(no verb)"
        key = (w[-1], "আগে" in w, any(x.startswith("খাওয়া") for x in w), len(w) > 1 and w[-2] == "হবে")
        table[key][tail] += 1
    return sorted(table.items(), key=lambda kv: -sum(kv[1].values()))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["bansign9k"]
    xlsx = ROOT / c["xlsx"]
    if download(cfg, xlsx):
        print(f"Downloaded {xlsx.relative_to(ROOT)}")

    sentences = B.read_sentences(xlsx)
    write_csv(ROOT / c["sentences"], sentences, ["sentence_id", "natural", "sign"])
    timings = B.read_word_timings(xlsx)
    write_csv(ROOT / c["word_timings"], timings)

    builder = SentenceBuilder.from_config(cfg, write=False)
    vocab = B.vocabulary(sentences)
    known = set(builder.lx.surface)
    for v in vocab:
        v["in_lexicon"] = v["word"] in known or all(p in known for p in v["word"].split("-"))
    write_csv(ROOT / c["vocabulary"], vocab)
    typos = B.possible_typos(vocab, known)
    write_csv(ROOT / c["vocabulary"].replace("vocabulary.csv", "typos.csv"), typos,
              ["word", "count", "looks_like", "reason"])
    print(f"{len(sentences)} sentences, {len(vocab)} sign words, {len(timings)} word timings, "
          f"{len(typos)} possible typos")

    results = []
    for s in sentences:
        out, trace, unknown = builder.build(s["sign"])
        results.append(dict(sentence_id=s["sentence_id"], marker=B.marker(s["sign"]), sign=s["sign"],
                            natural=s["natural"], output=out, exact=E.same(out, s["natural"]),
                            chrf=round(E.chrf_from_stats(E.chrf_stats(out, s["natural"])), 1),
                            unknown=" ".join(unknown), trace=" | ".join(trace)))
    groups = defaultdict(list)
    for r in results:
        groups[r["marker"]].append(r)
    unknown = Counter(w for r in results for w in r["unknown"].split())
    summary = dict(all=score(results), metric_source="sacrebleu" if E.sacrebleu else "built-in chrF and BLEU",
                   by_last_word={m: score(groups[m]) for m in (*B.MARKERS, "other") if groups[m]},
                   unknown_words=dict(unknown.most_common()),
                   copy_baseline=score([dict(r, copy=" ".join(B.sign_words(r["sign"])) + "।") for r in results], "copy"))
    out_dir = subdir(cfg, c["eval_dir"])
    write_csv(out_dir / "results.csv", results)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    a = summary["all"]
    lines = [f"Ban-Sign-Sent-9K: Module 3 builder on {a['n']} sign sentences: {a['exact']} exact "
             f"({100 * a['exact_rate']:.1f}%), chrF {a['chrf']}, BLEU {a['bleu']} ({summary['metric_source']})",
             f"Sign words copied unchanged (for comparison): chrF {summary['copy_baseline']['chrf']}, "
             f"BLEU {summary['copy_baseline']['bleu']}", "",
             "| Last sign word | Rows | Exact | chrF | BLEU |", "|---|---:|---:|---:|---:|"]
    for m, s in summary["by_last_word"].items():
        lines.append(f"| {m} | {s['n']} | {s['exact']} | {s['chrf']} | {s['bleu']} |")
    lines += ["", "Words the builder does not know: " + ", ".join(f"{w} ({n})" for w, n in unknown.most_common())]
    lines += ["", "How the natural sentences say each sign pattern (verb and what follows it):", "",
              "| Last sign | আগে | verb sign | হবে না | Rows | Natural verb |", "|---|---|---|---|---:|---|"]
    for (last, before, verb, will), cnt in verb_patterns(sentences):
        yn = lambda x: "yes" if x else ""  # noqa: E731
        lines.append(f"| {last} | {yn(before)} | {yn(verb)} | {yn(will)} | {sum(cnt.values())} | "
                     + ", ".join(f"{t} {n}" for t, n in cnt.most_common(3)) + " |")
    lines += ["", "Examples per last sign word:"]
    for m, rs in groups.items():
        for r in rs[:3]:
            lines.append(f"- [{m}] {r['sign']} -> {r['output']} (natural: {r['natural']})")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir.relative_to(ROOT)}: results.csv, summary.json, summary.md")


if __name__ == "__main__":
    main()
