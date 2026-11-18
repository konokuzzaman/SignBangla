"""MODULE 3 on Ban-Sign-Sent-9K: split the sentence pairs, count rules from the train split, evaluate on dev.

    python scripts/53_bansign9k_module3.py            (splits.csv if missing, rules from train, scores on train + dev)
    python scripts/53_bansign9k_module3.py --final    (also the test split, once; it is not run again after that)

Reads data/bansign9k/sentences.csv (scripts/46). Writes:
    data/bansign9k/splits.csv           train / dev / test by (subject + object) group, fixed seed; never overwritten
    data/module3/corpus_rules.json      rules counted from the train rows only (corpus_rules.derive)
    work/module3/eval_bansign9k_v2/     results_<split>.csv, summary.json, summary.md, errors.md, rules.md
Scores: exact match, chrF and BLEU (same code as scripts/50), next to the ceiling (the most common natural sentence for
each distinct sign sentence of that split) and the builder before these rules (Part A, scripts/46).
"""
import argparse
import csv
import importlib.util
import json
import sys
import unicodedata
from collections import Counter, defaultdict

import _path  # noqa: F401

from signbangla.sign2text import corpus_rules as CR
from signbangla.sign2text.builder import SentenceBuilder, corpus_rules_path
from signbangla.utils import ROOT, load_config, subdir, work_dir

_spec = importlib.util.spec_from_file_location("builder_eval", ROOT / "scripts" / "50_sentence_builder_eval.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)  # same(), chrF and BLEU
MARKER_GROUPS = ("না", "নেই", "ছিল", "হবে", "হয়েছে")


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def last_sign(sign):
    w = CR.words_of(sign)
    return w[-1] if w and w[-1] in {CR.norm(m) for m in MARKER_GROUPS} else "other"


def nfc(text):
    """chrF and BLEU compare characters, so both sides get the same Unicode form first (the dataset writes য় as one
    character, the builder as য + nukta; exact match already ignores this)."""
    return unicodedata.normalize("NFC", text)


def score(rows, key="output"):
    if not rows:
        return dict(n=0, exact=0, exact_rate=0.0, chrf=0.0, bleu=0.0)
    hyps, refs = [nfc(r[key]) for r in rows], [nfc(r["natural"]) for r in rows]
    exact = sum(E.same(h, r) for h, r in zip(hyps, refs))
    return dict(n=len(rows), exact=exact, exact_rate=round(exact / len(rows), 3),
                chrf=round(E.corpus_chrf(hyps, refs), 1), bleu=round(E.corpus_bleu(hyps, refs), 1))


def ceiling(rows):
    """Exact-match rate when each distinct sign sentence always gets its most common natural sentence."""
    by = defaultdict(Counter)
    for r in rows:
        by[" ".join(CR.words_of(r["sign"]))][" ".join(CR.words_of(r["natural"]))] += 1
    best = sum(c.most_common(1)[0][1] for c in by.values())
    return dict(n=len(rows), exact=best, exact_rate=round(best / len(rows), 3) if rows else 0.0,
                ambiguous_sign_sentences=sum(1 for c in by.values() if len(c) > 1))


def evaluate(builder, rows, baseline):
    out = []
    for r in rows:
        sentence, trace, unknown = builder.build(r["sign"])
        out.append(dict(sentence_id=r["sentence_id"], split=r["split"], group=r["group"], marker=last_sign(r["sign"]),
                        sign=r["sign"], natural=r["natural"], output=sentence, exact=E.same(sentence, r["natural"]),
                        baseline=baseline.get(r["sentence_id"], ""),
                        chrf=round(E.chrf_from_stats(E.chrf_stats(nfc(sentence), nfc(r["natural"]))), 1),
                        unknown=" ".join(unknown), trace=" | ".join(trace)))
    return out


def classify(r, builder, pool_naturals, forms):
    """Why a row is wrong: ambiguity no rule can fix, a data inconsistency, or fixable."""
    sign, out, nat = " ".join(CR.words_of(r["sign"])), CR.words_of(r["output"]), CR.words_of(r["natural"])
    if " ".join(out) in pool_naturals.get(sign, set()):
        return "ambiguous: the same signs have this sentence elsewhere"
    key = CR.marker_key(builder.normalise(r["sign"], [], []))
    rule = builder.corpus.markers.get(key) if key and builder.corpus else None
    got = CR.natural_outcome(nat, forms)
    if rule and got and f"{got[1]}|{got[3]}" in rule["others"]:
        return "ambiguous: train has both forms for these signs"
    if out and nat and out[0] != nat[0]:
        return "data: the subject sign and the sentence's subject differ"
    return "fixable"


def signature(r):
    out, nat = CR.words_of(r["output"]), CR.words_of(r["natural"])
    extra = [w for w in out if w not in nat]
    missing = [w for w in nat if w not in out]
    return f"{' '.join(extra) or '-'} -> {' '.join(missing) or '-'}"


def error_table(results, builder, pool_rows, top=20):
    pool = defaultdict(set)
    for r in pool_rows:
        pool[" ".join(CR.words_of(r["sign"]))].add(" ".join(CR.words_of(r["natural"])))
    forms = CR.verb_forms(builder.lx)
    groups = defaultdict(list)
    for r in results:
        if not r["exact"]:
            r["error_kind"] = classify(r, builder, pool, forms)
            groups[signature(r)].append(r)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top], Counter(
        r["error_kind"] for r in results if not r["exact"])


def rules_md(rules):
    lines = [f"# Rules counted from {rules['train_rows']} train rows (min support {rules['settings']['min_support']})",
             "", "Support = train rows that agree / train rows with that key.", "",
             "## a. Compounds", "", "| Signs | Word | Support |", "|---|---|---:|"]
    lines += [f"| {c['signs']} | {c['word']} | {c['support']}/{c['of']} |" for c in rules["compounds"]]
    lines += ["", "## a. Time of day -> locative (lexicon forms; how often the train sentence has it)", "",
              "| Sign | Written | Support |", "|---|---|---:|"]
    lines += [f"| {t['word']} | {t['locative']} | {t['support']}/{t['of']} |" for t in rules["time_locative"]]
    lines += ["", "## b. Verb for sentences without a verb sign", "", "| Object | Verb | Support |", "|---|---|---:|"]
    lines += [f"| {k} | {v['verb']} | {v['support']}/{v['of']} |" for k, v in rules["verb_by_object"].items()]
    lines += [f"| (any {k} word) | {v['verb']} | {v['support']}/{v['of']} |" for k, v in rules["verb_by_semantic"].items()]
    lines += ["", "## c. Final marker -> tense and negation", "",
              "| Marker | Tense | Negation | Support | Other forms in train |", "|---|---|---|---:|---|"]
    for m in rules["markers"]:
        key = CR.describe_marker((m["marker"], m["habitual"], m["will"]))
        lines.append(f"| {key} | {m['tense']} | {m['neg'] or '-'} | {m['support']}/{m['of']} | "
                     + ", ".join(f"{k} {n}" for k, n in m["others"].items()) + " |")
    lines += ["", "## d. Person for kin subjects", "", "| Subject | Tense | Negation | Person | Support |",
              "|---|---|---|---|---:|"]
    lines += [f"| {p['subject']} | {p['tense']} | {p['neg'] or '-'} | {p['person']} | {p['support']}/{p['of']} |"
              for p in rules["person_by_subject"]]
    lines += [f"| (any kin) | {p['tense']} | {p['neg'] or '-'} | {p['person']} | {p['support']}/{p['of']} |"
              for p in rules["person_by_tense"]]
    return "\n".join(lines) + "\n"


def report(split, results, rows, baseline_rows):
    by = defaultdict(list)
    for r in results:
        by[r["marker"]].append(r)
    return dict(scores=score(results), ceiling=ceiling(rows), baseline=score(baseline_rows, "baseline"),
                by_marker={m: dict(score(rs), ceiling=ceiling(rs)["exact_rate"], baseline=score(rs, "baseline"))
                           for m, rs in sorted(by.items())})


def summary_lines(s, title):
    a, c, b = s["scores"], s["ceiling"], s["baseline"]
    lines = [f"## {title}", "", "| | Exact | Exact % | chrF | BLEU |", "|---|---:|---:|---:|---:|",
             f"| builder with corpus rules | {a['exact']}/{a['n']} | {100 * a['exact_rate']:.1f} | {a['chrf']} | "
             f"{a['bleu']} |",
             f"| ceiling (most common sentence per sign sentence) | {c['exact']}/{c['n']} | "
             f"{100 * c['exact_rate']:.1f} | | |",
             f"| Part A builder (before) | {b['exact']}/{b['n']} | {100 * b['exact_rate']:.1f} | {b['chrf']} | "
             f"{b['bleu']} |", "", "| Last sign | Rows | Exact % | Ceiling % | chrF | BLEU | Part A chrF |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m, x in s["by_marker"].items():
        lines.append(f"| {m} | {x['n']} | {100 * x['exact_rate']:.1f} | {100 * x['ceiling']:.1f} | {x['chrf']} | "
                     f"{x['bleu']} | {x['baseline']['chrf']} |")
    return lines + [""]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--final", action="store_true", help="evaluate the test split (once)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["module3"]["corpus"]
    sentences = read_csv(ROOT / cfg["bansign9k"]["sentences"])
    split_path = ROOT / c["splits"]
    if not split_path.exists():
        split_rows = CR.make_splits(sentences, c["time_words"], tuple(c["split_fractions"]), c["split_seed"])
        write_csv(split_path, split_rows)
        print(f"Wrote {split_path.relative_to(ROOT)}")
    splits = {r["sentence_id"]: r for r in read_csv(split_path)}
    rows = [dict(s, split=splits[s["sentence_id"]]["split"], group=splits[s["sentence_id"]]["group"])
            for s in sentences]
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    print("Split rows: " + ", ".join(f"{k} {len(v)} ({len({r['group'] for r in v})} groups)"
                                    for k, v in sorted(by_split.items())))

    plain = SentenceBuilder.from_config(cfg, corpus=False)
    rules = CR.derive_from_split_rows(rows, plain, min_support=c["min_support"], min_share=c["min_share"])
    CR.save(rules, corpus_rules_path(cfg))
    builder = SentenceBuilder(plain.lx, cfg["module3"], CR.CorpusRules(rules))
    out_dir = subdir(cfg, c["eval_dir"])
    (out_dir / "rules.md").write_text(rules_md(rules), encoding="utf-8")
    base_path = work_dir(cfg) / c["baseline"]
    baseline = {r["sentence_id"]: r["output"] for r in read_csv(base_path)} if base_path.exists() else {}

    summary, results = {}, {}
    names = ["train", "dev"] + (["test"] if args.final else [])
    test_done = out_dir / "test_summary.json"
    if args.final and test_done.exists():
        print(f"The test split was already evaluated ({test_done.relative_to(ROOT)}); it is not run again.")
        names.remove("test")
    for name in names:
        results[name] = evaluate(builder, by_split[name], baseline)
        write_csv(out_dir / f"results_{name}.csv", results[name])
        summary[name] = report(name, results[name], by_split[name], results[name])
    summary["rules"] = rules_path = str(corpus_rules_path(cfg).relative_to(ROOT))
    if "test" in results:
        test_done.write_text(json.dumps(summary["test"], ensure_ascii=False, indent=1), encoding="utf-8")
    elif test_done.exists():
        summary["test"] = json.loads(test_done.read_text(encoding="utf-8"))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    top, kinds = error_table(results["dev"], builder, by_split["train"] + by_split["dev"])
    err = ["# Dev errors: the 20 most common (output words not in the sentence -> sentence words missing)", "",
           "Kinds of error on dev: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common()), ""]
    for sig, rs in top:
        r = rs[0]
        err += [f"## {len(rs)} x  {sig}  [{Counter(x['error_kind'] for x in rs).most_common(1)[0][0]}]", "",
                f"- signs: {r['sign']}", f"- output: {r['output']}", f"- natural: {r['natural']}",
                f"- trace: {r['trace']}", ""]
    (out_dir / "errors.md").write_text("\n".join(err), encoding="utf-8")

    lines = ["# Module 3 on Ban-Sign-Sent-9K with rules counted from train", "",
             f"Rules: {rules_path} ({rules['train_rows']} train rows). Test split: "
             + ("evaluated once, below." if "test" in summary else "not evaluated yet (--final)."), ""]
    for name in ("dev", "test", "train"):
        if name in summary:
            lines += summary_lines(summary[name], name)
    lines += ["Dev error kinds: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common())]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir.relative_to(ROOT)}: summary.md, summary.json, errors.md, rules.md, results_*.csv")


if __name__ == "__main__":
    main()
