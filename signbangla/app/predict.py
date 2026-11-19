"""Context prediction: when a sign is unclear (Sign -> Text) or a spoken word has no sign (Speech -> Sign), suggest the
word that fits the OTHER words of the same sentence, before and after the gap. Always shown as a suggestion (orange, with
accept / undo / alternatives), never applied silently, and logged to work/app/predictions.csv.

Language models (app.predict.model):
  ngram  word pair and triple counts with smoothing, from our own Bangla sentences at the level of dictionary words
         (খাই -> খাওয়া): Ban-Sign-Sent-9K natural sentences (train split only, so its test split stays unseen), Module
         3's test sentences and the DU phrases. A gap is scored with the words on both sides:
         log P(w | l2 l1) + log P(r1 | l1 w) + log P(r2 | w r1).
  mlm    a small pretrained Bangla fill-mask model (app.predict.mlm_model, BanglaBERT generator), offline once
         downloaded: the sentence with the gap masked, scoring only the given candidate words.

Rules:
  - never add or remove না / নেই / নয় or question words: they are never candidates, and a sign whose best word is one of
    them is not predicted;
  - two neighbouring unclear signs, or a best score below app.predict.min_score -> "…" instead of a guess.
"""
import csv
import json
import math
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..speech2sign import core_vocab as cv
from ..speech2sign.text_processing import UNKNOWN, clean_text
from ..utils import work_dir

PROTECTED = frozenset(cv.NEGATION_WORDS | cv.QUESTION_WORDS)
GAP = "…"
BOS, EOS = "<s>", "</s>"
LOG_COLUMNS = ["time", "direction", "sentence", "position", "candidates", "chosen", "accepted"]


def softmax(scores):
    v = np.array(list(scores.values()), float)
    if not len(v):
        return {}
    p = np.exp(v - v.max())
    p /= p.sum()
    return dict(zip(scores, p.tolist()))


# ------------------------------------------------------------------ sentences for the n-gram model
class Lemmatiser:
    """Bangla text -> dictionary words (খাই -> খাওয়া, বাবার -> বাবা), word by word with a cache; dropped words left out,
    unknown words kept as they are."""

    def __init__(self, matcher):
        self.matcher, self.cache = matcher, {}

    def word(self, w):
        if w not in self.cache:
            m = self.matcher.match(w)
            self.cache[w] = [x.word if x.word != UNKNOWN else x.source for x in m if x.how != "dropped"]
        return self.cache[w]

    def __call__(self, text):
        return [lemma for w in clean_text(text).split() for lemma in self.word(w)]


def training_sentences(cfg, root, lemmatise):
    """Lists of dictionary words from Ban-Sign-Sent-9K (train split), Module 3's test sentences and the DU phrases."""
    out = []
    b = cfg["bansign9k"]
    sentences, splits = Path(root) / b["sentences"], Path(root) / cfg["module3"]["corpus"]["splits"]
    if sentences.exists():
        train = None
        if splits.exists():
            with open(splits, encoding="utf-8-sig", newline="") as f:
                train = {r["sentence_id"] for r in csv.DictReader(f) if r["split"] == "train"}
        with open(sentences, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if train is None or r["sentence_id"] in train:
                    out.append(lemmatise(r["natural"]))
    tests = Path(root) / cfg["module3"]["test_sentences"]
    if tests.exists():
        with open(tests, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                out.append(lemmatise(r["expected_bangla"]))
    du = work_dir(cfg) / cfg["dynamic_du"]["dir"] / "class_table.csv"
    if du.exists():
        with open(du, encoding="utf-8-sig", newline="") as f:
            out += [lemmatise(r["bangla"]) for r in csv.DictReader(f)]
    return [s for s in out if s]


# ------------------------------------------------------------------ language models
class NgramLM:
    """Interpolated trigram / bigram / unigram model with add-k smoothing of the unigrams."""
    name = "ngram"

    def __init__(self, sentences, lambdas=(0.6, 0.3, 0.1), k=0.1):
        self.l3, self.l2, self.l1 = lambdas
        self.k = k
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        self.ctx1, self.ctx2 = Counter(), Counter()
        for s in sentences:
            t = [BOS, BOS] + list(s) + [EOS]
            self.uni.update(t[2:])
            for a, b in zip(t[1:], t[2:]):
                self.bi[a, b] += 1
                self.ctx1[a] += 1
            for a, b, c in zip(t, t[1:], t[2:]):
                self.tri[a, b, c] += 1
                self.ctx2[a, b] += 1
        self.total = sum(self.uni.values())
        self.vocab = set(self.uni)
        self.n_sentences = len(sentences)

    def prob(self, w, a, b):
        """P(w | a b)."""
        p1 = (self.uni[w] + self.k) / (self.total + self.k * (len(self.vocab) + 1))
        p2 = self.bi[b, w] / self.ctx1[b] if self.ctx1[b] else 0.0
        p3 = self.tri[a, b, w] / self.ctx2[a, b] if self.ctx2[a, b] else 0.0
        return self.l3 * p3 + self.l2 * p2 + self.l1 * p1

    def gap_logprob(self, words, i, w):
        t = [BOS, BOS] + list(words[:i]) + [w] + list(words[i + 1:]) + [EOS]
        j = i + 2
        terms = [(t[j], t[j - 2], t[j - 1])]
        for n in (1, 2):
            if j + n < len(t):
                terms.append((t[j + n], t[j + n - 2], t[j + n - 1]))
        return sum(math.log(self.prob(*x)) for x in terms)

    def scores(self, words, i, candidates):
        """Log score of each candidate in the gap words[i] (words[i] itself is ignored)."""
        return {c: self.gap_logprob(words, i, c) for c in candidates}

    def evidence(self, words, i, w):
        """True when a word pair with a neighbour was seen in the training sentences."""
        left = words[i - 1] if i > 0 else BOS
        right = words[i + 1] if i + 1 < len(words) else EOS
        return self.bi[left, w] > 0 or self.bi[w, right] > 0


class MaskedLM:
    """A pretrained fill-mask model: the sentence with the gap masked; candidates of k word pieces are scored with k
    masks (mean log probability of their pieces), one forward pass per piece count."""
    name = "mlm"

    def __init__(self, model_name, device="auto"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        self.pieces = {}

    def _ids(self, w):
        if w not in self.pieces:
            self.pieces[w] = self.tok.convert_tokens_to_ids(self.tok.tokenize(w))
        return self.pieces[w]

    def scores(self, words, i, candidates):
        by_len = defaultdict(list)
        for c in candidates:
            ids = self._ids(c)
            if ids and self.tok.unk_token_id not in ids:
                by_len[len(ids)].append(c)
        out = {c: -50.0 for c in candidates}  # words the model cannot spell get a very low score
        for n, group in by_len.items():
            text = " ".join(list(words[:i]) + [" ".join([self.tok.mask_token] * n)] + list(words[i + 1:]))
            x = self.tok(text, return_tensors="pt").to(self.device)
            pos = (x["input_ids"][0] == self.tok.mask_token_id).nonzero().flatten().tolist()
            with self.torch.no_grad():
                logp = self.torch.log_softmax(self.model(**x).logits[0, pos], -1).cpu().numpy()
            for c in group:
                ids = self._ids(c)
                out[c] = float(np.mean([logp[k, t] for k, t in enumerate(ids)]))
        return out

    def evidence(self, words, i, w):
        return True


# ------------------------------------------------------------------ log
class PredictionLog:
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()

    def write(self, direction, sentence, position, candidates, chosen, accepted):
        with self.lock:
            new = not self.path.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(LOG_COLUMNS)
                w.writerow([datetime.now().isoformat(timespec="seconds"), direction, sentence, position,
                            json.dumps([[c, round(float(s), 4)] for c, s in candidates], ensure_ascii=False),
                            chosen, accepted])


# ------------------------------------------------------------------ the predictor
class ContextPredictor:
    def __init__(self, lm, pcfg, sign_vocabulary=(), log=None):
        """sign_vocabulary: the words that have a sign (gap suggestions are taken from these)."""
        self.lm, self.p, self.log = lm, pcfg, log
        self.vocabulary = sorted({w for w in sign_vocabulary if w and w not in PROTECTED})
        self.signs = {}  # word -> sign key, for Speech -> Sign
        self.categories = {}

    # ---- Sign -> Text
    def rescore(self, words, i, candidates):
        """candidates [(word, recogniser probability)] of the sign at i; words: the sentence (words[i] is ignored).
        -> dict(kind 'top5' | 'gap' | 'none', word, ranked [(word, combined probability)])"""
        lam = self.p["lm_weight"]
        cands = [(w, max(float(s), 1e-6)) for w, s in candidates if w not in PROTECTED]
        if not cands:
            return dict(kind="none", word=GAP, ranked=[])
        lm = softmax(self.lm.scores(words, i, [w for w, _ in cands]))
        combined = {w: (1 - lam) * math.log(s) + lam * math.log(max(lm[w], 1e-9)) for w, s in cands}
        ranked = sorted(softmax(combined).items(), key=lambda kv: -kv[1])
        fits = any(self.lm.evidence(words, i, w) for w, _ in cands)
        if not fits:
            gap = self.fill_gap(words, i)
            if gap["word"] != GAP:
                return gap
        if ranked[0][1] < self.p["min_score"]:
            return dict(kind="none", word=GAP, ranked=ranked)
        return dict(kind="top5", word=ranked[0][0], ranked=ranked)

    def fill_gap(self, words, i, extra=()):
        """The sign vocabulary word that fits the gap best (from the context alone)."""
        cands = [w for w in self.vocabulary if self.lm.evidence(words, i, w)] or []
        cands = list(dict.fromkeys(list(cands) + [w for w in extra if w not in PROTECTED]))
        if not cands:
            return dict(kind="none", word=GAP, ranked=[])
        ranked = sorted(softmax(self.lm.scores(words, i, cands)).items(), key=lambda kv: -kv[1])
        if ranked[0][1] < self.p["gap_min_score"]:
            return dict(kind="none", word=GAP, ranked=ranked[:5])
        return dict(kind="gap", word=ranked[0][0], ranked=ranked[:5])

    def predict_items(self, items):
        """Sign -> Text. items: dicts with word, candidates [[w, p], ...] (camera) and user (chosen by hand).
        Sets 'prediction' on unclear camera signs: dict(word, kind, alternatives, top1). Returns the changed indices."""
        threshold = self.p["unclear_below"]

        def unclear(it):
            c = it.get("candidates") or []
            return (it.get("source") == "camera" and not it.get("user") and c and float(c[0][1]) < threshold
                    and c[0][0] not in PROTECTED)

        flags = [unclear(it) for it in items]
        words = [it["word"] for it in items]
        changed = []
        for i, it in enumerate(items):
            old = (it.get("prediction") or {}).get("word")
            if not flags[i]:
                if it.get("prediction") and not it.get("user"):
                    it["prediction"] = None
                continue
            top1 = it["candidates"][0][0]
            if (i > 0 and flags[i - 1]) or (i + 1 < len(items) and flags[i + 1]):
                pred = dict(kind="none", word=GAP, ranked=[])  # two unclear neighbours: no guess
            else:
                ctx = [w if not flags[k] else GAP for k, w in enumerate(words)]
                pred = self.rescore(ctx, i, it["candidates"])
            it["prediction"] = dict(word=pred["word"], kind=pred["kind"], top1=top1,
                                    alternatives=[w for w, _ in pred["ranked"] if w != pred["word"]][:4],
                                    ranked=[[w, round(s, 4)] for w, s in pred["ranked"][:6]])
            if pred["word"] != old:
                changed.append(i)
                if self.log:
                    self.log.write("sign", " ".join(words), i, pred["ranked"], pred["word"], "pending")
        return changed

    # ---- Speech -> Sign
    def fill_speech_gaps(self, tokens):
        """Speech -> Sign: a word without a sign (after the synonym table) -> the signed word that fits the sentence
        best, by context and closeness of spelling to the spoken word (a word of a known category only gets words of
        that category); shown as '≈word'. Never for না / নেই / question words."""
        import difflib
        if not self.p.get("speech", True) or not self.signs:
            return tokens
        words = [t.word or t.text for t in tokens]
        for i, t in enumerate(tokens):
            if t.signed or t.status == "dropped" or t.text in PROTECTED or t.word in PROTECTED:
                continue
            cands = [w for w in self.signs if self.lm.evidence(words, i, w)
                     and (not t.category or self.categories.get(w) == t.category)]  # a known verb stays a verb
            lm = softmax(self.lm.scores(words, i, cands))
            if not lm:
                continue
            w_sim = self.p["speech_similarity_weight"]
            scored = {}
            for w, p in lm.items():
                sim = difflib.SequenceMatcher(None, clean_text(t.word or t.text), w, autojunk=False).ratio()
                scored[w] = (1 - w_sim) * p + w_sim * sim
            ranked = sorted(scored.items(), key=lambda kv: -kv[1])
            if ranked[0][1] < self.p["speech_min_score"]:
                continue
            best = ranked[0][0]
            t.status, t.word, t.key = "predicted", best, self.signs[best]
            t.alternatives = [w for w, _ in ranked[1:5]]
            t.note = f"no sign for {t.text}: ≈ {best} from the sentence context"
            if self.log:
                self.log.write("speech", " ".join(words), i, ranked[:6], best, "pending")
        return tokens


def build_predictor(cfg, root, matcher, sign_words, signs=None, categories=None, log=True):
    """-> (ContextPredictor, load seconds, message). sign_words: vocabulary for gap suggestions."""
    p = cfg["app"]["predict"]
    started = time.perf_counter()
    message = ""
    lm = None
    if p["model"] == "mlm":
        try:
            lm = MaskedLM(p["mlm_model"], p["device"])
        except Exception as e:  # noqa: BLE001 - not downloaded and offline, no transformers, ...
            message = f"fill-mask model not available ({type(e).__name__}: {str(e)[:120]}); using the n-gram model"
    if lm is None:
        lm = NgramLM(training_sentences(cfg, root, Lemmatiser(matcher)))
    path = work_dir(cfg) / p["log"]
    pred = ContextPredictor(lm, p, sign_words, PredictionLog(path) if log else None)
    pred.signs = dict(signs or {})
    pred.categories = dict(categories or {})
    return pred, round(time.perf_counter() - started, 2), message
