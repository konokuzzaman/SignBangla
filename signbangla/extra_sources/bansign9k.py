"""Ban-Sign-Sent-9K (Hugging Face banglagov/Ban-Sign-Sent-9K-V1): 1,922 Bangla sentences, each with its sign sentence
(sign words in BdSL order), and 9,610 videos of them. No licence is stated: research use only, never redistributed.

Bangla_Sign_Sentence.xlsx has two sheets:
- "Training Data": Sentence ID, Natural Sentence, Sign Sentence (1,922 rows).
- "Test Data Set": 44 word annotations (one sign word with its start and end time in one video). Its `metas` column
  holds the signers' names and e-mail addresses; only the id, word and timing columns are ever read from it.
"""
import difflib
import re
from collections import Counter

from ..speech2sign.text_processing import clean_text
from .xlsx import read_sheet

SENTENCE_SHEET, TIMING_SHEET = "Training Data", "Test Data Set"
SENTENCE_COLUMNS = ("Sentence ID", "Natural Sentence", "Sign Sentence")
TIMING_COLUMNS = ("annotatedGlossId", "sentenceId", "userId", "word", "start", "end", "contentUrl")
MARKERS = tuple(clean_text(w) for w in ("না", "নেই", "ছিল", "হবে", "হয়েছে"))  # last sign words that carry negation
# or tense in this data
SOURCE = "BanSign9K"


def sign_words(sign_sentence):
    """'বাবা সকাল শাক-সব্জী না।' -> ['বাবা', 'সকাল', 'শাক-সব্জী', 'না']: the project's text cleanup (NFC, punctuation
    removed), except that a hyphen inside a word is kept, since শাক-সব্জী is written as one sign word."""
    out = []
    for token in (sign_sentence or "").split():
        word = "-".join(p for p in (clean_text(x) for x in token.split("-")) if p)
        out.extend(word.split())
    return out


def _table(rows, wanted, sheet):
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    missing = [c for c in wanted if c not in header]
    if missing:
        raise ValueError(f"sheet {sheet!r}: missing columns {missing} (found {header})")
    idx = [header.index(c) for c in wanted]
    return [[r[i] for i in idx] for r in rows[1:] if any(r[i] is not None for i in idx)]


def read_sentences(xlsx_path):
    """-> [{sentence_id, natural, sign}] in file order. Text is kept exactly as in the file (no fixes)."""
    out = []
    for sid, natural, sign in _table(read_sheet(xlsx_path, SENTENCE_SHEET), SENTENCE_COLUMNS, SENTENCE_SHEET):
        out.append(dict(sentence_id=int(sid), natural=str(natural or "").strip(), sign=str(sign or "").strip()))
    return out


def read_word_timings(xlsx_path):
    """-> [{gloss_id, sentence_id, user_id, word, start, end, video}]; video is the file name from contentUrl."""
    out = []
    for gid, sid, uid, word, start, end, url in _table(read_sheet(xlsx_path, TIMING_SHEET), TIMING_COLUMNS,
                                                       TIMING_SHEET):
        out.append(dict(gloss_id=int(gid), sentence_id=int(sid), user_id=int(uid), word=str(word).strip(),
                        start=float(start), end=float(end), video=str(url).rsplit("/", 1)[-1]))
    return out


def vocabulary(sentences):
    """-> [{word, count, last_count, sentences}] sorted by count: every sign word (after cleanup), how often it occurs,
    how often as the last sign, and in how many sentences."""
    count, last, in_sent = Counter(), Counter(), Counter()
    for s in sentences:
        words = sign_words(s["sign"])
        count.update(words)
        in_sent.update(set(words))
        if words:
            last[words[-1]] += 1
    return [dict(word=w, count=c, last_count=last[w], sentences=in_sent[w])
            for w, c in sorted(count.items(), key=lambda x: (-x[1], x[0]))]


_DOUBLE_SIGN = re.compile(r"([া-ৌৗ])\1|য়ায়|ওয়ায়")


def possible_typos(vocab, known=(), max_count=2, cutoff=0.85):
    """Words that look like a misspelling: a doubled vowel sign or য়া twice (খাওয়ায়া, at any count), or a rare word
    (at most max_count times) very close to a word that is much more common here or known to the project.
    -> [{word, count, looks_like, reason}]. Only a list for a person to check; the source file is never changed."""
    counts = {v["word"]: v["count"] for v in vocab}
    known = [clean_text(w) for w in known]
    common = [w for w, c in counts.items() if c > max_count * 3] + [w for w in known if w not in counts]
    out = []
    for w, c in counts.items():
        if w in known:
            continue
        reason, like = "", ""
        if _DOUBLE_SIGN.search(w):
            reason = "repeated vowel sign or য়া"
        close = [x for x in difflib.get_close_matches(w, common, n=3, cutoff=cutoff) if x != w]
        if close and (reason or c <= max_count or close[0] in known):
            like = close[0]
            reason = reason or ("close to a known word" if close[0] in known else "close to a more common word")
        if reason:
            out.append(dict(word=w, count=c, looks_like=like, reason=reason))
    return out


def marker(sign_sentence):
    """The last sign word if it is one of MARKERS, else 'other'."""
    words = sign_words(sign_sentence)
    return words[-1] if words and words[-1] in MARKERS else "other"


# ---------------------------------------------------------------- videos (Sign_Videos.z01 ... z09 + Sign_Videos.zip)
# Sign_Videos/<sentence id>/REC_CS_S<sentence id>_U<signer>_<yyyymmdd>_<ms since 1970>.mp4 (REC = recorded in the
# crowd-sourcing app; 20 videos of 4 sentences start with UL, probably uploaded). Five signers (U4, U5, U6, U7, U19)
# each signed every sentence once.
VIDEO_RE = re.compile(r"^Sign_Videos/(\d+)/(REC|UL)_CS_S(\d+)_U(\d+)_(\d{8})_(\d+)\.mp4$")


def parse_video(name):
    """-> {sentence_id, signer, date, stamp, kind, file} or None. signer is 'U4' etc."""
    m = VIDEO_RE.match(name)
    if not m or m.group(1) != m.group(3):
        return None
    return dict(sentence_id=int(m.group(1)), signer=f"U{m.group(4)}", date=m.group(5), stamp=m.group(6),
                kind=m.group(2), file=name.rsplit("/", 1)[-1])


def video_stem(info):
    """A file name for the keypoints: S<sentence id>_<signer>."""
    return f"S{info['sentence_id']}_{info['signer']}"


def last_word_partners(sentences):
    """{sentence_id: [sentence ids whose sign words are the same except a different last word]}."""
    groups = {}
    words = {s["sentence_id"]: sign_words(s["sign"]) for s in sentences}
    for sid, w in words.items():
        if len(w) >= 2:
            groups.setdefault(tuple(w[:-1]), []).append(sid)
    out = {}
    for sid, w in words.items():
        out[sid] = [o for o in groups.get(tuple(w[:-1]), []) if o != sid and words[o][-1] != w[-1]] if len(w) >= 2 else []
    return out


def choose_subset(videos, sentences, per_marker, seed=0, max_partners=1):
    """Videos to extract: for each marker up to per_marker[marker] videos, spread evenly over the signers.

    Per signer, groups of sentences that share every sign word but the last are taken in random order, and from each
    group the members whose marker still has room are chosen, so the chosen videos are each other's DTW partners.
    A chosen video still without a partner gets up to max_partners partner videos (role 'partner'; these are cut too
    but do not count for the quota). -> [video dict + role ('target' or 'partner'), marker]."""
    import random
    rng = random.Random(seed)
    by_id = {s["sentence_id"]: s for s in sentences}
    words = {sid: sign_words(s["sign"]) for sid, s in by_id.items()}
    partners = last_word_partners(sentences)
    have = {(v["sentence_id"], v["signer"]): v for v in videos if v["sentence_id"] in by_id}
    signers = sorted({v["signer"] for v in have.values()}, key=lambda x: int(x[1:]))
    quota = {clean_text(m): n for m, n in per_marker.items()}
    groups = {}
    for sid, w in words.items():
        groups.setdefault(tuple(w[:-1]), []).append(sid)
    chosen, keys = [], set()
    for i, signer in enumerate(signers):
        room = {m: n // len(signers) + (1 if i < n % len(signers) else 0) for m, n in quota.items()}
        order = list(groups.values())
        rng.shuffle(order)
        order.sort(key=lambda g: -len({marker(by_id[x]["sign"]) for x in g}))  # groups with more markers first
        for g in order:
            take = []
            for sid in sorted(g, key=lambda x: rng.random()):
                mk = marker(by_id[sid]["sign"])
                if room.get(mk, 0) > 0 and (sid, signer) in have and mk not in {t[1] for t in take}:
                    take.append((sid, mk))
            if len(take) < 2 and any(room.values()) and len(g) > 1:
                continue  # a group giving one video adds no partner; used only in the second pass
            for sid, mk in take:
                room[mk] -= 1
                keys.add((sid, signer))
                chosen.append(dict(have[(sid, signer)], role="target", marker=mk))
        for g in order:  # second pass: fill what is left, one video at a time
            for sid in g:
                mk = marker(by_id[sid]["sign"])
                if room.get(mk, 0) > 0 and (sid, signer) in have and (sid, signer) not in keys:
                    room[mk] -= 1
                    keys.add((sid, signer))
                    chosen.append(dict(have[(sid, signer)], role="target", marker=mk))
    for v in list(chosen):
        mine = [p for p in partners[v["sentence_id"]] if (p, v["signer"]) in keys]
        for p in [p for p in partners[v["sentence_id"]] if (p, v["signer"]) in have and (p, v["signer"]) not in keys]:
            if len(mine) >= max_partners:
                break
            keys.add((p, v["signer"]))
            mine.append(p)
            chosen.append(dict(have[(p, v["signer"])], role="partner", marker=marker(by_id[p]["sign"])))
    return chosen


# Every sign word of the data: (word for the word list, English, category). খাওয়ায়া is written as খাওয়া here (the
# source spells it খাওয়ায়া everywhere); the source file itself is not changed.
WORDS = {
    "না": ("না", "not (present / future negation)", "negation"),
    "নেই": ("নেই", "not (past negation in this data)", "negation"),
    "ছিল": ("ছিল", "was (past tense marker)", "other"),
    "হবে": ("হবে", "will (future tense marker)", "other"),
    "হয়েছে": ("হয়েছে", "has (present perfect marker)", "other"),
    "আগে": ("আগে", "before, used to (habitual past)", "time"),
    "খাওয়ায়া": ("খাওয়া", "eat / drink", "verb"),
    "খাওয়া": ("খাওয়া", "eat / drink", "verb"),
    "বাবা": ("বাবা", "father", "noun"), "মা": ("মা", "mother", "noun"),
    "চাচা": ("চাচা", "uncle (father's brother)", "noun"), "খালা": ("খালা", "aunt (mother's sister)", "noun"),
    "দাদা": ("দাদা", "grandfather (father's father)", "noun"),
    "দাদি": ("দাদি", "grandmother (father's mother)", "noun"),
    "বউ": ("বউ", "wife (চাচা বউ = aunt)", "noun"), "জামাই": ("জামাই", "husband (খালা জামাই = uncle)", "noun"),
    "সকাল": ("সকাল", "morning", "time"), "দুপুর": ("দুপুর", "noon", "time"), "বিকাল": ("বিকাল", "afternoon", "time"),
    "সন্ধ্যা": ("সন্ধ্যা", "evening", "time"), "রাত": ("রাত", "night", "time"),
    "চা": ("চা", "tea", "noun"), "কফি": ("কফি", "coffee", "noun"), "মাংস": ("মাংস", "meat", "noun"),
    "ডাল": ("ডাল", "lentils", "noun"), "ফল": ("ফল", "fruit", "noun"),
    "শাক-সব্জী": ("শাক-সব্জী", "vegetables", "noun"),
}


def word_list_rows(vocab, cut_words=None):
    """Rows for extra_sources.word_list.add_words: every sign word. cut_words: {word: file prefix} of the last signs
    that have clips; those rows get has_clips and the prefix as folder."""
    table = {sign_words(k)[0]: v for k, v in WORDS.items()}
    cut_words = {sign_words(k)[0]: v for k, v in (cut_words or {}).items()}
    out = []
    for v in vocab:
        w = v["word"]
        bangla, english, category = table.get(w, (w, "", "other"))
        folder = cut_words.get(w, "")
        out.append(dict(bangla=bangla, english=english, category=category, folder=folder, has_clips=bool(folder)))
    return out
