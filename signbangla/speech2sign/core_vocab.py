"""The core vocabulary for Module 2: pronouns, verbs, question words, negation, answers, numbers, time words, greetings.

There is no BdSL signer in the team, so every sign comes from an existing dataset:
    BdSL40    INCLUDE (Indian SL) videos whose sign its authors found to be the same in BdSL, using the BdSL dictionary.
              The BdSL40 download link is dead, so the same videos are taken from INCLUDE, using the BdSL40 mapping
              (which INCLUDE word is which BdSL40 word). verified = yes.
    BDSL49    still photos of the BdSL digits 0-9, turned into short signs (scripts/42). verified = yes.
              10, 20 and 100 are signed digit by digit, which is an assumption: verified = no.
    INCLUDE   Indian SL videos of words on our list. Indian SL is a different language, so verified = no; they are used
              only with speech2sign.core_vocab.use_unverified_signs: true.
Words with no sign stay in the list, so text processing still recognises them and shows them as text.

This table is the source of data/core_vocab/word_list.csv (written by scripts/40_core_vocab_download.py), because data/
is not in git.
"""
import csv

CATEGORIES = ("pronoun", "verb", "question", "negation", "answer", "number", "time", "greeting", "extra")
# Text colour (BGR) for a word that has no sign yet, by category. Words that are not in any list are red.
CATEGORY_COLORS = {  # BGR: pronoun blue, verb purple, question orange, negation magenta, answer green, number teal,
    # time brown, greeting olive, extra grey
    "pronoun": (160, 110, 20), "verb": (170, 50, 140), "question": (20, 120, 220), "negation": (130, 20, 180),
    "answer": (60, 140, 40), "number": (130, 100, 0), "time": (30, 90, 120), "greeting": (0, 130, 150),
    "extra": (90, 90, 90), "word": (90, 90, 90),
}
UNKNOWN_TEXT_COLOR = (40, 40, 200)

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"

# (id, Bangla, English, category). Spelling variants are joined with "/" like labels.py; the first one is shown.
CORE_WORDS = [
    ("C001", "আমি", "I", "pronoun"), ("C002", "তুমি", "you", "pronoun"), ("C003", "আপনি", "you (formal)", "pronoun"),
    ("C004", "সে", "he / she", "pronoun"), ("C005", "আমরা", "we", "pronoun"), ("C006", "তারা", "they", "pronoun"),
    ("C007", "আমার", "my", "pronoun"), ("C008", "তোমার", "your", "pronoun"),
    ("C009", "খাওয়া", "eat", "verb"), ("C010", "পান করা", "drink", "verb"), ("C011", "যাওয়া", "go", "verb"),
    ("C012", "আসা", "come", "verb"), ("C013", "চাওয়া", "want", "verb"), ("C014", "দেওয়া/দেয়া", "give", "verb"),
    ("C015", "নেওয়া/নেয়া", "take", "verb"), ("C016", "ঘুমানো", "sleep", "verb"), ("C017", "বসা", "sit", "verb"),
    ("C018", "দেখা", "see", "verb"), ("C019", "শোনা", "hear", "verb"), ("C020", "বলা", "say", "verb"),
    ("C021", "জানা", "know", "verb"), ("C022", "করা", "do", "verb"), ("C023", "সাহায্য করা", "help", "verb"),
    ("C024", "অপেক্ষা করা", "wait", "verb"), ("C025", "কেনা", "buy", "verb"), ("C026", "পড়া", "read", "verb"),
    ("C027", "লেখা", "write", "verb"),
    ("C028", "কী/কি", "what", "question"), ("C029", "কে", "who", "question"), ("C030", "কোথায়", "where", "question"),
    ("C031", "কখন", "when", "question"), ("C032", "কেন", "why", "question"), ("C033", "কেমন", "how", "question"),
    ("C034", "কত/কতো", "how much", "question"),
    ("C035", "না/নয়", "no / not", "negation"), ("C036", "নেই/নাই", "there is not", "negation"),
    ("C037", "হ্যাঁ/হ্যা/হাঁ", "yes", "answer"), ("C038", "ঠিক আছে", "all right", "answer"),
    ("C039", "০", "0", "number"), ("C040", "১", "1", "number"), ("C041", "২", "2", "number"), ("C042", "৩", "3", "number"),
    ("C043", "৪", "4", "number"), ("C044", "৫", "5", "number"), ("C045", "৬", "6", "number"), ("C046", "৭", "7", "number"),
    ("C047", "৮", "8", "number"), ("C048", "৯", "9", "number"), ("C049", "১০", "10", "number"),
    ("C050", "২০", "20", "number"), ("C051", "১০০", "100", "number"),
    ("C052", "আজ/আজকে", "today", "time"), ("C053", "গতকাল", "yesterday", "time"),
    ("C054", "আগামীকাল", "tomorrow", "time"), ("C055", "এখন", "now", "time"), ("C056", "পরে", "later", "time"),
    ("C057", "সকাল", "morning", "time"), ("C058", "রাত", "night", "time"), ("C059", "দিন", "day", "time"),
    ("C060", "আসসালামু আলাইকুম", "peace be upon you (greeting)", "greeting"), ("C061", "ধন্যবাদ", "thank you", "greeting"),
    ("C062", "দুঃখিত", "sorry", "greeting"), ("C063", "দয়া করে", "please", "greeting"),
    ("C064", "ব্যথা/ব্যাথা", "pain", "extra"), ("C065", "পানি/জল", "water", "extra"), ("C066", "বাড়ি/বাসা", "home", "extra"),
    ("C067", "বেশি", "more", "extra"), ("C068", "কম", "less", "extra"), ("C069", "বড়", "big", "extra"),
    ("C070", "ছোট", "small", "extra"),
    # extra BdSL40 words, not on the original list
    ("C071", "সময়", "time", "extra"), ("C072", "দুপুর", "noon", "time"), ("C073", "বন্ধু", "friend", "extra"),
    ("C074", "শিক্ষক", "teacher", "extra"), ("C075", "ছাত্র/শিক্ষার্থী", "student", "extra"), ("C076", "টাকা", "money", "extra"),
    ("C077", "বধির", "deaf", "extra"), ("C078", "নতুন", "new", "extra"), ("C079", "খারাপ", "bad", "extra"),
    ("C080", "খুশি", "glad", "extra"),
]
CORE_IDS = {row[0] for row in CORE_WORDS}

# ------------------------------------------------------------------ sources
INCLUDE_RECORD = "https://zenodo.org/records/4010759/files"
# INCLUDE category -> its zip files on Zenodo
INCLUDE_ZIPS = {
    "Pronouns": 2, "Greetings": 2, "Days_and_Time": 3, "Adjectives": 8, "People": 5, "Jobs": 2, "Society": 3,
}


def include_zip_urls(category):
    n = INCLUDE_ZIPS[category]
    return [f"{INCLUDE_RECORD}/{category}_{i}of{n}.zip?download=1" for i in range(1, n + 1)]


# id -> (INCLUDE category, INCLUDE word folder, BdSL40 label). From the BdSL40 ISL-to-BdSL mapping: BdSL40 "more" is the
# INCLUDE sign "wide", "noon" is "Afternoon", "glad" is "expensive" (video counts match: 21, 14 and 8).
BDSL40_SOURCES = {
    "C001": ("Pronouns", "I", "i"), "C002": ("Pronouns", "you", "you"),
    "C053": ("Days_and_Time", "Yesterday", "yesterday"), "C067": ("Adjectives", "wide", "more"),
    "C071": ("Days_and_Time", "Time", "time"), "C072": ("Days_and_Time", "Afternoon", "noon"),
    "C073": ("People", "Friend", "friend"), "C074": ("Jobs", "Teacher", "teacher"), "C075": ("Jobs", "Student", "student"),
    "C076": ("Society", "Money", "money"), "C077": ("Adjectives", "Deaf", "deaf"), "C078": ("Adjectives", "new", "new"),
    "C079": ("Adjectives", "bad", "bad"), "C080": ("Adjectives", "expensive", "glad"),
}
# The mapping notes that INCLUDE's I and you folders have 3 videos each the wrong way round.
INCLUDE_LABEL_FIXES = {
    ("I", "MVI_0001"): "you", ("I", "MVI_0002"): "you", ("I", "MVI_0003"): "you",
    ("you", "MVI_9996"): "I", ("you", "MVI_9997"): "I", ("you", "MVI_9998"): "I",
}
# Indian SL signs for words on our list: not verified for BdSL. (BdSL40's authors list "she" as different in BdSL.)
INCLUDE_SOURCES = {
    "C004": ("Pronouns", "he"), "C005": ("Pronouns", "we"), "C006": ("Pronouns", "they"),
    "C038": ("Greetings", "Alright"), "C061": ("Greetings", "Thank you"),
    "C052": ("Days_and_Time", "Today"), "C054": ("Days_and_Time", "Tomorrow"),
    "C057": ("Days_and_Time", "Morning"), "C058": ("Days_and_Time", "Night"),
}
# BDSL49 recognition set: folder number -> digit. The paper's Table 1 numbers labels from 1 (37 = ০ ... 46 = ৯) and the
# folders from 0; the image counts per folder confirm the shift.
BDSL49_URL = ("https://data.mendeley.com/public-files/datasets/k5yk4j8z8s/files/"
              "677428a7-dd66-4bc2-9c41-2ce044d512fb/file_downloaded")
BDSL49_DIGIT_FOLDERS = {36 + i: d for i, d in enumerate(BANGLA_DIGITS)}
DIGIT_IDS = {d: f"C{39 + i:03d}" for i, d in enumerate(BANGLA_DIGITS)}
COMPOSED_NUMBERS = {"C049": "১০", "C050": "২০", "C051": "১০০"}


def source_of(word_id):
    """(source, verified) as planned in this table; whether a sign was really made is in the dictionary."""
    if word_id in BDSL40_SOURCES:
        return "BdSL40", "yes"
    if word_id in DIGIT_IDS.values():
        return "BDSL49", "yes"
    if word_id in COMPOSED_NUMBERS:
        return "BDSL49 digit by digit", "no"
    if word_id in INCLUDE_SOURCES:
        return "INCLUDE", "no"
    return "", ""


WORD_LIST_COLUMNS = ["id", "bangla", "english", "category", "source", "verified", "folder", "core_id"]


def write_word_list(path):
    """data/core_vocab/word_list.csv. utf-8-sig so Excel shows Bangla correctly. Rows from other sources already in
    the file (BdSL-300, added by scripts/43 with a verified flag set by hand) are kept as they are."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if path.exists():
        with open(path, encoding="utf-8-sig") as f:
            core = {r[0] for r in CORE_WORDS}
            kept = [r for r in csv.DictReader(f) if r.get("id") not in core]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(WORD_LIST_COLUMNS)
        for word_id, bangla, english, category in CORE_WORDS:
            w.writerow([word_id, bangla, english, category, *source_of(word_id), "", ""])
        for r in kept:
            w.writerow([r.get(k, "") for k in WORD_LIST_COLUMNS])


# ------------------------------------------------------------------ verb forms
# For each verb: present forms, then stems that take the usual endings. Future stem + ব/বো/বে/বেন, continuous and perfect
# stems + ি/-/ে/েন (and ...িলাম, ...িল for the past continuous / past perfect), simple past stem + াম/ে/-/েন/ো.
# "other" are participles, infinitives and imperatives. Compound verbs (সাহায্য করা) reuse the forms of করা.
VERBS = {
    "খাওয়া": dict(present="খাই খাও খায় খান খাস", future="খা", cont="খাচ্ছ", perf="খেয়েছ", past="খেল",
                 other="খেয়ে খেতে খাওয়া"),
    "যাওয়া": dict(present="যাই যাও যায় যান যাস", future="যা", cont="যাচ্ছ", perf="গিয়েছ গেছ", past="গেল",
                 other="গিয়ে যেতে যাওয়া"),
    "আসা": dict(present="আসি আসো আসে আসেন আসিস", future="আস", cont="আসছ", perf="এসেছ", past="এল",
               other="এসে আসতে আসুন"),
    "চাওয়া": dict(present="চাই চাও চায় চান", future="চা", cont="চাইছ চাচ্ছ", perf="চেয়েছ", past="চাইল",
                 other="চেয়ে চাইতে"),
    "দেওয়া": dict(present="দিই দেই দাও দেয় দেন", future="দে দি", cont="দিচ্ছ", perf="দিয়েছ", past="দিল",
                 other="দিয়ে দিতে দেয়া"),
    "নেওয়া": dict(present="নিই নাও নেয় নেন", future="নে নি", cont="নিচ্ছ", perf="নিয়েছ", past="নিল",
                 other="নিয়ে নিতে নিন নেয়া"),
    "ঘুমানো": dict(present="ঘুমাই ঘুমাও ঘুমায় ঘুমান", future="ঘুমা", cont="ঘুমাচ্ছ", perf="ঘুমিয়েছ", past="ঘুমাল",
                  other="ঘুমিয়ে ঘুমাতে ঘুম ঘুমান"),
    "বসা": dict(present="বসি বসো বসে বসেন", future="বস", cont="বসছ", perf="বসেছ", past="বসল", other="বসতে বসুন"),
    "দেখা": dict(present="দেখি দেখো দেখে দেখেন", future="দেখ", cont="দেখছ", perf="দেখেছ", past="দেখল",
                other="দেখতে দেখুন"),
    "শোনা": dict(present="শুনি শোনো শোনে শোনেন", future="শুন", cont="শুনছ", perf="শুনেছ", past="শুনল",
                other="শুনে শুনতে শুনুন"),
    "বলা": dict(present="বলি বলো বলে বলেন", future="বল", cont="বলছ", perf="বলেছ", past="বলল", other="বলতে বলুন"),
    "জানা": dict(present="জানি জানো জানে জানেন", future="জান", cont="জানছ", perf="জেনেছ", past="জানল",
                other="জেনে জানতে"),
    "করা": dict(present="করি করো করে করেন করিস", future="কর", cont="করছ", perf="করেছ", past="করল",
               other="করতে করুন"),
    "কেনা": dict(present="কিনি কেনো কেনে কেনেন", future="কিন", cont="কিনছ", perf="কিনেছ", past="কিনল",
                other="কিনে কিনতে কিনুন"),
    "পড়া": dict(present="পড়ি পড়ো পড়ে পড়েন", future="পড়", cont="পড়ছ", perf="পড়েছ", past="পড়ল", other="পড়তে পড়ুন"),
    "লেখা": dict(present="লিখি লেখো লেখে লেখেন", future="লিখ", cont="লিখছ", perf="লিখেছ", past="লিখল",
                other="লিখে লিখতে লিখুন"),
}
COMPOUND_VERBS = {"পান করা": "পান", "সাহায্য করা": "সাহায্য", "অপেক্ষা করা": "অপেক্ষা"}
# Forms that are also other words are left out: দিন is "day", নেই is "there is not". (চা + বি would be চাবি, "key",
# so the future ending বি is not used.)
NOT_VERB_FORMS = {"দিন", "নেই", "পরে", "আজ"}


def _forms(v):
    out = set(v["present"].split()) | set(v.get("other", "").split())
    for stem in v["future"].split():
        out |= {stem + e for e in ("ব", "বো", "বে", "বেন")}
    for key in ("cont", "perf"):
        for stem in v[key].split():
            out |= {stem + e for e in ("ি", "", "ে", "েন", "িলাম", "িলে", "িল", "িলেন", "ো")}
    for stem in v["past"].split():
        out |= {stem + e for e in ("াম", "ে", "", "েন", "ো", "ি")}
    return out - NOT_VERB_FORMS


def verb_forms():
    """Inflected form -> base verb (as written in CORE_WORDS), for every verb on the list."""
    table = {}
    for base, v in VERBS.items():
        for form in _forms(v):
            table.setdefault(form, base)
    for base, noun in COMPOUND_VERBS.items():
        for form in _forms(VERBS["করা"]):
            table.setdefault(f"{noun} {form}", base)
    return table


# Other spellings and everyday words for words on the list (form -> word).
CORE_SYNONYMS = {
    "আজকে": "আজ", "আগামীকালকে": "আগামীকাল", "সকালে": "সকাল", "রাতে": "রাত", "দিনে": "দিন",
    "এখনই": "এখন", "পরের": "পরে", "আমাকে": "আমি", "তোমাকে": "তুমি", "আপনাকে": "আপনি", "তাকে": "সে", "তিনি": "সে",
    "উনি": "সে", "ওরা": "তারা", "তাঁরা": "তারা", "আমাদের": "আমরা", "কোথা": "কোথায়", "কেমনে": "কেমন",
    "সরি": "দুঃখিত", "প্লিজ": "দয়া করে", "সালাম": "আসসালামু আলাইকুম", "আসসালামুআলাইকুম": "আসসালামু আলাইকুম",
    "আসসালামু আলায়কুম": "আসসালামু আলাইকুম", "বাসায়": "বাড়ি", "বাড়িতে": "বাড়ি", "বেশী": "বেশি", "শিক্ষিকা": "শিক্ষক",
    "স্যার": "শিক্ষক", "খুশী": "খুশি", "বন্ধুরা": "বন্ধু", "টাকার": "টাকা", "পয়সা": "টাকা", "ঠিকাছে": "ঠিক আছে",
}
# Number words -> digits. নয় (nine) is also "is not", and is read as negation.
NUMBER_WORDS = {
    "শূন্য": "০", "এক": "১", "দুই": "২", "তিন": "৩", "চার": "৪", "পাঁচ": "৫", "ছয়": "৬", "সাত": "৭", "আট": "৮",
    "দশ": "১০", "বিশ": "২০", "কুড়ি": "২০", "একশ": "১০০", "একশো": "১০০", "একশত": "১০০",
}
NUMBER_SUFFIXES = ("টা", "টি", "টো", "জন", "খানা", "বার", "টাকা")
# Copula forms: BdSL usually leaves them out, so they are dropped quietly instead of being shown as unknown.
DROPPED_WORDS = {"আছ", "আছে", "আছি", "আছেন", "আছো", "আছিস", "হয়", "হই", "হও", "হন", "হয়ে", "তো"}
QUESTION_WORDS = {"কী", "কি", "কে", "কোথায়", "কখন", "কেন", "কেমন", "কত", "কতো"}
NEGATION_WORDS = {"না", "নেই", "নয়", "নাই"}


def to_bangla_digits(text):
    return text.translate(str.maketrans("0123456789", BANGLA_DIGITS))
