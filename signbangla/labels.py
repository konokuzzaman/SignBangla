"""The 60 BdSLW60 classes (word number -> class index), taken from Table 2 of the dataset paper.
Check the Bangla spelling against the dataset's own label file before using it for speech/TTS."""
import re

# (word number, Bangla, romanized, English)
LABELS = [
    (1, "বাবা/আব্বা", "baba/abba", "Father"), (2, "আত্মীয়", "atmiyo", "Relative"),
    (3, "ভাই", "bhai", "Brother"), (4, "বোন", "bon", "Sister"), (5, "বউ", "bou", "Wife"),
    (6, "চাচা", "chacha", "Paternal Uncle"), (7, "চাচী", "chachi", "Paternal Aunt"),
    (8, "দাদা/নানা", "dada/nana", "Grandfather"), (9, "দাদী/নানী", "dadi/nani", "Grandmother"),
    (10, "দায়িত্ব", "dayitto", "Responsibility"), (11, "দেবর", "debor", "Husband's Younger Brother"),
    (12, "দুলাভাই", "dulabhai", "Sister's Husband"), (19, "কন্যা", "konna", "Daughter"),
    (20, "মা", "ma", "Mother"), (37, "আম", "am", "Mango"), (38, "আলু", "alu", "Potato"),
    (39, "আনারস", "anaras", "Pineapple"), (40, "আঙ্গুর", "angur", "Grapes"),
    (41, "আপেল", "apel", "Apple"), (42, "বিস্কুট", "biskut", "Biscuits"),
    (43, "বরই", "boroi", "Jujube"), (44, "কেক", "kek", "Cake"), (45, "চা", "cha", "Tea"),
    (46, "চাল", "chal", "Rice"), (47, "চিনি", "chini", "Sugar"), (48, "চিপস", "chips", "Chips"),
    (49, "চকলেট", "chokolet", "Chocolate"), (50, "ডাল", "dal", "Lentils"),
    (91, "বোতাম", "botam", "Button"), (92, "টুপি", "tupi", "Cap"), (93, "চাদর", "chador", "Shawl"),
    (94, "চিরুনি", "chiruni", "Comb"), (95, "চশমা", "chosma", "Spectacles"),
    (96, "চুড়ি", "churi", "Bangles"), (97, "ক্লিপ", "clip", "Clip"), (98, "ক্রিম", "cream", "Cream"),
    (99, "তথ্য", "tottho", "Data"), (100, "দেনাদার", "denadar", "Indebted"),
    (111, "জমজ", "jomoj", "Twin baby"), (112, "জুতা", "juta", "Shoe"),
    (211, "টুথপেস্ট", "toothpaste", "Toothpaste"), (212, "টিশার্ট", "t-shirt", "T-shirt"),
    (213, "টিউবলাইট", "tubelight", "Tubelight"), (214, "টিভি", "tv", "Television"),
    (215, "এসি", "ac", "Air-conditioner"), (216, "অ্যাপার্টমেন্ট", "apartment", "Apartment"),
    (217, "অডিও ক্যাসেট", "audio cassette", "Audio Cassette"), (218, "আয়না", "ayna", "Mirror"),
    (219, "বালতি", "balti", "Water Bucket"), (220, "বালু", "balu", "Sand"),
    (351, "এইডস", "aids", "AIDS"), (352, "বাত", "baat", "Arthritis"),
    (353, "ব্যান্ডেজ", "bandage", "Bandage"), (354, "ক্যাপসুল", "capsule", "Capsule"),
    (355, "চিকিৎসা", "chikitsa", "Treatment"), (356, "চোখ ওঠা", "chokh otha", "Conjunctivitis"),
    (357, "ডেঙ্গু", "dengue", "Dengue"), (358, "ডক্টর", "doctor", "Doctor"),
    (359, "দংশন", "donshon", "Bite"), (360, "দুর্বল", "durbol", "Weak"),
]
N_CLASSES = len(LABELS)
WORD_TO_CLASS = {f"W{w}": i for i, (w, *_rest) in enumerate(LABELS)}
# Optional "_<n>" suffix: one recording split into parts (BdSLW60: U1W357F_1, _2, _3), same signer and word.
NAME_RE = re.compile(r"^U(?P<user>\d+)W(?P<word>\d+)(?P<view>[A-Za-z]*)(?:_(?P<part>\d+))?$", re.IGNORECASE)
# Odd view letters in file names -> the real view. U18W358FL is annotated "View: Front".
VIEW_ALIASES = {"FL": "F"}


def parse_video_stem(stem):
    """'U11W1F' -> {'signer': 11, 'word': 'W1', 'view': 'F', 'part': None, 'class_idx': 0}; None if it does not match.
    'U1W357F_2' -> part 2 of signer 1, word W357. The stem itself stays the key that links video, keypoints and trials."""
    m = NAME_RE.match(stem)
    if not m:
        return None
    word = f"W{int(m['word'])}"
    view = m["view"].upper()
    return dict(signer=int(m["user"]), word=word, view=VIEW_ALIASES.get(view, view),
                part=int(m["part"]) if m["part"] else None, class_idx=WORD_TO_CLASS.get(word, -1))


def class_names(kind="english"):
    col = {"bangla": 1, "roman": 2, "english": 3}[kind]
    return [row[col] for row in LABELS]


def short_names(max_len=14):
    return [f"{i}:{n[:max_len]}" for i, n in enumerate(class_names("english"))]
