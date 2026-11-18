"""Dhaka University "Dynamic" BdSL videos (Kaggle hassan0008jhh/dataset-bangla-sign-language, folder dynamic_dataset).
40 classes of .avi videos in train/val/test; no licence stated: research use only, never redistributed.

Video files are named by a Unix time stamp (dynamic_dataset/dataset/test/hello/1668402433.avi); there is no signer id,
so a signer-wise split cannot be made from the names. dataset_numpy holds the authors' own hand keypoints (not used).

CLASSES maps the class folders that are downloaded to Bangla. Every row needs a check by someone who knows BdSL.
"""
import re

SOURCE = "DynamicDU"
VIDEO_RE = re.compile(r"^dynamic_dataset/dataset/(train|val|test)/([^/]+)/([^/]+)\.avi$")

# folder: (bangla, english, category, note)
CLASSES = {
    "Please": ("দয়া করে", "please", "other", ""),
    "ghumano": ("ঘুমানো", "sleep", "verb", ""),
    "hello": ("হ্যালো", "hello", "greeting", ""),
    "Leave": ("বিদায়", "leave / goodbye", "greeting", "may mean leaving (চলে যাওয়া) rather than a goodbye"),
    "apni kemon achen": ("আপনি কেমন আছেন", "how are you", "phrase", "one sign sequence for the whole question"),
    "apnar nam ki": ("আপনার নাম কী", "what is your name", "phrase", ""),
    "Where Do You Live": ("আপনি কোথায় থাকেন", "where do you live", "phrase", ""),
    "What Time is It": ("কয়টা বাজে", "what time is it", "phrase", ""),
    "ami valo achi": ("আমি ভালো আছি", "I am fine", "phrase", ""),
    "abar dekha hbe": ("আবার দেখা হবে", "see you again", "phrase", ""),
    "shuvo sokal": ("শুভ সকাল", "good morning", "greeting", ""),
    "shuvo ratri": ("শুভ রাত্রি", "good night", "greeting", ""),
    "Good to See You": ("আপনাকে দেখে ভালো লাগলো", "good to see you", "phrase", ""),
    "apnake kivabe sahajjo korte pari": ("আপনাকে কীভাবে সাহায্য করতে পারি", "how can I help you", "phrase", ""),
}


def parse_video(name):
    """'dynamic_dataset/dataset/test/hello/1668402433.avi' -> {split, folder, file, stamp}, or None."""
    m = VIDEO_RE.match(name)
    if not m:
        return None
    return dict(split=m.group(1), folder=m.group(2), file=m.group(3) + ".avi", stamp=m.group(3))


def clip_stem(info):
    """A file name for the keypoints: <folder with _ for spaces>__<split>__<stamp>."""
    return f"{info['folder'].replace(' ', '_')}__{info['split']}__{info['stamp']}"
