"""BdSL-300 (Kaggle hasibturjo/bdsl-300, CC BY-SA 4.0; built on SignBD-Word): word table, clip names, overlap with the
words the project already has.

The dataset has 300 word folders in two layouts:
- 100 English folders (Eat, Go, Today): <Folder>_<signer A-D>_<take>.mp4/.mov, 4 signers x 5 takes.
- 200 romanised Bangla folders from SignBD-Word (kothay, noy, apni): p<signer 1-16>_<view>_<word>.mp4. Every signing is
  filmed at once by a close camera (view c, upper body) and a full-body camera (view f); p2 and p3 have only f.

FOLDERS holds, for each folder, a reviewed comparison by meaning with the existing words (BdSLW60, BdSLW401 and the core
words that have a sign):
    english, bangla_guess, category, status, matches, core_id, note
- bangla_guess is a guess (romanised spellings are ambiguous); it is never presented as verified.
- status: "new" (no existing word has this meaning), "duplicate" (an existing word clearly has it; matches = which),
  "ambiguous" (could be the same or different; matches = the candidates).
- matches: existing keys, "W<n>" = BdSLW60/BdSLW401 word number, "C<nnn>" = core word with a sign.
- core_id: the core word (C001-C080) this folder would give a sign to, when that core word has none yet. A verified
  folder's sign is then used for that core word, so verb forms (খাব -> খাওয়া) and questions work with it.
"""
import csv
import re
from pathlib import Path

from ..labels import LABELS
from ..labels401 import LABELS_401
from . import core_vocab as cv

NEW, DUPLICATE, AMBIGUOUS = "new", "duplicate", "ambiguous"
STATUSES = (NEW, DUPLICATE, AMBIGUOUS)
CATEGORIES = ("verb", "pronoun", "question", "time", "number", "noun", "adjective", "other")
SOURCE = "BdSL300"
_N, _D, _A = NEW, DUPLICATE, AMBIGUOUS

# folder: (english, bangla_guess, category, status, matches, core_id, note)
FOLDERS = {
    # ---- English folders (signers A-D)
    "Afternoon": ("afternoon", "বিকেল", "time", _A, "C072", "", "BdSL40 'noon' (দুপুর) is INCLUDE 'Afternoon'; বিকেল and দুপুর are different times"),
    "April": ("April", "এপ্রিল", "time", _N, "", "", ""),
    "Attitude": ("attitude", "মনোভাব", "noun", _N, "", "", ""),
    "August": ("August", "আগস্ট", "time", _N, "", "", ""),
    "Aunty": ("aunty", "আন্টি / খালা", "noun", _A, "W17|W7|W15|W22", "", "a general word for an aunt; BdSLW401 has the specific aunts"),
    "Autumn": ("autumn", "শরৎ", "time", _N, "", "", ""),
    "Awareness": ("awareness", "সচেতনতা", "noun", _N, "", "", ""),
    "Bad Smell": ("bad smell", "দুর্গন্ধ", "noun", _N, "", "", ""),
    "Banana": ("banana", "কলা", "noun", _D, "W63", "", ""),
    "Birth": ("birth", "জন্ম", "noun", _N, "", "", ""),
    "Blackberry": ("blackberry / black plum", "কালোজাম / জাম", "noun", _A, "W60", "", "কালোজাম is both the fruit and a sweet"),
    "Boy": ("boy", "ছেলে", "noun", _A, "W25", "", "ছেলে means boy and son; BdSLW401 has পুত্র (son)"),
    "Call": ("call", "ডাকা", "verb", _N, "", "", ""),
    "Coffee": ("coffee", "কফি", "noun", _N, "", "", ""),
    "Cold": ("cold", "ঠান্ডা / সর্দি", "adjective", _A, "W393", "", "cold weather, or a cold (সর্দি, BdSLW401)"),
    "Come": ("come", "আসা", "verb", _N, "", "C012", ""),
    "Cover": ("cover", "ঢাকা / ঢাকনা", "verb", _A, "W309", "", "to cover, or a lid (ঢাকনা, BdSLW401)"),
    "Curry": ("curry", "তরকারি", "noun", _N, "", "", ""),
    "Daughter": ("daughter", "মেয়ে / কন্যা", "noun", _D, "W19", "", ""),
    "Death": ("death", "মৃত্যু", "noun", _N, "", "", ""),
    "December": ("December", "ডিসেম্বর", "time", _N, "", "", ""),
    "Diarrhea": ("diarrhoea", "ডায়রিয়া", "noun", _N, "", "", ""),
    "Disease": ("disease", "রোগ", "noun", _A, "W387", "", "রোগ (disease) vs অসুস্থ (sick); the signs may be the same"),
    "Do": ("do", "করা", "verb", _N, "", "C022", ""),
    "Drainage": ("drainage", "নর্দমা", "noun", _N, "", "", ""),
    "Eat": ("eat", "খাওয়া", "verb", _A, "W61", "C009", "খাবার (food, BdSLW401) and খাওয়া (eat) often share a sign"),
    "Egg": ("egg", "ডিম", "noun", _D, "W51", "", ""),
    "Evening": ("evening", "সন্ধ্যা", "time", _N, "", "", ""),
    "February": ("February", "ফেব্রুয়ারি", "time", _N, "", "", ""),
    "Feet": ("feet", "পা", "noun", _N, "", "", ""),
    "Flood": ("flood", "বন্যা", "noun", _N, "", "", ""),
    "Fruit": ("fruit", "ফল", "noun", _D, "W54", "", ""),
    "Girl": ("girl", "মেয়ে", "noun", _A, "W19", "", "মেয়ে means girl and daughter; BdSLW401 has কন্যা (daughter)"),
    "Go": ("go", "যাওয়া", "verb", _N, "", "C011", ""),
    "Gram": ("gram", "গ্রাম (ওজন)", "number", _N, "", "", "the unit of weight"),
    "Health": ("health", "স্বাস্থ্য", "noun", _A, "W391", "", "স্বাস্থ্য (health) vs সুস্থ (healthy, BdSLW401)"),
    "Itching": ("itching", "চুলকানি", "noun", _N, "", "", ""),
    "Jackfruit": ("jackfruit", "কাঁঠাল", "noun", _N, "", "", ""),
    "January": ("January", "জানুয়ারি", "time", _N, "", "", ""),
    "July": ("July", "জুলাই", "time", _N, "", "", ""),
    "June": ("June", "জুন", "time", _N, "", "", ""),
    "Kill": ("kill", "মারা / হত্যা", "verb", _N, "", "", ""),
    "Kilogram": ("kilogram", "কেজি", "number", _N, "", "", ""),
    "Kilometer": ("kilometre", "কিলোমিটার", "number", _N, "", "", ""),
    "Knock": ("knock", "টোকা দেওয়া", "verb", _N, "", "", ""),
    "Late Autumn": ("late autumn", "হেমন্ত", "time", _N, "", "", ""),
    "Leafy vegetable": ("leafy vegetable", "শাক", "noun", _A, "W76", "", "শাক vs সবজি (vegetable, BdSLW401)"),
    "Lemon": ("lemon", "লেবু", "noun", _D, "W65", "", ""),
    "Like": ("like", "পছন্দ", "verb", _N, "", "", "same meaning as the BdSL-300 folder pochondo"),
    "Liter": ("litre", "লিটার", "number", _N, "", "", ""),
    "Maintenance": ("maintenance", "রক্ষণাবেক্ষণ", "noun", _N, "", "", ""),
    "March": ("March", "মার্চ", "time", _N, "", "", ""),
    "May": ("May", "মে", "time", _N, "", "", ""),
    "Menstrual": ("menstruation", "মাসিক", "noun", _N, "", "", ""),
    "Month": ("month", "মাস", "time", _N, "", "", ""),
    "Morning": ("morning", "সকাল", "time", _N, "", "C057", "C057 has only an unverified INCLUDE sign"),
    "Need": ("need", "দরকার", "verb", _N, "", "", ""),
    "Neighbor": ("neighbour", "প্রতিবেশী", "noun", _D, "W135", "", ""),
    "November": ("November", "নভেম্বর", "time", _N, "", "", ""),
    "October": ("October", "অক্টোবর", "time", _N, "", "", ""),
    "Oil": ("oil", "তেল", "noun", _N, "", "", ""),
    "One year": ("one year", "এক বছর", "time", _N, "", "", ""),
    "Onion": ("onion", "পেঁয়াজ", "noun", _N, "", "", ""),
    "Pain": ("pain", "ব্যথা", "noun", _A, "W376", "C064", "ব্যথা (pain) vs মাথা ব্যাথা (headache, BdSLW401)"),
    "Pineapple": ("pineapple", "আনারস", "noun", _D, "W39", "", ""),
    "Plan": ("plan", "পরিকল্পনা", "noun", _N, "", "", ""),
    "Potato": ("potato", "আলু", "noun", _D, "W38", "", ""),
    "Pregnant": ("pregnant", "গর্ভবতী", "adjective", _N, "", "", ""),
    "Pulses": ("pulses / lentils", "ডাল", "noun", _D, "W50", "", ""),
    "Rainy season": ("rainy season", "বর্ষা", "time", _N, "", "", ""),
    "Receive": ("receive", "গ্রহণ করা", "verb", _A, "W108", "", "to receive vs গ্রহীতা (receiver, BdSLW401)"),
    "Relatives": ("relatives", "আত্মীয়", "noun", _D, "W2", "", ""),
    "Repair": ("repair", "মেরামত", "verb", _N, "", "", ""),
    "Rice": ("rice", "ভাত / চাল", "noun", _A, "W81|W46", "", "cooked rice (ভাত) or raw rice (চাল); BdSLW401 has both"),
    "Salt": ("salt", "লবণ", "noun", _D, "W67", "", ""),
    "Say": ("say", "বলা", "verb", _N, "", "C020", ""),
    "See": ("see", "দেখা", "verb", _N, "", "C018", ""),
    "September": ("September", "সেপ্টেম্বর", "time", _N, "", "", ""),
    "Session": ("session", "সেশন / অধিবেশন", "noun", _N, "", "", ""),
    "Shoes": ("shoes", "জুতা", "noun", _D, "W112", "", ""),
    "Soap": ("soap", "সাবান", "noun", _D, "W262", "", ""),
    "Spring": ("spring", "বসন্ত", "time", _N, "", "", ""),
    "Stop": ("stop", "থামা", "verb", _N, "", "", ""),
    "Sugar": ("sugar", "চিনি", "noun", _D, "W47", "", ""),
    "Summer": ("summer", "গ্রীষ্ম", "time", _N, "", "", ""),
    "Tea": ("tea", "চা", "noun", _D, "W45", "", ""),
    "Think": ("think", "চিন্তা করা", "verb", _N, "", "", "same meaning as the BdSL-300 folder chinta"),
    "Today": ("today", "আজ", "time", _N, "", "C052", "C052 has only an unverified INCLUDE sign"),
    "Toilet": ("toilet", "টয়লেট", "noun", _A, "W161", "", "টয়লেট vs বাথরুম (BdSLW401)"),
    "Tomorrow": ("tomorrow", "আগামীকাল", "time", _N, "", "C054", "C054 has only an unverified INCLUDE sign"),
    "Vegetable": ("vegetable", "সবজি", "noun", _D, "W76", "", ""),
    "Want": ("want", "চাওয়া", "verb", _N, "", "C013", ""),
    "Wash": ("wash", "ধোয়া", "verb", _N, "", "", ""),
    "Wastebin": ("wastebin", "ময়লার ঝুড়ি", "noun", _A, "W183", "", "a waste basket vs ঝুড়ি (basket, BdSLW401)"),
    "Week": ("week", "সপ্তাহ", "time", _N, "", "", ""),
    "Weight": ("weight", "ওজন", "noun", _N, "", "", ""),
    "Winter": ("winter", "শীত", "time", _N, "", "", ""),
    "Write": ("write", "লেখা", "verb", _N, "", "C027", "same meaning as the BdSL-300 folder lekha"),
    "Year": ("year", "বছর", "time", _N, "", "", ""),
    "Yesterday": ("yesterday", "গতকাল", "time", _D, "C053", "", ""),
    # ---- romanised Bangla folders (SignBD-Word, signers p1-p16)
    "aam": ("mango", "আম", "noun", _D, "W37", "", ""),
    "aangti": ("ring", "আংটি", "noun", _D, "W83", "", ""),
    "aat": ("eight", "আট", "number", _D, "C047", "", "the existing ৮ is built from a still photo (BDSL49)"),
    "abhawa": ("weather", "আবহাওয়া", "noun", _N, "", "", ""),
    "ador": ("affection", "আদর", "noun", _N, "", "", ""),
    "agun": ("fire", "আগুন", "noun", _N, "", "", ""),
    "akash": ("sky", "আকাশ", "noun", _N, "", "", ""),
    "alo": ("light", "আলো", "noun", _D, "W157", "", ""),
    "amader": ("our", "আমাদের", "pronoun", _N, "", "", ""),
    "amar": ("my", "আমার", "pronoun", _N, "", "C007", ""),
    "ami": ("I", "আমি", "pronoun", _D, "C001", "", ""),
    "amra": ("we", "আমরা", "pronoun", _N, "", "C005", "C005 has only an unverified INCLUDE sign"),
    "apnader": ("your (formal, plural)", "আপনাদের", "pronoun", _N, "", "", ""),
    "apnara": ("you (formal, plural)", "আপনারা", "pronoun", _N, "", "", ""),
    "apni": ("you (formal)", "আপনি", "pronoun", _N, "", "C003", ""),
    "attio": ("relative", "আত্মীয়", "noun", _D, "W2", "", ""),
    "baari": ("home", "বাড়ি", "noun", _N, "", "C066", ""),
    "baba": ("father", "বাবা", "noun", _D, "W1", "", ""),
    "bagan": ("garden", "বাগান", "noun", _N, "", "", ""),
    "bah": ("wow / nice", "বাহ", "other", _N, "", "", "guess: an exclamation"),
    "bangladesh": ("Bangladesh", "বাংলাদেশ", "noun", _N, "", "", ""),
    "batash": ("wind", "বাতাস", "noun", _N, "", "", ""),
    "bathroom": ("bathroom", "বাথরুম", "noun", _D, "W161", "", ""),
    "bhai": ("brother", "ভাই", "noun", _D, "W3", "", ""),
    "bhul": ("mistake", "ভুল", "noun", _N, "", "", ""),
    "bichana": ("bed", "বিছানা", "noun", _D, "W164", "", ""),
    "boi": ("book", "বই", "noun", _N, "", "", ""),
    "boka": ("foolish", "বোকা", "adjective", _N, "", "", ""),
    "bon": ("sister", "বোন", "noun", _D, "W4", "", ""),
    "bondhu": ("friend", "বন্ধু", "noun", _D, "W89|C073", "", ""),
    "boro": ("big", "বড়", "adjective", _N, "", "C069", ""),
    "brihospotibar": ("Thursday", "বৃহস্পতিবার", "time", _N, "", "", ""),
    "bristi": ("rain", "বৃষ্টি", "noun", _N, "", "", ""),
    "budhbar": ("Wednesday", "বুধবার", "time", _N, "", "", ""),
    "chabi": ("key", "চাবি", "noun", _N, "", "", ""),
    "chacha": ("uncle (father's brother)", "চাচা", "noun", _D, "W6", "", ""),
    "chachi": ("aunt (chacha's wife)", "চাচী", "noun", _D, "W7", "", ""),
    "chad": ("moon / roof", "চাঁদ / ছাদ", "noun", _A, "W229", "", "BdSLW401 W229 is চাদ in the house section, maybe ছাদ (roof)"),
    "chagol": ("goat", "ছাগল", "noun", _N, "", "", ""),
    "chair": ("chair", "চেয়ার", "noun", _D, "W168", "", ""),
    "chalak": ("clever", "চালাক", "adjective", _N, "", "", ""),
    "char": ("four", "চার", "number", _D, "C043", "", "the existing ৪ is built from a still photo (BDSL49)"),
    "chata": ("umbrella", "ছাতা", "noun", _D, "W170", "", ""),
    "chesta": ("try", "চেষ্টা", "verb", _N, "", "", ""),
    "chih": ("shame on you / yuck", "ছি", "other", _N, "", "", "guess: an exclamation"),
    "chikitsha": ("treatment", "চিকিৎসা", "noun", _D, "W355", "", ""),
    "chinta": ("think / worry", "চিন্তা (করা)", "verb", _N, "", "", "file names say chintakora; same meaning as the folder Think"),
    "chocolate": ("chocolate", "চকলেট", "noun", _D, "W49", "", ""),
    "chosma": ("glasses", "চশমা", "noun", _D, "W95", "", ""),
    "choto": ("small", "ছোট", "adjective", _N, "", "C070", ""),
    "choy": ("six", "ছয়", "number", _D, "C045", "", "the existing ৬ is built from a still photo (BDSL49)"),
    "chup": ("quiet", "চুপ", "other", _N, "", "", ""),
    "cricket": ("cricket", "ক্রিকেট", "noun", _N, "", "", ""),
    "dengue": ("dengue", "ডেঙ্গু", "noun", _D, "W357", "", ""),
    "dhoni": ("rich", "ধনী", "adjective", _N, "", "", ""),
    "dhonnobad": ("thank you", "ধন্যবাদ", "other", _N, "", "C061", "C061 has only an unverified INCLUDE sign"),
    "dhorjo": ("patience", "ধৈর্য", "noun", _N, "", "", ""),
    "din": ("day", "দিন", "time", _N, "", "C059", "guess: দিন can also mean 'give' (imperative)"),
    "diye": ("with / by giving", "দিয়ে", "other", _N, "", "", "guess"),
    "doctor": ("doctor", "ডাক্তার", "noun", _D, "W358", "", ""),
    "dorja": ("door", "দরজা", "noun", _D, "W175", "", ""),
    "dorshok": ("audience", "দর্শক", "noun", _N, "", "", ""),
    "dos": ("ten", "দশ", "number", _A, "C049", "", "the existing ১০ is signed digit by digit (১ then ০), an assumption"),
    "dosh": ("fault / blame", "দোষ", "noun", _N, "", "", "guess; the folder dos is ten"),
    "doya": ("please", "দয়া করে", "other", _N, "", "C063", "file names say doyakore"),
    "dudh": ("milk", "দুধ", "noun", _D, "W52", "", ""),
    "dui": ("two", "দুই", "number", _D, "C041", "", "the existing ২ is built from a still photo (BDSL49)"),
    "dukkhito": ("sorry", "দুঃখিত", "other", _N, "", "C062", ""),
    "durbol": ("weak", "দুর্বল", "adjective", _D, "W360", "", ""),
    "dustu": ("naughty", "দুষ্টু", "adjective", _N, "", "", ""),
    "ebong": ("and", "এবং", "other", _N, "", "", ""),
    "ek": ("one", "এক", "number", _D, "C040", "", "the existing ১ is built from a still photo (BDSL49)"),
    "ekhane": ("here", "এখানে", "other", _N, "", "", ""),
    "ekhon": ("now", "এখন", "time", _N, "", "C055", ""),
    "football": ("football", "ফুটবল", "noun", _N, "", "", ""),
    "ful": ("flower", "ফুল", "noun", _N, "", "", ""),
    "gach": ("tree", "গাছ", "noun", _N, "", "", ""),
    "ghori": ("clock / watch", "ঘড়ি", "noun", _D, "W105", "", ""),
    "glass": ("drinking glass", "গ্লাস", "noun", _N, "", "", "not চশমা (eyeglasses)"),
    "golarhaar": ("necklace", "গলার হার", "noun", _D, "W107", "", ""),
    "gopon": ("secret", "গোপন", "adjective", _N, "", "", ""),
    "gorib": ("poor", "গরিব", "adjective", _N, "", "", ""),
    "goru": ("cow", "গরু", "noun", _N, "", "", ""),
    "hash": ("duck", "হাঁস", "noun", _N, "", "", "guess: হাঁস (duck); could be হাসি (laugh)"),
    "hata": ("walk", "হাঁটা", "verb", _N, "", "", "guess: হাঁটা (walk); could be হাতা (sleeve)"),
    "hospital": ("hospital", "হাসপাতাল", "noun", _D, "W365", "", ""),
    "hridrog": ("heart disease", "হৃদরোগ", "noun", _D, "W366", "", ""),
    "icecream": ("ice cream", "আইসক্রিম", "noun", _D, "W58", "", ""),
    "janala": ("window", "জানালা", "noun", _D, "W179", "", ""),
    "jiggasha": ("ask", "জিজ্ঞাসা", "verb", _N, "", "", ""),
    "jodi": ("if", "যদি", "other", _N, "", "", ""),
    "jokhon": ("when (that)", "যখন", "other", _N, "", "", "not the question word কখন"),
    "jor": ("fever", "জ্বর", "noun", _D, "W371", "", ""),
    "kanna": ("crying", "কান্না", "noun", _N, "", "", ""),
    "kashi": ("cough", "কাশি", "noun", _D, "W372", "", ""),
    "ke": ("who", "কে", "question", _N, "", "C029", ""),
    "khabar": ("food", "খাবার", "noun", _D, "W61", "", ""),
    "kharap": ("bad", "খারাপ", "adjective", _D, "C079", "", ""),
    "khela": ("play / game", "খেলা", "verb", _N, "", "", ""),
    "khelowar": ("player", "খেলোয়াড়", "noun", _N, "", "", ""),
    "khoj": ("search", "খোঁজ", "verb", _N, "", "", ""),
    "khushi": ("glad", "খুশি", "adjective", _D, "C080", "", ""),
    "ki": ("what", "কী", "question", _N, "", "C028", ""),
    "kintu": ("but", "কিন্তু", "other", _N, "", "", ""),
    "konna": ("daughter", "কন্যা", "noun", _D, "W19", "", ""),
    "kothay": ("where", "কোথায়", "question", _N, "", "C030", ""),
    "kukur": ("dog", "কুকুর", "noun", _N, "", "", ""),
    "lekha": ("write", "লেখা", "verb", _N, "", "C027", "same meaning as the folder Write"),
    "lojja": ("shame / shy", "লজ্জা", "noun", _N, "", "", ""),
    "maa": ("mother", "মা", "noun", _D, "W20", "", ""),
    "mach": ("fish", "মাছ", "noun", _D, "W68", "", ""),
    "mangsho": ("meat", "মাংস", "noun", _D, "W69", "", ""),
    "mask": ("mask", "মাস্ক", "noun", _D, "W375", "", ""),
    "mathabetha": ("headache", "মাথা ব্যথা", "noun", _D, "W376", "", ""),
    "mehman": ("guest", "মেহমান", "noun", _D, "W120", "", ""),
    "moja": ("fun / socks", "মজা / মোজা", "noun", _A, "W121", "", "মজা (fun) or মোজা (socks, BdSLW401)"),
    "mongolbar": ("Tuesday", "মঙ্গলবার", "time", _N, "", "", ""),
    "mosha": ("mosquito", "মশা", "noun", _N, "", "", "not মশারি (mosquito net)"),
    "murgi": ("chicken", "মুরগি", "noun", _N, "", "", ""),
    "nana": ("grandfather (mother's father)", "নানা", "noun", _D, "W8", "", ""),
    "nani": ("grandmother (mother's mother)", "নানী", "noun", _D, "W9", "", ""),
    "nije": ("self", "নিজে", "pronoun", _N, "", "", ""),
    "nischit": ("sure", "নিশ্চিত", "adjective", _N, "", "", ""),
    "nodi": ("river", "নদী", "noun", _N, "", "", ""),
    "noy": ("nine / not", "নয়", "number", _A, "C048", "C035", "নয় is nine (৯ exists, from a still photo) or 'not' (negation, no sign)"),
    "nurse": ("nurse", "নার্স", "noun", _D, "W380", "", ""),
    "olosh": ("lazy", "অলস", "adjective", _N, "", "", ""),
    "onno": ("other", "অন্য", "other", _N, "", "", ""),
    "onumoti": ("permission", "অনুমতি", "noun", _N, "", "", ""),
    "opoman": ("insult", "অপমান", "noun", _N, "", "", ""),
    "orna": ("scarf (orna)", "ওড়না", "noun", _D, "W123", "", ""),
    "osubidha": ("problem / inconvenience", "অসুবিধা", "noun", _N, "", "", ""),
    "osudh": ("medicine", "ঔষধ", "noun", _D, "W389", "", ""),
    "ovivabok": ("guardian", "অভিভাবক", "noun", _D, "W124", "", ""),
    "ovvash": ("habit", "অভ্যাস", "noun", _N, "", "", ""),
    "pach": ("five", "পাঁচ", "number", _D, "C044", "", "the existing ৫ is built from a still photo (BDSL49)"),
    "pakhi": ("bird", "পাখি", "noun", _N, "", "", "not পাখা (hand fan)"),
    "pant": ("trousers", "প্যান্ট", "noun", _D, "W127", "", ""),
    "plastic": ("plastic", "প্লাস্টিক", "noun", _D, "W339", "", ""),
    "plate": ("plate", "প্লেট", "noun", _D, "W199", "", ""),
    "pochondo": ("like", "পছন্দ", "verb", _N, "", "", "same meaning as the folder Like"),
    "pora": ("read / study", "পড়া", "verb", _N, "", "C026", "guess: পড়া can also mean 'fall' or 'wear'"),
    "poribar": ("family", "পরিবার", "noun", _D, "W24", "", ""),
    "poriskar": ("clean", "পরিষ্কার", "adjective", _N, "", "", ""),
    "porisrom": ("hard work", "পরিশ্রম", "noun", _N, "", "", ""),
    "prosno": ("question", "প্রশ্ন", "noun", _N, "", "", ""),
    "protibeshi": ("neighbour", "প্রতিবেশী", "noun", _D, "W135", "", ""),
    "protibondhi": ("disabled", "প্রতিবন্ধী", "adjective", _N, "", "", ""),
    "protigga": ("promise", "প্রতিজ্ঞা", "noun", _N, "", "", ""),
    "protijogita": ("competition", "প্রতিযোগিতা", "noun", _N, "", "", ""),
    "purush": ("man", "পুরুষ", "noun", _N, "", "", ""),
    "putro": ("son", "পুত্র", "noun", _D, "W25", "", ""),
    "raag": ("anger", "রাগ", "noun", _N, "", "", ""),
    "raat": ("night", "রাত", "time", _N, "", "C058", "C058 has only an unverified INCLUDE sign"),
    "robibar": ("Sunday", "রবিবার", "time", _N, "", "", ""),
    "rupa": ("silver", "রূপা", "noun", _D, "W136", "", ""),
    "saat": ("seven", "সাত", "number", _D, "C046", "", "the existing ৭ is built from a still photo (BDSL49)"),
    "sahajjo": ("help", "সাহায্য", "verb", _N, "", "C023", ""),
    "sajano": ("decorate", "সাজানো", "verb", _D, "W263", "", ""),
    "sandal": ("sandal", "স্যান্ডেল", "noun", _D, "W137", "", ""),
    "school": ("school", "স্কুল", "noun", _N, "", "", ""),
    "se": ("he / she", "সে", "pronoun", _N, "", "C004", "C004 has only an unverified INCLUDE sign"),
    "shami": ("husband", "স্বামী", "noun", _A, "W16", "", "BdSLW401 W16 জামাই is labelled Husband; জামাই usually means son-in-law"),
    "shanti": ("peace", "শান্তি", "noun", _N, "", "", ""),
    "sharee": ("sari", "শাড়ি", "noun", _D, "W139", "", ""),
    "shikkhito": ("educated", "শিক্ষিত", "adjective", _N, "", "", "not শিক্ষক (teacher)"),
    "shikkhok": ("teacher", "শিক্ষক", "noun", _D, "C074", "", ""),
    "shirt": ("shirt", "শার্ট", "noun", _D, "W141", "", ""),
    "shona": ("hear / gold", "শোনা / সোনা", "verb", _A, "W149", "C019", "শোনা (hear, no sign) or সোনা (gold, BdSLW401)"),
    "shunno": ("zero", "শূন্য", "number", _D, "C039", "", "the existing ০ is built from a still photo (BDSL49)"),
    "soktishali": ("strong", "শক্তিশালী", "adjective", _N, "", "", ""),
    "sombar": ("Monday", "সোমবার", "time", _N, "", "", ""),
    "somoy": ("time", "সময়", "time", _D, "C071", "", ""),
    "somporko": ("relationship", "সম্পর্ক", "noun", _D, "W148", "", ""),
    "somudro": ("sea", "সমুদ্র", "noun", _N, "", "", ""),
    "sonibar": ("Saturday", "শনিবার", "time", _N, "", "", ""),
    "sotru": ("enemy", "শত্রু", "noun", _D, "W150", "", ""),
    "sotto": ("truth", "সত্য", "noun", _N, "", "", ""),
    "sroddha": ("respect", "শ্রদ্ধা", "noun", _N, "", "", ""),
    "stree": ("wife", "স্ত্রী", "noun", _D, "W13", "", ""),
    "subidha": ("convenience", "সুবিধা", "noun", _N, "", "", ""),
    "sukrobar": ("Friday", "শুক্রবার", "time", _N, "", "", ""),
    "surjo": ("sun", "সূর্য", "noun", _N, "", "", ""),
    "sustho": ("healthy", "সুস্থ", "adjective", _D, "W391", "", ""),
    "table": ("table", "টেবিল", "noun", _D, "W206", "", ""),
    "tader": ("their", "তাদের", "pronoun", _N, "", "", ""),
    "tara": ("they", "তারা", "pronoun", _N, "", "C006", "guess: তারা can also mean stars; C006 has only an unverified INCLUDE sign"),
    "taratari": ("quickly", "তাড়াতাড়ি", "other", _N, "", "", ""),
    "telephone": ("telephone", "টেলিফোন", "noun", _D, "W207", "", ""),
    "theke": ("from", "থেকে", "other", _N, "", "", ""),
    "tin": ("three", "তিন", "number", _D, "C042", "", "the existing ৩ is built from a still photo (BDSL49)"),
    "tokhon": ("then", "তখন", "time", _N, "", "", ""),
    "tv": ("television", "টিভি", "noun", _D, "W214", "", ""),
    "unmad": ("insane", "উন্মাদ", "adjective", _D, "W398", "", ""),
    "upodesh": ("advice", "উপদেশ", "noun", _N, "", "", ""),
    "uttor": ("answer", "উত্তর", "noun", _N, "", "", ""),
    "uvoy": ("both", "উভয়", "other", _N, "", "", ""),
    "valo": ("good", "ভালো", "adjective", _D, "W33", "", ""),
    "valobasha": ("love", "ভালোবাসা", "noun", _N, "", "", ""),
    "virus": ("virus", "ভাইরাস", "noun", _D, "W399", "", ""),
}
# English folders whose Bangla is an uncertain translation (romanised folders are always marked)
UNSURE_ENGLISH = {"Afternoon", "Attitude", "Aunty", "Blackberry", "Boy", "Cold", "Cover", "Disease", "Girl", "Gram",
                  "Health", "Knock", "Leafy vegetable", "Maintenance", "Pain", "Receive", "Rice", "Session", "Toilet",
                  "Wastebin", "Kill", "Eat", "Like", "Need", "Think"}

ENGLISH_CLIP = re.compile(r"^(?P<word>.+)_(?P<signer>[A-D])_(?P<take>\d+)\.(?P<ext>\w+)$")
SIGNBD_CLIP = re.compile(r"^p(?P<signer>\d+)_(?P<view>[cf])_(?P<word>.+)\.(?P<ext>\w+)$", re.IGNORECASE)
VIDEO_EXT = {"mp4", "mov"}


def is_romanised(folder):
    return folder[:1].islower()


def needs_check(folder):
    return is_romanised(folder) or folder in UNSURE_ENGLISH


def parse_clip(name):
    """'Eat/Eat_A_3.mp4' or 'kothay/p4_c_kothay.mp4' -> dict(folder, file, signer, view, take), or None.
    Signers: A-D for the English folders, p1-p16 for SignBD-Word. View: 'c' (close), 'f' (full body) or '' (one camera)."""
    folder, _, file = name.partition("/")
    if not file or "/" in file:
        return None
    m = SIGNBD_CLIP.match(file)
    if m and m.group("ext").lower() in VIDEO_EXT:
        return dict(folder=folder, file=file, signer=f"p{int(m.group('signer'))}", view=m.group("view").lower(), take=1)
    m = ENGLISH_CLIP.match(file)
    if m and m.group("ext").lower() in VIDEO_EXT:
        return dict(folder=folder, file=file, signer=m.group("signer"), view="", take=int(m.group("take")))
    return None


def clip_stem(name):
    """A file-system-safe, unique name for a clip's outputs: '<folder>__<file stem>', spaces as '-'."""
    folder, _, file = name.partition("/")
    return f"{folder}__{Path(file).stem}".replace(" ", "-")


def select_clips(entries, views):
    """Zip entries of one or more folders -> those to use. views 'both': every clip. 'close_first': for SignBD-Word
    clips the close view (c) only, and the full-body view (f) only where that signer has no close view; English folders
    have one camera, so all their clips are kept."""
    if views not in ("both", "close_first"):
        raise ValueError(f"bdsl300.views must be 'both' or 'close_first', not {views!r}")
    parsed = [(e, parse_clip(e["name"])) for e in entries]
    parsed = [(e, p) for e, p in parsed if p]
    if views == "both":
        return [e for e, _ in parsed]
    close = {(p["folder"], p["signer"]) for _, p in parsed if p["view"] == "c"}
    return [e for e, p in parsed if p["view"] != "f" or (p["folder"], p["signer"]) not in close]


# ------------------------------------------------------------------ existing words
def existing_words(word_list_401=None, core_coverage=None, core_word_list=None):
    """{key: (bangla, english, source)} for the words that already have a sign: BdSLW60 (labels.py), BdSLW401
    (labels401.py and its word_list.csv) and the core words with a sign (core_coverage.csv from scripts/20 --core; if
    missing, word_list.csv rows with a source other than INCLUDE)."""
    out = {}
    for n, bangla, _roman, english in LABELS:
        out[f"W{n}"] = (bangla, english, "BdSLW60")
    for n, bangla, _roman, english in LABELS_401:
        out.setdefault(f"W{n}", (bangla, english, "BdSLW401"))
    if word_list_401 and Path(word_list_401).exists():
        with open(word_list_401, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                out.setdefault(f"W{int(r['number'])}", (r["bangla"], r["english"], "BdSLW401"))
    if core_coverage and Path(core_coverage).exists():
        with open(core_coverage, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["has_sign"] == "True" and not r["source"].startswith(SOURCE):  # not our own verified words
                    out[r["id"]] = (r["bangla"], r["english"], r["source"])
    elif core_word_list and Path(core_word_list).exists():
        with open(core_word_list, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("source") and r["source"] not in ("INCLUDE", SOURCE) and r["id"].startswith("C"):
                    out[r["id"]] = (r["bangla"], r["english"], r["source"])
    return out


def _norm(text):
    return re.sub(r"[^a-z]", "", text.lower())


def spelling_clashes(existing):
    """Folders marked 'new' whose name or English matches an existing word's English or romanised name: these need a
    second look (the table is by meaning, so a clash is not proof of a duplicate). Returns [(folder, key)]."""
    names = {}
    for n, _b, roman, english in LABELS_401:
        for t in [english, *roman.split("/")]:
            names.setdefault(_norm(t), set()).add(f"W{n}")
    for key, (_b, english, _s) in existing.items():
        names.setdefault(_norm(english), set()).add(key)
    out = []
    for folder, (english, *_rest) in FOLDERS.items():
        if FOLDERS[folder][3] != NEW:
            continue
        for t in {_norm(folder), _norm(english)}:
            for key in sorted(names.get(t, ())):
                if key in existing:
                    out.append((folder, key))
    return out


def overlap_rows(folders, existing):
    """One row per dataset folder for overlap.csv. Raises if a folder is missing from FOLDERS or a match is unknown."""
    rows = []
    for i, folder in enumerate(sorted(folders), 1):
        if folder not in FOLDERS:
            raise KeyError(f"BdSL-300 folder {folder!r} is not in bdsl300.FOLDERS; add it before downloading")
        english, bangla, category, status, matches, core_id, note = FOLDERS[folder]
        keys = [k for k in matches.split("|") if k]
        unknown = [k for k in keys if k not in existing]
        if unknown:
            raise KeyError(f"{folder}: {unknown} are not existing words")
        if (status == NEW) == bool(keys):
            raise ValueError(f"{folder}: status {status} with matches {keys}")
        rows.append(dict(id=f"B{i:03d}", folder=folder, english=english, bangla_guess=bangla, needs_check=needs_check(folder),
                         category=category, status=status,
                         matches_existing_word="; ".join(f"{k} {existing[k][0]} ({existing[k][1]}, {existing[k][2]})"
                                                          for k in keys),
                         core_id=core_id, note=note))
    return rows


# ------------------------------------------------------------------ word list
WORD_LIST_COLUMNS = cv.WORD_LIST_COLUMNS


def is_true(value):
    return str(value).strip().lower() in ("true", "yes", "1", "y")


def read_word_list(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def add_to_word_list(path, rows):
    """Add BdSL-300 rows (new and ambiguous folders) to data/core_vocab/word_list.csv with verified = false. Rows already
    there are kept as they are, so a 'true' set by hand is never undone. Returns the number of rows added."""
    existing = read_word_list(path)
    have = {r["id"] for r in existing}
    added = [dict(id=r["id"], bangla=r["bangla_guess"], english=r["english"], category=r["category"], source=SOURCE,
                  verified="false", folder=r["folder"], core_id=r["core_id"])
             for r in rows if r["status"] in (NEW, AMBIGUOUS) and r["id"] not in have]
    if not existing:
        existing = [dict(id=w, bangla=b, english=e, category=c, source=s, verified=v)
                    for (w, b, e, c), (s, v) in ((r, cv.source_of(r[0])) for r in cv.CORE_WORDS)]
    write_rows(path, existing + added)
    return len(added)


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, WORD_LIST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in WORD_LIST_COLUMNS})


def verified_words(path):
    """BdSL-300 rows of the word list marked verified: [(id, folder, core_id, bangla, english, category)]."""
    return [(r["id"], r["folder"], r.get("core_id", ""), r["bangla"], r["english"], r["category"])
            for r in read_word_list(path) if r.get("source") == SOURCE and is_true(r.get("verified"))]
