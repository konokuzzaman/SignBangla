"""MODULE 3 demo: sign words (dictionary forms, BdSL order) -> a Bangla sentence, with the steps the rules took.

    python scripts/51_sentence_demo.py --words "আমি পানি চাওয়া"
    python scripts/51_sentence_demo.py --words "তুমি কোথায় যাওয়া"
    python scripts/51_sentence_demo.py --words "তুমি পানি চাওয়া" --question      (yes/no question)
    python scripts/51_sentence_demo.py --words "আমি পানি চাওয়া" --speak         (also say it aloud)

--speak uses the project's Bangla speech output, signbangla/sign2speech/speech.py (edge-tts, needs the internet and
`pip install -r requirements-speech.txt`). That module comes with the texttovoice branch; without it the sentence is
only printed.
"""
import argparse
import os
import shutil
import subprocess
import sys

import _path  # noqa: F401

from signbangla.sign2text.builder import SentenceBuilder
from signbangla.utils import load_config, subdir


def speak(text, cache_dir):
    try:
        from signbangla.sign2speech.speech import synthesize
    except ImportError:
        print("--speak: signbangla/sign2speech/speech.py is not on this branch (it is on texttovoice); not spoken.")
        return
    try:
        audio = synthesize(text, cache_dir)
    except Exception as e:  # noqa: BLE001 - no internet, edge-tts missing, ...
        print(f"--speak: could not make the audio ({type(e).__name__}: {e}); not spoken.")
        return
    player = shutil.which("ffplay")
    if player:
        subprocess.run([player, "-nodisp", "-autoexit", "-loglevel", "error", str(audio)], check=False)
    elif sys.platform == "win32":
        os.startfile(audio)  # the default audio player
    else:
        print(f"--speak: no audio player found; the audio is in {audio}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--words", required=True, help="sign words in dictionary form, separated by spaces")
    ap.add_argument("--question", action="store_true", help="a yes/no question (BdSL marks it with a head tilt)")
    ap.add_argument("--speak", action="store_true", help="also say the sentence aloud")
    args = ap.parse_args()
    cfg = load_config(args.config)
    builder = SentenceBuilder.from_config(cfg)
    sentence, trace, unknown = builder.build(args.words, is_question=args.question)
    print(f"Signs:    {args.words}")
    print(f"Sentence: {sentence}")
    print("Steps:")
    for step in trace:
        print(f"  {step}")
    if unknown:
        print(f"Unknown words (kept as they are): {' '.join(unknown)}")
    if args.speak and sentence:
        speak(sentence, subdir(cfg, cfg["module3"]["speech_cache"]))


if __name__ == "__main__":
    main()
