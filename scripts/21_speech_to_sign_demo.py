"""MODULE 2 demo: speak a word and see its sign. Record -> recognize -> match -> stitch -> play in an OpenCV window.

    python scripts/21_speech_to_sign_demo.py                        (microphone: Enter to start, Enter to stop, q to quit)
    python scripts/21_speech_to_sign_demo.py --text "বাবা"          (typed text: no microphone or speech model needed)
    python scripts/21_speech_to_sign_demo.py --text "মা চা" --save work/demo.gif --save work/demo.mp4

In the window: r plays it again, any other key closes it.
Needs work/sign_dictionary/ and, for the microphone, requirements-m2.txt.
"""
import argparse
import sys
import time

import _path  # noqa: F401
import cv2
import numpy as np

from signbangla.speech2sign.dictionary import SignDictionary
from signbangla.speech2sign.render import CaptionDrawer, render_frames, save_animation
from signbangla.speech2sign.stitch import frame_captions, stitch_words
from signbangla.speech2sign.text_processing import WordMatcher, known_words, unknown_words
from signbangla.utils import ROOT, load_config

WINDOW = "SignBangla - r: play again, any other key: close"
HEARD_LABEL = "শোনা গেছে"
UNKNOWN_LABEL = "অজানা"
UNKNOWN_COLOR = (40, 40, 200)  # BGR
IDLE_SECONDS = 2  # how long to show the rest pose when no word has a sign


def build_animation(text, matcher, dictionary, stitch_params, variant):
    """Text -> (sequence, one caption per frame, unknown words). With no known word: the rest pose and no caption."""
    matches = matcher.match(text)
    words, unknown = known_words(matches), unknown_words(matches)
    if not words:
        seq = np.repeat(dictionary.rest_pose[None], IDLE_SECONDS * dictionary.fps, axis=0)
        return seq, [""] * len(seq), unknown
    stitched = stitch_words(words, dictionary, variant=variant, **stitch_params)
    return stitched.seq, frame_captions(stitched, " ".join(words)), unknown


def screen_frames(seq, captions, text, unknown, s2s):
    """Yield what the window shows: recognized text on top, skeleton and current word, unknown words at the bottom."""
    font, render, demo = ROOT / s2s["font"], s2s["render"], s2s["demo"]
    size = (font, render["size"], demo["bar_height"], demo["bar_font_size"])
    top = CaptionDrawer(*size).bar(f"{HEARD_LABEL}: {text}")
    bottom = CaptionDrawer(*size, color=UNKNOWN_COLOR).bar(f"{UNKNOWN_LABEL}: {' '.join(unknown)}" if unknown else "")
    for img in render_frames(seq, captions=captions, font_path=font, **render):
        yield np.vstack([top, img, bottom])


def play(make_frames, fps, window_height, wait_for_key=True):
    """Show the frames at their real speed. Holds the last frame until a key is pressed or the window is closed."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            start, last = time.perf_counter(), None
            for i, img in enumerate(make_frames()):
                last = img
                scale = window_height / img.shape[0]
                cv2.imshow(WINDOW, cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                wait_ms = max(1, int((start + (i + 1) / fps - time.perf_counter()) * 1000))
                if cv2.waitKey(wait_ms) & 0xFF in (ord("q"), 27) or not _window_open():
                    return
            if not wait_for_key or last is None:
                return
            print("In the window: r plays it again, any other key closes it.")
            key = -1
            while key == -1 and _window_open():
                key = cv2.waitKey(100)
            if key & 0xFF != ord("r"):
                return
    finally:
        cv2.destroyAllWindows()


def _window_open():
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1


def show_text(text, matcher, dictionary, s2s, args):
    """From text to the screen, and to any --save files."""
    params = s2s["stitch"]
    seq, captions, unknown = build_animation(text, matcher, dictionary, params, args.variant)
    matches = matcher.match(text)
    print(f"Recognized text: {text}")
    for m in matches:
        print(f"  {m.source} -> {m.word} ({m.how}" + (f", {m.score:.2f})" if m.how == "fuzzy" else ")"))
    if unknown:
        print(f"Unknown words: {' '.join(unknown)}")
    for path in args.save or []:
        n = save_animation(seq, path, fps=dictionary.fps, gif_fps=s2s["gif"]["fps"], gif_width=s2s["gif"]["width"],
                           captions=captions, font_path=ROOT / s2s["font"], **s2s["render"])
        print(f"Saved {path} ({n} frames)")
    if not args.no_window:
        play(lambda: screen_frames(seq, captions, text, unknown, s2s), dictionary.fps, s2s["demo"]["window_height"],
             wait_for_key=not args.no_wait)


def wait_for_enter(message):
    """input() for audio.record: typing q (then Enter) ends the demo."""
    if input(message).strip().lower() == "q":
        raise EOFError
    return ""


def microphone_loop(matcher, dictionary, s2s, args):
    from signbangla.speech2sign.asr import WhisperASR, vocabulary_words
    from signbangla.speech2sign.audio import has_speech, loudest_rms, record

    acfg, audio_cfg = s2s["asr"], s2s["audio"]
    print(f"Loading speech model {acfg['model']} (the first run downloads it)...")
    asr = WhisperASR(acfg["model"], acfg["language"], acfg["device"], acfg["max_new_tokens"], vocabulary_words(),
                     acfg["generation_config_from"])
    print(f"Ready on {asr.device}. Type q then Enter to quit.")
    rate = audio_cfg["sample_rate"]
    while True:
        try:
            audio = record(rate, audio_cfg["device"], wait=wait_for_enter)
        except (EOFError, KeyboardInterrupt):
            return
        if not has_speech(audio, rate, audio_cfg["min_seconds"], audio_cfg["min_rms"]):
            print(f"Heard nothing: {len(audio) / rate:.1f} s, volume {loudest_rms(audio, rate):.4f} (needs at least "
                  f"{audio_cfg['min_seconds']} s and {audio_cfg['min_rms']}). Speak closer to the microphone, or lower "
                  "speech2sign.audio.min_rms in the config.")
        else:
            show_text(asr.transcribe(audio, matcher, rate), matcher, dictionary, s2s, args)
        if args.once:
            return


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console default cannot print Bangla
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--text", help="use this text instead of the microphone")
    ap.add_argument("--save", action="append", help="also save the animation: a .gif or .mp4 path (can be repeated)")
    ap.add_argument("--variant", type=int, default=0, help="which of a word's trials to use (0, 1, 2)")
    ap.add_argument("--once", action="store_true", help="microphone: stop after one recording")
    ap.add_argument("--no-wait", action="store_true", help="close the window as soon as the animation ends")
    ap.add_argument("--no-window", action="store_true", help="do not open a window (use with --save)")
    args = ap.parse_args()
    for path in args.save or []:
        if not path.lower().endswith((".gif", ".mp4")):
            ap.error(f"--save {path}: use a .gif or .mp4 path")
    cfg = load_config(args.config)
    s2s = cfg["speech2sign"]
    m = s2s["matching"]
    matcher = WordMatcher(cutoff=m["fuzzy_cutoff"], min_fuzzy_chars=m["min_fuzzy_chars"], margin=m["fuzzy_margin"])
    try:
        dictionary = SignDictionary.from_config(cfg)
    except FileNotFoundError:
        raise SystemExit("work/sign_dictionary/ not found. Unzip signbangla_keypoints.zip (see README).")
    if args.text is not None:
        show_text(args.text, matcher, dictionary, s2s, args)
    else:
        microphone_loop(matcher, dictionary, s2s, args)


if __name__ == "__main__":
    main()
