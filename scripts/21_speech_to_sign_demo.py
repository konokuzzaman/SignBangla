"""MODULE 2 demo: speak a word and see its sign. Record -> recognize -> match -> smooth motion -> play in a window.

    python scripts/21_speech_to_sign_demo.py                        (microphone: Enter to start, Enter to stop, q to quit)
    python scripts/21_speech_to_sign_demo.py --text "বাবা"          (typed text: no microphone or speech model needed)
    python scripts/21_speech_to_sign_demo.py --text "মা চা" --speed 0.75 --save work/demo.gif --save work/demo.mp4
    python scripts/21_speech_to_sign_demo.py --text "মা চা" --style neon        (figure | neon | skeleton | studio3d)
    python scripts/21_speech_to_sign_demo.py --text "বাবা ভাই" --style studio3d   (3D mannequin in the browser)

In the window: + and - change the speed (0.5x, 0.75x, 1x), r plays it again, q or Esc closes it. Defaults come from
configs/config.yaml -> speech2sign.render. Needs work/sign_dictionary/ and, for the microphone, requirements-m2.txt.

With speech2sign.use_core_dictionary and work/sign_dictionary_core/ (scripts/20 --core), it also knows BdSLW401 and the
core vocabulary. Words without a sign stay in the caption as text, coloured by category (verbs purple, questions orange,
unknown words red). Questions and negation add head movement in the figure and neon styles (nonmanual.py).

The studio3d style plays a 3D mannequin in the browser instead of the OpenCV window. A local server on 127.0.0.1
(speech2sign.studio3d) serves the page; each new sentence appears there by itself. It cannot be saved as MP4/GIF yet.
"""
import argparse
import sys
import time

import _path  # noqa: F401
import cv2
import numpy as np

from signbangla.speech2sign.core_vocab import CATEGORY_COLORS, UNKNOWN_TEXT_COLOR
from signbangla.speech2sign.dictionary import SignDictionary
from signbangla.speech2sign.nonmanual import STYLES_WITH_MARKERS, apply_markers, sentence_markers
from signbangla.speech2sign.render import STYLES, CaptionDrawer, caption_drawer, draw_frame, render_options, save_animation
from signbangla.speech2sign.stitch import sentence_motion
from signbangla.speech2sign.studio3d import STYLE as STUDIO3D
from signbangla.speech2sign.studio3d import Studio3DServer, sequence_payload
from signbangla.speech2sign.text_processing import UNKNOWN, WordMatcher, dictionary_words, unknown_words
from signbangla.utils import ROOT, load_config, work_dir

WINDOW = "SignBangla"
HEARD_LABEL = "শোনা গেছে"
NO_SIGN_LABEL = "চিহ্ন নেই"  # "no sign"
LABEL_COLOR = (90, 90, 90)  # BGR
SPEEDS = (0.5, 0.75, 1.0)
LIVE_SUPERSAMPLE = 2  # the window draws a little faster than saved videos (3), with slightly rougher edges
STYLE_CHOICES = (*STYLES, STUDIO3D)
LOAD_TIMEOUT_S = 30  # --no-wait with studio3d: how long to wait for the browser to load the sentence


def word_color(m):
    """Caption colour of a word that has no sign: by category, red when the word is not known at all."""
    return UNKNOWN_TEXT_COLOR if m.word == UNKNOWN else CATEGORY_COLORS.get(m.category, CATEGORY_COLORS["extra"])


def plan_sentence(matches, dictionary):
    """Matches -> (dictionary keys to sign, caption layout (tokens, colours, token of each sign), matches without a
    sign). Dropped words (the copula আছ, আছে) are left out of both."""
    keys, tokens, colors, positions, no_sign = [], [], [], [], []
    for m in matches:
        if m.how == "dropped":
            continue
        key = m.key or m.word
        if m.word != UNKNOWN and key and dictionary.has(key):
            positions.append(len(tokens))
            keys.append(key)
            colors.append(None)
        else:
            colors.append(word_color(m))
            no_sign.append(m)
        tokens.append(m.source)
    return keys, (tokens, colors, positions), no_sign


def with_markers(sentence, n_signs, markers, style, s2s):
    """(keypoints, eyebrow raise per frame): head movement for questions and negation, figure and neon only."""
    if style not in STYLES_WITH_MARKERS or not (markers.question or markers.negation):
        return sentence.seq, None
    return apply_markers(sentence.seq, sentence.words, n_signs, markers, sentence.fps, s2s["nonmanual"])


class Screen:
    """What the window shows at one moment: recognized text on top, the animation, the sentence with the current word
    highlighted (words without a sign coloured by category), and the words without a sign."""

    def __init__(self, sentence, layout, n_signs, markers, text, no_sign, s2s):
        self.sentence = sentence
        self.captions = sentence.captions(layout)
        self.opts = render_options(s2s["render"])
        self.seq, self.brows = with_markers(sentence, n_signs, markers, self.opts["style"], s2s)
        font, size, demo = ROOT / s2s["font"], self.opts["size"], s2s["demo"]
        bars = (font, size, demo["bar_height"], demo["bar_font_size"])
        self.top = CaptionDrawer(*bars).bar(f"{HEARD_LABEL}: {text}")
        bottom = CaptionDrawer(*bars)
        self.bottom = bottom.bar(((f"{NO_SIGN_LABEL}:", *[m.source for m in no_sign]), -1,
                                  (LABEL_COLOR, *[word_color(m) for m in no_sign]))) if no_sign else bottom.bar("")
        self.caption = caption_drawer(font, self.opts["style"], size, self.opts["caption_height"], self.opts["font_size"])

    @property
    def duration(self):
        return (len(self.sentence.seq) - 1) / self.sentence.fps

    def image(self, t):
        i = min(int(round(t * self.sentence.fps)), len(self.sentence.seq) - 1)
        brow = float(self.brows[i]) if self.brows is not None else 0.0
        img = draw_frame(self.seq[i], self.opts["style"], self.opts["size"], self.opts["view"], LIVE_SUPERSAMPLE, brow)
        return np.vstack([self.top, img, self.caption.bar(self.captions[i]), self.bottom])


def play(screen, fps, window_height, speed, wait_for_key=True):
    """Play at steady frame times. Each frame shows the moment it is due, so the timing stays right even when drawing a
    frame is slow (figure and neon draw at about 15-20 frames per second here), and the speed can change while it plays."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def title():
        cv2.setWindowTitle(WINDOW, f"SignBangla - speed {speed:g}x  (+/- speed, r replay, q close)")

    title()
    t, finished, next_time = 0.0, False, time.perf_counter()
    try:
        while True:
            img = screen.image(t)
            scale = window_height / img.shape[0]
            cv2.imshow(WINDOW, cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
            next_time += 1 / fps
            wait_ms = max(1, int((next_time - time.perf_counter()) * 1000))
            key = cv2.waitKey(wait_ms if not finished else 50) & 0xFF
            if not _window_open() or key in (ord("q"), 27):
                return
            if key in (ord("+"), ord("=")):
                speed = SPEEDS[min(SPEEDS.index(speed) + 1, len(SPEEDS) - 1)]
                title()
            elif key in (ord("-"), ord("_")):
                speed = SPEEDS[max(SPEEDS.index(speed) - 1, 0)]
                title()
            elif key == ord("r"):
                t, finished = 0.0, False
                next_time = time.perf_counter()
                continue
            elif finished and key != 255:
                return
            if time.perf_counter() > next_time + 0.25:  # fell far behind (window dragged, PC busy): do not rush
                next_time = time.perf_counter()
            if not finished:
                t += speed / fps
                if t >= screen.duration:
                    t, finished = screen.duration, True
                    if not wait_for_key:
                        return
    finally:
        cv2.destroyAllWindows()


def _window_open():
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1


def show_text(text, matcher, dictionary, s2s, args, browser=None):
    """From text to the screen (or the browser for studio3d), and to any --save files."""
    matches = matcher.match(text)
    keys, layout, no_sign = plan_sentence(matches, dictionary)
    markers = sentence_markers(text, matches)
    print(f"Recognized text: {text}")
    for m in matches:
        sign = "sign" if m.how != "dropped" and m.word != UNKNOWN and dictionary.has(m.key or m.word) else "no sign"
        detail = f"{m.how}, {m.score:.2f}" if m.how == "fuzzy" else m.how
        print(f"  {m.source} -> {m.word or '-'} ({detail}; {m.category or ('left out' if m.how == 'dropped' else 'unknown')}; {m.key or '-'}; {sign})")
    if unknown_words(matches):
        print(f"Unknown words: {' '.join(unknown_words(matches))}")
    if markers.question or markers.negation:
        print(f"Non-manual markers: {'question ' if markers.question else ''}{'negation' if markers.negation else ''}")
    sentence = sentence_motion(keys, dictionary, variant=args.variant, **s2s["sentence"])
    render = s2s["render"]
    if render["style"] == STUDIO3D:
        show_text_3d(sentence, layout, dictionary.rest_pose, s2s, args, browser)
        return
    for path in args.save or []:
        played = sentence.retimed(render["speed"])
        seq, brows = with_markers(played, len(keys), markers, render["style"], s2s)
        n = save_animation(seq, path, fps=sentence.fps, gif_fps=s2s["gif"]["fps"], gif_width=s2s["gif"]["width"],
                           captions=played.captions(layout), font_path=ROOT / s2s["font"], brows=brows,
                           **render_options(render))
        print(f"Saved {path} ({n} frames, {sentence.fps:g} fps, {render['style']} style, speed {render['speed']:g}x)")
    if not args.no_window:
        screen = Screen(sentence, layout, len(keys), markers, text, no_sign, s2s)
        play(screen, render["fps"], s2s["demo"]["window_height"], render["speed"], wait_for_key=not args.no_wait)


class Browser3D:
    """The studio3d viewer: the local server starts with the first sentence and keeps running for the next ones."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.server = None

    def show(self, sentence, layout, rest, s2s):
        if self.server is None:
            self.server = Studio3DServer.from_config(self.cfg, ROOT)
            print(f"studio3d viewer: {self.server.url}")
        played = sentence.retimed(s2s["render"]["speed"])
        version = self.server.publish(sequence_payload(played, layout, rest, s2s["studio3d"]["lift"]))
        return version, (len(played.seq) - 1) / played.fps

    def close(self):
        if self.server is not None:
            self.server.close()
            self.server = None


def show_text_3d(sentence, layout, rest, s2s, args, browser):
    """studio3d: send the sentence to the browser page. Saving is not supported for this style yet."""
    if args.save:
        print(f"Saving MP4/GIF is not supported for the studio3d style yet; not saved: {', '.join(args.save)}. "
              "Use --style figure, neon or skeleton to save.")
    if args.no_window:
        return
    version, duration = browser.show(sentence, layout, rest, s2s)
    if args.no_wait:  # wait until the page has it and played it once
        if browser.server.wait_until_loaded(version, LOAD_TIMEOUT_S):
            time.sleep(duration + 0.5)
        else:
            print(f"The page did not load the sentence within {LOAD_TIMEOUT_S} s. Open {browser.server.url}")


def wait_for_enter(message):
    """input() for audio.record: typing q (then Enter) ends the demo."""
    if input(message).strip().lower() == "q":
        raise EOFError
    return ""


def microphone_loop(matcher, dictionary, s2s, args, browser=None):
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
            show_text(asr.transcribe(audio, matcher, rate), matcher, dictionary, s2s, args, browser)
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
    ap.add_argument("--speed", type=float, choices=SPEEDS, help="0.5, 0.75 or 1 (default: speech2sign.render.speed)")
    ap.add_argument("--style", choices=STYLE_CHOICES,
                    help="figure, neon, skeleton or studio3d (default: speech2sign.render.style)")
    args = ap.parse_args()
    for path in args.save or []:
        if not path.lower().endswith((".gif", ".mp4")):
            ap.error(f"--save {path}: use a .gif or .mp4 path")
    cfg = load_config(args.config)
    s2s = cfg["speech2sign"]
    if args.speed is not None:
        s2s["render"] = dict(s2s["render"], speed=args.speed)
    if args.style is not None:
        s2s["render"] = dict(s2s["render"], style=args.style)
    if s2s["render"]["speed"] not in SPEEDS:
        raise SystemExit(f"speech2sign.render.speed must be one of {SPEEDS}")
    if s2s["render"]["style"] not in STYLE_CHOICES:
        raise SystemExit(f"speech2sign.render.style must be one of {STYLE_CHOICES}")
    m = s2s["matching"]
    fuzzy = dict(cutoff=m["fuzzy_cutoff"], min_fuzzy_chars=m["min_fuzzy_chars"], margin=m["fuzzy_margin"])
    core = work_dir(cfg) / s2s["core_vocab"]["dictionary_dir"]
    if s2s["use_core_dictionary"] and (core / "dictionary.json").exists():
        dictionary = SignDictionary(core)
        matcher = WordMatcher.full(extra=dictionary_words(dictionary, "B"), **fuzzy)  # + verified BdSL-300 words
        print(f"Dictionary: {core.name} ({len(dictionary.words)} words"
              f"{', with unverified INCLUDE signs' if dictionary.info.get('use_unverified_signs') else ''})")
    else:
        matcher = WordMatcher(**fuzzy)
        try:
            dictionary = SignDictionary.from_config(cfg)
        except FileNotFoundError:
            raise SystemExit("work/sign_dictionary/ not found. Unzip signbangla_keypoints.zip (see README).")
    browser = Browser3D(cfg) if s2s["render"]["style"] == STUDIO3D else None
    try:
        if args.text is not None:
            show_text(args.text, matcher, dictionary, s2s, args, browser)
        else:
            microphone_loop(matcher, dictionary, s2s, args, browser)
        if browser is not None and browser.server is not None and not args.no_wait and (args.text is not None or args.once):
            try:
                input(f"Playing in the browser at {browser.server.url} (Replay button or r to play again). "
                      "Press Enter here to stop the viewer.")
            except (EOFError, KeyboardInterrupt):
                pass
    finally:
        if browser is not None:
            browser.close()


if __name__ == "__main__":
    main()
