"""MODULE 2 demo: words -> stitched sign sequence -> skeleton video with a Bangla caption.

    python scripts/20_m2_demo.py
    python scripts/20_m2_demo.py --text "বাবা আপেল" --n-transition 20 --easing minimum_jerk --variant 1 --out work/test.mp4

Needs work/sign_dictionary/ (python scripts/20_build_sign_dictionary.py).
"""
import argparse

import _path  # noqa: F401

from signbangla.speech2sign.dictionary import SignDictionary
from signbangla.speech2sign.render import render_video
from signbangla.speech2sign.stitch import EASINGS, frame_captions, stitch_words
from signbangla.utils import ROOT, load_config, work_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--text", default="মা চা আপেল")
    ap.add_argument("--out", help="default: m2_demo.mp4 in the work folder")
    ap.add_argument("--n-transition", type=int, help="frames between signs (default: speech2sign.stitch in the config)")
    ap.add_argument("--easing", choices=sorted(EASINGS), help="default: speech2sign.stitch in the config")
    ap.add_argument("--variant", type=int, default=0, help="which of a word's trials to use (0, 1, 2)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s2s = cfg["speech2sign"]

    params = dict(s2s["stitch"])
    if args.n_transition is not None:
        params["n_transition"] = args.n_transition
    if args.easing:
        params["easing"] = args.easing

    d = SignDictionary.from_config(cfg)
    try:
        words = d.tokenize(args.text)
    except KeyError as e:  # the Windows console usually cannot print Bangla
        raise SystemExit(e.args[0].encode("ascii", "backslashreplace").decode())
    st = stitch_words(words, d, variant=args.variant, **params)
    out = args.out or work_dir(cfg) / "m2_demo.mp4"
    n = render_video(st.seq, out, fps=d.fps, captions=frame_captions(st, args.text), font_path=ROOT / s2s["font"], **s2s["render"])

    for seg in st.segments:
        name = d.words[d.find(seg["label"])]["romanized"][0] if seg["label"] else ""
        print(f"  {seg['start']:4d}-{seg['end']:4d}  {seg['kind']:10s} {name}")
    print(f"\n{n} frames, {n / d.fps:.1f} s at {d.fps} fps (n_transition={params['n_transition']}, easing={params['easing']})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
