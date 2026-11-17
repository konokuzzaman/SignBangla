"""MODULE 2 core vocabulary, step 3: the BdSL digits 0-9 as short signs, from the BDSL49 photos.

    python scripts/42_digits_from_images.py
    python scripts/42_digits_from_images.py --folder D:/photos/digits     (any folder with one subfolder per digit:
                                                                           ০..৯ or 0..9)

MediaPipe Hands runs in static-image mode on every photo. Per digit, the most typical detection is kept (the medoid of all
handshapes, which ignores odd photos), held at the median angle of all photos. Each sign is: rest pose -> handshape ->
hold 0.5 s -> rest, at the dictionary's 30 fps, in the 67-keypoint format (the rest pose from work/sign_dictionary).
    work/core_vocab/BDSL49/hands.npz         every detected hand per digit (pixels), with file names and scores
    work/core_vocab/BDSL49/digit_<d>.npy     the sign; digit_<d>.json says which photo was used
    work/core_vocab/BDSL49/best_hands.png    the chosen photo of each digit with its landmarks, to check by eye
Next: python scripts/20_build_sign_dictionary.py --core
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import cv2
import numpy as np

from signbangla.keypoints import GROUPS
from signbangla.speech2sign import core_vocab as cv
from signbangla.speech2sign.core_signs import (best_photo, elbow_from_signs, handshape_pose, hold_sign, photo_hand,
                                               place_hand)
from signbangla.speech2sign.dictionary import SignDictionary
from signbangla.utils import ROOT, atomic_save_npy, atomic_write_json, load_config, progress, subdir

HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def read_image(path):
    """cv2.imread cannot open paths with Bangla letters on Windows (the folders are named ০-৯), so decode the bytes."""
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def digit_of(name):
    name = cv.to_bangla_digits(name.strip())
    return name if len(name) == 1 and name in cv.BANGLA_DIGITS else None


def detect(folder, min_score):
    import mediapipe as mp
    out = {}
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=1,
                                  min_detection_confidence=0.5) as hands:
        for sub in sorted(p for p in folder.iterdir() if p.is_dir() and digit_of(p.name)):
            d = digit_of(sub.name)
            found = []
            for path in progress(sorted(sub.glob("*.jp*g")) + sorted(sub.glob("*.png")), desc=f"digit {d}"):
                img = read_image(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not res.multi_hand_landmarks:
                    continue
                cls = res.multi_handedness[0].classification[0]
                if cls.score < min_score:
                    continue
                found.append((path.name, photo_hand(res.multi_hand_landmarks[0], w, h, cls.label), cls.label, cls.score))
            out[d] = found
            print(f"  {d}: {len(found)} hands in {len(list(sub.iterdir()))} photos")
    return out


def hand_length(dictionary):
    """Typical wrist-to-middle-knuckle length (shoulder widths) of the right hand in the BdSLW60 dictionary."""
    lengths = []
    for word in list(dictionary.words)[:30]:
        seq = dictionary.sequence(word)
        rh = seq[:, GROUPS["rh"]]
        have = np.abs(rh).sum((1, 2)) > 0
        lengths += list(np.linalg.norm(rh[have, 9, :2] - rh[have, 0, :2], axis=1))
    return float(np.median(lengths))


def contact_sheet(chosen, folder, path):
    tiles = []
    for d, (name, hand) in sorted(chosen.items()):
        img = read_image(folder / d / name) if (folder / d / name).exists() else None
        if img is None:
            continue
        for a, b in HAND_LINES:
            cv2.line(img, tuple(int(v) for v in hand[a, :2]), tuple(int(v) for v in hand[b, :2]), (0, 200, 255), 8)
        s = 240 / max(img.shape[:2])
        t = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        t = cv2.copyMakeBorder(t, 0, 240 - t.shape[0], 0, 240 - t.shape[1], cv2.BORDER_CONSTANT)
        cv2.putText(t, cv.DIGIT_IDS[d] + " " + str(cv.BANGLA_DIGITS.index(d)), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        tiles.append(t)
    if tiles:
        cv2.imwrite(str(path), np.hstack(tiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--folder", help="photo folder with one subfolder per digit (default: data/BDSL49)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["speech2sign"]["core_vocab"]
    folder = Path(args.folder) if args.folder else ROOT / c["bdsl49_dir"]
    out = subdir(cfg, f"{c['work_subdir']}/BDSL49")
    if not folder.exists():
        raise SystemExit(f"{folder} not found. Run scripts/40_core_vocab_download.py --only bdsl49 first.")
    found = detect(folder, c["digit_min_score"])
    np.savez(out / "hands.npz", **{d: np.stack([f[1] for f in v]) for d, v in found.items() if v},
             **{f"{d}_files": np.array([f[0] for f in v]) for d, v in found.items() if v})

    base = SignDictionary.from_config(cfg)  # the BdSLW60 dictionary: rest pose, fps and hand size
    rest, fps = base.rest_pose, base.fps
    length = hand_length(base)
    elbow = elbow_from_signs([base.sequence(w, v) for w in base.words for v in range(len(base.words[w]["trials"]))],
                             c["digit_wrist"])
    print(f"Hand length {length:.3f} shoulder widths; elbow from BdSLW60 signs: "
          f"{None if elbow is None else elbow.round(2).tolist()}")
    hold = int(round(c["digit_hold_s"] * fps))
    chosen = {}
    for d, items in sorted(found.items()):
        if not items:
            print(f"  {d}: no usable photo, no sign")
            continue
        i, angle = best_photo([h for _, h, _, _ in items])
        name, hand, label, score = items[i]
        chosen[d] = (name, hand)
        pose = handshape_pose(rest, place_hand(hand, c["digit_wrist"], length, angle), cfg, elbow)
        seq = hold_sign([pose], rest, c["digit_move_frames"], hold, c["digit_change_frames"])
        atomic_save_npy(out / f"digit_{d}.npy", seq)
        atomic_write_json(out / f"digit_{d}.json", dict(
            digit=d, id=cv.DIGIT_IDS[d], photo=name, handedness=label, score=round(float(score), 3),
            angle_deg=round(float(np.degrees(angle)), 1), photos_detected=len(items), frames=len(seq), fps=fps,
            hand_length=round(length, 3), pose=pose.tolist()))
    contact_sheet(chosen, folder, out / "best_hands.png")
    print(f"Signs for {len(chosen)} / 10 digits in {out}. Check best_hands.png.")
    print("Next: python scripts/20_build_sign_dictionary.py --core")


if __name__ == "__main__":
    main()
