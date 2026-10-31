"""WEEK 1 (Member 3): test MediaPipe Holistic on sample videos and on the webcam; confirm the 67-keypoint output.

    python scripts/02_test_mediapipe.py                  # 5 sample dataset videos (needs step 1 first)
    python scripts/02_test_mediapipe.py --video my.mp4   # one specific video
    python scripts/02_test_mediapipe.py --webcam         # live skeleton overlay, press q to quit
"""
import argparse

import _path  # noqa: F401
import numpy as np
import pandas as pd

from signbangla.keypoints import GROUPS, N_KP, extract_video, make_holistic, results_to_frame
from signbangla.utils import load_config, subdir, work_dir
from signbangla.viz import CONNECTIONS, plot_trial_grid


def test_videos(cfg, paths, n_frames):
    out = subdir(cfg, "mediapipe_test")
    hol = make_holistic(cfg)
    all_ok = True
    for p in paths:
        arr, meta = extract_video(p, hol, max_frames=n_frames, rotate_180=p.stem in (cfg["video"].get("rotate_180") or []))
        shape_ok = arr.ndim == 3 and arr.shape[1:] == (N_KP, 3)
        det = {k: float((~np.isnan(arr[:, s.start, 0])).mean()) if len(arr) else 0.0 for k, s in GROUPS.items()}
        print(f"{p.name}: shape {arr.shape} -> {'OK 67 keypoints' if shape_ok else 'WRONG SHAPE'} | "
              f"detected pose {det['pose']:.0%}, left hand {det['lh']:.0%}, right hand {det['rh']:.0%} | {meta}")
        all_ok &= shape_ok
        if len(arr):
            plot_trial_grid(arr, out / f"{p.stem}.png", title=p.name, aspect=meta["width"] / max(meta["height"], 1))
    print(f"\nSnapshots saved in {out}")
    print("RESULT:", "all videos give (T, 67, 3)" if all_ok else "problem found")


def run_webcam(cfg, camera):
    import cv2
    hol = make_holistic(cfg)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {camera}. Try --camera 1")
    print("Webcam running - press q to quit. (Note: the image is NOT mirrored here; training videos are not mirrored either.)")
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        res = hol.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        f = results_to_frame(res)
        good = ~np.isnan(f).any(-1)
        colors = {"black": (0, 255, 0), "tab:blue": (255, 120, 0), "tab:red": (0, 0, 255)}
        for a, b, c in CONNECTIONS:
            if good[a] and good[b]:
                cv2.line(bgr, (int(f[a, 0] * w), int(f[a, 1] * h)), (int(f[b, 0] * w), int(f[b, 1] * h)), colors[c], 2)
        txt = f"keypoints found: {int(good.sum())}/{N_KP}  (pose {int(good[:25].sum())}, L {int(good[25:46].sum())}, R {int(good[46:].sum())})"
        cv2.putText(bgr, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("SignBangla - MediaPipe test", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", nargs="*")
    ap.add_argument("--n-videos", type=int, default=5)
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.webcam:
        return run_webcam(cfg, args.camera)
    from pathlib import Path
    if args.video:
        paths = [Path(v) for v in args.video]
    else:
        man = work_dir(cfg) / "manifest.csv"
        if not man.exists():
            raise SystemExit("Run scripts/01_inspect_dataset.py first (or pass --video file.mp4).")
        m = pd.read_csv(man)
        m = m[m["parsed"]]
        pick = np.linspace(0, len(m) - 1, args.n_videos).astype(int)
        paths = [Path(p) for p in m.iloc[pick]["path"]]
    test_videos(cfg, paths, args.frames)


if __name__ == "__main__":
    main()
