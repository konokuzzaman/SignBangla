"""Live Webcam Bengali Sign Language Recognition using trained BiGRU model.

Recognizes Bengali signs in real-time (e.g. W1: Father / বাবা, W2: Relative / আত্মীয়)
using MediaPipe Holistic 67-keypoint extraction and our BiGRU neural network.

Usage:
    python scripts/11_live_webcam_demo.py
    python scripts/11_live_webcam_demo.py --camera 0
    python scripts/11_live_webcam_demo.py --checkpoint work/results/bigru/best_model.pt
"""
import argparse
import collections
from pathlib import Path
import time

import _path  # noqa: F401
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from signbangla.keypoints import GROUPS, N_KP, make_holistic, results_to_frame
from signbangla.labels import LABELS
from signbangla.sign2speech.predictor import load_model
from signbangla.sign2speech.stability import StabilityFilter
from signbangla.sign2speech.speech import SpeechWorker, VOICE, synthesize
import matplotlib.ft2font as ft2font
from functools import lru_cache
from signbangla.preprocess import PreprocessError, preprocess_trial
from signbangla.utils import load_config, work_dir
from signbangla.viz import CONNECTIONS




def draw_hud(
    frame,
    pred_label,
    pred_bangla,
    confidence,
    probs_dict,
    buffer_len,
    target_len,
    fps,
    speech_status="",
):
    h, w = frame.shape[:2]

    # Semi-transparent top HUD card
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (w - 15, 125), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (15, 15), (w - 15, 125), (70, 80, 100), 1)

    # Status & Buffer info
    buf_pct = min(1.0, buffer_len / target_len)
    cv2.putText(frame, f"SignBangla Live AI | FPS: {fps:.0f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 220), 1, cv2.LINE_AA)

    # Buffer bar
    bar_w = 120
    bar_x = w - 160
    cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_w, 42), (50, 55, 70), -1)
    cv2.rectangle(frame, (bar_x, 30), (bar_x + int(bar_w * buf_pct), 42), (13, 148, 136), -1)
    cv2.putText(frame, f"{buffer_len}/{target_len} frames", (bar_x - 110, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

    # Main Recognition Result
    if pred_label is not None and confidence is not None:
        color = (34, 197, 94) if confidence > 0.80 else (245, 158, 11)  # Green / Orange
        pred_text = pred_label
        cv2.putText(frame, pred_text, (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

        # Confidence bar & text
        conf_str = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, conf_str, (30, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 225, 235), 1, cv2.LINE_AA)

        # Draw probability meters for active classes on right
        y_off = 75
        for word, p in probs_dict.items():
            txt = f"{word}: {p:.1%}"
            cv2.putText(frame, txt, (w - 220, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 210, 225), 1, cv2.LINE_AA)
            y_off += 25
    else:
        cv2.putText(frame, "Waiting for sign gesture...", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 160, 180), 2, cv2.LINE_AA)
        cv2.putText(frame, "Perform sign in view of the camera", (30, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 125, 145), 1, cv2.LINE_AA)

    # Matplotlib 3.11 bundles the shaper required for Bengali conjuncts.
    if pred_bangla:
        mask = bangla_mask(pred_bangla)
        mh, mw = mask.shape
        mh, mw = min(mh, max(0, h - 132)), min(mw, max(0, w - 30))
        if mh and mw:
            alpha = mask[:mh, :mw, None] / 255.0
            region = frame[132:132 + mh, 30:30 + mw]
            region[:] = (region * (1 - alpha) + 255 * alpha).astype(np.uint8)
    cv2.putText(frame, speech_status, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 225, 235), 1, cv2.LINE_AA)

    # Bottom Instructions
    cv2.putText(frame, "Press 'q': Quit  |  'c': Clear buffer  |  'm': Mirror video", (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 145), 1, cv2.LINE_AA)


@lru_cache(maxsize=128)
def bangla_mask(text):
    if not hasattr(ft2font, "__libraqm_version__"):
        raise RuntimeError("Bengali display needs matplotlib >= 3.11")
    font = ft2font.FT2Font(str(Path(__file__).resolve().parents[1] / "assets/fonts/NotoSansBengali.ttf"))
    font.set_size(32, 72)
    font.set_text(text)
    font.draw_glyphs_to_bitmap(antialiased=True)
    return np.array(font.get_image(), dtype=np.uint8)


def run_live_recognition(cfg, checkpoint_path, camera=0, min_buffer=25, window_size=60, speech=True, threshold=0.8, confirmations=3):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading BiGRU model from {checkpoint_path} on {device}...")
    model = load_model(checkpoint_path, device)
    gate = StabilityFilter(threshold, confirmations)
    speaker = SpeechWorker(work_dir(cfg) / "speech_cache") if speech else None
    hol = None
    cap = None
    try:
        hol = make_holistic(cfg)
        cap = cv2.VideoCapture(camera)
        _camera_loop(cfg, model, device, hol, cap, camera, min_buffer, window_size, gate, speaker)
    finally:
        if cap is not None:
            cap.release()
        if hol is not None:
            hol.close()
        cv2.destroyAllWindows()
        if speaker is not None:
            speaker.close()


def _camera_loop(cfg, model, device, hol, cap, camera, min_buffer, window_size, gate, speaker):

    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {camera}. Try --camera 1")

    print("\n" + "=" * 60)
    print("      SignBangla Real-Time Gesture Recognition Running      ")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  'c' - Clear sliding frame buffer")
    print("  'm' - Toggle mirroring (flip horizontal)")
    print("=" * 60 + "\n")

    # Sliding keypoints buffer (stores raw frames: shape (67, 3) each)
    raw_buffer = collections.deque(maxlen=window_size)
    pred_label = None
    pred_bangla = None
    confidence = None
    probs_dict = {}

    # Preserve the training orientation by default.
    mirror = False
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    colors = {
        "black": (0, 255, 128),     # Pose (Neon green)
        "tab:blue": (255, 160, 50), # Left hand (Cyan/Blue)
        "tab:red": (50, 80, 255),   # Right hand (Orange/Red)
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(cur_time - prev_time, 1e-5))
        prev_time = cur_time
        frame_count += 1

        # MediaPipe Holistic Extraction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hol.process(rgb)
        kp_frame = results_to_frame(res)
        raw_buffer.append(kp_frame)

        # Draw Skeleton
        good = ~np.isnan(kp_frame).any(axis=-1)
        for a, b, c in CONNECTIONS:
            if good[a] and good[b]:
                pt1 = (int(kp_frame[a, 0] * w), int(kp_frame[a, 1] * h))
                pt2 = (int(kp_frame[b, 0] * w), int(kp_frame[b, 1] * h))
                cv2.line(frame, pt1, pt2, colors[c], 2, cv2.LINE_AA)

        # Keypoint circles for hands
        for idx in range(25, N_KP):
            if good[idx]:
                pt = (int(kp_frame[idx, 0] * w), int(kp_frame[idx, 1] * h))
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Run Neural Model Inference periodically (every 4 frames once buffer is warm)
        has_hands = (good[25:46].sum() > 4) or (good[46:].sum() > 4)
        if gate.observe_hands(has_hands, time.monotonic()):
            raw_buffer.clear()
            pred_label = pred_bangla = confidence = None
            probs_dict = {}
        if len(raw_buffer) >= min_buffer and (frame_count % 4 == 0) and has_hands:
            raw_arr = np.array(list(raw_buffer))  # (T, 67, 3)
            try:
                processed, _ = preprocess_trial(raw_arr, w, h, cfg["preprocess"])  # (64, 67, 3)
                feat = processed.reshape(1, 64, -1)  # (1, 64, 201)
                feat_tensor = torch.from_numpy(feat).float().to(device)

                with torch.no_grad():
                    logits = model(feat_tensor)
                    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

                top_class = int(np.argmax(probs))
                conf = float(probs[top_class])

                # Get class labels
                # LABELS: (word number, Bangla, romanized, English)
                w_num, b_text, r_text, en_text = LABELS[top_class]
                pred_label = en_text
                pred_bangla = b_text
                confidence = conf
                if speaker is None or not speaker.busy:
                    accepted = gate.update(top_class, conf)
                    if accepted is not None and speaker is not None:
                        speaker.submit(b_text)
                        print(f"Confirmed: {en_text} / {b_text} ({conf:.1%})", flush=True)

                # Top 3 candidate predictions with probabilities
                top3_indices = np.argsort(probs)[::-1][:3]
                probs_dict = {
                    LABELS[i][3]: float(probs[i])
                    for i in top3_indices
                    if probs[i] > 0.03
                }

            except PreprocessError:
                gate.update(None, 0.0)

        # If hands were missing for consecutive frames, gradually decay prediction
        if not has_hands and frame_count % 30 == 0:
            pred_label = None
            pred_bangla = None
            confidence = None
            probs_dict = {}

        # Draw Modern HUD overlay
        draw_hud(
            frame,
            pred_label,
            pred_bangla,
            confidence,
            probs_dict,
            len(raw_buffer),
            window_size,
            fps,
            (speaker.status if speaker else "Speech disabled") + (" | Move hands out of view or press c for next sign" if gate.latched else " | Waiting for a stable sign"),
        )

        cv2.imshow("SignBangla - Live Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            raw_buffer.clear()
            gate.reset()
            pred_label = None
            pred_bangla = None
            confidence = None
            probs_dict = {}
        elif key == ord("m"):
            mirror = not mirror
            raw_buffer.clear()
            gate.reset()
            pred_label = pred_bangla = confidence = None
            probs_dict = {}



def main():
    ap = argparse.ArgumentParser(description="Live Bengali Sign Language Recognition")
    ap.add_argument("--config", help="Path to config file")
    ap.add_argument("--checkpoint", default="work/results/bigru/best_model.pt", help="Path to trained model checkpoint")
    ap.add_argument("--camera", type=int, default=0, help="Webcam device index")
    ap.add_argument("--min-buffer", type=int, default=25, help="Minimum frames before predicting")
    ap.add_argument("--window", type=int, default=50, help="Sliding window size (frames)")
    ap.add_argument("--no-speech", action="store_true", help="Run recognition without speech")
    ap.add_argument("--confidence", type=float, default=0.8, help="Minimum confidence to speak")
    ap.add_argument("--confirmations", type=int, default=3, help="Consecutive predictions required")
    ap.add_argument("--test-speech", action="store_true", help="Speak Baba (Father) without opening camera")
    ap.add_argument("--cache-speech", action="store_true", help="Generate audio for all 60 words without opening camera")
    args = ap.parse_args()
    if not 1 <= args.min_buffer <= args.window:
        ap.error("Require 1 <= --min-buffer <= --window")
    if not 0 <= args.confidence <= 1 or args.confirmations < 1:
        ap.error("Confidence must be 0..1 and confirmations must be positive")

    cfg = load_config(args.config)
    if args.cache_speech:
        for _, bangla, _, english in LABELS:
            synthesize(bangla, work_dir(cfg) / "speech_cache")
            print(f"Cached: {english} / {bangla}", flush=True)
        return
    if args.test_speech:
        worker = SpeechWorker(work_dir(cfg) / "speech_cache")
        try:
            worker.submit("বাবা")
            worker.future.result()
            if worker.status != "Ready":
                raise SystemExit(worker.status)
        finally:
            worker.close()
        return
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Please run scripts/10_train_baseline.py first.")

    run_live_recognition(cfg, ckpt, camera=args.camera, min_buffer=args.min_buffer, window_size=args.window, speech=not args.no_speech, threshold=args.confidence, confirmations=args.confirmations)


if __name__ == "__main__":
    main()
