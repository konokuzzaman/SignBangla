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
from signbangla.models import BiGRUClassifier, BiLSTMClassifier
from signbangla.preprocess import PreprocessError, preprocess_trial
from signbangla.utils import load_config, work_dir
from signbangla.viz import CONNECTIONS


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    model_type = cfg.get("model_type", "bigru")
    model_cls = BiLSTMClassifier if model_type == "bilstm" else BiGRUClassifier
    model = model_cls(
        input_dim=cfg.get("input_dim", 201),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        num_classes=cfg.get("num_classes", 60),
        dropout=cfg.get("dropout", 0.3),
        use_attention=cfg.get("use_attention", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def draw_hud(
    frame,
    pred_label,
    pred_bangla,
    confidence,
    probs_dict,
    buffer_len,
    target_len,
    fps,
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
        pred_text = f"{pred_label} ({pred_bangla})"
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

    # Bottom Instructions
    cv2.putText(frame, "Press 'q': Quit  |  'c': Clear buffer  |  'm': Mirror video", (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 145), 1, cv2.LINE_AA)


def run_live_recognition(cfg, checkpoint_path, camera=0, min_buffer=25, window_size=60):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading BiGRU model from {checkpoint_path} on {device}...")
    model = load_model(checkpoint_path, device)
    hol = make_holistic(cfg)

    cap = cv2.VideoCapture(camera)
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

    mirror = True
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

                # Top 3 candidate predictions with probabilities
                top3_indices = np.argsort(probs)[::-1][:3]
                probs_dict = {
                    LABELS[i][3]: float(probs[i])
                    for i in top3_indices
                    if probs[i] > 0.03
                }

            except PreprocessError:
                pass

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
        )

        cv2.imshow("SignBangla - Live Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            raw_buffer.clear()
            pred_label = None
            pred_bangla = None
            confidence = None
            probs_dict = {}
        elif key == ord("m"):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Live Bengali Sign Language Recognition")
    ap.add_argument("--config", help="Path to config file")
    ap.add_argument("--checkpoint", default="work/results/bigru/best_model.pt", help="Path to trained model checkpoint")
    ap.add_argument("--camera", type=int, default=0, help="Webcam device index")
    ap.add_argument("--min-buffer", type=int, default=25, help="Minimum frames before predicting")
    ap.add_argument("--window", type=int, default=50, help="Sliding window size (frames)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Please run scripts/10_train_baseline.py first.")

    run_live_recognition(cfg, ckpt, camera=args.camera, min_buffer=args.min_buffer, window_size=args.window)


if __name__ == "__main__":
    main()
