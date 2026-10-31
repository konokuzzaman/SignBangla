"""MediaPipe Holistic -> 67 keypoints per frame.

Layout of one frame, shape (67, 3) with (x, y, z):
    0..24   pose  (nose, eyes, ears, mouth, shoulders 11/12, elbows, wrists, hand points, hips 23/24)
    25..45  left hand  (21 points; the signer's own left hand)
    46..66  right hand (21 points)
Anything MediaPipe did not detect is stored as NaN (not 0), so "missing" and "at the origin" never mix.
x, y are MediaPipe's normalised image coordinates (0..1, y points down).
"""
import numpy as np

N_POSE, N_HAND = 25, 21
N_KP = N_POSE + 2 * N_HAND  # 67
POSE = slice(0, N_POSE)
LH = slice(N_POSE, N_POSE + N_HAND)
RH = slice(N_POSE + N_HAND, N_KP)
GROUPS = {"pose": POSE, "lh": LH, "rh": RH}
L_SHOULDER, R_SHOULDER = 11, 12


def _lm_to_array(landmark_list, n):
    return np.array([[p.x, p.y, p.z] for p in landmark_list.landmark[:n]], dtype=np.float32)


def results_to_frame(results):
    """MediaPipe Holistic results -> (67, 3) float32 array with NaN for missing parts."""
    out = np.full((N_KP, 3), np.nan, dtype=np.float32)
    if results.pose_landmarks:
        out[POSE] = _lm_to_array(results.pose_landmarks, N_POSE)
    if results.left_hand_landmarks:
        out[LH] = _lm_to_array(results.left_hand_landmarks, N_HAND)
    if results.right_hand_landmarks:
        out[RH] = _lm_to_array(results.right_hand_landmarks, N_HAND)
    return out


def group_valid(seq):
    """(T, 67, 3) with NaN for missing -> {'pose','lh','rh'} -> bool array (T,). Detection is per group."""
    return {name: ~np.isnan(seq[:, sl.start, 0]) for name, sl in GROUPS.items()}


def make_holistic(cfg):
    import mediapipe as mp  # imported here so the rest of the package works without mediapipe
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "This mediapipe version has no legacy 'solutions' API. Install the pinned version: "
            "pip install mediapipe==0.10.14 (Python 3.10-3.12)")
    k = cfg["keypoints"]
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=k["model_complexity"],
        smooth_landmarks=k["smooth_landmarks"],
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=k["min_detection_confidence"],
        min_tracking_confidence=k["min_tracking_confidence"],
    )


def extract_video(path, holistic, max_frames=None, rotate_180=False):
    """Run Holistic over every frame of a video. Returns (array (T, 67, 3), meta dict).
    rotate_180: turn each frame upright first, for the videos listed in config video.rotate_180."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    if hasattr(holistic, "reset"):
        holistic.reset()  # forget tracking state from the previous video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, w, h = [], 0, 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        if not w:
            h, w = bgr.shape[:2]
        if rotate_180:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        frames.append(results_to_frame(holistic.process(rgb)))
    cap.release()
    arr = np.stack(frames) if frames else np.zeros((0, N_KP, 3), np.float32)
    meta = dict(fps=float(fps) if fps and fps > 0 else None, width=int(w), height=int(h), n_frames=len(frames))
    if rotate_180:
        meta["rotated_180"] = True
    return arr, meta
