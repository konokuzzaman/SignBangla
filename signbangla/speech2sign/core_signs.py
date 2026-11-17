"""Turning the core-vocabulary sources into signs in the 67-keypoint format: trimming whole-video recordings to the
signing part, and building short signs from still photos of a handshape (the BDSL49 digits)."""
import numpy as np

from ..keypoints import GROUPS, L_SHOULDER, R_SHOULDER, group_valid
from .stitch import minimum_jerk

L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 13, 14, 15, 16, 23, 24
R_PINKY, R_INDEX, R_THUMB = 18, 20, 22  # the pose model's own right-hand points


def signing_span(raw, raise_frac=0.7, margin=3):
    """Raw video keypoints (T, 67, 3), NaN = missing -> (start, end) frames of the signing part, end exclusive, or None.

    A frame is signing when a wrist is above the line `raise_frac` of the way from the shoulders (0) down to the hips (1).
    The pose model estimates the wrists even when the hand model loses a lowered hand, so the pose wrists are used.
    The span runs from the first to the last signing frame, plus `margin` frames on each side."""
    v = group_valid(raw)["pose"]
    if v.sum() < 2:
        return None
    pose = raw[v]
    shoulder_y = np.median(pose[:, [L_SHOULDER, R_SHOULDER], 1])
    hip_y = np.median(pose[:, [L_HIP, R_HIP], 1])
    if not hip_y > shoulder_y:
        return None
    line = shoulder_y + raise_frac * (hip_y - shoulder_y)
    wrist_y = np.nanmin(raw[:, [L_WRIST, R_WRIST], 1], axis=1)
    hands = ~np.isnan(raw[:, GROUPS["lh"].start, 0]) | ~np.isnan(raw[:, GROUPS["rh"].start, 0])
    active = np.where(np.isnan(wrist_y), hands, wrist_y < line)
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return None
    return max(0, int(idx[0]) - margin), min(len(raw), int(idx[-1]) + 1 + margin)


def is_upright_video(raw):
    """True when most of 15 sampled frames have the head above the shoulders (bdslw401.upright_vote)."""
    from ..bdslw401 import upright_vote
    votes = [v for v in (upright_vote(f) for f in raw[:: max(1, len(raw) // 15)]) if v is not None]
    return bool(votes) and sum(votes) > len(votes) / 2


def check_clip(raw, qcfg, raise_frac=0.7, margin=3):
    """Whole-video keypoints (NaN = missing) -> (signing part or None, checks). The signing part is cut out with
    signing_span; the checks are those of step 5 (quality.trial_quality) on that part plus the upright check. ok is True
    when no flag is raised."""
    from ..quality import trial_quality
    up = is_upright_video(raw)
    span = signing_span(raw, raise_frac, margin)
    if span is None:
        return None, dict(upright=up, start=-1, end=-1, n_frames=0, flags="no_signing", ok=False)
    trial = raw[span[0]:span[1]]
    q = trial_quality(trial, qcfg)
    flags = [f for f in q["flags"].split("|") if f] + ([] if up else ["not_upright"])
    return trial, dict(upright=up, start=span[0], end=span[1], **{k: v for k, v in q.items() if k not in ("flags", "ok")},
                       flags="|".join(flags), ok=not flags)


# ------------------------------------------------------------------ handshapes from photos
def photo_hand(landmarks, width, height, handedness):
    """MediaPipe Hands result for one photo -> (21, 3) points in pixels (z in the same unit as x), as the signer's right
    hand seen from the front. MediaPipe names the hand as if the photo were mirrored, so on an ordinary photo "Left"
    is the signer's right hand; a "Right" (the signer's left hand) is mirrored to look like a right hand."""
    pts = np.array([[p.x * width, p.y * height, p.z * width] for p in landmarks.landmark], np.float32)
    if handedness == "Right":
        pts[:, 0] = 2 * pts[0, 0] - pts[:, 0]
        pts[:, 2] *= -1
    return pts


def hand_angle(hand):
    """Direction from the wrist to the middle-finger knuckle, in radians (image axes: x right, y down)."""
    v = hand[9, :2] - hand[0, :2]
    return float(np.arctan2(v[1], v[0]))


def canonical_shape(hand):
    """Wrist at 0, wrist-to-middle-knuckle length 1, that direction pointing up: only the handshape is left."""
    p = hand - hand[0]
    length = np.linalg.norm(p[9, :2]) + 1e-9
    a = -np.pi / 2 - hand_angle(hand)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    out = p / length
    out[:, :2] = out[:, :2] @ rot.T
    return out


def medoid(shapes):
    """Index of the shape with the smallest total distance to all the others: a typical example, robust to bad
    detections."""
    flat = np.stack([s[:, :2].ravel() for s in shapes])
    d = np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum(1)
    return int(np.argmin(d))


def best_photo(hands):
    """hands: list of (21, 3) photo hands of one digit -> (index of the most typical one, median wrist angle)."""
    i = medoid([canonical_shape(h) for h in hands])
    angles = np.array([hand_angle(h) for h in hands])
    angle = float(np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles))))
    return i, angle


def place_hand(hand, wrist, length, angle):
    """Photo hand -> normalized right-hand points: wrist at `wrist`, wrist-to-middle-knuckle `length` shoulder widths,
    pointing at `angle`."""
    shape = canonical_shape(hand) * length
    a = angle + np.pi / 2
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    out = shape.copy()
    out[:, :2] = shape[:, :2] @ rot.T + np.asarray(wrist, np.float32)
    return out


def elbow_for(shoulder, wrist, upper, fore, outward=-1.0):
    """2D two-bone arm: elbow position for given bone lengths, bent outward (x < 0 side for the right arm) and down."""
    s, w = np.asarray(shoulder[:2], float), np.asarray(wrist[:2], float)
    d = np.linalg.norm(w - s)
    d = np.clip(d, abs(upper - fore) + 1e-6, upper + fore - 1e-6)
    u = (w - s) / (np.linalg.norm(w - s) + 1e-9)
    along = (upper ** 2 - fore ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(upper ** 2 - along ** 2, 0.0))
    n = np.array([-u[1], u[0]])
    if n[0] * outward < 0:
        n = -n
    return s + u * along + n * h


def elbow_from_signs(sequences, wrist, radius=0.12, min_frames=20):
    """Median right elbow (x, y, z) in real signing frames whose right wrist is within `radius` of `wrist`, or None.
    A hand raised in front of the chest points the forearm at the camera, so in 2D the arm looks much shorter than at
    rest; real frames give a natural elbow where a flat two-bone arm cannot."""
    elbows = []
    for seq in sequences:
        have = np.abs(seq[:, GROUPS["rh"]]).sum((1, 2)) > 0
        near = have & (np.linalg.norm(seq[:, R_WRIST, :2] - np.asarray(wrist, np.float32), axis=1) < radius)
        elbows += list(seq[near, R_ELBOW])
    return np.median(elbows, axis=0).astype(np.float32) if len(elbows) >= min_frames else None


def handshape_pose(rest, hand, cfg, elbow=None):
    """The rest pose with the right arm raised and the right hand replaced by `hand` (normalized, 21 x 3). `elbow`:
    from elbow_from_signs; without it a 2D two-bone arm with the rest pose's bone lengths is used."""
    pose = rest.copy()
    rh = GROUPS["rh"]
    upper = np.linalg.norm(rest[R_ELBOW, :2] - rest[R_SHOULDER, :2])
    fore = np.linalg.norm(rest[R_WRIST, :2] - rest[R_ELBOW, :2])
    if elbow is not None:
        pose[R_ELBOW] = elbow
    else:
        pose[R_ELBOW, :2] = elbow_for(rest[R_SHOULDER], hand[0], upper, fore)
    pose[R_WRIST] = hand[0]
    pose[R_PINKY], pose[R_INDEX], pose[R_THUMB] = hand[17], hand[5], hand[2]
    pose[rh] = hand
    return pose


def hold_sign(poses, rest, move_frames, hold_frames, change_frames):
    """Rest -> first handshape -> hold -> (next handshape -> hold ...) -> rest, with minimum-jerk moves. (T, 67, 3)."""
    parts = [minimum_jerk(rest, poses[0], move_frames)]
    for k, p in enumerate(poses):
        parts.append(np.repeat(p[None], hold_frames, 0))
        nxt = poses[k + 1] if k + 1 < len(poses) else rest
        parts.append(minimum_jerk(p, nxt, change_frames if k + 1 < len(poses) else move_frames)[1:])
    return np.concatenate(parts).astype(np.float32)
