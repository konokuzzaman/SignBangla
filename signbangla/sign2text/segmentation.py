"""Cutting a live keypoint stream into single signs.

A sign starts when a wrist rises above a line between the shoulders and the hips, and ends when both hands have been
lowered, or have stayed still, for `end_s` seconds. Segments shorter than `min_s` are dropped; a sign longer than
`max_s` is cut there. Frames are the raw (67, 3) MediaPipe keypoints of signbangla.keypoints (image coordinates,
y down, NaN = missing), so the same code works on a recorded sequence and on the camera.
"""
from collections import deque

import numpy as np

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 11, 12, 15, 16, 23, 24
HAND_WRISTS = (25, 46)  # hand landmark 0 of the left and right hand

READY, SIGNING = "ready", "signing"


def body_line(frame, raise_frac, torso_ratio):
    """-> (y of the 'hands up' line, shoulder width), or None without both shoulders. Without visible hips (a webcam
    often shows only the upper body) the hips are assumed torso_ratio shoulder widths below the shoulders."""
    ls, rs = frame[L_SHOULDER], frame[R_SHOULDER]
    if np.isnan(ls[:2]).any() or np.isnan(rs[:2]).any():
        return None
    width = float(np.linalg.norm(ls[:2] - rs[:2]))
    if width < 1e-4:
        return None
    shoulder_y = (ls[1] + rs[1]) / 2
    hips = frame[[L_HIP, R_HIP], 1]
    hip_y = float(np.nanmean(hips)) if not np.isnan(hips).all() else np.nan
    if not np.isfinite(hip_y) or hip_y <= shoulder_y:
        hip_y = shoulder_y + torso_ratio * width
    return shoulder_y + raise_frac * (hip_y - shoulder_y), width


def wrists(frame):
    """(2, 2) x/y of the two wrists: the pose wrists, or the hand model's wrist where the pose has none."""
    out = frame[[L_WRIST, R_WRIST], :2].copy()
    for i, h in enumerate(HAND_WRISTS):
        if np.isnan(out[i]).any():
            out[i] = frame[h, :2]
    return out


class SignSegmenter:
    def __init__(self, raise_frac=0.5, torso_ratio=1.6, end_s=0.4, min_s=0.4, max_s=4.0, still_speed=0.2,
                 preroll_s=0.2, margin_s=0.1):
        self.raise_frac, self.torso_ratio = raise_frac, torso_ratio
        self.end_s, self.min_s, self.max_s = end_s, min_s, max_s
        self.still_speed, self.preroll_s, self.margin_s = still_speed, preroll_s, margin_s
        self.state = READY
        self.history = deque()  # recent (t, frame) for the pre-roll
        self.frames = []  # (t, frame) of the current sign
        self.prev = None  # (t, wrists, width) of the previous frame
        self.need_move = False  # after a sign ended because the hands stood still, wait for movement
        self.last_event = ""

    @classmethod
    def from_config(cls, scfg):
        return cls(**scfg)

    def _measure(self, frame, t):
        """-> (hands up?, wrist speed in shoulder widths per second)."""
        line = body_line(frame, self.raise_frac, self.torso_ratio)
        if line is None:
            self.prev = None
            return False, 0.0
        y_line, width = line
        w = wrists(frame)
        up = bool(np.any(w[:, 1] < y_line))
        speed = 0.0
        if self.prev is not None and t > self.prev[0]:
            moved = np.linalg.norm(np.nan_to_num(w - self.prev[1]), axis=1)
            moved = moved[~np.isnan(w).any(1) & ~np.isnan(self.prev[1]).any(1)]
            speed = float(moved.max() / width / (t - self.prev[0])) if len(moved) else 0.0
        self.prev = (t, w, width)
        return up, speed

    def update(self, frame, t):
        """Add one frame taken at time t (seconds). -> the finished sign (T, 67, 3) when one ends here, else None."""
        frame = np.asarray(frame, np.float32)
        up, speed = self._measure(frame, t)
        moving = speed > self.still_speed
        if self.state == READY:
            self.history.append((t, frame))
            while self.history and self.history[0][0] < t - self.preroll_s:
                self.history.popleft()
            if not up:
                self.need_move = False
            elif not self.need_move or moving:
                self.state, self.start = SIGNING, t
                self.last_up = self.last_move = t
                self.frames = list(self.history)
                self.history.clear()
                self.last_event = "sign started"
            return None
        self.frames.append((t, frame))
        if up:
            self.last_up = t
        if moving:
            self.last_move = t
        lowered, still, too_long = t - self.last_up >= self.end_s, t - self.last_move >= self.end_s, \
            t - self.start >= self.max_s
        if not (lowered or still or too_long):
            return None
        end = (self.last_up if lowered else self.last_move if still else t) + self.margin_s
        segment = [f for ts, f in self.frames if ts <= end]
        duration = min(end - self.margin_s, t) - self.start
        self.state, self.frames = READY, []
        self.need_move = still and not lowered
        if duration < self.min_s:
            self.last_event = f"ignored a {duration:.2f} s movement (shorter than {self.min_s} s)"
            return None
        self.last_event = f"sign of {duration:.2f} s ({'hands lowered' if lowered else 'hands still' if still else 'too long'})"
        return np.stack(segment)

    def segments(self, frames, fps):
        """All signs in a recorded sequence (frames at a fixed fps)."""
        out = []
        for i, f in enumerate(frames):
            s = self.update(f, i / fps)
            if s is not None:
                out.append(s)
        return out
