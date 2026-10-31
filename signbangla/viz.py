"""Plots: skeleton snapshots, quality summary, trial preview video. Matplotlib uses the headless 'Agg' backend."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .keypoints import LH, RH

_HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
_POSE = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22)]
CONNECTIONS = ([(a, b, "black") for a, b in _POSE]
               + [(LH.start + a, LH.start + b, "tab:blue") for a, b in _HAND]
               + [(RH.start + a, RH.start + b, "tab:red") for a, b in _HAND])


def _present(frame):
    return ~np.isnan(frame).any(-1) & (np.abs(np.nan_to_num(frame)).sum(-1) > 0)


def draw_skeleton(ax, frame, invert_y=True):
    """frame (67, 3) -> draws x/y. NaN or all-zero points are treated as missing. Left hand blue, right hand red."""
    ok = _present(frame)
    for a, b, c in CONNECTIONS:
        if ok[a] and ok[b]:
            ax.plot([frame[a, 0], frame[b, 0]], [frame[a, 1], frame[b, 1]], color=c, lw=1)
    ax.scatter(frame[ok, 0], frame[ok, 1], s=3, color="black")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()


def plot_trial_grid(seq, path, n_frames=6, title="", aspect=1.0):
    """Snapshots of one trial at evenly spaced times. For raw trials pass aspect=width/height."""
    seq = np.array(seq, dtype=np.float32)
    seq[..., 0] *= aspect
    idx = np.linspace(0, len(seq) - 1, n_frames).astype(int)
    fig, axes = plt.subplots(1, n_frames, figsize=(2.2 * n_frames, 2.8))
    pts = seq[~np.isnan(seq).any(-1)]
    pts = pts[np.abs(pts).sum(-1) > 0]
    for ax, i in zip(np.atleast_1d(axes), idx):
        draw_skeleton(ax, seq[i])
        if len(pts):
            ax.set_xlim(pts[:, 0].min() - 0.05, pts[:, 0].max() + 0.05)
            ax.set_ylim(pts[:, 1].max() + 0.05, pts[:, 1].min() - 0.05)
        ax.set_title(f"frame {i}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_quality_summary(df, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].hist(df["n_frames"], bins=40, color="gray")
    ax[0, 0].set(title="Trial length (frames)", xlabel="frames", ylabel="trials")
    ax[0, 1].hist(df["best_hand_frac"], bins=20, color="gray")
    ax[0, 1].set(title="Best hand: fraction of frames detected", xlabel="fraction")
    ax[1, 0].hist(df["pose_frac"], bins=20, color="gray")
    ax[1, 0].set(title="Upper body: fraction of frames detected", xlabel="fraction")
    tab = df.groupby(["signer", "dominant_hand"]).size().unstack(fill_value=0)
    tab.plot(kind="bar", stacked=True, ax=ax[1, 1], color={"R": "tab:red", "L": "tab:blue", "none": "lightgray"})
    ax[1, 1].set(title="Trials per signer by dominant hand", xlabel="signer", ylabel="trials")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_preview_video(video_path, raw_trial, start, out_path, fps=10, rotate_180=False):
    """Draw the extracted keypoints on top of the original frames of one trial -> mp4 (to check annotation alignment).
    rotate_180: turn the frames upright first, as extraction did for the videos in config video.rotate_180."""
    import cv2
    from .keypoints import N_KP  # noqa: F401
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = None
    for i in range(len(raw_trial)):
        ok, img = cap.read()
        if not ok:
            break
        if rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        h, w = img.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        f = raw_trial[i]
        good = ~np.isnan(f).any(-1)
        for a, b, c in CONNECTIONS:
            if good[a] and good[b]:
                col = {"black": (0, 255, 0), "tab:blue": (255, 120, 0), "tab:red": (0, 0, 255)}[c]
                cv2.line(img, (int(f[a, 0] * w), int(f[a, 1] * h)), (int(f[b, 0] * w), int(f[b, 1] * h)), col, 2)
        cv2.putText(img, f"frame {start + i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(img)
    cap.release()
    if writer:
        writer.release()
