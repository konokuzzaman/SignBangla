"""Module 1's 401-word BdSLW401 model on one segmented sign: top-k words with scores.

The model is built from the configuration saved in its checkpoint, with signbangla.models (loads
BiGRU, BiLSTM, and the state-of-the-art DualStreamBiGRU).
The input is prepared as in training: preprocess_trial (aspect fix, gap filling, shoulder normalisation, 64 frames).
For DualStreamBiGRU: extracts Stream 1 (402d global motion + velocity) and Stream 2 (246d hand shapes + bone vectors).
For single-stream models: flattened to (64, 201), then velocity features appended when trained with them.
Class index -> word: class_idx = W number - 1 (data/BdSLW401/word_list.csv).
"""
import csv

import numpy as np

from ..preprocess import PreprocessError, preprocess_trial

HAND_PARENTS = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


def velocity(x):
    """(T, F) -> (T, 2F): the coordinates and their frame-to-frame differences (first difference 0)."""
    return np.concatenate([x, np.vstack([np.zeros_like(x[:1]), np.diff(x, axis=0)])], axis=1)


def extract_dual_streams(x_seq, torch_mod):
    """x_seq: (B, 64, 67, 3) torch.Tensor
    Returns:
      stream1: (B, 64, 402) -> shoulder coords + velocity
      stream2: (B, 64, 246) -> wrist-rel hand shapes (126) + hand bones (120)
    """
    B, T, K, C = x_seq.shape

    # Stream 1: Global Motion (402 dims)
    shoulder_norm = x_seq.reshape(B, T, 201)
    vel = torch_mod.cat([torch_mod.zeros_like(shoulder_norm[:, :1]), shoulder_norm[:, 1:] - shoulder_norm[:, :-1]], dim=1)
    stream1 = torch_mod.cat([shoulder_norm, vel], dim=-1)

    # Stream 2: Hand Articulation & Bones (246 dims)
    # Left Hand (25:46)
    lh = x_seq[:, :, 25:46, :]
    lh_size = torch_mod.norm(lh[:, :, 9, :] - lh[:, :, 0, :], dim=-1, keepdim=True)  # (B, T, 1)
    lh_valid = (lh_size > 1e-4).float()
    lh_norm = (lh - lh[:, :, 0:1, :]) / torch_mod.clamp(lh_size.unsqueeze(-1), min=1e-4) * lh_valid.unsqueeze(-1)
    lh_flat = lh_norm.reshape(B, T, 63)

    lh_bones = []
    for child, parent in enumerate(HAND_PARENTS, start=1):
        bone = (lh[:, :, child, :] - lh[:, :, parent, :]) / torch_mod.clamp(lh_size, min=1e-4) * lh_valid
        lh_bones.append(bone)
    lh_bones_flat = torch_mod.cat(lh_bones, dim=-1)  # (B, T, 60)

    # Right Hand (46:67)
    rh = x_seq[:, :, 46:67, :]
    rh_size = torch_mod.norm(rh[:, :, 9, :] - rh[:, :, 0, :], dim=-1, keepdim=True)
    rh_valid = (rh_size > 1e-4).float()
    rh_norm = (rh - rh[:, :, 0:1, :]) / torch_mod.clamp(rh_size.unsqueeze(-1), min=1e-4) * rh_valid.unsqueeze(-1)
    rh_flat = rh_norm.reshape(B, T, 63)

    rh_bones = []
    for child, parent in enumerate(HAND_PARENTS, start=1):
        bone = (rh[:, :, child, :] - rh[:, :, parent, :]) / torch_mod.clamp(rh_size, min=1e-4) * rh_valid
        rh_bones.append(bone)
    rh_bones_flat = torch_mod.cat(rh_bones, dim=-1)  # (B, T, 60)

    stream2 = torch_mod.cat([lh_flat, rh_flat, lh_bones_flat, rh_bones_flat], dim=-1)  # (B, T, 246)
    return stream1, stream2


def read_word_list(path):
    """-> [(bangla, english)] indexed by class_idx (W number - 1). Bangla: its first spelling."""
    with open(path, encoding="utf-8-sig") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["number"]))
    words = [(r["bangla"].split("/")[0].strip(), r["english"]) for r in rows]
    if [int(r["number"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: word numbers are not 1..{len(rows)}")
    return words


class SignRecognizer:
    def __init__(self, checkpoint, word_list, pcfg, device="auto"):
        import torch

        from .. import models
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        c = ck["config"]
        m_type = c.get("model_type", "")

        if m_type == "dual_stream_bigru":
            model = models.DualStreamBiGRU(
                stream1_dim=c.get("stream1_dim", 402),
                stream2_dim=c.get("stream2_dim", 246),
                hidden_dim=c.get("hidden_dim", 128),
                num_layers=c.get("num_layers", 2),
                num_classes=c["num_classes"],
                dropout=c.get("dropout", 0.3),
            )
            self.is_dual_stream = True
        else:
            cls = getattr(models, {"bigru": "BiGRUClassifier", "bilstm": "BiLSTMClassifier"}.get(m_type, ""), None)
            if cls is None:
                raise ValueError(f"model type {m_type} cannot be built with this branch's signbangla/models.py")
            model = cls(input_dim=c["input_dim"], hidden_dim=c["hidden_dim"], num_layers=c["num_layers"],
                        num_classes=c["num_classes"], dropout=c["dropout"], use_attention=True)
            self.is_dual_stream = False

        model.load_state_dict(ck["model_state_dict"], strict=True)
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch, self.device = torch, torch.device(device)
        self.model = model.to(self.device).eval()
        self.config, self.pcfg = c, pcfg
        self.use_velocity = c.get("input_dim", 0) == 2 * 201
        self.words = read_word_list(word_list)
        if len(self.words) != c["num_classes"]:
            raise ValueError(f"{word_list} has {len(self.words)} words, the model {c['num_classes']} classes")
        self.name = f"{m_type} ({c['num_classes']} words, epoch {ck.get('epoch')})"

    def features(self, raw, width, height):
        out, _ = preprocess_trial(raw, width, height, self.pcfg)
        if self.is_dual_stream:
            return out  # (64, 67, 3)
        x = out.reshape(out.shape[0], -1)
        return velocity(x) if self.use_velocity else x

    def top_k(self, raw, width, height, k=5):
        """raw (T, 67, 3) NaN = missing, in image coordinates -> [dict(word, english, score, class_idx)] best first.
        Raises PreprocessError when the sign has no usable body."""
        feats = self.features(raw, width, height)
        if self.is_dual_stream:
            x_t = self.torch.from_numpy(feats[None].astype(np.float32)).to(self.device)
            s1, s2 = extract_dual_streams(x_t, self.torch)
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(s1, s2), -1)[0].cpu().numpy()
        else:
            x = self.torch.from_numpy(feats[None].astype(np.float32)).to(self.device)
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(x), -1)[0].cpu().numpy()
        best = np.argsort(p)[::-1][:k]
        return [dict(word=self.words[i][0], english=self.words[i][1], score=round(float(p[i]), 4), class_idx=int(i))
                for i in best]


__all__ = ["SignRecognizer", "PreprocessError", "read_word_list", "velocity", "extract_dual_streams"]
