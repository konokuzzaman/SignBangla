"""Module 1's 401-word BdSLW401 model on one segmented sign: top-k words with scores.

The model is built from the configuration saved in its checkpoint, with signbangla.models (this branch loads the
module-1 BiGRU and BiLSTM checkpoints exactly). The input is prepared as in training: preprocess_trial (aspect fix, gap
filling, shoulder normalisation, 64 frames), flattened to (64, 201), then velocity features appended (module-1
scripts/13_train_bdslw401.py, compute_velocity) when the checkpoint was trained with them.
Class index -> word: class_idx = W number - 1 (data/BdSLW401/word_list.csv).
"""
import csv

import numpy as np

from ..preprocess import PreprocessError, preprocess_trial


def velocity(x):
    """(T, F) -> (T, 2F): the coordinates and their frame-to-frame differences (first difference 0)."""
    return np.concatenate([x, np.vstack([np.zeros_like(x[:1]), np.diff(x, axis=0)])], axis=1)


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
        cls = getattr(models, {"bigru": "BiGRUClassifier", "bilstm": "BiLSTMClassifier"}.get(c["model_type"], ""), None)
        if cls is None:
            raise ValueError(f"model type {c['model_type']} cannot be built with this branch's signbangla/models.py")
        model = cls(input_dim=c["input_dim"], hidden_dim=c["hidden_dim"], num_layers=c["num_layers"],
                    num_classes=c["num_classes"], dropout=c["dropout"], use_attention=True)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch, self.device = torch, torch.device(device)
        self.model = model.to(self.device).eval()
        self.config, self.pcfg = c, pcfg
        self.use_velocity = c["input_dim"] == 2 * 201
        self.words = read_word_list(word_list)
        if len(self.words) != c["num_classes"]:
            raise ValueError(f"{word_list} has {len(self.words)} words, the model {c['num_classes']} classes")
        self.name = f"{c['model_type']} ({c['num_classes']} words, epoch {ck.get('epoch')})"

    def features(self, raw, width, height):
        out, _ = preprocess_trial(raw, width, height, self.pcfg)
        x = out.reshape(out.shape[0], -1)
        return velocity(x) if self.use_velocity else x

    def top_k(self, raw, width, height, k=5):
        """raw (T, 67, 3) NaN = missing, in image coordinates -> [dict(word, english, score, class_idx)] best first.
        Raises PreprocessError when the sign has no usable body."""
        x = self.torch.from_numpy(self.features(raw, width, height)[None].astype(np.float32)).to(self.device)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(x), -1)[0].cpu().numpy()
        best = np.argsort(p)[::-1][:k]
        return [dict(word=self.words[i][0], english=self.words[i][1], score=round(float(p[i]), 4), class_idx=int(i))
                for i in best]


__all__ = ["SignRecognizer", "PreprocessError", "read_word_list", "velocity"]
