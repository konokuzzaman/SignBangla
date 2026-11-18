"""Load the existing 60-word checkpoint without changing its architecture."""
import torch
from signbangla.models import BiGRUClassifier, BiLSTMClassifier


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

