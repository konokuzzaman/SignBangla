"""SignBangla neural network architectures for Bengali Sign Language recognition.

Includes BiGRU baseline classifier with temporal attention pooling.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import N_CLASSES


class TemporalAttention(nn.Module):
    """Learned attention over sequence frames (batch, T, dim) -> (batch, dim)."""

    def __init__(self, feat_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        att_dim = hidden_dim or max(32, feat_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D)
        scores = self.net(x)  # (B, T, 1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, D)
        return pooled, weights


class BiGRUClassifier(nn.Module):
    """Bidirectional GRU with input projection and temporal attention pooling.

    Input shape: (batch, seq_len=64, input_dim=201)
    Output shape: (batch, num_classes=60)
    """

    def __init__(
        self,
        input_dim: int = 201,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = N_CLASSES,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Spatial projection & input normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )

        # Multi-layer Bidirectional GRU
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )

        gru_out_dim = hidden_dim * 2  # Bidirectional

        # Temporal pooling
        if use_attention:
            self.attention = TemporalAttention(gru_out_dim)
        else:
            self.attention = None

        self.layer_norm = nn.LayerNorm(gru_out_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_dim) or (batch, seq_len, 67, 3)

        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        if x.ndim == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        # Project input to hidden_dim
        h = self.input_proj(x)  # (B, T, hidden_dim)

        # BiGRU encoding
        out, _ = self.gru(h)  # (B, T, 2 * hidden_dim)

        # Pooling across time
        if self.attention is not None:
            pooled, _ = self.attention(out)
        else:
            pooled = out.mean(dim=1)

        pooled = self.layer_norm(pooled)

        # Classification
        logits = self.classifier(pooled)
        return logits


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with input projection and temporal attention pooling.

    Input shape: (batch, seq_len=64, input_dim=201)
    Output shape: (batch, num_classes=60)
    """

    def __init__(
        self,
        input_dim: int = 201,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = N_CLASSES,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Spatial projection & input normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )

        # Multi-layer Bidirectional LSTM
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        lstm_out_dim = hidden_dim * 2  # Bidirectional

        # Temporal pooling
        if use_attention:
            self.attention = TemporalAttention(lstm_out_dim)
        else:
            self.attention = None

        self.layer_norm = nn.LayerNorm(lstm_out_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_dim) or (batch, seq_len, 67, 3)

        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        if x.ndim == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        # Project input to hidden_dim
        h = self.input_proj(x)  # (B, T, hidden_dim)

        # BiLSTM encoding
        out, _ = self.lstm(h)  # (B, T, 2 * hidden_dim)

        # Pooling across time
        if self.attention is not None:
            pooled, _ = self.attention(out)
        else:
            pooled = out.mean(dim=1)

        pooled = self.layer_norm(pooled)

        # Classification
        logits = self.classifier(pooled)
        return logits

