"""SignBangla neural network architectures for Bengali Sign Language recognition.

Includes BiGRU baseline classifier with temporal attention pooling.
"""
import math
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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BdSLSpoterClassifier(nn.Module):
    """BdSL-SPOTER: Sign POse-based TransformER for Bengali Sign Language.

    Input shape:  (batch, seq_len=64, input_dim=201 or 402)
    Output shape: (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 201,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = N_CLASSES,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=128, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        if use_attention:
            self.attention = TemporalAttention(d_model)
        else:
            self.attention = None

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        h = self.input_proj(x)
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)

        if self.attention is not None:
            pooled, _ = self.attention(encoded)
        else:
            pooled = encoded.mean(dim=1)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        return logits


def get_skeleton_parents_and_adj():
    """Construct kinematic bone parent indices and normalized adjacency for the 67-joint skeleton."""
    parents = [0] * 67
    parents[1] = 0; parents[2] = 1; parents[3] = 2; parents[4] = 0; parents[5] = 4; parents[6] = 5
    parents[7] = 3; parents[8] = 6; parents[9] = 0; parents[10] = 0; parents[11] = 0; parents[12] = 0
    parents[13] = 11; parents[14] = 12; parents[15] = 13; parents[16] = 14
    parents[17] = 15; parents[18] = 16; parents[19] = 15; parents[20] = 16; parents[21] = 15; parents[22] = 16
    parents[23] = 11; parents[24] = 12

    def set_hand(base, wrist):
        parents[base] = wrist
        for finger_start in [1, 5, 9, 13, 17]:
            parents[base + finger_start] = base
            for k in range(1, 4):
                parents[base + finger_start + k] = base + finger_start + k - 1

    set_hand(25, 15)  # Left hand to left arm wrist
    set_hand(46, 16)  # Right hand to right arm wrist

    A = torch.eye(67)
    for i in range(1, 67):
        p = parents[i]
        A[i, p] = 1.0
        A[p, i] = 1.0

    deg = A.sum(dim=-1)
    deg_inv = torch.pow(deg, -0.5)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    A_norm = torch.diag(deg_inv) @ A @ torch.diag(deg_inv)

    return torch.tensor(parents, dtype=torch.long), A_norm


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution over the anatomical skeleton with adaptive topology learning."""

    def __init__(self, in_c: int, out_c: int, adj: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("A", adj)
        self.A_learnable = nn.Parameter(torch.zeros(67, 67))
        self.fc = nn.Linear(in_c, out_c, bias=False)
        self.norm = nn.LayerNorm(out_c)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C)
        A_eff = self.A + torch.tanh(self.A_learnable) * 0.25
        out = torch.einsum("vw,btwc->btvc", A_eff, x)
        out = self.fc(out)
        out = self.norm(out)
        out = self.act(out)
        return self.dropout(out)


class BdSLKinematicGraphTransformer(nn.Module):
    """Phase 3 SOTA: Spatio-Temporal Kinematic Graph Transformer for Bengali Sign Language.

    Combines:
      1. 4-Stream Kinematic Features: Joint positions (J), velocities (dJ),
         bone vectors (B = J - J_parent), and bone velocities (dB) -> (B, T, 67, 12).
      2. Spatial Graph Convolution: Uses physical skeletal bone topology + adaptive attention.
      3. Spatial Joint Attention Pooling: Learns dynamic frame-by-frame joint saliency.
      4. Multi-Layer Temporal Transformer: Models global temporal signing dynamics.
      5. Temporal Attention Pooling & Regularized Classification Head.
    """

    def __init__(
        self,
        num_classes: int = N_CLASSES,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        parents, adj = get_skeleton_parents_and_adj()
        self.register_buffer("parents", parents)

        # 1. Kinematic projection & Spatial GCN
        self.joint_proj = nn.Sequential(
            nn.Linear(12, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )
        self.spatial_gcn = SpatialGraphConv(d_model, d_model, adj, dropout=dropout / 2.0)

        # 2. Spatial Joint Attention Pooling (per-frame across 67 joints)
        self.spatial_attn = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # 3. Temporal Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=128, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 4. Temporal Attention Pooling across frames
        self.temporal_attn = TemporalAttention(d_model)
        self.norm = nn.LayerNorm(d_model)

        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def extract_kinematics(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 67, 3)
        J = x
        B = x - x[:, :, self.parents, :]
        B[:, :, 0, :] = 0.0  # Root has no parent bone
        dJ = torch.cat([torch.zeros_like(J[:, :1]), J[:, 1:] - J[:, :-1]], dim=1)
        dB = torch.cat([torch.zeros_like(B[:, :1]), B[:, 1:] - B[:, :-1]], dim=1)
        return torch.cat([J, dJ, B, dB], dim=-1)  # (B, T, 67, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[-1] == 201:
            x = x.reshape(x.shape[0], x.shape[1], 67, 3)
        elif x.ndim == 3 and x.shape[-1] == 402:
            x = x[:, :, :201].reshape(x.shape[0], x.shape[1], 67, 3)

        # 1. 4-Stream Kinematics: (B, T, 67, 12)
        feats = self.extract_kinematics(x)

        # 2. Joint embedding & Spatial GCN: (B, T, 67, d_model)
        h = self.joint_proj(feats)
        h = h + self.spatial_gcn(h)

        # 3. Spatial Joint Attention Pooling across 67 joints: (B, T, d_model)
        scores = self.spatial_attn(h)
        weights = torch.softmax(scores, dim=2)
        h_frame = (h * weights).sum(dim=2)

        # 4. Temporal Transformer Encoding: (B, T, d_model)
        h_temporal = self.pos_encoder(h_frame)
        encoded = self.transformer(h_temporal)

        # 5. Temporal Attention Pooling: (B, d_model)
        pooled, _ = self.temporal_attn(encoded)
        pooled = self.norm(pooled)

        # 6. Classification: (B, num_classes)
        return self.classifier(pooled)


class DualStreamBiGRU(nn.Module):
    """Dual-Stream Late-Fusion BiGRU for Sign Language Recognition (Module 1 Step 5).

    Stream 1 (Global Motion): Shoulder-normalized pose/hands coordinates + joint velocity (402 dims).
    Stream 2 (Hand Articulation & Bones): Wrist-relative hand shapes + finger bone vectors (246 dims).

    Both streams pass through dedicated BiGRU encoders with temporal attention pooling,
    and are fused into a unified classification head.
    """

    def __init__(
        self,
        stream1_dim: int = 402,
        stream2_dim: int = 246,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stream1_dim = stream1_dim
        self.stream2_dim = stream2_dim
        self.hidden_dim = hidden_dim

        # Stream 1: Global Motion Branch
        self.proj1 = nn.Sequential(
            nn.Linear(stream1_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru1 = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.attn1 = TemporalAttention(hidden_dim * 2)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)

        # Stream 2: Hand Articulation & Bones Branch
        self.proj2 = nn.Sequential(
            nn.Linear(stream2_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )
        self.gru2 = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.attn2 = TemporalAttention(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        # Late Fusion Classifier Head
        fused_dim = (hidden_dim * 2) * 2  # 256 + 256 = 512
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        # Stream 1
        h1 = self.proj1(s1)
        out1, _ = self.gru1(h1)
        pooled1, _ = self.attn1(out1)
        pooled1 = self.norm1(pooled1)

        # Stream 2
        h2 = self.proj2(s2)
        out2, _ = self.gru2(h2)
        pooled2, _ = self.attn2(out2)
        pooled2 = self.norm2(pooled2)

        # Late fusion
        fused = torch.cat([pooled1, pooled2], dim=-1)
        return self.classifier(fused)


