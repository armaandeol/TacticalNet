"""
Phase 2 & 3: Model Architecture

Contains:
  - PlayerLSTM:     Encodes rolling match sequences into dynamic form embeddings.
  - StyleAutoencoder: Compresses team style metrics into a latent style vector.
  - TacticalNet:    GCN-based team graph encoder + MLP match outcome predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


# ─────────────────────────────────────────────
# Phase 2a: Player Form Trajectory Encoder
# ─────────────────────────────────────────────

class PlayerLSTM(nn.Module):
    """
    Encodes a rolling window of per-match performance metrics into a
    fixed-size dynamic Form Embedding via an LSTM.

    Input:  (batch, seq_len=20, input_size)  — e.g. [xG, prog_passes, def_interventions, distance]
    Output: (batch, hidden_size)             — final hidden state = Form Embedding
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) — take last layer
        embedding = self.layer_norm(h_n[-1])  # (batch, hidden_size)
        return embedding


# ─────────────────────────────────────────────
# Phase 2b: Playing Style Autoencoder
# ─────────────────────────────────────────────

class StyleAutoencoder(nn.Module):
    """
    Unsupervised autoencoder that compresses high-dimensional team style
    metrics (PPDA, directness, field tilt, crossing freq, etc.) into a
    low-dimensional latent style vector.

    Input:  (batch, input_dim)   — raw team style features
    Output: (batch, latent_dim)  — compressed style embedding
    """

    def __init__(self, input_dim: int = 16, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z  # reconstruction + latent vector


# ─────────────────────────────────────────────
# Phase 3: Graph Neural Network + MLP Predictor
# ─────────────────────────────────────────────

class TeamGraphEncoder(nn.Module):
    """
    Encodes a team's player graph (nodes = players with LSTM embeddings,
    edges = passing chemistry) into a single Team Synergy Vector.

    Supports both GCN and GAT variants.
    """

    def __init__(
        self,
        player_feature_dim: int = 64,
        hidden_dim: int = 128,
        gnn_type: str = "gcn",  # "gcn" or "gat"
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout

        if gnn_type == "gat":
            self.conv1 = GATConv(player_feature_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        else:  # default GCN
            self.conv1 = GCNConv(player_feature_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # Global pooling: aggregate 11 player nodes → 1 team vector
        team_vector = global_mean_pool(x, batch)  # (batch_size, hidden_dim)
        return team_vector


class TacticalNet(nn.Module):
    """
    Full end-to-end match outcome predictor.

    Combines:
      - Team A graph synergy vector  (hidden_dim)
      - Team B graph synergy vector  (hidden_dim)
      - Concatenated style latent vectors for both teams (latent_dim * 2)

    Outputs logits for [Win, Draw, Loss] via a 3-class MLP head.
    """

    def __init__(
        self,
        player_feature_dim: int = 64,
        hidden_dim: int = 128,
        style_latent_dim: int = 4,
        gnn_type: str = "gcn",
        dropout: float = 0.4,
    ):
        super().__init__()
        self.team_encoder = TeamGraphEncoder(
            player_feature_dim=player_feature_dim,
            hidden_dim=hidden_dim,
            gnn_type=gnn_type,
            dropout=dropout,
        )

        # MLP head: [team_a_vec | team_b_vec | style_a | style_b]
        mlp_input_dim = hidden_dim * 2 + style_latent_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # [Win, Draw, Loss]
        )

    def forward(self, data_a, data_b, style_a: torch.Tensor, style_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data_a:   PyG Data object for Team A (x, edge_index, batch)
            data_b:   PyG Data object for Team B
            style_a:  (batch, style_latent_dim) — Team A style embedding
            style_b:  (batch, style_latent_dim) — Team B style embedding

        Returns:
            logits: (batch, 3) — raw scores for [Win, Draw, Loss]
        """
        vec_a = self.team_encoder(data_a.x, data_a.edge_index, data_a.batch)
        vec_b = self.team_encoder(data_b.x, data_b.edge_index, data_b.batch)

        combined = torch.cat([vec_a, vec_b, style_a, style_b], dim=1)
        logits = self.classifier(combined)
        return logits

    def predict_proba(self, data_a, data_b, style_a, style_b) -> torch.Tensor:
        """Returns softmax probabilities: [P(Win), P(Draw), P(Loss)]."""
        with torch.no_grad():
            logits = self.forward(data_a, data_b, style_a, style_b)
            return F.softmax(logits, dim=-1)
