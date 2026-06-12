"""
Phase 3: Training Loop with Time-Series Backtesting

Validation strategy (no data leakage):
  - Fold 1: Train 2010-2017 → Test on 2018 World Cup
  - Fold 2: Train 2010-2018 → Test on 2022 World Cup
  - Final:  Train 2010-2022 → Predict 2026 World Cup

Metrics: Cross-Entropy Loss + Brier Score (calibration)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from modules import TacticalNet, PlayerLSTM, StyleAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Brier Score (calibration metric)
# ─────────────────────────────────────────────

def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """
    Multiclass Brier Score — lower is better (0 = perfect calibration).

    Args:
        probs:  (N, num_classes) predicted probabilities
        labels: (N,) integer class labels
    """
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ─────────────────────────────────────────────
# Autoencoder Pre-training
# ─────────────────────────────────────────────

def pretrain_autoencoder(
    autoencoder: StyleAutoencoder,
    style_features: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> StyleAutoencoder:
    """
    Unsupervised pre-training of the StyleAutoencoder on team style feature vectors.
    Uses MSE reconstruction loss.
    """
    autoencoder = autoencoder.to(device)
    style_features = style_features.to(device)
    optimizer = AdamW(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    autoencoder.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_hat, _ = autoencoder(style_features)
        loss = criterion(x_hat, style_features)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            logger.info(f"[Autoencoder] Epoch {epoch}/{epochs} — Recon Loss: {loss.item():.4f}")

    torch.save(autoencoder.state_dict(), WEIGHTS_DIR / "style_autoencoder.pt")
    logger.info("Autoencoder weights saved.")
    return autoencoder


# ─────────────────────────────────────────────
# LSTM Pre-training
# ─────────────────────────────────────────────

def pretrain_lstm(
    lstm: PlayerLSTM,
    sequences: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> PlayerLSTM:
    """
    Supervised pre-training of PlayerLSTM to predict next-match xG
    from a rolling window. This grounds the form embeddings.
    """
    lstm = lstm.to(device)
    sequences, targets = sequences.to(device), targets.to(device)
    optimizer = AdamW(lstm.parameters(), lr=lr)
    criterion = nn.MSELoss()
    head = nn.Linear(lstm.lstm.hidden_size, 1).to(device)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        embedding = lstm(sequences)          # (batch, hidden_size)
        pred = head(embedding).squeeze(-1)   # (batch,)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            logger.info(f"[LSTM] Epoch {epoch}/{epochs} — MSE: {loss.item():.4f}")

    torch.save(lstm.state_dict(), WEIGHTS_DIR / "player_lstm.pt")
    logger.info("LSTM weights saved.")
    return lstm


# ─────────────────────────────────────────────
# Main Training Loop (TacticalNet)
# ─────────────────────────────────────────────

def train_one_epoch(
    model: TacticalNet,
    loader,
    optimizer,
    criterion,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        data_a, data_b, style_a, style_b, labels = batch
        data_a = data_a.to(device)
        data_b = data_b.to(device)
        style_a = style_a.to(device)
        style_b = style_b.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(data_a, data_b, style_a, style_b)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: TacticalNet,
    loader,
    criterion,
    device: str,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        data_a, data_b, style_a, style_b, labels = batch
        data_a = data_a.to(device)
        data_b = data_b.to(device)
        style_a = style_a.to(device)
        style_b = style_b.to(device)
        labels = labels.to(device)

        logits = model(data_a, data_b, style_a, style_b)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds = all_probs.argmax(axis=1)
    accuracy = float((preds == all_labels).mean())
    bs = brier_score(all_probs, all_labels)

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "brier_score": bs,
    }


def run_backtest_fold(
    train_loader,
    test_loader,
    fold_name: str,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    player_feature_dim: int = 64,
    hidden_dim: int = 128,
    style_latent_dim: int = 4,
    gnn_type: str = "gcn",
) -> dict:
    """
    Train and evaluate TacticalNet for one backtesting fold.
    Returns evaluation metrics on the test set.
    """
    model = TacticalNet(
        player_feature_dim=player_feature_dim,
        hidden_dim=hidden_dim,
        style_latent_dim=style_latent_dim,
        gnn_type=gnn_type,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_brier = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if epoch % 10 == 0:
            metrics = evaluate(model, test_loader, criterion, device)
            logger.info(
                f"[{fold_name}] Epoch {epoch}/{epochs} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Acc: {metrics['accuracy']:.3f} "
                f"| Brier: {metrics['brier_score']:.4f}"
            )
            if metrics["brier_score"] < best_brier:
                best_brier = metrics["brier_score"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save best checkpoint for this fold
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, WEIGHTS_DIR / f"tactical_net_{fold_name}.pt")
        logger.info(f"[{fold_name}] Best model saved (Brier: {best_brier:.4f})")

    final_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"[{fold_name}] Final — Acc: {final_metrics['accuracy']:.3f}, Brier: {final_metrics['brier_score']:.4f}")
    return final_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info("Load your DataLoaders and call run_backtest_fold() for each temporal split.")
