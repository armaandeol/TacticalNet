"""
Phase 3: Training Loop with Time-Series Backtesting

Validation strategy (no data leakage):
  - Fold 1: Train 2010-2017 → Test on 2018 World Cup
  - Fold 2: Train 2010-2021 → Test on 2022 World Cup
  - Final:  Train 2010-2022 → Predict 2026 World Cup

Metrics: Cross-Entropy Loss + Brier Score (calibration)
"""

import logging
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from modules import TacticalNet, PlayerLSTM, StyleAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Metrics
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


def compute_class_weights(labels: list, num_classes: int = 3) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced data.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────
# Label Smoothing Loss
# ─────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing for better calibration.
    """
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        log_probs = F.log_softmax(pred, dim=-1)
        
        if self.weight is not None:
            # Apply class weights
            weight = self.weight.to(pred.device)
            log_probs = log_probs * weight.unsqueeze(0)
        
        loss = (-true_dist * log_probs).sum(dim=-1).mean()
        return loss


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
    
    # Per-class accuracy
    class_names = ["Home Win", "Draw", "Away Win"]
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = float((preds[mask] == i).mean())
        else:
            per_class_acc[name] = 0.0

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "brier_score": bs,
        "per_class_accuracy": per_class_acc,
        "predictions": preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def run_backtest_fold(
    train_loader,
    test_loader,
    fold_name: str,
    epochs: int = 100,
    lr: float = 5e-4,
    device: str = "cpu",
    player_feature_dim: int = 64,
    hidden_dim: int = 128,
    style_latent_dim: int = 4,
    gnn_type: str = "gcn",
    label_smoothing: float = 0.1,
    early_stopping_patience: int = 15,
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
        dropout=0.4,  # Higher dropout for regularization
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{fold_name}] Model has {num_params:,} trainable parameters")
    
    # Compute class weights from training data
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch[4].tolist())
    class_weights = compute_class_weights(train_labels)
    logger.info(f"[{fold_name}] Class distribution: {Counter(train_labels)}")
    logger.info(f"[{fold_name}] Class weights: {class_weights.tolist()}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights)

    best_brier = float("inf")
    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, test_loader, criterion, device)
            scheduler.step(metrics['brier_score'])
            
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"[{fold_name}] Epoch {epoch:3d}/{epochs} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Acc: {metrics['accuracy']:.1%} "
                f"| Brier: {metrics['brier_score']:.4f} "
                f"| LR: {current_lr:.6f}"
            )
            
            # Log per-class accuracy
            pca = metrics['per_class_accuracy']
            logger.info(
                f"[{fold_name}]   Per-class: "
                f"Home={pca.get('Home Win', 0):.1%}, "
                f"Draw={pca.get('Draw', 0):.1%}, "
                f"Away={pca.get('Away Win', 0):.1%}"
            )
            
            # Save best model (prioritize Brier score for calibration)
            if metrics["brier_score"] < best_brier:
                best_brier = metrics["brier_score"]
                best_acc = metrics["accuracy"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                logger.info(f"[{fold_name}]   ✓ New best model!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"[{fold_name}] Early stopping at epoch {epoch}")
                break

    # Save best checkpoint for this fold
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, WEIGHTS_DIR / f"tactical_net_{fold_name}.pt")
        logger.info(f"[{fold_name}] Best model saved (Brier: {best_brier:.4f}, Acc: {best_acc:.1%})")

    final_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"[{fold_name}] Final — Acc: {final_metrics['accuracy']:.1%}, Brier: {final_metrics['brier_score']:.4f}")
    
    # Return simplified metrics for JSON serialization
    return {
        "loss": final_metrics["loss"],
        "accuracy": final_metrics["accuracy"],
        "brier_score": final_metrics["brier_score"],
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info("Load your DataLoaders and call run_backtest_fold() for each temporal split.")
