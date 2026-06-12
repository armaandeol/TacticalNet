"""
World Cup 2026 Match Outcome Predictor

A multimodal deep learning system combining Graph Neural Networks,
LSTMs, and computer vision to predict FIFA World Cup 2026 match outcomes.
"""

from .modules import PlayerLSTM, StyleAutoencoder, TeamGraphEncoder, TacticalNet
from .data_prep import (
    fetch_statsbomb_matches,
    fetch_statsbomb_events,
    build_player_match_stats,
    build_lstm_sequences,
    scrape_fbref_team_stats,
    compute_team_style_features,
)

__version__ = "0.1.0"
__all__ = [
    "PlayerLSTM",
    "StyleAutoencoder",
    "TeamGraphEncoder",
    "TacticalNet",
    "fetch_statsbomb_matches",
    "fetch_statsbomb_events",
    "build_player_match_stats",
    "build_lstm_sequences",
    "scrape_fbref_team_stats",
    "compute_team_style_features",
]
