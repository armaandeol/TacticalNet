"""
PyTorch Dataset and DataLoader utilities for match prediction.

Creates proper DataLoaders from processed data for training TacticalNet.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MatchDataset(Dataset):
    """
    Dataset for match outcome prediction.
    
    Each sample contains:
    - Team A player graph (11 nodes with LSTM embeddings)
    - Team B player graph (11 nodes with LSTM embeddings)
    - Team A style vector
    - Team B style vector
    - Label: 0 = Team A wins, 1 = Draw, 2 = Team B wins
    """
    
    def __init__(
        self,
        matches: List[dict],
        player_embeddings: dict,
        team_styles: dict,
        feature_dim: int = 64,
    ):
        """
        Args:
            matches: List of match dicts with keys:
                     'match_id', 'home_team', 'away_team', 'home_players',
                     'away_players', 'home_score', 'away_score'
            player_embeddings: Dict mapping player_id -> embedding tensor (feature_dim,)
            team_styles: Dict mapping team_name -> style vector (4,)
            feature_dim: Dimension of player embeddings
        """
        self.matches = matches
        self.player_embeddings = player_embeddings
        self.team_styles = team_styles
        self.feature_dim = feature_dim
        
    def __len__(self) -> int:
        return len(self.matches)
    
    def _build_team_graph(self, player_ids: List[int]) -> Data:
        """
        Build a fully-connected graph for a team's players.
        
        Nodes: 11 players with their LSTM embeddings
        Edges: All pairs (passing network approximation)
        """
        num_players = len(player_ids)
        
        # Get player embeddings (use random if not found)
        node_features = []
        for pid in player_ids:
            if pid in self.player_embeddings:
                emb = self.player_embeddings[pid]
            else:
                # Fallback: random embedding for unknown players
                emb = torch.randn(self.feature_dim) * 0.1
            node_features.append(emb)
        
        # Pad to 11 players if needed
        while len(node_features) < 11:
            node_features.append(torch.zeros(self.feature_dim))
        
        x = torch.stack(node_features[:11])  # (11, feature_dim)
        
        # Fully connected edges
        src = [i for i in range(11) for j in range(11) if i != j]
        dst = [j for i in range(11) for j in range(11) if i != j]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    def _get_label(self, home_score: int, away_score: int) -> int:
        """Convert match result to label: 0=home win, 1=draw, 2=away win."""
        if home_score > away_score:
            return 0
        elif home_score == away_score:
            return 1
        else:
            return 2
    
    def __getitem__(self, idx: int) -> Tuple[Data, Data, torch.Tensor, torch.Tensor, int]:
        match = self.matches[idx]
        
        # Build team graphs
        graph_a = self._build_team_graph(match.get('home_players', []))
        graph_b = self._build_team_graph(match.get('away_players', []))
        
        # Get style vectors
        home_team = match['home_team']
        away_team = match['away_team']
        
        style_a = self.team_styles.get(
            home_team, 
            torch.tensor([0.5, 0.5, 0.5, 0.5])
        )
        style_b = self.team_styles.get(
            away_team,
            torch.tensor([0.5, 0.5, 0.5, 0.5])
        )
        
        if not isinstance(style_a, torch.Tensor):
            style_a = torch.tensor(style_a, dtype=torch.float32)
        if not isinstance(style_b, torch.Tensor):
            style_b = torch.tensor(style_b, dtype=torch.float32)
        
        # Get label
        label = self._get_label(match['home_score'], match['away_score'])
        
        return graph_a, graph_b, style_a, style_b, label


def collate_match_batch(
    batch: List[Tuple[Data, Data, torch.Tensor, torch.Tensor, int]]
) -> Tuple[Batch, Batch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching match data.
    
    Batches PyG Data objects properly and stacks tensors.
    """
    graphs_a, graphs_b, styles_a, styles_b, labels = zip(*batch)
    
    # Batch PyG graphs
    batch_a = Batch.from_data_list(list(graphs_a))
    batch_b = Batch.from_data_list(list(graphs_b))
    
    # Stack style vectors and labels
    styles_a = torch.stack(styles_a)
    styles_b = torch.stack(styles_b)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_a, batch_b, styles_a, styles_b, labels


def create_dataloaders(
    train_matches: List[dict],
    val_matches: List[dict],
    player_embeddings: dict,
    team_styles: dict,
    batch_size: int = 32,
    feature_dim: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_matches: List of training match dicts
        val_matches: List of validation match dicts
        player_embeddings: Dict mapping player_id -> embedding
        team_styles: Dict mapping team_name -> style vector
        batch_size: Batch size for training
        feature_dim: Player embedding dimension
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = MatchDataset(
        matches=train_matches,
        player_embeddings=player_embeddings,
        team_styles=team_styles,
        feature_dim=feature_dim,
    )
    
    val_dataset = MatchDataset(
        matches=val_matches,
        player_embeddings=player_embeddings,
        team_styles=team_styles,
        feature_dim=feature_dim,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_match_batch,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_match_batch,
        num_workers=num_workers,
    )
    
    logger.info(f"Created DataLoaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Demo/Synthetic Data Generation
# ─────────────────────────────────────────────

def generate_synthetic_data(
    num_matches: int = 200,
    num_teams: int = 32,
    feature_dim: int = 64,
    seed: int = 42,
) -> Tuple[List[dict], dict, dict]:
    """
    Generate synthetic match data for testing the pipeline.
    
    Returns:
        Tuple of (matches, player_embeddings, team_styles)
    """
    rng = np.random.default_rng(seed)
    
    # Generate team names
    teams = [
        "Argentina", "France", "Brazil", "England", "Spain", "Germany",
        "Portugal", "Netherlands", "Italy", "Belgium", "USA", "Mexico",
        "Morocco", "Japan", "South Korea", "Senegal", "Australia", "Croatia",
        "Uruguay", "Colombia", "Switzerland", "Denmark", "Poland", "Austria",
        "Wales", "Serbia", "Sweden", "Ukraine", "Chile", "Peru", "Ecuador", "Canada"
    ][:num_teams]
    
    # Generate player embeddings (11 players per team, ~350 total)
    player_embeddings = {}
    player_id = 1
    team_players = {}
    
    for team in teams:
        team_players[team] = []
        for _ in range(11):
            # Embeddings with team-specific bias for realism
            team_strength = rng.uniform(0.3, 0.9)
            emb = torch.tensor(
                rng.normal(loc=team_strength, scale=0.2, size=feature_dim),
                dtype=torch.float32
            )
            player_embeddings[player_id] = emb
            team_players[team].append(player_id)
            player_id += 1
    
    # Generate team styles
    team_styles = {}
    for team in teams:
        style = torch.tensor(
            rng.uniform(0.2, 0.9, size=4),
            dtype=torch.float32
        )
        team_styles[team] = style
    
    # Generate matches
    matches = []
    for i in range(num_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        
        # Simulate scores based on team "strength" (mean of style vector)
        home_strength = team_styles[home].mean().item()
        away_strength = team_styles[away].mean().item()
        
        home_score = int(rng.poisson(1.5 * home_strength + 0.5))
        away_score = int(rng.poisson(1.5 * away_strength + 0.5))
        
        matches.append({
            'match_id': i + 1,
            'home_team': home,
            'away_team': away,
            'home_players': team_players[home],
            'away_players': team_players[away],
            'home_score': home_score,
            'away_score': away_score,
            'year': rng.choice([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]),
        })
    
    logger.info(f"Generated {num_matches} synthetic matches with {len(player_embeddings)} players")
    return matches, player_embeddings, team_styles


def save_synthetic_data(output_dir: str = "data/processed") -> None:
    """Generate and save synthetic data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    matches, player_embeddings, team_styles = generate_synthetic_data()
    
    # Save matches
    with open(output_path / "matches.json", "w") as f:
        json.dump(matches, f, indent=2)
    
    # Save player embeddings
    emb_dict = {str(k): v.tolist() for k, v in player_embeddings.items()}
    with open(output_path / "player_embeddings.json", "w") as f:
        json.dump(emb_dict, f)
    
    # Save team styles
    style_dict = {k: v.tolist() for k, v in team_styles.items()}
    with open(output_path / "team_styles.json", "w") as f:
        json.dump(style_dict, f)
    
    logger.info(f"Saved synthetic data to {output_path}")


def load_processed_data(data_dir: str = "data/processed") -> Tuple[List[dict], dict, dict]:
    """Load processed data from disk."""
    data_path = Path(data_dir)
    
    with open(data_path / "matches.json", "r") as f:
        matches = json.load(f)
    
    with open(data_path / "player_embeddings.json", "r") as f:
        emb_dict = json.load(f)
        player_embeddings = {
            int(k): torch.tensor(v, dtype=torch.float32) 
            for k, v in emb_dict.items()
        }
    
    with open(data_path / "team_styles.json", "r") as f:
        style_dict = json.load(f)
        team_styles = {
            k: torch.tensor(v, dtype=torch.float32) 
            for k, v in style_dict.items()
        }
    
    logger.info(f"Loaded {len(matches)} matches from {data_path}")
    return matches, player_embeddings, team_styles


if __name__ == "__main__":
    # Generate synthetic data for testing
    save_synthetic_data()
