"""
End-to-End Pipeline Orchestrator

Run the complete pipeline with:
    python src/pipeline.py

Or run individual stages:
    python src/pipeline.py --stage data
    python src/pipeline.py --stage train
    python src/pipeline.py --stage evaluate

Use real StatsBomb data:
    python src/pipeline.py --real-data
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

try:
    from statsbombpy import sb
    STATSBOMB_AVAILABLE = True
except ImportError:
    sb = None
    STATSBOMB_AVAILABLE = False

from dataset import (
    create_dataloaders,
    generate_synthetic_data,
    load_processed_data,
    save_synthetic_data,
)
from modules import PlayerLSTM, StyleAutoencoder, TacticalNet
from train import run_backtest_fold, pretrain_autoencoder, pretrain_lstm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def get_device(config: dict) -> str:
    """Determine the device to use for training."""
    device_setting = config["training"]["device"]
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


# ─────────────────────────────────────────────
# StatsBomb Data Fetching
# ─────────────────────────────────────────────

def fetch_statsbomb_competitions() -> pd.DataFrame:
    """Fetch available competitions from StatsBomb."""
    if not STATSBOMB_AVAILABLE:
        raise RuntimeError("statsbombpy is required. Install with: pip install statsbombpy")
    return sb.competitions()


def fetch_statsbomb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fetch all matches for a competition/season."""
    if not STATSBOMB_AVAILABLE:
        raise RuntimeError("statsbombpy is required.")
    return sb.matches(competition_id=competition_id, season_id=season_id)


def fetch_statsbomb_lineups(match_id: int) -> dict:
    """Fetch lineups for a match."""
    if not STATSBOMB_AVAILABLE:
        raise RuntimeError("statsbombpy is required.")
    return sb.lineups(match_id=match_id)


def fetch_statsbomb_events(match_id: int) -> pd.DataFrame:
    """Fetch all events for a match."""
    if not STATSBOMB_AVAILABLE:
        raise RuntimeError("statsbombpy is required.")
    return sb.events(match_id=match_id)


# ─────────────────────────────────────────────
# Real Data Processing
# ─────────────────────────────────────────────

def extract_player_stats_from_events(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """
    Extract per-player statistics from StatsBomb events.
    
    Returns DataFrame with columns:
    - player_id, player_name, team
    - xG, shots, passes, progressive_passes
    - tackles, interceptions, pressures
    """
    stats = []
    
    # Get unique players
    players = events[['player_id', 'player', 'team']].dropna().drop_duplicates()
    
    for _, player_row in players.iterrows():
        player_id = player_row['player_id']
        player_name = player_row['player']
        team = player_row['team']
        
        player_events = events[events['player_id'] == player_id]
        
        # xG from shots
        shots = player_events[player_events['type'] == 'Shot']
        xg = 0.0
        if len(shots) > 0 and 'shot_statsbomb_xg' in shots.columns:
            xg = shots['shot_statsbomb_xg'].sum()
        elif len(shots) > 0:
            # Try to extract from nested shot column
            for _, shot in shots.iterrows():
                if isinstance(shot.get('shot'), dict):
                    xg += shot['shot'].get('statsbomb_xg', 0.0)
        
        # Passes
        passes = player_events[player_events['type'] == 'Pass']
        total_passes = len(passes)
        
        # Progressive passes (moves ball significantly toward goal)
        progressive_passes = 0
        if len(passes) > 0:
            for _, p in passes.iterrows():
                pass_data = p.get('pass', {})
                if isinstance(pass_data, dict):
                    length = pass_data.get('length', 0)
                    if length >= 10:  # 10+ meters
                        progressive_passes += 1
        
        # Defensive actions
        tackles = len(player_events[player_events['type'] == 'Tackle'])
        interceptions = len(player_events[player_events['type'] == 'Interception'])
        pressures = len(player_events[player_events['type'] == 'Pressure'])
        
        # Dribbles and carries
        dribbles = len(player_events[player_events['type'] == 'Dribble'])
        carries = len(player_events[player_events['type'] == 'Carry'])
        
        stats.append({
            'match_id': int(match_id),
            'player_id': int(player_id) if pd.notna(player_id) else 0,
            'player_name': str(player_name),
            'team': str(team),
            'xG': float(xg),
            'shots': int(len(shots)),
            'passes': int(total_passes),
            'progressive_passes': int(progressive_passes),
            'tackles': int(tackles),
            'interceptions': int(interceptions),
            'pressures': int(pressures),
            'dribbles': int(dribbles),
            'carries': int(carries),
        })
    
    return pd.DataFrame(stats)


def compute_team_style_from_events(events: pd.DataFrame, team_name: str) -> List[float]:
    """
    Compute team playing style vector from match events.
    
    Returns 4-dim style vector:
    - [0] Possession tendency (0-1)
    - [1] Pressing intensity (0-1) 
    - [2] Directness (0-1)
    - [3] Width of play (0-1)
    """
    team_events = events[events['team'] == team_name]
    
    if len(team_events) == 0:
        return [0.5, 0.5, 0.5, 0.5]
    
    # Possession: ratio of passes to total actions
    passes = len(team_events[team_events['type'] == 'Pass'])
    total_actions = len(team_events)
    possession = min(passes / max(total_actions, 1), 1.0)
    
    # Pressing: pressures per opponent action
    pressures = len(team_events[team_events['type'] == 'Pressure'])
    pressing_intensity = min(pressures / 50, 1.0)  # Normalize to ~50 pressures per game
    
    # Directness: long passes / total passes
    long_passes = 0
    pass_events = team_events[team_events['type'] == 'Pass']
    for _, p in pass_events.iterrows():
        pass_data = p.get('pass', {})
        if isinstance(pass_data, dict) and pass_data.get('length', 0) > 30:
            long_passes += 1
    directness = min(long_passes / max(passes, 1), 1.0)
    
    # Width: crosses and wide passes
    crosses = 0
    for _, p in pass_events.iterrows():
        pass_data = p.get('pass', {})
        if isinstance(pass_data, dict) and pass_data.get('cross', False):
            crosses += 1
    width = min(crosses / max(passes, 1) * 10, 1.0)  # Scale up crosses
    
    return [float(possession), float(pressing_intensity), float(directness), float(width)]


def create_player_embeddings(
    player_stats: pd.DataFrame,
    feature_dim: int = 64,
) -> Dict[int, torch.Tensor]:
    """
    Create player embeddings from aggregated statistics.
    
    Uses a simple approach: normalize stats and project to embedding space.
    In production, this would use the pre-trained LSTM on sequences.
    """
    embeddings = {}
    
    # Aggregate stats per player across all matches
    agg_stats = player_stats.groupby('player_id').agg({
        'xG': 'sum',
        'shots': 'sum',
        'passes': 'sum',
        'progressive_passes': 'sum',
        'tackles': 'sum',
        'interceptions': 'sum',
        'pressures': 'sum',
        'dribbles': 'sum',
        'carries': 'sum',
    }).reset_index()
    
    # Normalize each stat column
    stat_cols = ['xG', 'shots', 'passes', 'progressive_passes', 
                 'tackles', 'interceptions', 'pressures', 'dribbles', 'carries']
    
    for col in stat_cols:
        max_val = agg_stats[col].max()
        if max_val > 0:
            agg_stats[col] = agg_stats[col] / max_val
    
    # Create embeddings
    rng = np.random.default_rng(42)
    
    for _, row in agg_stats.iterrows():
        player_id = int(row['player_id'])
        
        # Base embedding from stats (9 features)
        base_features = np.array([row[col] for col in stat_cols], dtype=np.float32)
        
        # Project to higher dimension with some randomness for diversity
        # This simulates what an LSTM would learn
        projection = rng.normal(0, 0.1, size=(len(stat_cols), feature_dim)).astype(np.float32)
        embedding = np.tanh(base_features @ projection + rng.normal(0, 0.1, size=feature_dim))
        
        embeddings[player_id] = torch.tensor(embedding, dtype=torch.float32)
    
    return embeddings


def process_real_statsbomb_data(
    config: dict,
    output_dir: Path,
) -> Tuple[List[dict], Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Fetch and process real data from StatsBomb free API.
    
    Returns:
        Tuple of (matches, player_embeddings, team_styles)
    """
    if not STATSBOMB_AVAILABLE:
        raise RuntimeError(
            "statsbombpy is required for real data. "
            "Install with: pip install statsbombpy"
        )
    
    logger.info("Fetching real data from StatsBomb API...")
    
    all_matches = []
    all_player_stats = []
    team_events_agg = defaultdict(list)  # team -> list of event DataFrames
    
    competition_id = config["competitions"]["fifa_world_cup"]
    seasons = config["seasons"]
    
    for season_name, season_id in seasons.items():
        logger.info(f"\nProcessing {season_name} (season_id={season_id})...")
        
        try:
            matches_df = fetch_statsbomb_matches(competition_id, season_id)
            logger.info(f"  Found {len(matches_df)} matches")
            
            # Extract year from season name (e.g., "wc_2022" -> 2022)
            year = int(season_name.split('_')[1]) if '_' in season_name else 2020
            
            for idx, match_row in tqdm(matches_df.iterrows(), total=len(matches_df), desc=f"  {season_name}"):
                match_id = match_row['match_id']
                
                try:
                    # Fetch events
                    events = fetch_statsbomb_events(match_id)
                    
                    if events is None or len(events) == 0:
                        continue
                    
                    # Fetch lineups to get player IDs
                    lineups = fetch_statsbomb_lineups(match_id)
                    
                    home_team = str(match_row['home_team'])
                    away_team = str(match_row['away_team'])
                    
                    # Get starting XI player IDs
                    home_players = []
                    away_players = []
                    
                    if home_team in lineups:
                        home_lineup = lineups[home_team]
                        if isinstance(home_lineup, pd.DataFrame) and 'player_id' in home_lineup.columns:
                            home_players = home_lineup['player_id'].head(11).tolist()
                    
                    if away_team in lineups:
                        away_lineup = lineups[away_team]
                        if isinstance(away_lineup, pd.DataFrame) and 'player_id' in away_lineup.columns:
                            away_players = away_lineup['player_id'].head(11).tolist()
                    
                    # Extract player stats
                    player_stats = extract_player_stats_from_events(events, match_id)
                    player_stats['year'] = year
                    all_player_stats.append(player_stats)
                    
                    # Store events for team style computation
                    team_events_agg[home_team].append(events[events['team'] == home_team])
                    team_events_agg[away_team].append(events[events['team'] == away_team])
                    
                    # Create match record
                    all_matches.append({
                        'match_id': int(match_id),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_players': [int(p) for p in home_players if pd.notna(p)],
                        'away_players': [int(p) for p in away_players if pd.notna(p)],
                        'home_score': int(match_row['home_score']),
                        'away_score': int(match_row['away_score']),
                        'year': year,
                        'match_date': str(match_row.get('match_date', '')),
                        'competition': season_name,
                    })
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"  Failed to process match {match_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to fetch {season_name}: {e}")
            continue
    
    if not all_matches:
        raise RuntimeError("No matches were successfully processed")
    
    logger.info(f"\nProcessed {len(all_matches)} matches total")
    
    # Combine all player stats
    combined_stats = pd.concat(all_player_stats, ignore_index=True)
    logger.info(f"Collected stats for {combined_stats['player_id'].nunique()} unique players")
    
    # Save raw stats
    combined_stats.to_csv(output_dir / "player_stats.csv", index=False)
    
    # Create player embeddings
    logger.info("Creating player embeddings...")
    player_embeddings = create_player_embeddings(
        combined_stats, 
        feature_dim=config["model"]["player_feature_dim"]
    )
    logger.info(f"Created embeddings for {len(player_embeddings)} players")
    
    # Compute team styles
    logger.info("Computing team style vectors...")
    team_styles = {}
    for team_name, events_list in team_events_agg.items():
        if events_list:
            # Combine all events for this team
            combined_events = pd.concat(events_list, ignore_index=True)
            style_vector = compute_team_style_from_events(combined_events, team_name)
            team_styles[team_name] = torch.tensor(style_vector, dtype=torch.float32)
    logger.info(f"Computed styles for {len(team_styles)} teams")
    
    return all_matches, player_embeddings, team_styles


def save_real_data(
    matches: List[dict],
    player_embeddings: Dict[int, torch.Tensor],
    team_styles: Dict[str, torch.Tensor],
    output_dir: str = "data/processed",
) -> None:
    """Save processed real data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    
    logger.info(f"Saved real data to {output_path}")


# ─────────────────────────────────────────────
# Stage 1: Data Collection & Processing
# ─────────────────────────────────────────────

def run_data_stage(config: dict, use_synthetic: bool = True) -> None:
    """
    Run the data collection and processing stage.
    
    Args:
        config: Configuration dictionary
        use_synthetic: If True, generate synthetic data for demo.
                      If False, fetch real data from StatsBomb.
    """
    logger.info("=" * 50)
    logger.info("STAGE 1: Data Collection & Processing")
    logger.info("=" * 50)
    
    output_dir = Path(config["paths"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if use_synthetic:
        logger.info("Generating synthetic data for demo...")
        save_synthetic_data(str(output_dir))
        return
    
    # Real data collection from StatsBomb
    try:
        matches, player_embeddings, team_styles = process_real_statsbomb_data(
            config, output_dir
        )
        save_real_data(matches, player_embeddings, team_styles, str(output_dir))
        
    except Exception as e:
        logger.error(f"Failed to fetch real data: {e}")
        logger.warning("Falling back to synthetic data...")
        save_synthetic_data(str(output_dir))


# ─────────────────────────────────────────────
# Stage 2: Model Training
# ─────────────────────────────────────────────

def run_train_stage(config: dict) -> dict:
    """
    Run the model training stage with time-series backtesting.
    
    Returns:
        Dictionary with training results for each fold.
    """
    logger.info("=" * 50)
    logger.info("STAGE 2: Model Training")
    logger.info("=" * 50)
    
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Load processed data
    try:
        matches, player_embeddings, team_styles = load_processed_data(
            config["paths"]["processed_dir"]
        )
    except FileNotFoundError:
        logger.warning("Processed data not found. Generating synthetic data...")
        save_synthetic_data(config["paths"]["processed_dir"])
        matches, player_embeddings, team_styles = load_processed_data(
            config["paths"]["processed_dir"]
        )
    
    results = {}
    model_config = config["model"]
    train_config = config["training"]
    
    # Fold 1: Train on pre-2018, test on 2018
    logger.info("\n--- Fold 1: Train ≤2017, Test 2018 ---")
    train_matches_f1 = [m for m in matches if m.get("year", 2020) <= 2017]
    test_matches_f1 = [m for m in matches if m.get("year", 2020) == 2018]
    
    if len(train_matches_f1) < 10 or len(test_matches_f1) < 5:
        # Fallback: random split for insufficient temporal data
        logger.info("Insufficient temporal split, using 80/20 random split")
        matches_copy = matches.copy()
        np.random.shuffle(matches_copy)
        split_idx = int(len(matches_copy) * 0.8)
        train_matches_f1 = matches_copy[:split_idx]
        test_matches_f1 = matches_copy[split_idx:]
    
    train_loader_f1, test_loader_f1 = create_dataloaders(
        train_matches=train_matches_f1,
        val_matches=test_matches_f1,
        player_embeddings=player_embeddings,
        team_styles=team_styles,
        batch_size=train_config["batch_size"],
        feature_dim=model_config["player_feature_dim"],
    )
    
    results["fold1"] = run_backtest_fold(
        train_loader=train_loader_f1,
        test_loader=test_loader_f1,
        fold_name="fold1",
        epochs=train_config["epochs"],
        lr=train_config["learning_rate"],
        device=device,
        player_feature_dim=model_config["player_feature_dim"],
        hidden_dim=model_config["hidden_dim"],
        style_latent_dim=model_config["style_latent_dim"],
        gnn_type=model_config["gnn_type"],
    )
    
    # Fold 2: Train on pre-2022, test on 2022
    logger.info("\n--- Fold 2: Train ≤2021, Test 2022 ---")
    train_matches_f2 = [m for m in matches if m.get("year", 2020) <= 2021]
    test_matches_f2 = [m for m in matches if m.get("year", 2020) == 2022]
    
    if len(train_matches_f2) < 10 or len(test_matches_f2) < 5:
        # Use same split as fold1 for insufficient data
        train_matches_f2 = train_matches_f1
        test_matches_f2 = test_matches_f1
    
    train_loader_f2, test_loader_f2 = create_dataloaders(
        train_matches=train_matches_f2,
        val_matches=test_matches_f2,
        player_embeddings=player_embeddings,
        team_styles=team_styles,
        batch_size=train_config["batch_size"],
        feature_dim=model_config["player_feature_dim"],
    )
    
    results["fold2"] = run_backtest_fold(
        train_loader=train_loader_f2,
        test_loader=test_loader_f2,
        fold_name="fold2",
        epochs=train_config["epochs"],
        lr=train_config["learning_rate"],
        device=device,
        player_feature_dim=model_config["player_feature_dim"],
        hidden_dim=model_config["hidden_dim"],
        style_latent_dim=model_config["style_latent_dim"],
        gnn_type=model_config["gnn_type"],
    )
    
    # Save results
    results_path = Path(config["paths"]["weights_dir"]) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Training results saved to {results_path}")
    
    return results


# ─────────────────────────────────────────────
# Stage 3: Evaluation & Summary
# ─────────────────────────────────────────────

def run_evaluate_stage(config: dict) -> None:
    """
    Run evaluation and print summary of model performance.
    """
    logger.info("=" * 50)
    logger.info("STAGE 3: Evaluation Summary")
    logger.info("=" * 50)
    
    results_path = Path(config["paths"]["weights_dir"]) / "training_results.json"
    
    if not results_path.exists():
        logger.error("No training results found. Run training first.")
        return
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    for fold_name, metrics in results.items():
        print(f"\n{fold_name.upper()}:")
        print(f"  Accuracy:    {metrics['accuracy']:.1%}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"  Loss:        {metrics['loss']:.4f}")
    
    # Average metrics
    avg_acc = np.mean([m["accuracy"] for m in results.values()])
    avg_brier = np.mean([m["brier_score"] for m in results.values()])
    
    print(f"\nAVERAGE:")
    print(f"  Accuracy:    {avg_acc:.1%}")
    print(f"  Brier Score: {avg_brier:.4f}")
    print("=" * 50)
    
    # Check for trained weights
    weights_dir = Path(config["paths"]["weights_dir"])
    checkpoints = list(weights_dir.glob("tactical_net_*.pt"))
    
    if checkpoints:
        print(f"\nTrained model checkpoints:")
        for cp in checkpoints:
            print(f"  - {cp}")
        print(f"\nRun the dashboard with: streamlit run src/app.py")
    else:
        print("\nNo model checkpoints found.")
    
    # Show data summary
    processed_dir = Path(config["paths"]["processed_dir"])
    matches_file = processed_dir / "matches.json"
    if matches_file.exists():
        with open(matches_file, "r") as f:
            matches = json.load(f)
        
        # Count matches by year
        year_counts = defaultdict(int)
        for m in matches:
            year_counts[m.get('year', 'unknown')] += 1
        
        print(f"\nData Summary:")
        print(f"  Total matches: {len(matches)}")
        print(f"  Matches by year:")
        for year in sorted(year_counts.keys()):
            print(f"    {year}: {year_counts[year]}")


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Predictor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py                    # Run full pipeline with synthetic data
  python src/pipeline.py --real-data        # Run full pipeline with real StatsBomb data
  python src/pipeline.py --stage data       # Only run data stage
  python src/pipeline.py --stage train      # Only run training
  python src/pipeline.py --stage evaluate   # Only run evaluation
  
  # Fetch real data, then train
  python src/pipeline.py --stage data --real-data
  python src/pipeline.py --stage train
        """
    )
    parser.add_argument(
        "--stage",
        choices=["data", "train", "evaluate", "all"],
        default="all",
        help="Pipeline stage to run (default: all)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real StatsBomb data instead of synthetic"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    for path_key in ["data_dir", "raw_dir", "processed_dir", "video_dir", "weights_dir"]:
        Path(config["paths"][path_key]).mkdir(parents=True, exist_ok=True)
    
    logger.info("World Cup 2026 Predictor Pipeline")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data source: {'Real (StatsBomb)' if args.real_data else 'Synthetic'}")
    
    if args.real_data and not STATSBOMB_AVAILABLE:
        logger.error("statsbombpy is not installed. Install with: pip install statsbombpy")
        logger.info("Falling back to synthetic data.")
        args.real_data = False
    
    # Run requested stage(s)
    if args.stage in ["data", "all"]:
        run_data_stage(config, use_synthetic=not args.real_data)
    
    if args.stage in ["train", "all"]:
        run_train_stage(config)
    
    if args.stage in ["evaluate", "all"]:
        run_evaluate_stage(config)
    
    logger.info("\nPipeline complete!")


if __name__ == "__main__":
    main()
