"""
End-to-End Pipeline Orchestrator

Run the complete pipeline with:
    python src/pipeline.py

Or run individual stages:
    python src/pipeline.py --stage data
    python src/pipeline.py --stage train
    python src/pipeline.py --stage evaluate
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from data_prep import (
    fetch_statsbomb_matches,
    fetch_statsbomb_events,
    build_player_match_stats,
    build_lstm_sequences,
    RAW_DIR,
    PROCESSED_DIR,
)
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
    
    if use_synthetic:
        logger.info("Generating synthetic data for demo...")
        save_synthetic_data(config["paths"]["processed_dir"])
        return
    
    # Real data collection from StatsBomb
    logger.info("Fetching data from StatsBomb API...")
    
    all_player_stats = []
    competition_id = config["competitions"]["fifa_world_cup"]
    
    for season_name, season_id in config["seasons"].items():
        logger.info(f"Processing {season_name} (season_id={season_id})...")
        
        try:
            matches = fetch_statsbomb_matches(competition_id, season_id)
            
            for _, match in matches.iterrows():
                match_id = match["match_id"]
                try:
                    events = fetch_statsbomb_events(match_id)
                    stats = build_player_match_stats(events, match_id)
                    stats["match_date"] = match.get("match_date", "")
                    stats["competition"] = season_name
                    all_player_stats.append(stats)
                except Exception as e:
                    logger.warning(f"Failed to process match {match_id}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to fetch {season_name}: {e}")
    
    if all_player_stats:
        import pandas as pd
        combined_stats = pd.concat(all_player_stats, ignore_index=True)
        combined_stats.to_csv(PROCESSED_DIR / "player_stats.csv", index=False)
        logger.info(f"Saved {len(combined_stats)} player-match records")
        
        # Build LSTM sequences
        sequences = build_lstm_sequences(combined_stats, window=config["training"]["sequence_window"])
        torch.save(sequences, PROCESSED_DIR / "lstm_sequences.pt")
        logger.info(f"Built LSTM sequences for {len(sequences)} players")
    else:
        logger.warning("No data collected. Using synthetic data as fallback.")
        save_synthetic_data(config["paths"]["processed_dir"])


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
        # Fallback: random split for synthetic data
        np.random.shuffle(matches)
        split_idx = int(len(matches) * 0.8)
        train_matches_f1 = matches[:split_idx]
        test_matches_f1 = matches[split_idx:]
    
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
    logger.info("\n--- Fold 2: Train ≤2018, Test 2022 ---")
    train_matches_f2 = [m for m in matches if m.get("year", 2020) <= 2018]
    test_matches_f2 = [m for m in matches if m.get("year", 2020) == 2022]
    
    if len(train_matches_f2) < 10 or len(test_matches_f2) < 5:
        # Use same split as fold1 for synthetic data
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


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Predictor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py                    # Run full pipeline
  python src/pipeline.py --stage data       # Only run data stage
  python src/pipeline.py --stage train      # Only run training
  python src/pipeline.py --stage evaluate   # Only run evaluation
  python src/pipeline.py --real-data        # Use real StatsBomb data
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
