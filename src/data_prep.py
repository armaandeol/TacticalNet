"""
Phase 1: Data Engineering & Multimodal Ingestion

Handles:
  - StatsBomb event data ingestion
  - FBref scraping for player/team stats
  - LSTM sequence generation (rolling 20-match windows)
  - YOLOv8 spatial coordinate extraction from video
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from statsbombpy import sb
except ImportError:
    sb = None
    logging.warning("statsbombpy not installed. StatsBomb ingestion disabled.")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics not installed. Vision pipeline disabled.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VIDEO_DIR = DATA_DIR / "video"

for d in [RAW_DIR, PROCESSED_DIR, VIDEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# StatsBomb Ingestion
# ─────────────────────────────────────────────

INTERNATIONAL_COMPETITION_IDS = {
    "FIFA World Cup": 43,
    "UEFA Euro": 55,
    "Copa America": 223,
    "AFC Asian Cup": 1,
    "CONMEBOL WC Qualifiers": 68,
    "UEFA WC Qualifiers": 67,
}


def fetch_statsbomb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fetch all matches for a given StatsBomb competition and season."""
    if sb is None:
        raise RuntimeError("statsbombpy is required for this function.")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    logger.info(f"Fetched {len(matches)} matches for competition {competition_id}, season {season_id}")
    return matches


def fetch_statsbomb_events(match_id: int) -> pd.DataFrame:
    """Fetch all events for a single match."""
    if sb is None:
        raise RuntimeError("statsbombpy is required for this function.")
    events = sb.events(match_id=match_id)
    return events


def build_player_match_stats(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """
    Aggregate per-player stats from raw StatsBomb events for one match.
    Returns a DataFrame with columns: player_id, xG, progressive_passes,
    defensive_interventions, distance_covered.
    """
    # xG: sum of shot statsbomb_xg per player
    shots = events[events["type"] == "Shot"].copy()
    if "shot" in shots.columns:
        shots["xg"] = shots["shot"].apply(
            lambda s: s.get("statsbomb_xg", 0.0) if isinstance(s, dict) else 0.0
        )
        xg_per_player = shots.groupby("player_id")["xg"].sum().reset_index()
        xg_per_player.columns = ["player_id", "xG"]
    else:
        xg_per_player = pd.DataFrame(columns=["player_id", "xG"])

    # Progressive passes: passes that move the ball ≥10 yards toward goal
    passes = events[events["type"] == "Pass"].copy()
    if not passes.empty and "pass" in passes.columns:
        passes["is_progressive"] = passes["pass"].apply(
            lambda p: p.get("length", 0) >= 10 and not p.get("outcome") if isinstance(p, dict) else False
        )
        prog_passes = passes[passes["is_progressive"]].groupby("player_id").size().reset_index(name="progressive_passes")
    else:
        prog_passes = pd.DataFrame(columns=["player_id", "progressive_passes"])

    # Defensive interventions: tackles + interceptions + blocks
    def_types = ["Tackle", "Interception", "Block"]
    def_events = events[events["type"].isin(def_types)]
    def_interventions = def_events.groupby("player_id").size().reset_index(name="defensive_interventions")

    # Merge all stats
    all_players = events[["player_id", "player"]].dropna().drop_duplicates()
    df = all_players.merge(xg_per_player, on="player_id", how="left")
    df = df.merge(prog_passes, on="player_id", how="left")
    df = df.merge(def_interventions, on="player_id", how="left")
    
    # Fix FutureWarning: use infer_objects after fillna
    df = df.fillna(0.0)
    df = df.infer_objects(copy=False)
    
    df["match_id"] = match_id
    return df


def build_lstm_sequences(
    player_stats_df: pd.DataFrame,
    window: int = 20,
    feature_cols: Optional[list] = None,
) -> dict:
    """
    Build rolling 20-match sequences per player for LSTM input.

    Returns:
        dict mapping player_id -> np.ndarray of shape (num_windows, window, num_features)
    """
    if feature_cols is None:
        feature_cols = ["xG", "progressive_passes", "defensive_interventions"]

    sequences = {}
    for player_id, group in player_stats_df.groupby("player_id"):
        group = group.sort_values("match_date") if "match_date" in group.columns else group
        values = group[feature_cols].values.astype(np.float32)
        if len(values) < window:
            # Pad with zeros if fewer than window matches available
            pad = np.zeros((window - len(values), len(feature_cols)), dtype=np.float32)
            values = np.vstack([pad, values])
        windows = np.stack(
            [values[i : i + window] for i in range(len(values) - window + 1)]
        )
        sequences[player_id] = windows
    return sequences


# ─────────────────────────────────────────────
# FBref Scraper
# ─────────────────────────────────────────────

FBREF_BASE = "https://fbref.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WC2026Bot/1.0)"}


def scrape_fbref_team_stats(team_url: str, retries: int = 3) -> pd.DataFrame:
    """
    Scrape team-level stats table from an FBref team page.
    Returns a DataFrame of the first stats table found.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(team_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"class": "stats_table"})
            if table is None:
                logger.warning(f"No stats table found at {team_url}")
                return pd.DataFrame()
            df = pd.read_html(str(table))[0]
            logger.info(f"Scraped {len(df)} rows from {team_url}")
            return df
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {team_url}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def compute_team_style_features(team_df: pd.DataFrame) -> dict:
    """
    Compute team playing-style metrics from scraped FBref data.
    Returns a dict with keys: ppda, directness_index, field_tilt, crossing_frequency.
    """
    features = {}
    # PPDA: Passes Per Defensive Action (lower = more pressing)
    if "Passes" in team_df.columns and "Def Actions" in team_df.columns:
        total_passes = team_df["Passes"].sum()
        total_def = team_df["Def Actions"].sum()
        features["ppda"] = float(total_passes / total_def) if total_def > 0 else 0.0
    else:
        features["ppda"] = 0.0

    # Directness index: forward passes / total passes
    if "Fwd" in team_df.columns and "Passes" in team_df.columns:
        features["directness_index"] = float(
            team_df["Fwd"].sum() / max(team_df["Passes"].sum(), 1)
        )
    else:
        features["directness_index"] = 0.0

    # Field tilt: % of touches in opponent's final third
    if "Att 3rd" in team_df.columns and "Touches" in team_df.columns:
        features["field_tilt"] = float(
            team_df["Att 3rd"].sum() / max(team_df["Touches"].sum(), 1)
        )
    else:
        features["field_tilt"] = 0.0

    # Crossing frequency: crosses per match
    if "Crs" in team_df.columns:
        features["crossing_frequency"] = float(team_df["Crs"].mean())
    else:
        features["crossing_frequency"] = 0.0

    return features


# ─────────────────────────────────────────────
# YOLOv8 Vision Pipeline
# ─────────────────────────────────────────────

YOLO_MODEL_PATH = "weights/yolov8_soccernet.pt"  # Fine-tuned checkpoint


def extract_spatial_coordinates(
    video_path: str,
    output_csv: str,
    model_path: str = YOLO_MODEL_PATH,
    conf_threshold: float = 0.4,
    frame_skip: int = 5,
) -> pd.DataFrame:
    """
    Run YOLOv8 on a video file and extract 2D (x, y) coordinates
    for detected players and balls per frame.

    Args:
        video_path:     Path to input video file.
        output_csv:     Where to save the coordinate CSV.
        model_path:     Path to fine-tuned YOLOv8 weights.
        conf_threshold: Minimum detection confidence.
        frame_skip:     Process every Nth frame to reduce compute.

    Returns:
        DataFrame with columns: frame, class, track_id, x_center, y_center, confidence
    """
    if YOLO is None:
        raise RuntimeError("ultralytics package is required for vision pipeline.")

    model = YOLO(model_path)
    records = []

    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    if cls_name not in ("person", "sports ball"):
                        continue
                    x_c, y_c = box.xywh[0][:2].tolist()
                    track_id = int(box.id[0]) if box.id is not None else -1
                    records.append({
                        "frame": frame_idx,
                        "class": cls_name,
                        "track_id": track_id,
                        "x_center": round(x_c, 2),
                        "y_center": round(y_c, 2),
                        "confidence": round(float(box.conf[0]), 3),
                    })
        frame_idx += 1

    cap.release()
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} detections to {output_csv}")
    return df


if __name__ == "__main__":
    logger.info("Data preparation pipeline ready. Run individual functions as needed.")
