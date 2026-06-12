# World Cup 2026 Match Outcome Predictor

A multimodal deep learning system combining Graph Neural Networks, LSTMs, and computer vision to predict FIFA World Cup 2026 match outcomes.

## Quick Start

```bash
# 1. Install dependencies
make setup

# 2. Run full pipeline (data → train → evaluate → app)
make quickstart
```

Or step by step:

```bash
pip install -r requirements.txt
python src/pipeline.py          # Runs full pipeline
streamlit run src/app.py        # Launch dashboard
```

## Architecture

```
[Broadcast Video / Stream] ──> [YOLOv8 Object Detection] ──> [Spatial Coordinate Streams] ┐
                                                                                           ├─> [GNN] ──> [Match Outcome MLP]
[FBref / StatsBomb Data]   ──> [LSTM Sequence Encoder]   ──> [Dynamic Form Embeddings]   ┘
```

## Project Structure

```
├── config.yaml           # Centralized configuration
├── Makefile              # Simplified commands
├── .gitlab-ci.yml        # CI/CD pipeline
├── data/
│   ├── raw/              # Scraped JSONs from StatsBomb & FBref
│   ├── processed/        # Cleaned sequences and graph objects
│   └── video/            # Sample spatial coordinate CSVs from YOLOv8
├── src/
│   ├── __init__.py       # Package exports
│   ├── data_prep.py      # FBref scraper and LSTM sequence generator
│   ├── dataset.py        # PyTorch Dataset and DataLoader utilities
│   ├── modules.py        # LSTM, Autoencoder, and GNN model classes
│   ├── pipeline.py       # End-to-end orchestration script
│   ├── train.py          # Training loop and backtesting validation
│   └── app.py            # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb
├── weights/              # Saved model checkpoints
└── requirements.txt
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies |
| `make data` | Generate synthetic data |
| `make train` | Train models with backtesting |
| `make evaluate` | Show model performance |
| `make app` | Run Streamlit dashboard |
| `make all` | Run full pipeline |
| `make quickstart` | Setup + full pipeline + app |
| `make demo` | Quick demo with synthetic data |
| `make clean` | Remove generated files |

### Using Real Data

```bash
# Fetch real data from StatsBomb API
make DATA_SOURCE=real data

# Or via pipeline script
python src/pipeline.py --stage data --real-data
```

## Pipeline Stages

### Stage 1: Data Collection

```bash
python src/pipeline.py --stage data
```

- Generates synthetic match data for demo (default)
- Or fetches real data from StatsBomb free API (`--real-data`)
- Creates player embeddings and team style vectors

### Stage 2: Model Training

```bash
python src/pipeline.py --stage train
```

Time-series backtesting to prevent data leakage:
- **Fold 1:** Train 2010–2017 → Validate on 2018 World Cup
- **Fold 2:** Train 2010–2018 → Validate on 2022 World Cup
- **Final:** Train on all data → Predict 2026 World Cup

### Stage 3: Evaluation

```bash
python src/pipeline.py --stage evaluate
```

Displays model performance metrics:
- Accuracy
- Brier Score (calibration)
- Cross-entropy loss

### Stage 4: Dashboard

```bash
streamlit run src/app.py
```

Interactive simulator where you can:
- Select teams and tactical setups
- Adjust form, pressing intensity, field tilt
- View predicted win/draw/loss probabilities

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  player_feature_dim: 64
  hidden_dim: 128
  gnn_type: "gcn"  # or "gat"

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
```

## CI/CD

The project includes a GitLab CI/CD pipeline (`.gitlab-ci.yml`) that:

1. **Lint:** Checks code style with flake8 and isort
2. **Test:** Validates module imports and model forward pass
3. **Build:** Runs full training pipeline on merge to main

## Model Components

| Component | Purpose |
|-----------|--------|
| `PlayerLSTM` | Encodes 20-match rolling performance into form embeddings |
| `StyleAutoencoder` | Compresses team style metrics (PPDA, directness, etc.) |
| `TeamGraphEncoder` | GCN/GAT that aggregates player nodes into team vector |
| `TacticalNet` | End-to-end predictor combining all components |

## License

MIT
