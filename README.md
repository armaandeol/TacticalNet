# World Cup 2026 Match Outcome Predictor

A multimodal deep learning system combining Graph Neural Networks, LSTMs, and computer vision to predict FIFA World Cup 2026 match outcomes.

## Architecture

```
[Broadcast Video / Stream] ──> [YOLOv8 Object Detection] ──> [Spatial Coordinate Streams] ┐
                                                                                           ├─> [GNN] ──> [Match Outcome MLP]
[FBref / StatsBomb Data]   ──> [LSTM Sequence Encoder]   ──> [Dynamic Form Embeddings]   ┘
```

## Project Structure

```
├── data/
│   ├── raw/          # Scraped JSONs from StatsBomb & FBref
│   ├── processed/    # Cleaned sequences and graph objects
│   └── video/        # Sample spatial coordinate CSVs from YOLOv8
├── src/
│   ├── data_prep.py  # FBref scraper and LSTM sequence generator
│   ├── modules.py    # LSTM, Autoencoder, and GNN model classes
│   ├── train.py      # Training loop and backtesting validation
│   └── app.py        # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb
├── weights/          # Saved model checkpoints
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/train.py
streamlit run src/app.py
```

## Validation Strategy

Time-series backtesting to prevent data leakage:
- Train: 2010–2017 → Validate: 2018 World Cup
- Retrain: 2010–2018 → Validate: 2022 World Cup
- Final model trained on all data → Predict: 2026 World Cup
