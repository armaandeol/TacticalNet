# World Cup 2026 Predictor - Makefile
# ===================================
#
# Quick commands:
#   make setup    - Install dependencies
#   make data     - Generate/fetch data
#   make train    - Train models
#   make app      - Run Streamlit dashboard
#   make all      - Run full pipeline
#

.PHONY: setup data train evaluate app all clean test lint help

PYTHON := python
PIP := pip
STREAMLIT := streamlit

# Default target
.DEFAULT_GOAL := help

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

setup: ## Install all dependencies
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

setup-dev: setup ## Install dev dependencies
	$(PIP) install pytest pytest-cov black flake8 isort
	@echo "✅ Dev dependencies installed"

# ─────────────────────────────────────────────
# Pipeline Stages
# ─────────────────────────────────────────────

data: ## Generate synthetic data (or fetch real data with DATA_SOURCE=real)
ifeq ($(DATA_SOURCE),real)
	$(PYTHON) src/pipeline.py --stage data --real-data
else
	$(PYTHON) src/pipeline.py --stage data
endif
	@echo "✅ Data stage complete"

train: ## Train models with time-series backtesting
	$(PYTHON) src/pipeline.py --stage train
	@echo "✅ Training complete"

evaluate: ## Evaluate model performance
	$(PYTHON) src/pipeline.py --stage evaluate
	@echo "✅ Evaluation complete"

all: ## Run full pipeline (data → train → evaluate)
	$(PYTHON) src/pipeline.py --stage all
	@echo "✅ Full pipeline complete"

# ─────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────

app: ## Run Streamlit dashboard
	$(STREAMLIT) run src/app.py

app-debug: ## Run Streamlit with debug logging
	STREAMLIT_LOG_LEVEL=debug $(STREAMLIT) run src/app.py

# ─────────────────────────────────────────────
# Development
# ─────────────────────────────────────────────

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint: ## Run linters
	flake8 src/ --max-line-length=120 --ignore=E501
	isort --check-only src/
	@echo "✅ Linting passed"

format: ## Format code
	black src/ --line-length=120
	isort src/
	@echo "✅ Code formatted"

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

clean: ## Remove generated files and caches
	rm -rf __pycache__ src/__pycache__ .pytest_cache
	rm -rf data/processed/*.json data/processed/*.csv data/processed/*.pt
	rm -rf weights/*.pt weights/*.json
	rm -rf htmlcov .coverage
	rm -f pipeline.log
	@echo "✅ Cleaned"

clean-all: clean ## Remove all generated files including data
	rm -rf data/raw/* data/video/*
	@echo "✅ Deep clean complete"

notebook: ## Launch Jupyter notebook
	jupyter notebook notebooks/

# ─────────────────────────────────────────────
# Quick Start
# ─────────────────────────────────────────────

quickstart: setup data train app ## Full quickstart: setup → data → train → app
	@echo "🚀 Quickstart complete!"

demo: data app ## Quick demo: generate data and run app
	@echo "🎮 Demo mode"

# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────

help: ## Show this help message
	@echo "World Cup 2026 Predictor - Available Commands"
	@echo "============================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make quickstart          # Full setup and run"
	@echo "  make demo                # Quick demo with synthetic data"
	@echo "  make DATA_SOURCE=real data  # Fetch real StatsBomb data"
