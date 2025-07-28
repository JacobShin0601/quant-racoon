# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running Strategies
```bash
# New unified pipeline (recommended)
./run_pipeline.sh                                    # Default: swing strategy
./run_pipeline.sh --time-horizon long                # Long-term strategy
./run_pipeline.sh --use-cached-data                  # Use cached data
./run_pipeline.sh --use-cached-data --cache-days 7  # Use cache valid for 7 days
./run_pipeline.sh --no-research                      # Skip research phase
./run_pipeline.sh --stages cleaner,scrapper          # Run specific stages only

# Individual strategies with research pipeline
./run_swing.sh                      # Swing trading (15-min bars, 60 days)
./run_swing.sh --use-cached-data    # Use cached data
./run_long.sh                       # Long-term (daily bars, 3 years)
./run_scalping.sh                   # Scalping (1-min bars, 7 days)
./run_ensemble.sh                   # Ensemble strategy with market regime detection

# Run without research phase
./run_swing_no_research.sh
python -m src.agent.orchestrator --time-horizon swing --no-research

# Run all strategies
./run_all_strategies.sh

# Clean directories
./run_clean_all.sh
```

### Data Management
```bash
# Unified data manager
python -m src.agent.data_manager --data-type stock --symbols AAPL MSFT --time-horizon swing
python -m src.agent.data_manager --data-type macro --time-horizon trader
python -m src.agent.data_manager --data-type all --use-cached-data --cache-days 3

# Direct orchestrator with cache options
python -m src.agent.orchestrator --time-horizon swing --use-cached-data --cache-days 1
```

### Testing & Validation
```bash
# Market sensor validation
./run_validation.sh                                    # Default comprehensive validation
./run_validation.sh comprehensive 2022-01-01 2022-12-31  # Specific period
./run_validation.sh backtest                          # Backtest validation
./run_validation.sh both                              # Both validation types

# Quick analysis
./quick_analysis.sh

# View results
python view_ensemble_results.py --uuid <UUID> --detailed
python view_swing_results.py
```

### Python Module Execution
```bash
# Orchestrator with time horizons
python -m src.agent.orchestrator --time-horizon swing
python -m src.agent.orchestrator --time-horizon long
python -m src.agent.orchestrator --time-horizon scalping

# Individual modules
python -m src.agent.cleaner --action clean-and-recreate
python -m src.agent.scrapper
python -m src.agent.researcher
python -m src.agent.evaluator
python -m src.agent.portfolio_manager --method sharpe_maximization --compare

# Market analysis
python -m src.actions.ensemble --config config/config_ensemble.json
python train_market_model.py --data-dir data/macro
```

## High-level Architecture

### Pipeline Flow
The system follows a multi-stage pipeline orchestrated by `src/agent/orchestrator.py`:

1. **Cleaner** → 2. **Scrapper** → 3. **Analyzer** → 4. **Researcher** → 5. **Evaluator** → 6. **Portfolio Manager**

Each stage can be run independently or as part of the full pipeline. The `enable_research` flag in config allows skipping the research phase.

### Core Components

#### Agent Layer (`src/agent/`)
- **orchestrator.py**: Controls the entire pipeline flow, manages configuration and stage execution
- **cleaner.py**: Manages directory structure, cleanup, and backup operations
- **data_manager.py**: Unified data management system with caching and deduplication
- **scrapper.py**: Collects market data from various sources (Yahoo Finance, Alpha Vantage, Finnhub)
- **researcher.py**: Performs hyperparameter optimization using Bayesian optimization
- **evaluator.py**: Backtests strategies and evaluates performance metrics
- **portfolio_manager.py**: Implements portfolio optimization methods (Sharpe, Sortino, Risk Parity, etc.)
- **market_sensor.py**: Detects market regimes using machine learning models

#### Actions Layer (`src/actions/`)
- **strategies.py**: Contains all trading strategy implementations
- **ensemble.py**: Manages ensemble strategy execution with market regime adaptation
- **backtest_strategies.py**: Core backtesting engine
- **portfolio_optimization.py**: Advanced portfolio optimization algorithms
- **hmm_regime_classifier.py**: Hidden Markov Model for regime detection
- **neural_stock_predictor.py**: Neural network models for price prediction
- **y_finance.py**, **alpha_vantage.py**, **finnhub.py**: Data collection interfaces

### Configuration System
- Time-horizon specific configs: `config_swing.json`, `config_long.json`, `config_scalping.json`
- Special configs: `config_ensemble.json`, `config_research.json`
- Config structure defines pipeline stages, data parameters, strategy settings, and optimization parameters

### Data Flow
- Raw data collected → `data/{time_horizon}/`
- Processed results → `results/{time_horizon}/`
- Logs → `log/{time_horizon}/`
- Backups → `backup/{time_horizon}/`
- Models → `models/market_regime/`

### Key Features

#### Market Regime Detection
The system uses Random Forest classifiers to detect four market regimes:
- TRENDING_UP: Upward trend strategies
- TRENDING_DOWN: Downward trend strategies
- VOLATILE: High volatility strategies
- SIDEWAYS: Range-bound strategies

#### Strategy Optimization
- Bayesian optimization for hyperparameter tuning
- Walk-forward analysis for robust testing
- Individual stock optimization + portfolio-level evaluation

#### Portfolio Management
Multiple optimization methods:
- Equal Weight
- Risk Parity
- Sharpe/Sortino Maximization
- Minimum Variance
- Maximum Diversification
- Hierarchical Risk Parity

### Environment Setup
Requires API keys in `.env` file:
```
FINNHUB_API_KEY=your_finnhub_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
```

### Important Notes
- The system is designed for financial analysis and backtesting, not live trading
- Default cleaner action is set to preserve data folders
- UUID-based execution tracking for reproducibility
- Comprehensive logging and backup system for all executions

### Data Caching System
The new unified data management system includes:
- **Automatic caching**: Data is cached to avoid redundant downloads
- **Cache validation**: Cache expires after specified days (default: 1 day)
- **Deduplication**: Prevents multiple downloads of the same data
- **Unified paths**: Consistent data directory structure across all components
- **Smart detection**: Automatically uses existing data when --use-cached-data is enabled

To use caching:
```bash
# First run - downloads data
./run_pipeline.sh --time-horizon swing

# Subsequent runs - uses cached data
./run_pipeline.sh --time-horizon swing --use-cached-data

# Force cache for longer period
./run_pipeline.sh --time-horizon swing --use-cached-data --cache-days 7
```