"""
Central configuration for DRL-based Options Trading project.
Optimized for hyperparameter tuning workflow.
"""

import os
from dataclasses import dataclass


# ==============================================================
# 📁 BASE PROJECT PATH
# ==============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TENSORBOARD_DIR = os.path.join(BASE_DIR, "tensorboard_logs")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")

# Auto-create required folders
for path in [
    RESULTS_DIR,
    FIGURES_DIR,
    TENSORBOARD_DIR,
    LOGS_DIR,
    MODEL_DIR,
]:
    os.makedirs(path, exist_ok=True)


# ==============================================================
# 📊 DATA CONFIG
# ==============================================================

@dataclass
class DataConfig:
    option_symbol: str = "GLD"
    options_data_path: str = r"C:\Users\arush\Desktop\Semester\BTP\Dataset\GLD.csv"

    train_split: float = 0.70
    val_split: float = 0.80  # rest = test

    min_dte: int = 10
    max_dte: int = 100
    moneyness_range: tuple = (0.8, 1.2)


# ==============================================================
# 💰 TRADING ENVIRONMENT CONFIG
# ==============================================================

@dataclass
class TradingConfig:
    initial_capital: float = 100_000
    transaction_cost_pct: float = 0.0005
    max_contracts_per_position: int = 100
    reward_scaling: float = 1.0
    contract_filter: str = "volume"  # volume / iv (implied volatility) / open_interest


# ==============================================================
# 🔍 HYPERPARAMETER TUNING CONFIG
# ==============================================================

@dataclass
class HyperOptConfig:
    algorithms: list = None

    n_trials: int = 50
    hopt_episodes_multiplier: int = 100
    final_train_multiplier: int = 500

    n_eval_episodes: int = 1
    eval_freq: int = 500
    timeout_seconds: int = 3600 * 6

    top_n_trials_to_compare: int = 5

    storage: str = "sqlite:///results/optuna/optuna.db"

    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ["a2c", "ppo"]


# ==============================================================
# 📈 GRAPH CONFIGURATION
# ==============================================================

@dataclass
class GraphConfig:
    save_figures: bool = True
    figure_dpi: int = 300
    figure_format: str = "png"   # png / pdf / jpg


# ==============================================================
# 🔁 REPRODUCIBILITY
# ==============================================================

RANDOM_SEED = 42