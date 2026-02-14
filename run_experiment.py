from __future__ import annotations

import warnings
import time
import os
import pandas as pd

warnings.filterwarnings("ignore")

from src.data_load import DataPreparation
from src.pipeline import OptionsTrainingPipeline
from src.hopt import OptionsHyperparameterTuning


def main():

    # ===============================================================
    # CONFIGURATION
    # ===============================================================
    OPTION_SYMBOL = "GLD"
    OPTIONS_DATA_PATH = f"C:\\Users\\arush\\Desktop\\Semester\\BTP\\Dataset\\{OPTION_SYMBOL}.csv" 

    INITIAL_CAPITAL = 100000
    TRANSACTION_COST = 0.0005
    MAX_CONTRACTS = 100
    REWARD_SCALING = 1

    ALGORITHMS = ["a2c", "ppo"]

    HOPT_TRIALS = 1  
    HOPT_EPISODES = 1  
    NUM_EPISODES = 1

    # ===============================================================
    # DATA PREPARATION
    # ===============================================================

    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    prep = DataPreparation(OPTION_SYMBOL)

    options_data = prep.load_options_data(OPTIONS_DATA_PATH)

    filtered_options = prep.filter_options(
        min_dte=10,
        max_dte=100,
        moneyness_range=(0.85, 1.15),
    )

    prep.fetch_index_data()
    prep.fetch_vix_data()
    prep.calculate_technical_indicators()
    combined_data = prep.combine_data()
    prep.get_data_summary()

    df_index = combined_data
    df_options = filtered_options

    print(f"\n✓ Index rows  : {len(df_index)}")
    print(f"✓ Options rows: {len(df_options)}")

    # ===============================================================
    # TRAIN / VAL / TEST SPLIT (70 / 10 / 20)
    # ===============================================================

    dates = sorted(df_index["date"].unique())
    n_dates = len(dates)

    train_end_idx = int(n_dates * 0.70)
    val_end_idx = int(n_dates * 0.80)

    train_start = dates[0]
    train_end = dates[train_end_idx]
    val_start = dates[train_end_idx]
    val_end = dates[val_end_idx]
    test_start = dates[val_end_idx]
    test_end = dates[-1]

    print("\nDate Ranges:")
    print(f"  Train: {train_start} → {train_end}")
    print(f"  Val  : {val_start} → {val_end}")
    print(f"  Test : {test_start} → {test_end}")

    # ===============================================================
    # PIPELINE INITIALIZATION
    # ===============================================================

    pipeline = OptionsTrainingPipeline(
        df_index=df_index,
        df_options=df_options,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST,
        max_contracts_per_position=MAX_CONTRACTS,
        reward_scaling=REWARD_SCALING,
    )

    best_models = []

    # ===============================================================
    # HYPERPARAMETER TUNING + TRAINING
    # ===============================================================

    for algorithm in ALGORITHMS:

        print("\n" + "=" * 70)
        print(f"HYPERPARAMETER OPTIMIZATION - {algorithm.upper()}")
        print("=" * 70)

        tuner = OptionsHyperparameterTuning(
            pipeline=pipeline,
            algorithm=algorithm,
            n_trials=HOPT_TRIALS,  # increase for real experiments
            n_timesteps=train_end_idx*HOPT_EPISODES,
            n_eval_episodes=1,
            study_name=f"options_{algorithm}_study",
            storage="sqlite:///results/optuna/optuna.db"
        )

        tuner.optimize_model(timeout=3600)  # 1 hour max

        tuner.compare_trials(top_n=5)

        print("\nTraining final model with best hyperparameters...")
        best_model = tuner.train_with_best_params(
            total_timesteps=train_end_idx*NUM_EPISODES
        )

        best_models.append(best_model)

    # ===============================================================
    # BACKTESTING
    # ===============================================================

    print("\n" + "=" * 70)
    print("BACKTESTING ON TEST SET")
    print("=" * 70)

    results, actions, stats, baseline_stats = pipeline.backtest(
        models=best_models,
        model_names=ALGORITHMS,
        option_symbol=OPTION_SYMBOL,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nOutputs saved inside:")
    print("  ./results/backtests/")
    print("  ./results/baselines/")
    print("  ./results/optuna/")
    print("  ./results/figures/")


if __name__ == "__main__":
    main()