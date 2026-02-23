from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from src.data_load import DataPreparation
from src.pipeline import OptionsTrainingPipeline
from src.hopt import OptionsHyperparameterTuning

from src.config import (
    DataConfig,
    TradingConfig,
    HyperOptConfig,
)

def main():

    # ===============================================================
    # LOAD CONFIGURATIONS
    # ===============================================================

    data_cfg = DataConfig()
    trade_cfg = TradingConfig()
    hopt_cfg = HyperOptConfig()

    OPTION_SYMBOL = data_cfg.option_symbol
    OPTIONS_DATA_PATH = data_cfg.options_data_path

    INITIAL_CAPITAL = trade_cfg.initial_capital
    TRANSACTION_COST = trade_cfg.transaction_cost_pct
    MAX_CONTRACTS = trade_cfg.max_contracts_per_position
    REWARD_SCALING = trade_cfg.reward_scaling
    CONTRACT_FILTER = trade_cfg.contract_filter

    ALGORITHMS = hopt_cfg.algorithms

    # ===============================================================
    # DATA PREPARATION
    # ===============================================================

    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    prep = DataPreparation(OPTION_SYMBOL)

    prep.load_options_data(OPTIONS_DATA_PATH)

    filtered_options = prep.filter_options(
        min_dte=data_cfg.min_dte,
        max_dte=data_cfg.max_dte,
        moneyness_range=data_cfg.moneyness_range,
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
    # TRAIN / VAL / TEST SPLIT
    # ===============================================================

    dates = sorted(df_index["date"].unique())
    n_dates = len(dates)

    train_end_idx = int(n_dates * data_cfg.train_split)
    val_end_idx = int(n_dates * data_cfg.val_split)

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
        contract_filter=CONTRACT_FILTER,
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
            n_trials=hopt_cfg.n_trials,
            n_timesteps=train_end_idx * hopt_cfg.hopt_episodes_multiplier,
            n_eval_episodes=hopt_cfg.n_eval_episodes,
            eval_freq=hopt_cfg.eval_freq,
            study_name=f"options_{algorithm}_study",
            storage=hopt_cfg.storage,
        )

        tuner.optimize_model(timeout=hopt_cfg.timeout_seconds)

        tuner.compare_trials(top_n=hopt_cfg.top_n_trials_to_compare)

        print("\nTraining final model with best hyperparameters...")
        best_model = tuner.train_with_best_params(
            total_timesteps=val_end_idx * hopt_cfg.final_train_multiplier
        )

        best_models.append(best_model)

    # ===============================================================
    # BACKTESTING
    # ===============================================================

    print("\n" + "=" * 70)
    print("BACKTESTING ON TEST SET")
    print("=" * 70)

    pipeline.backtest(
        models=best_models,
        model_names=ALGORITHMS,
        option_symbol=OPTION_SYMBOL,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()