from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_baseline

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.logger import configure

from src.trading_env import OptionsEnv

warnings.filterwarnings("ignore")

class OptionsTrainingPipeline:
    """
    Complete training pipeline for index options trading.

    Features:
    - Time-based train/validation/test splitting
    - DRL agent training via FinRL wrapper
    - Backtesting with benchmark comparison
    - Performance metrics and visualization
    """

    def __init__(
        self,
        df_index: pd.DataFrame,
        df_options: pd.DataFrame,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        test_end: str,
        initial_capital: float = 100000,
        transaction_cost_pct: float = 0.001,
        max_contracts_per_position: int = 10,
        reward_scaling: float = 1e-2,
        contract_filter: str = "volume",
    ):
        """
        Initialize the training pipeline.
        """

        # ------------------------------------------------------------------
        # Store data
        # ------------------------------------------------------------------
        self.df_index = df_index.copy()
        self.df_options = df_options.copy()

        # Ensure datetime format
        self.df_index["date"] = pd.to_datetime(self.df_index["date"])
        self.df_options["date"] = pd.to_datetime(self.df_options["date"])

        # ------------------------------------------------------------------
        # Date ranges
        # ------------------------------------------------------------------
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)

        self.val_start = pd.to_datetime(val_start)
        self.val_end = pd.to_datetime(val_end)

        self.test_start = pd.to_datetime(test_start)
        self.test_end = pd.to_datetime(test_end)

        # ------------------------------------------------------------------
        # Environment parameters
        # ------------------------------------------------------------------
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.max_contracts_per_position = max_contracts_per_position
        self.reward_scaling = reward_scaling
        self.contract_filter = contract_filter

        # ------------------------------------------------------------------
        # Directory setup
        # ------------------------------------------------------------------
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./trained_models", exist_ok=True)
        os.makedirs("./trained_models/checkpoints", exist_ok=True)
        os.makedirs("./results/logs", exist_ok=True)
        os.makedirs("./results/baseline", exist_ok=True)
        os.makedirs("./tensorboard_logs", exist_ok=True)

        # ------------------------------------------------------------------
        # Split data
        # ------------------------------------------------------------------
        self._split_data()

        print("\n" + "=" * 70)
        print("OPTIONS TRAINING PIPELINE INITIALIZED")
        print("=" * 70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Transaction Cost: {self.transaction_cost_pct * 100:.3f}%")
        print(f"Max Contracts per Position: {self.max_contracts_per_position}")

    def _split_data(self):
        """
        Split index and options data into train, validation, and test sets.
        Includes optional lookback buffer for validation and test.
        """

        lookback_buffer = pd.Timedelta(days=0)

        # ------------------------------------------------------------------
        # Index Data
        # ------------------------------------------------------------------
        self.train_index = (
            self.df_index[
                (self.df_index["date"] >= self.train_start)
                & (self.df_index["date"] < self.train_end)
            ]
            .copy()
            .reset_index(drop=True)
        )

        val_start_with_buffer = self.val_start - lookback_buffer

        self.val_index = (
            self.df_index[
                (self.df_index["date"] >= val_start_with_buffer)
                & (self.df_index["date"] < self.val_end)
            ]
            .copy()
            .reset_index(drop=True)
        )

        test_start_with_buffer = self.test_start - lookback_buffer

        self.test_index = (
            self.df_index[
                (self.df_index["date"] >= test_start_with_buffer)
                & (self.df_index["date"] < self.test_end)
            ]
            .copy()
            .reset_index(drop=True)
        )

        # ------------------------------------------------------------------
        # Options Data
        # ------------------------------------------------------------------
        self.train_options = (
            self.df_options[
                (self.df_options["date"] >= self.train_start)
                & (self.df_options["date"] < self.train_end)
            ]
            .copy()
            .reset_index(drop=True)
        )

        self.val_options = (
            self.df_options[
                (self.df_options["date"] >= val_start_with_buffer)
                & (self.df_options["date"] < self.val_end)
            ]
            .copy()
            .reset_index(drop=True)
        )

        self.test_options = (
            self.df_options[
                (self.df_options["date"] >= test_start_with_buffer)
                & (self.df_options["date"] < self.test_end)
            ]
            .copy()
            .reset_index(drop=True)
        )

        # ------------------------------------------------------------------
        # Logging summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("DATA SPLIT SUMMARY")
        print("=" * 70)

        print(
            f"\nTraining Period: {self.train_start.date()} to {self.train_end.date()}"
        )
        print(
            f"  index: {len(self.train_index)} rows, "
            f"{self.train_index['date'].nunique()} days"
        )
        print(f"  options: {len(self.train_options)} rows")

        print(
            f"\nValidation Period: {self.val_start.date()} to {self.val_end.date()}"
        )
        print(f"  (with buffer from: {val_start_with_buffer.date()})")
        print(
            f"  index: {len(self.val_index)} rows, "
            f"{self.val_index['date'].nunique()} days"
        )
        print(f"  options: {len(self.val_options)} rows")

        print(
            f"\nTest Period: {self.test_start.date()} to {self.test_end.date()}"
        )
        print(f"  (with buffer from: {test_start_with_buffer.date()})")
        print(
            f"  index: {len(self.test_index)} rows, "
            f"{self.test_index['date'].nunique()} days"
        )
        print(f"  options: {len(self.test_options)} rows")

    def create_env(
        self,
        df_index: pd.DataFrame,
        df_options: pd.DataFrame,
        mode: str = "train",
    ):
        """
        Create and return an OptionsEnv instance.

        Args:
            df_index: Index data for the environment.
            df_options: Options data for the environment.
            mode: One of {"train", "validation", "test"}.

        Returns:
            OptionsEnv instance.
        """

        env = OptionsEnv(
            df_index=df_index,
            df_options=df_options,
            initial_capital=self.initial_capital,
            transaction_cost_pct=self.transaction_cost_pct,
            max_contracts_per_position=self.max_contracts_per_position,
            reward_scaling=self.reward_scaling,
            contract_filter=self.contract_filter,
        )

        # Attach mode attribute (used for diagnostics / logging)
        env.mode = mode

        return env

    def _get_latest_checkpoint(
        self,
        checkpoint_dir: str,
        prefix: str = "a2c_index_options"
    ) -> str | None:
        """
        Retrieve the most recent checkpoint file based on timestep number
        embedded in the filename.

        Expected filename format:
            {prefix}_<timestep>.zip

        Args:
            checkpoint_dir: Directory containing checkpoint files.
            prefix: Filename prefix used during checkpoint saving.

        Returns:
            Path to latest checkpoint file or None if not found.
        """

        if not os.path.exists(checkpoint_dir):
            return None

        checkpoint_files = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.startswith(prefix) and f.endswith(".zip")
        ]

        if not checkpoint_files:
            return None

        def extract_timestep(filepath: str) -> int:
            filename = os.path.basename(filepath)
            try:
                return int(filename.split("_")[-1].replace(".zip", ""))
            except ValueError:
                return -1

        checkpoint_files.sort(key=extract_timestep)

        latest_checkpoint = checkpoint_files[-1]

        print(f"✓ Found latest checkpoint: {latest_checkpoint}")

        return latest_checkpoint
    
    def train_model(
        self,
        model_kwargs: dict,
        algorithms: list = ["a2c"],
        total_timesteps: int = 50000,
        tensorboard_log: str = "./tensorboard_logs/",
        eval_freq: int = 5000,
        n_eval_episodes: int = 3,
        resume: bool = True,
        checkpoint_dir: str = "./trained_models/checkpoints/",
    ):
        """
        Train one or more DRL agents using the FinRL DRLAgent wrapper.

        Args:
            model_kwargs: Hyperparameters passed to the SB3 model.
            algorithms: List of algorithms to train (e.g., ["a2c", "ppo"]).
            total_timesteps: Number of training timesteps.
            tensorboard_log: TensorBoard log directory.
            eval_freq: Evaluation frequency (in timesteps).
            n_eval_episodes: Episodes per evaluation.
            resume: Whether to resume from checkpoint.
            checkpoint_dir: Directory containing checkpoints.

        Returns:
            Dictionary mapping algorithm name → trained model.
        """

        trained_models = {}

        for algorithm in algorithms:
            algo_name = algorithm.lower()

            print("\n" + "=" * 70)
            print(f"TRAINING {algo_name.upper()} AGENT")
            print("=" * 70)

            # ------------------------------------------------------------------
            # Environment Setup
            # ------------------------------------------------------------------
            print("\nCreating training environment...")
            train_env = DummyVecEnv([
                lambda: Monitor(
                    self.create_env(self.train_index, self.train_options, mode="train"),
                    filename=f"./results/logs/{algo_name}/train_monitor.csv",
                )
            ])

            print("Creating validation environment...")
            val_env = DummyVecEnv([
                lambda: Monitor(
                    self.create_env(self.val_index, self.val_options, mode="validation"),
                    filename=f"./results/logs/{algo_name}/val_monitor.csv",
                )
            ])

            # ------------------------------------------------------------------
            # Callbacks
            # ------------------------------------------------------------------
            print("\nSetting up callbacks...")

            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5,
                min_evals=5,
                verbose=1,
            )

            eval_callback = EvalCallback(
                val_env,
                best_model_save_path="./trained_models/",
                log_path="./results/logs/",
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                n_eval_episodes=n_eval_episodes,
                callback_after_eval=stop_callback,
                verbose=1,
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=10000,
                save_path="./trained_models/checkpoints/",
                name_prefix=f"{algo_name}_index_options",
                verbose=1,
            )

            # ------------------------------------------------------------------
            # Model Initialization
            # ------------------------------------------------------------------
            print("\nInitializing DRLAgent...")
            agent = DRLAgent(env=train_env)

            print("\nModel Hyperparameters:")
            print("-" * 50)
            for key, value in model_kwargs.items():
                print(f"  {key:20s}: {value}")
            print(f"  {'total_timesteps':20s}: {total_timesteps}")
            print(f"  {'eval_freq':20s}: {eval_freq}")
            print("-" * 50)

            if algo_name == "a2c":
                ModelClass = A2C
            elif algo_name == "ppo":
                ModelClass = PPO
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            print("\nCreating model...")

            latest_ckpt = None
            if resume:
                latest_ckpt = self._get_latest_checkpoint(checkpoint_dir, prefix=f"{algo_name}_index_options")

            if latest_ckpt:
                print(f"🔁 Resuming from {latest_ckpt}")
                model = ModelClass.load(
                    latest_ckpt,
                    env=train_env,
                    tensorboard_log=tensorboard_log,
                    verbose=1,
                )
            else:
                print("\n🆕 Starting training from scratch")
                model = agent.get_model(
                    model_name=algo_name,
                    model_kwargs=model_kwargs,
                    policy="MlpPolicy",
                    tensorboard_log=tensorboard_log,
                    verbose=1,
                    seed=42,
                )

            # ------------------------------------------------------------------
            # Logger Setup
            # ------------------------------------------------------------------
            tmp_path = f"./results/logs/{algo_name}"
            os.makedirs(tmp_path, exist_ok=True)

            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)

            print(f"\nStarting training for {total_timesteps:,} timesteps...")
            print("Monitor training progress in TensorBoard:")
            print(f"  tensorboard --logdir {tensorboard_log}")
            print("\n" + "=" * 70)

            # ------------------------------------------------------------------
            # Training
            # ------------------------------------------------------------------
            trained_model = model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, checkpoint_callback],
                reset_num_timesteps=False,
                tb_log_name=f"{algo_name.upper()}_index_Options",
            )

            # ------------------------------------------------------------------
            # Save Model
            # ------------------------------------------------------------------
            final_model_path = (
                f"./trained_models/{algo_name}_index_options_final"
            )
            trained_model.save(final_model_path)

            trained_models[algo_name] = trained_model

            print("\n" + "=" * 70)
            print(f"TRAINING COMPLETE for {algo_name.upper()}")
            print("=" * 70)
            print(f"✓ Final model saved to: {final_model_path}")
            print("✓ Best model saved to: ./trained_models/best_model.zip")
            print("✓ Checkpoints saved to: ./trained_models/checkpoints/")
            print("✓ Logs saved to: ./logs/")

        return trained_models
    
    def run_option_baseline(
        self,
        df_index: pd.DataFrame,
        df_options: pd.DataFrame,
        baseline_type: str,
        initial_capital: float,
    ) -> pd.DataFrame:
        """
        Run deterministic option-based baseline strategy.

        Returns:
            DataFrame with columns ['date', 'account_value']
        """

        cash = initial_capital
        account_values = []
        dates = []

        trading_dates = sorted(df_index["date"].unique())

        for i in range(len(trading_dates) - 1):
            date_t = trading_dates[i]
            date_t1 = trading_dates[i + 1]

            opts_t = df_options[df_options["date"] == date_t]
            opts_t1 = df_options[df_options["date"] == date_t1]

            if opts_t.empty or opts_t1.empty:
                account_values.append(cash)
                dates.append(date_t)
                continue

            # Select ATM-like options (closest to moneyness = 1)
            opts_t = (
                opts_t.sort_values("moneyness", key=lambda x: abs(x - 1))
                .head(4)
            )

            # ------------------------------------------------------------------
            # Weight Computation
            # ------------------------------------------------------------------
            if baseline_type == "equal_weight":
                weights = np.ones(len(opts_t)) / len(opts_t)

            elif baseline_type == "delta_weighted":
                weights = 1 / opts_t["delta"].abs().clip(1e-4)
                weights /= weights.sum()

            elif baseline_type == "vega_weighted":
                weights = opts_t["vega"].abs()
                weights /= weights.sum()

            elif baseline_type == "theta_weighted":
                weights = opts_t["theta"].abs()
                weights /= weights.sum()

            elif baseline_type == "iv_based":
                weights = opts_t["iv"].rank(pct=True)
                weights /= weights.sum()

            elif baseline_type == "var_weighted":
                sigma = opts_t["iv"]
                var = sigma * np.sqrt(1 / 252)
                weights = 1 / var.clip(1e-4)
                weights /= weights.sum()

            else:
                raise ValueError(f"Unknown baseline_type: {baseline_type}")

            # ------------------------------------------------------------------
            # Daily PnL Computation
            # ------------------------------------------------------------------
            pnl = 0.0

            for w, (_, opt) in zip(weights, opts_t.iterrows()):
                opt_t1 = opts_t1[opts_t1["option_symbol"] == opt["option_symbol"]]
                if opt_t1.empty:
                    continue

                price_t = opt["price"]
                price_t1 = opt_t1.iloc[0]["price"]

                pnl += w * (price_t1 - price_t) * 100

            cash += pnl
            account_values.append(cash)
            dates.append(date_t)

        return pd.DataFrame({
            "date": dates,
            "account_value": account_values,
        })
    
    def backtest(
        self,
        models,
        model_names: list = ["A2C"],
        option_symbol: str = "SPX",
        deterministic: bool = True,
    ):
        """
        Backtest trained models on test data and compare against baselines.

        Returns:
            all_results, all_actions, all_stats, stats_baseline
        """

        all_results = {}
        all_actions = {}
        all_stats = {}

        # ======================================================================
        # MODEL BACKTESTING
        # ======================================================================
        for model, model_name in zip(models, model_names):

            print("\n" + "=" * 70)
            print(f"BACKTESTING {model_name.upper()} ON TEST DATA")
            print("=" * 70)

            # ------------------------------------------------------------------
            # Environment Setup
            # ------------------------------------------------------------------
            print("\nCreating test environment...")
            test_env = self.create_env(
                self.test_index,
                self.test_options,
                mode="test",
            )

            obs, info = test_env.reset()

            print("\nInitial state:")
            print(f"  Portfolio value: ${info['portfolio_value']:,.2f}")
            print(f"  Cash: ${info['cash']:,.2f}")
            print(f"  Trading dates available: {len(test_env.trading_dates)}")

            account_memory = [test_env.portfolio_value]
            dates_memory = [test_env.trading_dates[test_env.current_step]]
            actions_memory = []
            positions_memory = []
            rewards_memory = []

            done = False
            step = 0
            max_steps = (
                len(test_env.trading_dates)
                - test_env.current_step
                - 1
            )

            # ------------------------------------------------------------------
            # Backtest Loop
            # ------------------------------------------------------------------
            while not done and step < max_steps:

                action, _ = model.predict(obs, deterministic=deterministic)

                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated

                account_memory.append(info["portfolio_value"])
                dates_memory.append(info["date"])
                actions_memory.append(action)
                positions_memory.append(info["num_positions"])
                rewards_memory.append(reward)

                step += 1

                if step % 10 == 0 or step < 5:
                    print(
                        f"Step {step}/{max_steps} | "
                        f"Date={info['date']} | "
                        f"Portfolio=${info['portfolio_value']:,.2f} | "
                        f"PnL={info['pnl_pct']:.2f}% | "
                        f"Positions={info['num_positions']} | "
                        f"Reward={reward:.6f}"
                    )

            print(f"\nBacktest completed: {step} steps")

            # ------------------------------------------------------------------
            # Build Results DataFrame
            # ------------------------------------------------------------------
            df_account_value = pd.DataFrame({
                "date": pd.to_datetime(dates_memory),
                "account_value": account_memory,
            }).sort_values("date").reset_index(drop=True)

            df_account_value["daily_return"] = (
                df_account_value["account_value"].pct_change()
            )

            if actions_memory:
                df_actions = pd.DataFrame(actions_memory)
                df_actions["date"] = dates_memory[:-1]
                df_actions["num_positions"] = positions_memory
            else:
                df_actions = pd.DataFrame()

            all_results[model_name] = df_account_value
            all_actions[model_name] = df_actions

        # ======================================================================
        # BASELINE BENCHMARK (Underlying Index)
        # ======================================================================
        print("\n" + "=" * 70)
        print("FETCHING BASELINE BENCHMARK")
        print("=" * 70)

        baseline_ticker = "^GSPC" if option_symbol == "SPX" else option_symbol

        try:
            df_baseline_raw = get_baseline(
                ticker=baseline_ticker,
                start=df_account_value["date"].min(),
                end=df_account_value["date"].max(),
            )

            stats_baseline = backtest_stats(
                df_baseline_raw,
                value_col_name="close",
            )

            df_baseline = pd.DataFrame({
                "date": df_account_value["date"],
                "account_value": (
                    df_baseline_raw["close"]
                    / df_baseline_raw["close"].iloc[0]
                ) * self.initial_capital,
            })

            print(f"✓ Baseline data fetched for {baseline_ticker}")

        except Exception as e:
            print(f"⚠ Could not fetch baseline: {e}")
            df_baseline = None
            stats_baseline = None

        # ======================================================================
        # MODEL STATISTICS
        # ======================================================================
        for model_name in model_names:
            print("\n" + "=" * 70)
            print(f"{model_name} PERFORMANCE STATISTICS")
            print("=" * 70)

            all_stats[model_name] = backtest_stats(
                all_results[model_name],
                value_col_name="account_value",
            )

        # ======================================================================
        # OPTION-BASED BASELINES
        # ======================================================================
        print("\n" + "=" * 70)
        print("RUNNING OPTION-BASED BASELINES")
        print("=" * 70)

        option_baselines = [
            "equal_weight",
            "delta_weighted",
            "vega_weighted",
            "theta_weighted",
            "iv_based",
            "var_weighted",
        ]

        for b in option_baselines:
            df_base_opt = self.run_option_baseline(
                df_index=self.test_index,
                df_options=self.test_options,
                baseline_type=b,
                initial_capital=self.initial_capital,
            )

            backtest_stats(df_base_opt, value_col_name="account_value")
            df_base_opt.to_csv(f"./results/baseline/baseline_{b}.csv", index=False)

            print(f"✓ Completed baseline: {b}")

        # ======================================================================
        # Key Metrics
        # ======================================================================
        for algorithm, df_account_value in all_results.items():
            print("\n" + "=" * 70)
            print(f"{algorithm.upper()} KEY METRICS")
            print("=" * 70)
            total_return = (
                (df_account_value["account_value"].iloc[-1] - self.initial_capital)
                / self.initial_capital
                * 100
            )

            std = df_account_value["daily_return"].std()
            sharpe_ratio = (
                np.sqrt(252)
                * df_account_value["daily_return"].mean()
                / std
                if std != 0 else 0
            )

            max_drawdown = self._calculate_max_drawdown(
                df_account_value["account_value"]
            )

            print(f"\nKey Metrics:")
            print(f"  Initial Capital      : ${self.initial_capital:,.2f}")
            print(f"  Final Portfolio Value: ${df_account_value['account_value'].iloc[-1]:,.2f}")
            print(f"  Total Return         : {total_return:.2f}%")
            print(f"  Sharpe Ratio         : {sharpe_ratio:.3f}")
            print(f"  Max Drawdown         : {max_drawdown:.2f}%")
            print(f"  Total Trading Days   : {len(df_account_value)}")
            print(f"  Avg Daily Return     : {df_account_value['daily_return'].mean()*100:.3f}%")
            print(f"  Daily Return Std     : {df_account_value['daily_return'].std()*100:.3f}%")

        # ======================================================================
        # Plotting
        # ======================================================================
        self._plot_backtest_results(
            all_results,
            df_baseline,
            baseline_ticker,
            option_baselines={
                k: pd.read_csv(
                    f"./results/baseline/baseline_{k}.csv",
                    parse_dates=["date"],
                )
                for k in option_baselines
            },
        )

        if df_baseline is not None:
            try:
                backtest_plot(
                    df_account_value,
                    baseline_ticker=baseline_ticker,
                    baseline_start=df_account_value["date"].min(),
                    baseline_end=df_account_value["date"].max(),
                )
            except Exception:
                pass

        return all_results, all_actions, all_stats, stats_baseline
    
    def _calculate_max_drawdown(self, account_values):
        """
        Calculate maximum drawdown percentage.

        Args:
            account_values: Pandas Series of portfolio values

        Returns:
            Maximum drawdown in percentage (positive number)
        """
        account_values = pd.Series(account_values)

        cumulative_max = account_values.cummax()
        drawdown = (account_values - cumulative_max) / cumulative_max

        return abs(drawdown.min() * 100)
    
    def _plot_backtest_results(
        self,
        model_results: dict,
        df_baseline: pd.DataFrame,
        baseline_ticker: str,
        option_baselines: dict = None,
    ):
        """
        Plot separate backtest figures:
        1. Portfolio value
        2. Daily returns (all models + index)
        3. Cumulative returns
        """

        import os
        os.makedirs("./results/figures", exist_ok=True)

        # ==========================================================
        # 1️⃣ PORTFOLIO VALUE PLOT
        # ==========================================================
        plt.figure(figsize=(12, 6))

        for name, df_account_value in model_results.items():
            plt.plot(
                df_account_value["date"],
                df_account_value["account_value"],
                label=name,
                linewidth=2,
            )

        if df_baseline is not None:
            plt.plot(
                df_baseline["date"],
                df_baseline["account_value"],
                label=f"{baseline_ticker}",
                linewidth=2,
                alpha=0.7,
            )

        if option_baselines is not None:
            for name, df_base in option_baselines.items():
                plt.plot(
                    df_base['date'],
                    df_base['account_value'],
                    linestyle='--',
                    alpha=0.6,
                    label=f'Baseline: {name}'
                )

        plt.axhline(
            y=self.initial_capital,
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Initial Capital",
        )

        plt.title("Portfolio Value Comparison", fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        portfolio_path = "./results/figures/portfolio_value.png"
        plt.tight_layout()
        plt.savefig(portfolio_path, dpi=300)
        plt.close()

        print(f"✓ Portfolio value plot saved to {portfolio_path}")

        # ==========================================================
        # 2️⃣ DAILY RETURNS (ONE PLOT PER MODEL)
        # ==========================================================

        for name, df_account_value in model_results.items():

            plt.figure(figsize=(12, 6))

            # Model daily returns
            plt.plot(
                df_account_value["date"][1:],
                df_account_value["daily_return"][1:] * 100,
                label=f"{name}",
                alpha=0.8,
            )

            # Baseline / Underlying
            if df_baseline is not None:
                df_baseline["daily_return"] = df_baseline["account_value"].pct_change()

                plt.plot(
                    df_baseline["date"][1:],
                    df_baseline["daily_return"][1:] * 100,
                    label=f"{baseline_ticker}",
                    alpha=0.7,
                )

            plt.axhline(y=0, linewidth=0.8)

            plt.title(f"Daily Returns: {name} vs {baseline_ticker}", fontweight="bold")
            plt.xlabel("Date")
            plt.ylabel("Daily Return (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save separately per model
            daily_path = f"./results/figures/daily_returns_{name}.png"

            plt.tight_layout()
            plt.savefig(daily_path, dpi=300)
            plt.close()

            print(f"✓ Daily returns plot saved to {daily_path}")

        # ==========================================================
        # 3️⃣ CUMULATIVE RETURNS
        # ==========================================================
        plt.figure(figsize=(12, 6))

        for name, df_account_value in model_results.items():
            cumulative_returns = (
                df_account_value["account_value"] / self.initial_capital - 1
            ) * 100

            plt.plot(
                df_account_value["date"],
                cumulative_returns,
                label=name,
                linewidth=2,
            )

        if df_baseline is not None:
            baseline_returns = (
                df_baseline["account_value"] / self.initial_capital - 1
            ) * 100

            plt.plot(
                df_baseline["date"],
                baseline_returns,
                label=baseline_ticker,
                linewidth=2,
                alpha=0.7,
            )

        plt.axhline(y=0, linestyle="--", linewidth=1, alpha=0.5)

        plt.title("Cumulative Returns Comparison", fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        cumulative_path = "./results/figures/cumulative_returns.png"
        plt.tight_layout()
        plt.savefig(cumulative_path, dpi=300)
        plt.close()

        print(f"✓ Cumulative returns plot saved to {cumulative_path}")

    def diagnose_environment(self, mode="test", print_positions=True):
        """
        Run diagnostic checks on the environment configuration.
        """

        print("\n" + "=" * 70)
        print(f"ENVIRONMENT DIAGNOSTICS - {mode.upper()} MODE")
        print("=" * 70)

        # ------------------------------------------------------------------
        # Select dataset
        # ------------------------------------------------------------------
        if mode == "train":
            df_index, df_options = self.train_index, self.train_options
        elif mode == "validation":
            df_index, df_options = self.val_index, self.val_options
        else:
            df_index, df_options = self.test_index, self.test_options

        print("\nData Summary:")
        print(f"  index rows: {len(df_index)}")
        print(f"  Options rows: {len(df_options)}")
        print(f"  index date range: {df_index['date'].min()} → {df_index['date'].max()}")
        print(f"  Options date range: {df_options['date'].min()} → {df_options['date'].max()}")

        # ------------------------------------------------------------------
        # Required Column Check
        # ------------------------------------------------------------------
        print("\nColumn Validation:")

        index_required = ["date", "close", "open", "high", "low", "volume"]
        options_required = [
            "date",
            "option_symbol",
            "option_type",
            "strike",
            "expiration",
            "price",
            "dte",
            "moneyness",
        ]

        index_missing = [c for c in index_required if c not in df_index.columns]
        options_missing = [c for c in options_required if c not in df_options.columns]

        if index_missing:
            print(f"  ⚠ Missing index columns: {index_missing}")
        else:
            print("  ✓ Index columns OK")

        if options_missing:
            print(f"  ⚠ Missing options columns: {options_missing}")
        else:
            print("  ✓ Options columns OK")

        # ------------------------------------------------------------------
        # Environment Creation Test
        # ------------------------------------------------------------------
        print(f"\nCreating {mode} environment...")

        try:
            env = self.create_env(df_index, df_options, mode=mode)
            print("  ✓ Environment created")

            obs, info = env.reset()
            print("\nReset Check:")
            print(f"  Observation shape: {obs.shape}")
            print(f"  Expected shape: {env.observation_space.shape}")
            print(f"  Action space: {env.action_space}")
            print(f"  Initial portfolio: ${info['portfolio_value']:,.2f}")
            print(f"  Trading dates: {len(env.trading_dates)}")

            print("\nRandom Step Test:")
            for i in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                print(
                    f"  Step {i+1}: "
                    f"Portfolio=${info['portfolio_value']:,.2f}, "
                    f"Reward={reward:.6f}, "
                    f"Done={terminated or truncated}"
                )

                if terminated or truncated:
                    print("  ⚠ Episode terminated early")
                    break

            print("\n✓ Diagnostics complete")

        except Exception as e:
            print(f"  ✗ Environment error: {e}")
            import traceback
            traceback.print_exc()