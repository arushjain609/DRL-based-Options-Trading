from __future__ import annotations

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib

from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

warnings.filterwarnings("ignore")

class OptionsHyperparameterTuning:
    """
    Hyperparameter optimization for options trading using Optuna.

    Integrates with OptionsTrainingPipeline to tune SB3 algorithms
    such as A2C and PPO.
    """

    def __init__(
        self,
        pipeline,
        algorithm: str = "a2c",
        n_trials: int = 50,
        n_timesteps: int = 50_000,
        n_eval_episodes: int = 5,
        eval_freq: int = 5_000,
        study_name: str = "index_options_study",
        storage: str | None = None,
    ):
        self.pipeline = pipeline
        self.algorithm = algorithm.lower()

        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.study_name = study_name
        self.storage = storage

        os.makedirs('./results/optuna', exist_ok=True)
        os.makedirs('./results/optuna/optuna_models', exist_ok=True)

        self.best_params: Dict = {}
        self.best_model = None

    # ===============================================================
    # Objective Function
    # ===============================================================
    def objective_model(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Returns:
            Mean validation reward (to maximize)
        """

        # -----------------------------------------------------------
        # Sample hyperparameters
        # -----------------------------------------------------------
        if self.algorithm == "a2c":

            model_kwargs = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 32, 128),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
            }

        elif self.algorithm == "ppo":

            model_kwargs = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "n_steps": trial.suggest_int('n_steps', 128, 512, step=128),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "n_epochs": trial.suggest_int("n_epochs", 5, 20),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
            }

        else:
            raise ValueError("Unsupported algorithm")

        print("\n" + "=" * 70)
        print(f"Trial {trial.number}")
        print("=" * 70)

        try:
            # -------------------------------------------------------
            # Create environments
            # -------------------------------------------------------
            train_env = DummyVecEnv(
                [
                    lambda: Monitor(
                        self.pipeline.create_env(
                            self.pipeline.train_index,
                            self.pipeline.train_options,
                            mode="train",
                        )
                    )
                ]
            )

            val_env = DummyVecEnv(
                [
                    lambda: Monitor(
                        self.pipeline.create_env(
                            self.pipeline.val_index,
                            self.pipeline.val_options,
                            mode="validation",
                        )
                    )
                ]
            )

            # -------------------------------------------------------
            # Create agent and model
            # -------------------------------------------------------
            agent = DRLAgent(env=train_env)

            model = agent.get_model(
                model_name=self.algorithm,
                model_kwargs=model_kwargs,
                policy="MlpPolicy",
                verbose=0,
                seed=trial.number,
            )

            print(f"\nTraining for {self.n_timesteps:,} timesteps...")

            trained_model = agent.train_model(
                model=model,
                tb_log_name=f"{self.algorithm.upper()}_trial_{trial.number}",
                total_timesteps=self.n_timesteps,
                callbacks=[],
            )

            # -------------------------------------------------------
            # Validation Evaluation
            # -------------------------------------------------------
            print("Evaluating on validation set...")
            mean_reward, std_reward = self._evaluate_model(
                trained_model,
                val_env,
                n_eval_episodes=self.n_eval_episodes,
            )

            print(
                f"Trial {trial.number} Result → "
                f"{mean_reward:.4f} ± {std_reward:.4f}"
            )

            return mean_reward

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return -1e6

        finally:
            try:
                train_env.close()
                val_env.close()
            except Exception:
                pass

    # ===============================================================
    # Evaluation
    # ===============================================================
    def _evaluate_model(self, model, env, n_eval_episodes: int = 5):
        """
        Evaluate model on given environment.
        Compatible with Stable-Baselines3 VecEnv API.
        """

        episode_rewards = []

        for episode in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action, _states = model.predict(obs, deterministic=True)

                # VecEnv returns 4 values (SB3 API)
                obs, reward, done, info = env.step(action)

                episode_reward += reward[0]

            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward
    
    # ===============================================================
    # Optimization Runner
    # ===============================================================
    def optimize_model(self, timeout: int | None = None):
        """
        Run hyperparameter optimization.

        Returns:
            optuna.Study object
        """

        print("\n" + "=" * 70)
        print(f"STARTING HYPERPARAMETER OPTIMIZATION FOR {self.algorithm.upper()}")
        print("=" * 70)

        print("Configuration:")
        print(f"  Number of trials        : {self.n_trials}")
        print(f"  Timesteps per trial     : {self.n_timesteps:,}")
        print(f"  Evaluation episodes     : {self.n_eval_episodes}")
        print(f"  Timeout                 : {timeout if timeout else 'None'}")

        sampler = TPESampler(seed=42)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=self.eval_freq,
        )

        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        print(f"\nStudy: {self.study_name}")
        if self.storage:
            print(f"Storage: {self.storage}")

        try:
            # Count completed trials
            completed_trials = len(
                [
                    t
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
            )

            remaining_trials = max(self.n_trials - completed_trials, 0)

            if remaining_trials == 0:
                print("✓ Required number of trials already completed.")
            else:
                print(f"Running {remaining_trials} additional trials...")
                study.optimize(
                    self.objective_model,
                    n_trials=remaining_trials,
                    timeout=timeout,
                    show_progress_bar=True,
                )

        except KeyboardInterrupt:
            print("\n⚠ Optimization interrupted by user")

        # -----------------------------------------------------------
        # Results Summary
        # -----------------------------------------------------------
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        print(f"Total trials        : {len(study.trials)}")
        print(
            f"Pruned trials       : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
        )
        print(
            f"Completed trials    : {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )

        best_trial = study.best_trial
        print(f"\n✓ Best trial: #{best_trial.number}")
        print(f"  Mean Reward: {best_trial.value:.6f}")

        print("\nBest Hyperparameters:")
        for k, v in best_trial.params.items():
            print(f"  {k:25s}: {v}")

        # Store best parameters
        self.best_params[self.algorithm] = best_trial.params

        # Save study
        # Ensure directory exists
        results_dir = os.path.join(os.getcwd(), "results/optuna/optuna_results")
        os.makedirs(results_dir, exist_ok=True)

        study_path = os.path.join(results_dir, f"{self.study_name}.pkl")
        joblib.dump(study, study_path)
        print(f"\n✓ Study saved to: {study_path}")

        # Save best params
        import json

        params_path = f"./results/optuna/optuna_results/{self.study_name}_best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_trial.params, f, indent=2)

        print(f"✓ Best parameters saved to: {params_path}")

        self._visualize_study(study)

        return study

    # ===============================================================
    # Final Training with Best Params
    # ===============================================================
    def train_with_best_params(self, total_timesteps: int = 200_000):
        """
        Train final model using best hyperparameters.
        """

        if self.algorithm not in self.best_params:
            raise ValueError("No best parameters found. Run optimize_model() first.")

        best_params = self.best_params[self.algorithm]

        print("\n" + "=" * 70)
        print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
        print("=" * 70)

        print("\nBest hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k:25s}: {v}")

        # -----------------------------------------------------------
        # Update pipeline parameters
        # -----------------------------------------------------------
        self.pipeline.reward_scaling = best_params.get(
            "reward_scaling", self.pipeline.reward_scaling
        )

        self.pipeline.train_index = (
            self.pipeline.df_index[
                (self.pipeline.df_index["date"] >= self.pipeline.train_start)
                & (self.pipeline.df_index["date"] < self.pipeline.val_end)
            ]
            .copy()
            .reset_index(drop=True)
        )

        self.pipeline.train_options = (
            self.pipeline.df_options[
                (self.pipeline.df_options["date"] >= self.pipeline.train_start)
                & (self.pipeline.df_options["date"] < self.pipeline.val_end)
            ]
            .copy()
            .reset_index(drop=True)
        )

        # -----------------------------------------------------------
        # Extract model kwargs
        # -----------------------------------------------------------
        model_kwargs = {
            k: v
            for k, v in best_params.items()
            if k != "reward_scaling"
        }

        # -----------------------------------------------------------
        # Train model
        # -----------------------------------------------------------
        trained_model_dict = self.pipeline.train_model(
            algorithms=[self.algorithm],
            total_timesteps=total_timesteps,
            model_kwargs=model_kwargs,
            tensorboard_log="./tensorboard_logs/best_model/",
            eval_freq=self.eval_freq,
            n_eval_episodes=1,
            resume=True,
        )

        trained_model = trained_model_dict[self.algorithm]

        best_model_path = f"./results/optuna/optuna_models/{self.algorithm}_best_hyperparams_final"
        trained_model.save(best_model_path)

        print(f"\n✓ Best model saved to: {best_model_path}")

        self.best_model = trained_model

        return trained_model
    
    # ===============================================================
    # Study Visualization
    # ===============================================================
    def _visualize_study(self, study: optuna.Study):
        """
        Generate Optuna visualization plots and save them as HTML.
        """

        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATION PLOTS")
        print("=" * 70)

        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_slice,
            )

            output_dir = "./results/optuna/optuna_results"

            # 1️⃣ Optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_html(f"{output_dir}/{self.study_name}_optimization_history.html")
            print("✓ Optimization history saved")

            # 2️⃣ Parameter importances
            try:
                fig2 = plot_param_importances(study)
                fig2.write_html(f"{output_dir}/{self.study_name}_param_importances.html")
                print("✓ Parameter importances saved")
            except Exception:
                print("⚠ Could not generate parameter importances (insufficient trials)")

            # 3️⃣ Parallel coordinate plot
            try:
                fig3 = plot_parallel_coordinate(study)
                fig3.write_html(f"{output_dir}/{self.study_name}_parallel_coordinate.html")
                print("✓ Parallel coordinate plot saved")
            except Exception:
                print("⚠ Could not generate parallel coordinate plot")

            # 4️⃣ Slice plot
            try:
                fig4 = plot_slice(study)
                fig4.write_html(f"{output_dir}/{self.study_name}_slice.html")
                print("✓ Slice plot saved")
            except Exception:
                print("⚠ Could not generate slice plot")

            print("\n✓ Visualizations saved to ./optuna_results/")
            print("  Open HTML files in browser to view interactive plots")

        except ImportError:
            print("⚠ Plotly not installed. Skipping visualizations.")
            print("  Install with: pip install plotly kaleido")

    # ===============================================================
    # Trial Comparison
    # ===============================================================
    def compare_trials(self, top_n: int = 10):
        """
        Compare top N completed trials and display parameters.
        """

        if self.algorithm not in self.best_params:
            print("⚠ No study results found. Run optimize_model() first.")
            return

        study_path = f"./results/optuna/optuna_results/{self.study_name}.pkl"

        if not os.path.exists(study_path):
            print(f"⚠ Study file not found: {study_path}")
            return

        study = joblib.load(study_path)

        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed_trials:
            print("⚠ No completed trials available.")
            return

        completed_trials.sort(key=lambda t: t.value, reverse=True)

        print("\n" + "=" * 70)
        print(f"TOP {min(top_n, len(completed_trials))} TRIALS")
        print("=" * 70)

        comparison_data = []

        for rank, trial in enumerate(completed_trials[:top_n], start=1):
            row = {
                "Rank": rank,
                "Trial": trial.number,
                "Mean Reward": trial.value,
            }
            row.update(trial.params)
            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)

        print("\n", df_comparison.to_string(index=False))

        csv_path = f"./results/optuna/optuna_results/{self.study_name}_top_trials.csv"
        df_comparison.to_csv(csv_path, index=False)

        print(f"\n✓ Comparison saved to: {csv_path}")

# ===============================================================
# Optuna Trial Evaluation Callback
# ===============================================================
class TrialEvalCallback(EvalCallback):
    """
    Custom EvalCallback integrated with Optuna for pruning.
    """

    def __init__(self, eval_env, trial: optuna.Trial, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Only report after first evaluation
        if self.eval_idx > 0:
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True
                return False

        self.eval_idx += 1
        return result