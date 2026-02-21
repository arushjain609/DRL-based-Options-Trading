from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, LOGS_DIR, RESULTS_DIR, HyperOptConfig, DataConfig

hopt_cfg = HyperOptConfig()
data_cfg = DataConfig()

OPTION_SYMBOL = data_cfg.option_symbol
ALGORITHMS = hopt_cfg.algorithms

for algorithm in ALGORITHMS:

    progress_path = os.path.join(
        RESULTS_DIR, algorithm.lower(), "progress.csv"
    )

    df = pd.read_csv(progress_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    x = df["time/total_timesteps"]

    train_log_path = os.path.join(
        LOGS_DIR, algorithm.lower(), "train_monitor.csv"
    )

    train_log = pd.read_csv(train_log_path)
    train_rewards = np.array(
        train_log.iloc[1:].index.astype(float).to_list()
    )

    # ---------------------------------------------------------
    # 1️⃣ Training Episodic Reward
    # ---------------------------------------------------------
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(
            np.linspace(1, len(train_rewards), len(train_rewards)),
            train_rewards,
        )
        plt.xlabel("Episodes")
        plt.ylabel("Training Episodic Reward")
        plt.title(f"Training Reward - {algorithm.upper()} - {OPTION_SYMBOL}")
        plt.grid(True)

        save_path = os.path.join(
            FIGURES_DIR,
            f"{algorithm}_training_reward.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

    except:
        pass

    # ---------------------------------------------------------
    # 2️⃣ Evaluation Mean Reward
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(x, df["eval/mean_reward"])
    plt.xlabel("Timesteps")
    plt.ylabel("Eval Mean Reward")
    plt.title(f"Evaluation Reward - {algorithm.upper()} - {OPTION_SYMBOL}")
    plt.grid(True)

    save_path = os.path.join(
        FIGURES_DIR,
        f"{algorithm}_evaluation_reward.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3️⃣ Entropy
    # ---------------------------------------------------------
    if "train/entropy_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(x, df["train/entropy_loss"])
        plt.xlabel("Timesteps")
        plt.ylabel("Entropy Loss")
        plt.title(f"Entropy - {algorithm.upper()} - {OPTION_SYMBOL}")
        plt.grid(True)

        save_path = os.path.join(
            FIGURES_DIR,
            f"{algorithm}_entropy.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # 4️⃣ Policy Loss
    # ---------------------------------------------------------
    if algorithm.lower() == "a2c" and "train/policy_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(x, df["train/policy_loss"])
        plt.title(f"Policy Loss - {algorithm.upper()} - {OPTION_SYMBOL}")
        plt.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, f"{algorithm}_policy_loss.png"), dpi=300)
        plt.close()

    if algorithm.lower() == "ppo" and "train/policy_gradient_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(x, df["train/policy_gradient_loss"])
        plt.title(f"Policy Gradient Loss - {algorithm.upper()} - {OPTION_SYMBOL}")
        plt.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, f"{algorithm}_policy_loss.png"), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # 5️⃣ Value Loss
    # ---------------------------------------------------------
    if "train/value_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(x, df["train/value_loss"])
        plt.title(f"Value Loss - {algorithm.upper()} - {OPTION_SYMBOL}")
        plt.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, f"{algorithm}_value_loss.png"), dpi=300)
        plt.close()

print(f"\n✓ All graphs saved to: {FIGURES_DIR}")