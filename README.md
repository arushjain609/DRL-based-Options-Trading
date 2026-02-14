# Options Trading with Deep Reinforcement Learning

Bachelor Thesis Project\
Author: Arush Jain, Aryan Arya

------------------------------------------------------------------------

## 📌 Overview

This project implements a Deep Reinforcement Learning (DRL) framework
for systematic index options trading.

The framework combines:

-   Custom OpenAI Gymnasium trading environment
-   FinRL DRLAgent wrapper (Stable-Baselines3 backend)
-   A2C and PPO algorithms
-   Optuna-based hyperparameter optimization
-   Comprehensive backtesting with baseline comparisons

The goal is to evaluate whether DRL agents can learn profitable trading
policies in options markets using structured state representations and
realistic portfolio mechanics.

------------------------------------------------------------------------

## 🧠 Key Features

### 1️⃣ Custom Options Trading Environment

-   Characteristic-based action space (moneyness × DTE × option type)
-   Position consolidation within buckets
-   Realistic long/short margin mechanics
-   Option expiration handling
-   Portfolio Greek aggregation (Delta, Gamma, Vega, Theta)
-   Market-level metrics (Put/Call ratio, Skew)

### 2️⃣ Hierarchical State Representation

State includes: - Index technical indicators - Aggregated market options
metrics - Contract-specific features - Portfolio state & exposure -
Margin utilization

### 3️⃣ Training Pipeline

-   Time-based train / validation / test split
-   Early stopping
-   Checkpointing
-   TensorBoard logging
-   Baseline benchmark comparison

### 4️⃣ Hyperparameter Optimization

-   Optuna integration
-   Persistent SQLite storage
-   Median pruning
-   Automatic resume
-   Interactive visualization plots

### 5️⃣ Baseline Comparisons

-   Equal weight
-   Delta weighted
-   Vega weighted
-   Theta weighted
-   IV based
-   Variance weighted
-   Underlying benchmark

------------------------------------------------------------------------

## 📂 Project Structure

    options-trading-drl/
    │
    ├── src/
    │   ├── data_preparation.py
    │   ├── options_env.py
    │   ├── training_pipeline.py
    │   └── hyperparameter_tuning.py
    │
    ├── results/
    │   ├── backtests/
    │   ├── baselines/
    │   ├── optuna/
    │   └── figures/
    │
    ├── run_experiment.py
    ├── README.md
    ├── requirements.txt
    └── .gitignore

------------------------------------------------------------------------

## ⚙️ Installation

Create a virtual environment:

    python -m venv venv
    source venv/bin/activate   # Linux / Mac
    venv\Scripts\activate      # Windows

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

## 🚀 Running the Full Experiment

    python run_experiment.py

This will:

1.  Load and preprocess data
2.  Split into train/validation/test
3.  Run Optuna hyperparameter tuning
4.  Train final model with best parameters
5.  Backtest against benchmarks

------------------------------------------------------------------------


## 🏗 Algorithms Implemented

-   A2C (Advantage Actor Critic)
-   PPO (Proximal Policy Optimization)

Both implemented using Stable-Baselines3 via FinRL wrapper.

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   Total Return
-   Sharpe Ratio
-   Maximum Drawdown
-   Daily Return Statistics
-   Portfolio Greeks Exposure

------------------------------------------------------------------------

## 🔬 Research Contribution

This project:

-   Designs a characteristic-based action representation for options
    trading
-   Introduces a hierarchical observation space combining market and
    contract-level signals
-   Implements realistic margin mechanics for short options
-   Evaluates DRL agents against deterministic option allocation
    baselines

Note: The DRL backbone is based on FinRL and Stable-Baselines3
libraries.

------------------------------------------------------------------------

## 🛠 Technologies Used

-   Python
-   Gymnasium
-   Stable-Baselines3
-   FinRL
-   Optuna
-   Pandas
-   NumPy
-   Matplotlib

------------------------------------------------------------------------

## 📜 License

This project is for academic research purposes.
