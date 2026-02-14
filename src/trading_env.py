from __future__ import annotations

import warnings
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

warnings.filterwarnings("ignore")

class OptionsEnv(gym.Env):
    """
    Options Trading Environment with realistic position management.

    Key Features
    ------------
    - Unique option contract tracking using `option_symbol`
    - Relative characteristic-based actions (generalizable across datasets)
    - Hierarchical observation space:
        * Index-level technical features
        * Market-level aggregated option metrics
        * Contract-specific bucketed option features
        * Portfolio state metrics
    - Position consolidation within characteristic buckets
    - Automatic expiration handling
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df_index: pd.DataFrame,
        df_options: pd.DataFrame,
        initial_capital: float = 100000,
        transaction_cost_pct: float = 0.001,
        max_contracts_per_position: int = 10,
        risk_free_rate: float = 0.03,
        reward_scaling: float = 1e-2,
    ):
        """
        Initialize the options trading environment.

        Parameters
        ----------
        df_index : pd.DataFrame
            Index data with technical indicators.
        df_options : pd.DataFrame
            Options data (must include 'option_symbol').
        initial_capital : float
            Starting portfolio capital.
        transaction_cost_pct : float
            Transaction cost as a fraction of trade value.
        max_contracts_per_position : int
            Maximum contracts per bucket position.
        risk_free_rate : float
            Annual risk-free rate.
        reward_scaling : float
            Scaling factor applied to rewards.
        """
        super().__init__()

        # Validate required option columns
        required_options_cols = [
            "option_symbol",
            "date",
            "option_type",
            "strike",
            "expiration",
            "price",
            "dte",
            "moneyness",
        ]
        missing_cols = [
            col for col in required_options_cols if col not in df_options.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in df_options: {missing_cols}"
            )

        # Store copies of input data
        self.df_index = df_index.copy()
        self.df_options = df_options.copy()

        # Ensure datetime format
        self.df_index["date"] = pd.to_datetime(self.df_index["date"])
        self.df_options["date"] = pd.to_datetime(self.df_options["date"])
        self.df_options["expiration"] = pd.to_datetime(
            self.df_options["expiration"]
        )

        # Sort chronologically
        self.df_index = (
            self.df_index.sort_values("date").reset_index(drop=True)
        )
        self.df_options = (
            self.df_options.sort_values("date").reset_index(drop=True)
        )

        # Group data by date for fast lookup
        self.options_by_date = {
            d: df.to_dict("records")
            for d, df in self.df_options.groupby("date")
        }
        self.index_by_date = {
            d: df.iloc[0]
            for d, df in self.df_index.groupby("date")
        }

        # Unique trading dates
        self.trading_dates = sorted(self.df_index["date"].unique())

        # Environment parameters
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.max_contracts_per_position = max_contracts_per_position
        self.risk_free_rate = risk_free_rate
        self.reward_scaling = reward_scaling

        # Characteristic buckets
        self.moneyness_buckets = [
            ("deep_otm", 0.90, 0.95),
            ("otm", 0.95, 0.98),
            ("atm", 0.98, 1.02),
            ("itm", 1.02, 1.05),
            ("deep_itm", 1.05, 1.10),
        ]

        self.dte_buckets = [
            ("short", 25, 35),
            ("medium", 55, 65),
            ("long", 85, 95),
        ]

        # State tracking
        self.current_step = 0
        self.cash = initial_capital
        self.positions: Dict = {}
        self.portfolio_value = initial_capital
        self.portfolio_history = []

        # Index-level features
        self.index_features = [
            "close",
            "returns",
            "sma_20",
            "sma_50",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_width",
            "atr_14",
            "realized_vol_20",
            "momentum_10",
            "dist_from_sma20",
            "vix",
        ]

        # Contract-level features
        self.options_contract_features = [
            "moneyness",
            "dte",
            "iv",
            "delta",
            "gamma",
            "vega",
            "theta",
            "volume",
        ]

        self._setup_spaces()

    def _setup_spaces(self):
        """
        Configure observation and action spaces.

        Observation space includes:
        - Index-level features
        - Market-level aggregated option metrics
        - Bucketed contract-specific features
        - Portfolio state metrics

        Action space:
        - Continuous allocation values per (moneyness × DTE × type) bucket
        """

        num_index_features = len(self.index_features)
        num_contract_features = len(self.options_contract_features)

        # Index features
        index_history_size = num_index_features

        # Market-level metrics: [put_call_ratio, skew]
        market_options_metrics_size = 2

        # Bucketed option slots
        num_option_slots = (
            len(self.moneyness_buckets)
            * len(self.dte_buckets)
            * 2
        )
        options_state_size = (
            num_option_slots * num_contract_features
        )

        # Portfolio state metrics
        portfolio_state_size = 8

        total_state_size = (
            index_history_size
            + market_options_metrics_size
            + options_state_size
            + portfolio_state_size
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_state_size,),
            dtype=np.float32,
        )

        # Action space:
        # 5 moneyness × 3 DTE × 2 types = 30 categories
        num_categories = (
            len(self.moneyness_buckets)
            * len(self.dte_buckets)
            * 2
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_categories,),
            dtype=np.float32,
        )
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.

        Returns
        -------
        observation : np.ndarray
        info : dict
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.portfolio_history = [self.initial_capital]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one time step in the environment.

        Parameters
        ----------
        action : np.ndarray
            Continuous action vector of size 30
            (moneyness × DTE × call/put categories).

        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        current_date = self.trading_dates[self.current_step]

        # Update existing positions
        self._update_positions(current_date)

        positions_value = sum(
            pos["market_value"] for pos in self.positions.values()
        )
        margin_held = sum(
            pos["margin_held"] for pos in self.positions.values()
        )
        self.portfolio_value = (
            self.cash + positions_value + margin_held
        )

        # Termination condition
        terminated = (
            self.current_step >= len(self.trading_dates) - 1
            or self.portfolio_value <= self.initial_capital * 0.1
        )

        # Execute actions
        total_transaction_cost = self._execute_all_actions(
            action, current_date
        )

        # Update positions again after trades
        self._update_positions(current_date)

        positions_value = sum(
            pos["market_value"] for pos in self.positions.values()
        )
        margin_held = sum(
            pos["margin_held"] for pos in self.positions.values()
        )
        self.portfolio_value = (
            self.cash + positions_value + margin_held
        )

        self.portfolio_history.append(self.portfolio_value)

        # Reward calculation
        reward = self._calculate_reward()

        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        # Advance step
        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def _get_bucket_name(
        self, moneyness_idx: int, dte_idx: int
    ) -> Tuple[str, str]:
        """
        Get bucket names from index positions.

        Returns
        -------
        tuple[str, str]
            (moneyness_bucket_name, dte_bucket_name)
        """
        moneyness_name = self.moneyness_buckets[moneyness_idx][0]
        dte_name = self.dte_buckets[dte_idx][0]
        return moneyness_name, dte_name

    def _is_contract_in_category(
        self,
        position: Dict,
        target_moneyness_bucket: int,
        target_dte_bucket: int,
    ) -> bool:
        """
        Check whether a contract still belongs to its assigned bucket.

        Returns
        -------
        bool
            True if still in same category, False otherwise.
        """
        current_moneyness = position["moneyness"]
        current_dte = position["dte"]

        # Moneyness bucket check
        _, m_min, m_max = self.moneyness_buckets[
            target_moneyness_bucket
        ]
        in_moneyness_bucket = (
            m_min <= current_moneyness < m_max
        )

        # DTE bucket check
        _, d_min, d_max = self.dte_buckets[target_dte_bucket]
        in_dte_bucket = (
            d_min <= current_dte <= d_max
        )

        return in_moneyness_bucket and in_dte_bucket

    def _pos_key(self, opt_type, m_idx, d_idx):
        """
        Create a unique key for position bucket.
        """
        m_name = self.moneyness_buckets[m_idx][0]
        d_name = self.dte_buckets[d_idx][0]
        return (opt_type, m_name, d_name)
    
    def _execute_all_actions(
        self,
        actions: np.ndarray,
        current_date,
    ) -> float:
        """
        Execute all category-based actions using portfolio-weighted allocation.

        Capital Allocation Logic
        ------------------------
        - Total portfolio target = current portfolio value
        - Allocation proportional to |action|
        - Actions normalized to sum to 1.0
        - Two-phase execution: sell first, then buy
        """

        total_transaction_cost = 0.0

        # Normalize actions
        abs_actions = np.abs(actions)
        total_abs_action = abs_actions.sum()

        # If no meaningful signal → close all positions
        if total_abs_action < 0.01:
            for key in list(self.positions.keys()):
                total_transaction_cost += self._close_position(
                    key, current_date
                )
            return total_transaction_cost

        allocation_weights = abs_actions / total_abs_action
        total_portfolio_value = self.portfolio_value

        planned_trades = []
        action_idx = 0

        # ---------------------------------------------------------
        # PHASE 1: Build trade plan
        # ---------------------------------------------------------
        for m_idx, _ in enumerate(self.moneyness_buckets):
            for d_idx, _ in enumerate(self.dte_buckets):
                for option_type in ["C", "P"]:

                    action_value = actions[action_idx]
                    allocation_weight = allocation_weights[action_idx]
                    action_idx += 1

                    key = self._pos_key(option_type, m_idx, d_idx)

                    # Close category if near zero
                    if abs(action_value) < 0.01:
                        planned_trades.append({
                            "type": "close",
                            "option_type": option_type,
                            "m_idx": m_idx,
                            "d_idx": d_idx,
                            "key": key,
                            "priority": 0,
                        })
                        continue

                    direction = "long" if action_value > 0 else "short"
                    existing_position = self.positions.get(key)

                    option_symbol = (
                        existing_position["option_symbol"]
                        if existing_position is not None
                        else ""
                    )

                    # Find best contract for bucket
                    target_contract = self._find_contract_by_characteristics(
                        current_date,
                        option_type,
                        m_idx,
                        d_idx,
                    )

                    if target_contract is None:
                        if existing_position is not None:
                            planned_trades.append({
                                "type": "close_existing",
                                "option_type": option_type,
                                "m_idx": m_idx,
                                "d_idx": d_idx,
                                "existing_position": existing_position,
                                "key": key,
                                "priority": 0,
                            })
                        continue

                    contract_price = target_contract["price"]
                    if contract_price <= 0:
                        continue

                    # Determine target quantity
                    allocated_capital = (
                        total_portfolio_value * allocation_weight
                    )
                    max_contracts = int(
                        allocated_capital / (contract_price * 100)
                    )
                    target_quantity = min(
                        max_contracts,
                        self.max_contracts_per_position,
                    )
                    target_quantity = max(1, target_quantity)

                    current_quantity = (
                        existing_position["quantity"]
                        if existing_position
                        else 0
                    )

                    # Check if close + reopen required
                    needs_close_reopen = False
                    if existing_position is not None:
                        if (
                            existing_position["direction"] != direction
                            or option_symbol
                            != target_contract["option_symbol"]
                        ):
                            needs_close_reopen = True

                    # -------------------------------------------------
                    # Trade categorization
                    # -------------------------------------------------
                    if needs_close_reopen:
                        planned_trades.append({
                            "type": "close_existing",
                            "option_type": option_type,
                            "m_idx": m_idx,
                            "d_idx": d_idx,
                            "existing_position": existing_position,
                            "key": key,
                            "priority": 0,
                        })

                        planned_trades.append({
                            "type": "open_new",
                            "target_contract": target_contract,
                            "target_quantity": target_quantity,
                            "direction": direction,
                            "m_idx": m_idx,
                            "d_idx": d_idx,
                            "key": key,
                            "priority": 2,
                        })

                    elif existing_position:

                        if target_quantity < current_quantity:
                            priority = (
                                1 if direction == "long" else 3
                            )
                        elif target_quantity > current_quantity:
                            priority = (
                                3 if direction == "long" else 1
                            )
                        else:
                            continue

                        planned_trades.append({
                            "type": "rebalance",
                            "existing_position": existing_position,
                            "target_contract": target_contract,
                            "target_quantity": target_quantity,
                            "current_quantity": current_quantity,
                            "direction": direction,
                            "key": key,
                            "priority": priority,
                        })

                    else:
                        planned_trades.append({
                            "type": "open_new",
                            "target_contract": target_contract,
                            "target_quantity": target_quantity,
                            "direction": direction,
                            "m_idx": m_idx,
                            "d_idx": d_idx,
                            "key": key,
                            "priority": 3,
                        })

        # ---------------------------------------------------------
        # PHASE 2: Sort trades by execution priority
        # ---------------------------------------------------------
        planned_trades.sort(key=lambda x: x["priority"])

        # ---------------------------------------------------------
        # PHASE 3: Execute trades
        # ---------------------------------------------------------
        for trade in planned_trades:
            try:
                if trade["type"] in ("close", "close_existing"):
                    cost = self._close_position(
                        trade["key"], current_date
                    )

                elif trade["type"] == "rebalance":
                    cost = self._rebalance_position(
                        trade["target_contract"],
                        trade["target_quantity"],
                        trade["direction"],
                        trade["key"],
                        current_date,
                    )

                elif trade["type"] == "open_new":
                    cost = self._open_new_position(
                        trade["target_contract"],
                        trade["target_quantity"],
                        trade["direction"],
                        trade["m_idx"],
                        trade["d_idx"],
                        trade["key"],
                        current_date,
                    )

                else:
                    continue

                total_transaction_cost += cost

            except Exception as e:
                print(f"Trade execution error: {e}")
                continue

        return total_transaction_cost

    def _find_contract_by_characteristics(
        self,
        current_date,
        option_type: str,
        moneyness_bucket: int,
        dte_bucket: int
    ) -> Optional[pd.Series]:
        """
        Find highest-volume contract within a relative (moneyness, DTE) bucket.
        """

        options_today = self.options_by_date.get(current_date)
        if options_today is None:
            return None

        _, m_min, m_max = self.moneyness_buckets[moneyness_bucket]
        _, d_min, d_max = self.dte_buckets[dte_bucket]

        filtered = [
            o for o in options_today
            if (
                o["option_type"] == option_type
                and m_min <= o["moneyness"] < m_max
                and d_min <= o["dte"] <= d_max
            )
        ]

        if not filtered:
            return None

        df_filtered = pd.DataFrame(filtered)
        df_filtered = df_filtered.sort_values("volume", ascending=False)

        return df_filtered.iloc[0]
    
    def _open_new_position(
        self,
        contract: pd.Series,
        quantity: int,
        direction: str,
        moneyness_bucket: int,
        dte_bucket: int,
        key: tuple,
        current_date
    ) -> float:
        """
        Open a new long or short option position.
        """

        price = contract["price"]
        total_cost = price * quantity * 100
        transaction_cost = total_cost * self.transaction_cost_pct

        moneyness_name, dte_name = self._get_bucket_name(
            moneyness_bucket,
            dte_bucket
        )

        if direction == "long":

            total_debit = total_cost + transaction_cost
            if total_debit > self.cash:
                return 0.0

            position = {
                "option_symbol": contract["option_symbol"],
                "type": contract["option_type"],
                "direction": direction,
                "strike": contract["strike"],
                "expiration": contract["expiration"],
                "dte": contract["dte"],
                "moneyness": contract["moneyness"],
                "moneyness_bucket": moneyness_name,
                "dte_bucket": dte_name,
                "quantity": quantity,
                "entry_price": price,
                "entry_date": current_date,
                "delta": contract.get("delta", 0) * quantity,
                "gamma": contract.get("gamma", 0) * quantity,
                "vega": contract.get("vega", 0) * quantity,
                "theta": contract.get("theta", 0) * quantity,
                "market_value": total_cost,
                "margin_held": 0,
            }

            self.positions[key] = position
            self.cash -= total_debit

        else:  # short

            spot_price = self._get_current_spot_price(current_date)
            margin_required = spot_price * quantity * 100 * 0.2

            if margin_required > self.cash:
                return 0.0

            total_credit = total_cost - transaction_cost

            position = {
                "option_symbol": contract["option_symbol"],
                "type": contract["option_type"],
                "direction": direction,
                "strike": contract["strike"],
                "expiration": contract["expiration"],
                "dte": contract["dte"],
                "moneyness": contract["moneyness"],
                "moneyness_bucket": moneyness_name,
                "dte_bucket": dte_name,
                "quantity": quantity,
                "entry_price": price,
                "entry_date": current_date,
                "delta": -contract.get("delta", 0) * quantity,
                "gamma": -contract.get("gamma", 0) * quantity,
                "vega": -contract.get("vega", 0) * quantity,
                "theta": -contract.get("theta", 0) * quantity,
                "market_value": -total_cost,
                "margin_held": margin_required,
            }

            self.positions[key] = position
            self.cash += total_credit
            self.cash -= margin_required

        return transaction_cost
    
    def _rebalance_position(
        self,
        target_contract: pd.Series,
        target_quantity: int,
        direction: str,
        key: tuple,
        current_date
    ) -> float:
        """
        Adjust existing position quantity toward target_quantity.
        """

        position = self.positions.get(key)
        current_quantity = position["quantity"]
        quantity_diff = target_quantity - current_quantity

        if quantity_diff == 0:
            return 0.0

        current_price = self._get_option_price_by_symbol(
            position["option_symbol"],
            current_date,
        )

        if current_price is None:
            current_price = target_contract["price"]

        total_cost = abs(quantity_diff) * current_price * 100
        transaction_cost = total_cost * self.transaction_cost_pct

        if quantity_diff > 0:
            # Increase position
            if direction == "long":

                total_debit = total_cost + transaction_cost
                if total_debit > self.cash:
                    affordable_qty = int(
                        self.cash
                        / (current_price * 100 * (1 + self.transaction_cost_pct))
                    )
                    if affordable_qty <= 0:
                        return 0.0

                    quantity_diff = affordable_qty
                    total_cost = quantity_diff * current_price * 100
                    transaction_cost = total_cost * self.transaction_cost_pct
                    total_debit = total_cost + transaction_cost

                position["quantity"] += quantity_diff
                self.cash -= total_debit

            else:  # short

                spot_price = self._get_current_spot_price(current_date)
                margin_required = spot_price * quantity_diff * 100 * 0.2

                if margin_required > self.cash:
                    return 0.0

                total_credit = total_cost - transaction_cost
                position["quantity"] += quantity_diff
                position["margin_held"] += margin_required

                self.cash += total_credit
                self.cash -= margin_required

        else:
            # Reduce position
            quantity_diff = abs(quantity_diff)

            if direction == "long":
                proceeds = total_cost - transaction_cost
                self.cash += proceeds
                position["quantity"] -= quantity_diff

            else:  # short

                cost = total_cost + transaction_cost
                if cost > self.cash:
                    return 0.0

                margin_to_release = (
                    position.get("margin_held", 0)
                    * (quantity_diff / current_quantity)
                )

                self.cash -= cost
                self.cash += margin_to_release
                position["quantity"] -= quantity_diff
                position["margin_held"] -= margin_to_release

        self.positions[key] = position

        return transaction_cost
    
    def _close_position(self, key: tuple, current_date) -> float:
        """
        Close an open position and settle cash/margin.
        """

        position = self.positions.get(key)
        if position is None:
            return 0.0

        current_price = self._get_option_price_by_symbol(
            position["option_symbol"],
            current_date,
        )

        if current_price is None:
            current_price = position["entry_price"]

        total_value = current_price * position["quantity"] * 100
        transaction_cost = total_value * self.transaction_cost_pct

        if position["direction"] == "long":
            proceeds = total_value - transaction_cost
            self.cash += proceeds
        else:
            cost = total_value + transaction_cost
            self.cash -= cost
            self.cash += position.get("margin_held", 0)

        del self.positions[key]

        return transaction_cost
    
    def _update_positions(self, current_date):
        """
        Update open positions:
        - Handle expiration settlement
        - Mark-to-market pricing
        - Update Greeks
        """

        for key in list(self.positions.keys()):

            position = self.positions.get(key)

            # -------------------------------------------------
            # Expiration handling
            # -------------------------------------------------
            if current_date >= position["expiration"]:

                spot_price = self._get_current_spot_price(current_date)
                strike = position["strike"]

                if position["type"] == "C":
                    intrinsic_value = max(0, spot_price - strike)
                else:
                    intrinsic_value = max(0, strike - spot_price)

                settlement = intrinsic_value * position["quantity"] * 100

                if position["direction"] == "long":
                    self.cash += settlement
                else:
                    self.cash -= settlement
                    self.cash += position.get("margin_held", 0)

                del self.positions[key]
                continue

            # -------------------------------------------------
            # Mark-to-market update
            # -------------------------------------------------
            current_price = self._get_option_price_by_symbol(
                position["option_symbol"],
                current_date,
            )

            if current_price is None:
                continue

            if position["direction"] == "long":
                position["market_value"] = (
                    current_price * position["quantity"] * 100
                )
                multiplier = 1
            else:
                position["market_value"] = (
                    -current_price * position["quantity"] * 100
                )
                multiplier = -1

            option_data = self._get_option_data_by_symbol(
                position["option_symbol"],
                current_date,
            )

            if option_data is not None:
                position["delta"] = (
                    option_data.get("delta", 0)
                    * position["quantity"]
                    * multiplier
                )
                position["gamma"] = (
                    option_data.get("gamma", 0)
                    * position["quantity"]
                    * multiplier
                )
                position["vega"] = (
                    option_data.get("vega", 0)
                    * position["quantity"]
                    * multiplier
                )
                position["theta"] = (
                    option_data.get("theta", 0)
                    * position["quantity"]
                    * multiplier
                )
                position["dte"] = option_data.get("dte", position["dte"])
                position["moneyness"] = option_data.get(
                    "moneyness",
                    position["moneyness"],
                )

            self.positions[key] = position
    
    def _get_option_price_by_symbol(
        self,
        option_symbol: str,
        current_date
    ) -> Optional[float]:
        """
        Return option price for a given symbol on a given date.
        """

        options_today = self.options_by_date.get(current_date)
        if options_today is None:
            return None

        for o in options_today:
            if o["option_symbol"] == option_symbol:
                return o["price"]

        return None
    
    def _get_option_data_by_symbol(
        self,
        option_symbol: str,
        current_date
    ) -> Optional[dict]:
        """
        Return full option record for a given symbol and date.
        """

        options_today = self.options_by_date.get(current_date)
        if options_today is None:
            return None

        for o in options_today:
            if o["option_symbol"] == option_symbol:
                return o

        return None
    
    def _get_current_spot_price(self, current_date) -> float:
        """
        Return underlying spot price for the given date.
        """

        index_today = self.index_by_date.get(current_date)
        if index_today is not None:
            return index_today["close"]

        return 0.0
    
    def _get_available_options_by_buckets(self, current_date) -> Dict:
        """
        Return highest-volume option per (type, moneyness, DTE) bucket.
        Includes held contracts to ensure continuity.
        """

        options_today = self.options_by_date.get(current_date)
        if options_today is None:
            return {}

        held_symbols = {
            pos["option_symbol"] for pos in self.positions.values()
        }

        bucketed_options = {}

        for m_name, m_min, m_max in self.moneyness_buckets:
            for d_name, d_min, d_max in self.dte_buckets:
                for opt_type in ("C", "P"):

                    bucket_key = f"{opt_type}_{m_name}_{d_name}"
                    candidates = {}

                    for opt in options_today:

                        if opt["option_type"] != opt_type:
                            continue

                        if not (m_min <= opt["moneyness"] < m_max):
                            continue

                        if not (d_min <= opt["dte"] <= d_max):
                            continue

                        symbol = opt["option_symbol"]

                        if symbol in held_symbols or symbol not in candidates:
                            candidates[symbol] = opt

                    if not candidates:
                        continue

                    best_option = max(
                        candidates.values(),
                        key=lambda x: x.get("volume", 0.0),
                    )

                    bucketed_options[bucket_key] = best_option

        return bucketed_options

    def _calculate_market_options_metrics(
        self,
        current_date
    ) -> Dict[str, float]:
        """
        Compute:
        - Put-call ratio (volume based)
        - OTM skew (put IV - call IV)
        """

        options_today = self.options_by_date.get(current_date)
        if options_today is None:
            return {
                "put_call_ratio": 1.0,
                "skew": 0.0,
            }

        put_volume = 0.0
        call_volume = 0.0

        otm_put_iv_sum = 0.0
        otm_put_count = 0

        otm_call_iv_sum = 0.0
        otm_call_count = 0

        for opt in options_today:

            opt_type = opt["option_type"]
            volume = opt.get("volume", 0.0)
            moneyness = opt.get("moneyness", 0.0)
            iv = opt.get("iv", 0.0)

            if opt_type == "P":
                put_volume += volume
            elif opt_type == "C":
                call_volume += volume

            if opt_type == "P" and 0.90 <= moneyness < 0.95:
                otm_put_iv_sum += iv
                otm_put_count += 1

            elif opt_type == "C" and 1.05 < moneyness <= 1.10:
                otm_call_iv_sum += iv
                otm_call_count += 1

        put_call_ratio = (
            put_volume / (call_volume + 1e-6)
            if call_volume > 0
            else 1.0
        )

        otm_put_iv = (
            otm_put_iv_sum / otm_put_count
            if otm_put_count > 0
            else 0.0
        )

        otm_call_iv = (
            otm_call_iv_sum / otm_call_count
            if otm_call_count > 0
            else 0.0
        )

        skew = otm_put_iv - otm_call_iv

        return {
            "put_call_ratio": put_call_ratio,
            "skew": skew,
        }
    
    def _calculate_reward(self) -> float:
        """
        Compute step reward:
        - Portfolio return
        - Risk penalty (delta exposure)
        - Exploration bonus
        """

        if len(self.portfolio_history) < 2:
            return 0.0

        current_value = self.portfolio_history[-1]
        previous_value = self.portfolio_history[-2]

        returns = (current_value - previous_value) / previous_value

        portfolio_greeks = self._calculate_portfolio_greeks()

        risk_penalty = 0.0
        if abs(portfolio_greeks["delta"]) > 100:
            risk_penalty += (
                0.001 * abs(portfolio_greeks["delta"]) / 100
            )

        num_positions = len(self.positions)

        exploration_bonus = 0.0
        if num_positions == 0:
            exploration_bonus = -0.001
        elif 0 < num_positions < 5:
            exploration_bonus = 0.0005 * num_positions

        reward = (
            returns
            - risk_penalty
            + exploration_bonus
        ) * self.reward_scaling

        return reward
    
    def _calculate_portfolio_greeks(self) -> Dict[str, float]:
        """
        Aggregate portfolio Greeks across all open positions.
        """

        total_delta = sum(pos["delta"] for pos in self.positions.values())
        total_gamma = sum(pos["gamma"] for pos in self.positions.values())
        total_vega = sum(pos["vega"] for pos in self.positions.values())
        total_theta = sum(pos["theta"] for pos in self.positions.values())

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega": total_vega,
            "theta": total_theta,
        }
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct hierarchical observation vector:
        1. Index features
        2. Market-level option metrics
        3. Bucketed contract features
        4. Portfolio state
        """

        obs_components = []
        current_date = self.trading_dates[self.current_step]

        # -------------------------------------------------
        # 1. Index features
        # -------------------------------------------------
        historical_data = self.df_index.iloc[
            self.current_step : self.current_step
        ][self.index_features]

        historical_array = historical_data.values.flatten()

        expected_size = len(self.index_features)
        if len(historical_array) < expected_size:
            padding = np.zeros(expected_size - len(historical_array))
            historical_array = np.concatenate([padding, historical_array])

        obs_components.append(historical_array)

        # -------------------------------------------------
        # 2. Market-level metrics
        # -------------------------------------------------
        market_metrics = self._calculate_market_options_metrics(current_date)

        market_metrics_array = np.array([
            market_metrics["put_call_ratio"],
            market_metrics["skew"],
        ])

        obs_components.append(market_metrics_array)

        # -------------------------------------------------
        # 3. Bucketed contract features
        # -------------------------------------------------
        bucketed_options = self._get_available_options_by_buckets(current_date)

        num_buckets = (
            len(self.moneyness_buckets)
            * len(self.dte_buckets)
            * 2
        )

        num_features = len(self.options_contract_features)

        options_array = np.zeros(num_buckets * num_features)

        slot_idx = 0

        for m_bucket in self.moneyness_buckets:
            for d_bucket in self.dte_buckets:
                for opt_type in ("C", "P"):

                    bucket_key = f"{opt_type}_{m_bucket[0]}_{d_bucket[0]}"

                    if bucket_key in bucketed_options:
                        option_row = bucketed_options[bucket_key]

                        features = [
                            option_row.get(feat, 0.0)
                            for feat in self.options_contract_features
                        ]

                        start = slot_idx * num_features
                        end = start + num_features
                        options_array[start:end] = features

                    slot_idx += 1

        obs_components.append(options_array)

        # -------------------------------------------------
        # 4. Portfolio state
        # -------------------------------------------------
        portfolio_greeks = self._calculate_portfolio_greeks()

        max_positions = (
            len(self.moneyness_buckets)
            * len(self.dte_buckets)
            * 2
        )

        margin_used = sum(
            pos["margin_held"] for pos in self.positions.values()
        )

        portfolio_state = np.array([
            self.cash / self.initial_capital,
            len(self.positions) / max_positions,
            (self.portfolio_value - self.initial_capital)
            / self.initial_capital,
            portfolio_greeks["delta"] / 100,
            portfolio_greeks["gamma"] / 10,
            portfolio_greeks["vega"] / 1000,
            portfolio_greeks["theta"] / 100,
            margin_used / self.portfolio_value
            if self.portfolio_value != 0 else 0.0,
        ])

        obs_components.append(portfolio_state)

        observation = np.concatenate(obs_components).astype(np.float32)

        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        return observation
    
    def _get_info(self) -> Dict:
        """
        Return environment diagnostics.
        """

        portfolio_greeks = self._calculate_portfolio_greeks()

        return {
            "step": self.current_step,
            "date": str(self.trading_dates[self.current_step]),
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "num_positions": len(self.positions),
            "total_pnl": self.portfolio_value - self.initial_capital,
            "pnl_pct": (
                (self.portfolio_value - self.initial_capital)
                / self.initial_capital
                * 100
            ),
            "portfolio_greeks": portfolio_greeks,
            "positions": [
                {
                    "symbol": pos["option_symbol"],
                    "type": pos["type"],
                    "direction": pos["direction"],
                    "quantity": pos["quantity"],
                    "dte": pos["dte"],
                }
                for pos in self.positions.values()
            ],
        }
    
    def render(self, mode="human"):
        """
        Print current environment state.
        """

        if mode != "human":
            return

        info = self._get_info()

        print(f"\n{'=' * 60}")
        print(f"Step: {info['step']} | Date: {info['date']}")
        print(
            f"Portfolio Value: ${info['portfolio_value']:,.2f} "
            f"| Cash: ${info['cash']:,.2f}"
        )
        print(
            f"PnL: ${info['total_pnl']:,.2f} "
            f"({info['pnl_pct']:.2f}%)"
        )
        print(f"Positions: {info['num_positions']}")

        if info["positions"]:
            print("\nOpen Positions:")
            for i, pos in enumerate(info["positions"], 1):
                print(
                    f"  {i}. {pos['symbol']} | "
                    f"{pos['type'].upper()} {pos['direction'].upper()} | "
                    f"Qty: {pos['quantity']} | DTE: {pos['dte']}"
                )

        greeks = info["portfolio_greeks"]
        print(
            f"\nGreeks - "
            f"Delta: {greeks['delta']:.2f}, "
            f"Gamma: {greeks['gamma']:.4f}, "
            f"Vega: {greeks['vega']:.2f}, "
            f"Theta: {greeks['theta']:.2f}"
        )

        print(f"{'=' * 60}")