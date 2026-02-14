from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

# Technical Analysis library
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

warnings.filterwarnings("ignore")


class DataPreparation:
    """
    Data preparation pipeline for index options trading.

    This class:
    - Loads and cleans options data
    - Downloads index and VIX data
    - Computes technical indicators
    - Filters contracts based on liquidity & moneyness
    - Merges final dataset for RL training
    """

    def __init__(self, symbol: str):
        """
        Parameters
        ----------
        symbol : str
            Underlying symbol (e.g., 'SPX', 'GLD').
        """
        self.symbol = symbol
        self.und_symbol = "^GSPC" if symbol == "SPX" else symbol

        self.start_date = None
        self.end_date = None

        self.index_data = None
        self.vix_data = None
        self.options_data = None
        self.combined_data = None

    # ==========================================================
    # OPTIONS DATA
    # ==========================================================

    def load_options_data(self, path: str) -> pd.DataFrame:
        """
        Load and preprocess options data from CSV.

        Expected columns:
        - date
        - strike
        - expiration
        - bid, ask
        - Call_Put or Call/Put
        - iv, delta, gamma, vega, theta (optional)
        - volume, open_interest (optional)
        """
        print("Loading options data...")

        options_df = pd.read_csv(path)
        options_df = options_df[options_df["symbol"] == self.symbol]

        # Convert dates
        options_df["date"] = pd.to_datetime(
            options_df["date"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

        options_df.sort_values("date", inplace=True)
        options_df.drop_duplicates(inplace=True)
        options_df.reset_index(drop=True, inplace=True)

        # Expiration and DTE
        if "expiration" in options_df.columns:
            options_df["expiration"] = pd.to_datetime(
                options_df["expiration"],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )
            options_df["dte"] = (
                options_df["expiration"] - options_df["date"]
            ).dt.days

        # Mid price calculation
        if {"bid", "ask"}.issubset(options_df.columns):
            options_df["mid_price"] = np.where(
                (options_df["bid"] > 0) & (options_df["ask"] > 0),
                (options_df["bid"] + options_df["ask"]) / 2,
                np.maximum(options_df["bid"], options_df["ask"]),
            )

            options_df["price"] = options_df["mid_price"]
            options_df["bid_ask_spread"] = (
                options_df["ask"] - options_df["bid"]
            )
            options_df["spread_pct"] = (
                options_df["bid_ask_spread"] / options_df["mid_price"]
            )

        # Moneyness
        if {"strike", "Adjusted_close"}.issubset(options_df.columns):
            options_df["moneyness"] = (
                options_df["strike"] / options_df["Adjusted_close"]
            )

        # Option type harmonization
        if "Call_Put" in options_df.columns:
            options_df["option_type"] = options_df["Call_Put"]
        elif "Call/Put" in options_df.columns:
            options_df["option_type"] = options_df["Call/Put"]
        else:
            raise AttributeError(
                "Call/Put or Call_Put column not found in dataset."
            )

        options_df.sort_values("date", inplace=True)
        options_df.reset_index(drop=True, inplace=True)

        self.options_data = options_df

        print(f"Options data loaded: {len(options_df)} rows")
        print(
            f"Date range: {options_df['date'].min()} "
            f"to {options_df['date'].max()}"
        )

        return options_df

    # ==========================================================
    # INDEX & VIX DATA
    # ==========================================================

    def fetch_index_data(self) -> pd.DataFrame:
        """
        Download index OHLCV data using yfinance.
        """
        print("Fetching index data...")

        index = yf.download(
            self.und_symbol,
            start=self.start_date,
            end=self.end_date,
        )

        if isinstance(index.columns, pd.MultiIndex):
            index.columns = index.columns.get_level_values(0)

        index = index.reset_index()
        index.columns = [col.lower() for col in index.columns]

        self.index_data = index

        print(
            f"Index data fetched: {len(index)} rows "
            f"from {index['date'].min()} to {index['date'].max()}"
        )

        return index

    def fetch_vix_data(self) -> pd.DataFrame:
        """
        Download VIX index data.
        """
        print("Fetching VIX data...")

        vix = yf.download("^VIX", start=self.start_date, end=self.end_date)

        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        vix = vix.reset_index()
        vix.columns = [col.lower() for col in vix.columns]

        vix = vix[["date", "close"]].rename(columns={"close": "vix"})

        self.vix_data = vix

        print(f"VIX data fetched: {len(vix)} rows")

        return vix

    # ==========================================================
    # TECHNICAL INDICATORS
    # ==========================================================

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Compute technical indicators on index price data.
        """
        print("Calculating technical indicators...")

        if self.index_data is None:
            raise ValueError("Index data not loaded.")

        df = self.index_data.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(
            df["close"] / df["close"].shift(1)
        )

        # Moving averages
        df["sma_5"] = SMAIndicator(df["close"], 5).sma_indicator()
        df["sma_10"] = SMAIndicator(df["close"], 10).sma_indicator()
        df["sma_20"] = SMAIndicator(df["close"], 20).sma_indicator()
        df["sma_50"] = SMAIndicator(df["close"], 50).sma_indicator()

        df["ema_12"] = EMAIndicator(df["close"], 12).ema_indicator()
        df["ema_26"] = EMAIndicator(df["close"], 26).ema_indicator()

        # MACD
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # RSI
        df["rsi_14"] = RSIIndicator(df["close"], 14).rsi()

        # Bollinger Bands
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()

        # ATR
        df["atr_14"] = AverageTrueRange(
            df["high"], df["low"], df["close"], 14
        ).average_true_range()

        # Stochastic
        stoch = StochasticOscillator(
            df["high"], df["low"], df["close"], 14, 3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Realized volatility
        df["realized_vol_5"] = (
            df["returns"].rolling(5).std() * np.sqrt(252)
        )
        df["realized_vol_20"] = (
            df["returns"].rolling(20).std() * np.sqrt(252)
        )

        # Momentum
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

        # Volume
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # Distance from MA
        df["dist_from_sma20"] = (
            (df["close"] - df["sma_20"]) / df["sma_20"]
        )
        df["dist_from_sma50"] = (
            (df["close"] - df["sma_50"]) / df["sma_50"]
        )

        self.index_data = df

        print(f"Technical indicators calculated. Columns: {len(df.columns)}")

        return df

    # ==========================================================
    # FILTERING
    # ==========================================================

    def filter_options(
        self,
        min_dte: int = 7,
        max_dte: int = 60,
        min_volume: int = 10,
        moneyness_range: tuple = (0.9, 1.1),
    ) -> pd.DataFrame:
        """
        Filter options contracts based on:
        - DTE range
        - Non-zero price history
        - Moneyness band
        """
        print("Filtering options data...")

        if self.options_data is None:
            raise ValueError("Options data not loaded.")

        df = self.options_data.copy()
        initial_count = len(df)

        if "dte" in df.columns:
            df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]

        # Remove contracts with zero price at any time
        bad_contracts = (
            df.groupby("option_symbol")["price"]
            .transform(lambda x: (x <= 0).any())
        )
        df = df[~bad_contracts]

        # Moneyness filter
        df["moneyness"] = df["strike"] / df["Adjusted_close"]
        df = df[
            (df["moneyness"] >= moneyness_range[0])
            & (df["moneyness"] <= moneyness_range[1])
        ]

        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.options_data = df

        print(f"Filtered: {initial_count} → {len(df)} rows")

        self.start_date = df.loc[0, "date"].strftime("%Y-%m-%d")
        self.end_date = df.loc[len(df) - 1, "date"].strftime("%Y-%m-%d")

        return df

    # ==========================================================
    # MERGING
    # ==========================================================

    def combine_data(self) -> pd.DataFrame:
        """
        Merge index and VIX datasets.
        """
        print("Combining index and VIX data...")

        if self.index_data is None:
            raise ValueError("Index data not available.")

        combined = self.index_data.copy()

        if self.vix_data is not None:
            combined = combined.merge(
                self.vix_data, on="date", how="left"
            )
            combined["vix"] = combined["vix"].ffill()

        combined = combined.dropna(
            subset=["sma_50", "rsi_14", "atr_14"]
        )

        self.combined_data = combined.reset_index(drop=True)

        print(
            f"Combined shape: {combined.shape} "
            f"from {combined['date'].min()} "
            f"to {combined['date'].max()}"
        )

        return combined

    # ==========================================================
    # SUMMARY
    # ==========================================================

    def get_data_summary(self) -> None:
        """
        Print summary of prepared datasets.
        """
        print("\n" + "=" * 60)
        print("DATA PREPARATION SUMMARY")
        print("=" * 60)

        if self.index_data is not None:
            print(f"\nIndex Data: {len(self.index_data)} rows")

        if self.options_data is not None:
            print(f"\nOptions Data: {len(self.options_data)} rows")
            if "option_type" in self.options_data.columns:
                print(self.options_data["option_type"].value_counts())

        if self.combined_data is not None:
            print(
                f"\nCombined Dataset: "
                f"{len(self.combined_data)} rows x "
                f"{len(self.combined_data.columns)} columns"
            )

            missing = self.combined_data.isnull().sum()
            if missing.sum() > 0:
                print("\nColumns with missing values:")
                print(missing[missing > 0])