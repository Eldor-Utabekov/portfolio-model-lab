from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prepared_data(path: str | Path) -> pd.DataFrame:
    """Load the prepared price dataset used for feature generation."""
    return pd.read_parquet(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create backward-looking features for each ticker."""
    result = df.copy()

    grouped_returns = result.groupby("ticker")["return_1d"]
    grouped_close = result.groupby("ticker")["close"]

    result["volatility_20d"] = (
        grouped_returns.rolling(window=20).std().reset_index(level=0, drop=True)
    )

    result["mean_return_20d"] = (
        grouped_returns.rolling(window=20).mean().reset_index(level=0, drop=True)
    )

    result["momentum_20d"] = grouped_close.pct_change(periods=20)

    rolling_mean_20d = (
        grouped_close.rolling(window=20).mean().reset_index(level=0, drop=True)
    )
    result["ma_ratio_20d"] = result["close"] / rolling_mean_20d

    rolling_max_20d = (
        grouped_close.rolling(window=20).max().reset_index(level=0, drop=True)
    )
    result["drawdown_20d"] = result["close"] / rolling_max_20d - 1

    return result


def save_features(df: pd.DataFrame, path: str | Path) -> None:
    """Persist the feature dataset to parquet."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    input_path = Path("data/processed/etf_prices_prepared.parquet")
    output_path = Path("data/processed/etf_features.parquet")

    features = load_prepared_data(input_path)
    features = build_features(features)
    save_features(features, output_path)

    print(features.head())
    print(features.shape)