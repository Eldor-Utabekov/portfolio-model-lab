from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_backtest_inputs(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Attach next-day realized returns to each date-ticker row for backtesting."""
    df = prediction_df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["forward_return_1d"] = (
        df.groupby("ticker")["close"].shift(-1).div(df["close"]).sub(1.0)
    )

    return df


def compute_portfolio_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily portfolio returns and cumulative NAV."""
    if "weight" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'weight' column.")
    if "forward_return_1d" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'forward_return_1d' column.")

    result = df.copy()
    result = result.dropna(subset=["forward_return_1d"]).copy()
    result["weighted_return"] = result["weight"] * result["forward_return_1d"]

    portfolio_returns = (
        result.groupby("date", as_index=False)["weighted_return"]
        .sum()
        .rename(columns={"weighted_return": "portfolio_return"})
    )
    portfolio_returns["nav"] = (1.0 + portfolio_returns["portfolio_return"]).cumprod()

    return portfolio_returns


def save_backtest_results(df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist backtest results to parquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)