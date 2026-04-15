from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_price_data(input_path: str | Path) -> pd.DataFrame:
    """Load raw price data from parquet."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("Loaded price data is empty")

    return df


def validate_price_data(df: pd.DataFrame) -> None:
    """Run basic integrity checks on the raw price dataset."""
    required_columns = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df[["date", "ticker"]].duplicated().any():
        raise ValueError("Found duplicate (date, ticker) rows")

    if df["close"].isna().any():
        raise ValueError("Found missing values in close column")


def prepare_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize types and ordering before feature construction."""
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)

    return prepared


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple historical return features by ticker."""
    result = df.copy()

    grouped_close = result.groupby("ticker")["close"]
    result["return_1d"] = grouped_close.pct_change()
    result["return_5d"] = grouped_close.pct_change(periods=5)
    result["return_20d"] = grouped_close.pct_change(periods=20)

    return result


def save_prepared_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the prepared price dataset to parquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    raw_path = Path("data/raw/etf_prices.parquet")
    output_path = Path("data/processed/etf_prices_prepared.parquet")

    prices = load_price_data(raw_path)
    validate_price_data(prices)
    prices = prepare_price_data(prices)
    prices = add_return_features(prices)
    save_prepared_data(prices, output_path)

    preview_columns = ["ticker", "date", "close", "return_1d", "return_5d", "return_20d"]

    print(prices.head())
    print(prices[preview_columns].head(10))
    print(prices.shape)