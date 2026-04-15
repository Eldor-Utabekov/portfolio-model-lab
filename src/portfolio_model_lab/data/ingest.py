from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def download_price_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV data for the requested tickers from Yahoo Finance."""
    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("tickers must contain at least one symbol")

    raw = yf.download(
        tickers=ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
    )

    if raw.empty:
        raise ValueError("No data returned from yfinance")

    frames: list[pd.DataFrame] = []
    expected_columns = {"date", "open", "high", "low", "close", "volume", "ticker"}

    for ticker in ticker_list:
        if len(ticker_list) == 1:
            ticker_frame = raw.copy()
        else:
            if ticker not in raw.columns.get_level_values(0):
                continue
            ticker_frame = raw[ticker].copy()

        ticker_frame = ticker_frame.reset_index()
        ticker_frame.columns = [
            str(column).strip().lower().replace(" ", "_")
            for column in ticker_frame.columns
        ]
        ticker_frame["ticker"] = ticker

        missing_columns = expected_columns.difference(ticker_frame.columns)
        if missing_columns:
            raise ValueError(
                f"Missing expected columns for {ticker}: {sorted(missing_columns)}"
            )

        frames.append(ticker_frame[list(expected_columns)])

    if not frames:
        raise ValueError("No ticker data could be parsed from yfinance output")

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    return prices


def save_price_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist downloaded price data as a parquet file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    etf_universe = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD", "GLD", "VNQ", "XLE"]

    price_data = download_price_data(
        tickers=etf_universe,
        start_date="2018-01-01",
        end_date="2024-12-31",
    )
    save_price_data(price_data, Path("data/raw/etf_prices.parquet"))

    print(price_data.head())
    print(price_data.shape)