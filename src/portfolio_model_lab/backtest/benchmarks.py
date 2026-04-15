from __future__ import annotations

import pandas as pd


def equal_weight_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Assign equal weights across all assets on each date."""
    result = df.copy()

    weights = []
    for date, group in result.groupby("date"):
        group = group.copy()
        group["weight"] = 1.0 / len(group)
        weights.append(group)

    return pd.concat(weights, ignore_index=True)


def buy_and_hold_spy(df: pd.DataFrame) -> pd.DataFrame:
    """Allocate 100% to SPY and 0% to all other assets."""
    result = df.copy()
    result["weight"] = 0.0
    result.loc[result["ticker"] == "SPY", "weight"] = 1.0
    return result