from __future__ import annotations

import pandas as pd


def construct_weights(df: pd.DataFrame, prediction_col: str = "prediction") -> pd.DataFrame:
    """Convert model predictions into long-only portfolio weights by date."""
    result = df.copy()
    weighted_groups = []

    for _, group in result.groupby("date"):
        predictions = group[prediction_col]
        positive_predictions = predictions.clip(lower=0)

        if positive_predictions.sum() > 0:
            weights = positive_predictions / positive_predictions.sum()
        else:
            weights = pd.Series(1.0 / len(group), index=group.index)

        group = group.copy()
        group["weight"] = weights
        weighted_groups.append(group)

    return pd.concat(weighted_groups, ignore_index=True)


def construct_top_k_equal_weight_portfolio(
    df: pd.DataFrame,
    prediction_col: str = "prediction",
    top_k: int = 3,
) -> pd.DataFrame:
    """Select the top-k predicted assets on each date and allocate equally across them."""
    result = df.copy()
    portfolio_chunks = []

    for _, group in result.groupby("date"):
        ranked_group = group.copy().sort_values(prediction_col, ascending=False)

        selected = ranked_group.head(top_k).copy()
        selected["weight"] = 1.0 / len(selected)

        unselected = ranked_group.iloc[top_k:].copy()
        unselected["weight"] = 0.0

        combined = pd.concat([selected, unselected], ignore_index=False)
        portfolio_chunks.append(combined)

    return pd.concat(portfolio_chunks, ignore_index=True)