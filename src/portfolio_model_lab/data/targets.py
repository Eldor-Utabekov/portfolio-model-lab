from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_feature_data(path: str | Path) -> pd.DataFrame:
    """Load feature dataset used for modeling."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    df = pd.read_parquet(file_path)
    if df.empty:
        raise ValueError("Loaded feature data is empty")

    return df


def add_regression_target(df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """Create forward return target over the specified horizon."""
    result = df.copy()

    future_price = result.groupby("ticker")["close"].shift(-horizon_days)
    result["target_return_5d"] = future_price.div(result["close"]).sub(1.0)

    return result


def add_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary direction target from the regression target."""
    result = df.copy()

    if "target_return_5d" not in result.columns:
        raise ValueError("target_return_5d must exist before classification target is created")

    result["target_up_5d"] = (result["target_return_5d"] > 0).astype("float")
    result.loc[result["target_return_5d"].isna(), "target_up_5d"] = pd.NA

    return result


def save_target_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save dataset with targets to parquet."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    input_path = Path("data/processed/etf_features.parquet")
    output_path = Path("data/processed/etf_model_data.parquet")

    model_df = load_feature_data(input_path)
    model_df = add_regression_target(model_df, horizon_days=5)
    model_df = add_classification_target(model_df)
    save_target_data(model_df, output_path)

    preview_cols = ["ticker", "date", "close", "target_return_5d", "target_up_5d"]

    print(model_df[preview_cols].head(10))
    print(model_df[preview_cols].tail(10))
    print(model_df.shape)