from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from portfolio_model_lab.backtest.benchmarks import (
    buy_and_hold_spy,
    equal_weight_strategy,
)
from portfolio_model_lab.backtest.metrics import summarize_performance
from portfolio_model_lab.backtest.run_backtest import (
    build_backtest_inputs,
    compute_portfolio_returns,
)
from portfolio_model_lab.portfolio.construct_portfolio import construct_weights



FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "mean_return_20d",
    "momentum_20d",
    "ma_ratio_20d",
    "drawdown_20d",
]


def load_model_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_return_5d"])
    return df


def train_test_split_time(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    return train, test


def train_ridge_model(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target_return_5d"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model

def train_gradient_boosting_model(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target_return_5d"]

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)
    return model

def train_tuned_gradient_boosting_model(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target_return_5d"]

    base_model = GradientBoostingRegressor(random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    return best_model, grid.best_params_, grid.best_score_

def train_random_forest_model(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target_return_5d"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_model(model, df: pd.DataFrame) -> pd.Series:
    X = df[FEATURE_COLUMNS]
    return pd.Series(model.predict(X), index=df.index, name="prediction")


def evaluate_predictions(df: pd.DataFrame, prediction_col: str = "prediction") -> float:
    mse = mean_squared_error(df["target_return_5d"], df[prediction_col])
    return mse


def run_model_strategy_backtest(
    test_df: pd.DataFrame,
    prediction_col: str = "prediction",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_input = test_df.copy()
    portfolio_df = construct_weights(strategy_input, prediction_col=prediction_col)

    backtest_input_df = build_backtest_inputs(portfolio_df)
    portfolio_returns_df = compute_portfolio_returns(backtest_input_df)
    performance_summary = summarize_performance(portfolio_returns_df)

    return portfolio_returns_df, performance_summary


def run_equal_weight_backtest(test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    equal_weight_df = equal_weight_strategy(test_df)
    equal_weight_input = build_backtest_inputs(equal_weight_df)
    equal_weight_returns = compute_portfolio_returns(equal_weight_input)
    equal_weight_summary = summarize_performance(equal_weight_returns)

    return equal_weight_returns, equal_weight_summary


def run_spy_backtest(test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy_df = buy_and_hold_spy(test_df)
    spy_input = build_backtest_inputs(spy_df)
    spy_returns = compute_portfolio_returns(spy_input)
    spy_summary = summarize_performance(spy_returns)

    return spy_returns, spy_summary

def run_single_model_pipeline(
    input_path: str | Path,
    split_date: str,
    model_name: str,
    train_fn: Callable,
) -> dict[str, pd.DataFrame | float | str]:
    df = load_model_data(input_path)
    train_df, test_df = train_test_split_time(df, split_date=split_date)

    model = train_fn(train_df)

    test_df = test_df.copy()
    test_df["prediction"] = predict_model(model, test_df)

    mse = evaluate_predictions(test_df, prediction_col="prediction")

    model_returns, model_summary = run_model_strategy_backtest(test_df, prediction_col="prediction")
    equal_weight_returns, equal_weight_summary = run_equal_weight_backtest(test_df)
    spy_returns, spy_summary = run_spy_backtest(test_df)

    return {
        "model_name": model_name,
        "train_df": train_df,
        "test_df": test_df,
        "mse": mse,
        "model_returns": model_returns,
        "model_summary": model_summary,
        "equal_weight_returns": equal_weight_returns,
        "equal_weight_summary": equal_weight_summary,
        "spy_returns": spy_returns,
        "spy_summary": spy_summary,
    }


def run_ridge_pipeline(
    input_path: str | Path,
    split_date: str = "2022-01-01",
) -> dict[str, pd.DataFrame | float | str]:
    return run_single_model_pipeline(
        input_path=input_path,
        split_date=split_date,
        model_name="Ridge",
        train_fn=train_ridge_model,
    )

def run_gradient_boosting_pipeline(
    input_path: str | Path,
    split_date: str = "2022-01-01",
) -> dict[str, pd.DataFrame | float | str]:
    return run_single_model_pipeline(
        input_path=input_path,
        split_date=split_date,
        model_name="GradientBoosting",
        train_fn=train_gradient_boosting_model,
    )

def run_tuned_gradient_boosting_pipeline(
    input_path: str | Path,
    split_date: str = "2022-01-01",
) -> dict[str, pd.DataFrame | float | str | dict]:
    df = load_model_data(input_path)
    train_df, test_df = train_test_split_time(df, split_date=split_date)

    model, best_params, best_cv_score = train_tuned_gradient_boosting_model(train_df)

    test_df = test_df.copy()
    test_df["prediction"] = predict_model(model, test_df)

    mse = evaluate_predictions(test_df, prediction_col="prediction")

    model_returns, model_summary = run_model_strategy_backtest(test_df, prediction_col="prediction")
    equal_weight_returns, equal_weight_summary = run_equal_weight_backtest(test_df)
    spy_returns, spy_summary = run_spy_backtest(test_df)

    return {
        "model_name": "GradientBoostingTuned",
        "train_df": train_df,
        "test_df": test_df,
        "mse": mse,
        "model_returns": model_returns,
        "model_summary": model_summary,
        "equal_weight_returns": equal_weight_returns,
        "equal_weight_summary": equal_weight_summary,
        "spy_returns": spy_returns,
        "spy_summary": spy_summary,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
    }

def run_random_forest_pipeline(
    input_path: str | Path,
    split_date: str = "2022-01-01",
) -> dict[str, pd.DataFrame | float | str]:
    return run_single_model_pipeline(
        input_path=input_path,
        split_date=split_date,
        model_name="RandomForest",
        train_fn=train_random_forest_model,
    )