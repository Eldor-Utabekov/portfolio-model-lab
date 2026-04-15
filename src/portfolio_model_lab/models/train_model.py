from __future__ import annotations

from pathlib import Path

import pandas as pd

from portfolio_model_lab.backtest.plots import plot_nav_curves
from portfolio_model_lab.models.pipeline import (
    run_gradient_boosting_pipeline,
    run_random_forest_pipeline,
    run_ridge_pipeline,
    run_tuned_gradient_boosting_pipeline,
)


def build_comparison_table(*results_dicts: dict) -> pd.DataFrame:
    """Assemble a portfolio-level comparison table across model runs."""
    rows = []

    for results in results_dicts:
        summary = results["model_summary"].iloc[0].to_dict()
        summary["model_name"] = results["model_name"]
        summary["test_mse"] = results["mse"]
        rows.append(summary)

    comparison_df = pd.DataFrame(rows)[
        [
            "model_name",
            "test_mse",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]
    ]

    comparison_df = comparison_df.sort_values(
        "sharpe_ratio",
        ascending=False,
    ).reset_index(drop=True)

    return comparison_df


def print_results(results: dict) -> None:
    """Print a concise experiment summary for one model run."""
    print(f"\n=== {results['model_name']} ===")
    print(f"Train size: {len(results['train_df'])}")
    print(f"Test size: {len(results['test_df'])}")
    print(f"Test MSE: {results['mse']:.6f}")

    if "best_params" in results:
        print(f"Best params: {results['best_params']}")
        print(f"Best CV score (neg MSE): {results['best_cv_score']:.6f}")

    print("\nMODEL STRATEGY")
    print(results["model_summary"].round(6))

    print("\nEQUAL WEIGHT")
    print(results["equal_weight_summary"].round(6))

    print("\nSPY BUY & HOLD")
    print(results["spy_summary"].round(6))


if __name__ == "__main__":
    input_path = Path("data/processed/etf_model_data.parquet")

    ridge_results = run_ridge_pipeline(
        input_path=input_path,
        split_date="2022-01-01",
    )
    gbm_results = run_gradient_boosting_pipeline(
        input_path=input_path,
        split_date="2022-01-01",
    )
    rf_results = run_random_forest_pipeline(
        input_path=input_path,
        split_date="2022-01-01",
    )
    tuned_gbm_results = run_tuned_gradient_boosting_pipeline(
        input_path=input_path,
        split_date="2022-01-01",
    )

    print_results(ridge_results)
    print_results(gbm_results)
    print_results(rf_results)
    print_results(tuned_gbm_results)

    comparison_table = build_comparison_table(
        ridge_results,
        gbm_results,
        rf_results,
        tuned_gbm_results,
    )

    metric_columns = [
        "test_mse",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ]
    comparison_table[metric_columns] = comparison_table[metric_columns].round(6)

    output_path = Path("reports/model_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_table.to_csv(output_path, index=False)

    print("\n=== MODEL COMPARISON TABLE ===")
    print(comparison_table)

    plot_nav_curves(
        ("Ridge", ridge_results["model_returns"]),
        ("GradientBoosting", gbm_results["model_returns"]),
        ("RandomForest", rf_results["model_returns"]),
        ("GradientBoostingTuned", tuned_gbm_results["model_returns"]),
    )

    print("\nArtifacts saved:")
    print("- reports/model_comparison.csv")
    print("- reports/strategy_comparison.png")