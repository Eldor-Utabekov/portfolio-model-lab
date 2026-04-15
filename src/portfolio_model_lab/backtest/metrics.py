from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annualized_return(portfolio_returns: pd.Series) -> float:
    """Compute annualized return from a series of periodic returns."""
    returns = portfolio_returns.dropna()
    if returns.empty:
        return float("nan")

    cumulative_return = (1.0 + returns).prod()
    n_periods = len(returns)

    return cumulative_return ** (TRADING_DAYS_PER_YEAR / n_periods) - 1.0


def annualized_volatility(portfolio_returns: pd.Series) -> float:
    """Compute annualized volatility."""
    returns = portfolio_returns.dropna()
    if returns.empty:
        return float("nan")

    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute Sharpe ratio using annualized return and volatility."""
    ann_return = annualized_return(portfolio_returns)
    ann_vol = annualized_volatility(portfolio_returns)

    if ann_vol == 0 or np.isnan(ann_vol):
        return float("nan")

    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(nav: pd.Series) -> float:
    """Compute maximum drawdown from a NAV series."""
    nav_series = nav.dropna()
    if nav_series.empty:
        return float("nan")

    running_max = nav_series.cummax()
    drawdown = nav_series / running_max - 1.0

    return drawdown.min()


def summarize_performance(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key performance metrics for a backtest run."""
    if "portfolio_return" not in backtest_df.columns:
        raise ValueError("backtest_df must contain 'portfolio_return'")
    if "nav" not in backtest_df.columns:
        raise ValueError("backtest_df must contain 'nav'")

    returns = backtest_df["portfolio_return"]
    nav = backtest_df["nav"]

    summary = pd.DataFrame(
        {
            "annualized_return": [annualized_return(returns)],
            "annualized_volatility": [annualized_volatility(returns)],
            "sharpe_ratio": [sharpe_ratio(returns)],
            "max_drawdown": [max_drawdown(nav)],
        }
    )

    return summary