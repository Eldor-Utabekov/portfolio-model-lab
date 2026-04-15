from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_nav_curves(*series: tuple[str, pd.DataFrame]) -> None:
    """Plot cumulative NAV for multiple strategies and save the figure."""
    plt.figure(figsize=(10, 6))

    for label, df in series:
        plt.plot(df["date"], df["nav"], label=label)

    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title("Strategy Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = Path("reports/strategy_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()