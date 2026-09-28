"""
Plots cumulative en_predictions.json values across weeks 1-13,
featuring only the top 15 cumulative scorers (by final week-13 total).

Expects: data/forecasting/ml_data/week{n}/en_predictions.json for n in 1..13
Output: data/forecasting/ml_data/cumulative_top15.png
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("data/forecasting/ml_data")
NUM_WEEKS = 13
OUTPUT_PATH = BASE_DIR / "cumulative_top20.png"

EXCLUDED_TITLES = {"Kusuriya no Hitorigoto 3rd Season", "Black Clover 2nd Season"}

def load_week(week_num):
    file_path = BASE_DIR / f"week{week_num}" / "lgbm_predictions.json"
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cumulative_df():
    weekly_data = {w: load_week(w) for w in range(1, NUM_WEEKS + 1)}

    titles = [t for t in weekly_data[1].keys() if t not in EXCLUDED_TITLES]

    # Build raw weekly values dataframe: rows = weeks, cols = anime
    raw = pd.DataFrame(
        {title: [weekly_data[w].get(title, 0) for w in range(1, NUM_WEEKS + 1)] for title in titles},
        index=range(1, NUM_WEEKS + 1),
    )

    cumulative = raw.cumsum(axis=0)
    return cumulative


def plot_top20(cumulative_df):
    final_totals = cumulative_df.iloc[-1].sort_values(ascending=False)
    top15_titles = final_totals.head(20).index.tolist()

    plt.figure(figsize=(20, 12))

    for title in top15_titles:
        plt.plot(
            cumulative_df.index,
            cumulative_df[title],
            marker="o",
            linewidth=2,
            label=title,
        )

    plt.title("Top 20 Cumulative Predicted Points by Week", fontsize=20)
    plt.xlabel("Week", fontsize=16)
    plt.ylabel("Cumulative Predicted Points", fontsize=16)
    plt.xticks(range(1, NUM_WEEKS + 1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=12,
        title="Anime",
        title_fontsize=13,
    )
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved chart to: {OUTPUT_PATH}")


def main():
    cumulative_df = build_cumulative_df()
    plot_top20(cumulative_df)


if __name__ == "__main__":
    main()