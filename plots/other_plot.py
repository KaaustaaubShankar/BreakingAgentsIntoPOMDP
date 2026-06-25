"""Grouped bar charts: none vs medium reasoning across configs, for two models.

Metrics shown: avg_turns, avg_invalid_actions, avg_click_actions.
Produces 6 separate images (2 models x 3 metrics), each a grouped bar chart
with its own legend.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CONFIGS = ["baseline", "world_hard", "mechanics_hard", "feedback_hard"]
METRICS = ["avg_turns", "avg_invalid_actions", "avg_click_actions"]
METRIC_LABELS = {
    "avg_turns": "avg turns",
    "avg_invalid_actions": "avg invalid actions",
    "avg_click_actions": "avg click actions",
}

# data[model][metric] = {"none": [...per config...], "medium": [...]}
DATA = {
    "deepseek-v4-pro": {
        "avg_turns":          {"none": [21.4, 56.75, 42.35, 21.1], "medium": [18.3, 61.05, 26.35, 16.75]},
        "avg_invalid_actions":{"none": [0.0, 0.3, 2.3, 0.0],       "medium": [0.0, 0.0, 1.15, 0.0]},
        "avg_click_actions":  {"none": [9.1, 21.15, 8.55, 9.55],   "medium": [4.9, 25.45, 9.7, 4.6]},
    },
    "gpt-5.2": {
        "avg_turns":          {"none": [21.47, 37.47, 56.8, 20.0], "medium": [16.87, 48.13, 18.6, 17.33]},
        "avg_invalid_actions":{"none": [0.0, 0.0, 3.47, 0.0],      "medium": [0.0, 0.0, 0.2, 0.0]},
        "avg_click_actions":  {"none": [9.87, 6.33, 6.33, 8.0],    "medium": [4.07, 16.47, 4.87, 4.13]},
    },
}

COLOR_NONE = "#378ADD"
COLOR_MEDIUM = "#1D9E75"
EDGE_MEDIUM = "#0F6E56"
OUTPUT_DIR = Path(__file__).resolve().parent


def plot_one(model, metric):
    none = DATA[model][metric]["none"]
    medium = DATA[model][metric]["medium"]

    x = np.arange(len(CONFIGS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, none, width, label="none", color=COLOR_NONE)
    ax.bar(x + width / 2, medium, width, label="medium",
           color=COLOR_MEDIUM, edgecolor=EDGE_MEDIUM, linewidth=1.2)

    ax.set_title(f"{model} — {METRIC_LABELS[metric]}", fontsize=12)
    ax.set_xlabel("configuration", fontsize=10)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, rotation=25, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fname = f"ablation_{model.replace('.', '_').replace('-', '_')}_{metric}.png"
    out_path = OUTPUT_DIR / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)
    return fname


def plot_all():
    for model in DATA:
        for metric in METRICS:
            plot_one(model, metric)


if __name__ == "__main__":
    plot_all()