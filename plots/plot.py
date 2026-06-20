import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent

def wilson_interval(k, n, z=1.96):
    if k is None or n is None:
        return None, None
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    lo = max(0, (centre - margin) * 100)
    hi = min(100, (centre + margin) * 100)
    return lo, hi

def back_calculate(pct, n):
    if pct is None or n is None:
        return None
    return int(round(pct / 100 * n))

def format_cell(k, n, lo, hi):
    pct = int(round(k / n * 100))
    if pct == 100:
        return f"{k}/{n}\n100% (-{int(round(100 - lo))})"
    elif pct == 0:
        return f"{k}/{n}\n0% (+{int(round(hi))})"
    else:
        margin = (hi - lo) / 2
        return f"{k}/{n}\n{pct}% ±{int(round(margin))}"

data = {
    "BP35": {
        "rows": ["GPT-5.2 (med)", "GPT-5.2 (none)", "Deepseek (med)", "Deepseek (none)"],
        "cols": ["BASELINE", "WORLD_HARD", "MECHANICS_HARD", "FEEDBACK_HARD"],
        "vals": [
            [100,  0, 100, 100],
            [100,  0,  66, 100],
            [100,  0, 100, 100],
            [100,  0, 100, 100],
        ],
        "ns": [
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [20, 20, 20, 20],
            [20, 20, 20, 20],
        ]
    },
    "LS20": {
        "rows": ["GPT-5.2 (med)", "GPT-5.2 (none)", "Deepseek (med)", "Deepseek (none)"],
        "cols": ["BASELINE", "WORLD_HARD", "MECHANICS_HARD", "FEEDBACK_HARD"],
        "vals": [
            [100, 30, 100, 100],
            [100, 30,  50, 100],
            [100,  5,  45,  95],
            [100,  5,  45,  95],
        ],
        "ns": [
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [20, 20, 20, 20],
            [20, 20, 20, 20],
        ]
    },
    "KA59-Simple": {
        "rows": ["GPT-5.2 (med)", "GPT-5.2 (none)", "Deepseek (med)", "Deepseek (none)"],
        "cols": ["BASELINE", "WORLD_HARD", "MECHANICS_HARD", "MECHANICS_HARD\n_NORULES", "FEEDBACK_HARD"],
        "vals": [
            [ 90,  0,   0, None,  80],
            [ 90,  0,   0,    0,  80],
            [ 20,  0,   0,    0,  10],
            [ 60,  0,  20,   20,  75],
        ],
        "ns": [
            [10, 10, 10, None, 10],
            [10, 10, 10,   10, 10],
            [20, 20, 20,   20, 20],
            [20, 20, 20,   20, 20],
        ]
    },
}

cmap = plt.cm.RdYlGn

for game, d in data.items():
    fig, ax = plt.subplots(figsize=(7 if len(d["cols"]) < 5 else 9, 5))

    vals = np.array(
        [[v if v is not None else np.nan for v in row] for row in d["vals"]],
        dtype=float
    )

    im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(d["cols"])))
    ax.set_xticklabels(d["cols"], rotation=30, ha="right", fontsize=9,
                       fontfamily="monospace")
    ax.set_yticks(range(len(d["rows"])))
    ax.set_yticklabels(d["rows"], fontsize=10)
    ax.set_title(game, fontsize=13, fontweight="bold", pad=12)

    for i in range(len(d["rows"])):
        for j in range(len(d["cols"])):
            v = d["vals"][i][j]
            n = d["ns"][i][j]
            if v is None or n is None:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, color="#dddddd", zorder=2
                ))
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=9, color="gray", zorder=3)
            else:
                k = back_calculate(v, n)
                lo, hi = wilson_interval(k, n)
                label = format_cell(k, n, lo, hi)
                text_color = "white" if v < 40 or v > 75 else "black"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=text_color,
                        linespacing=1.5)

    plt.colorbar(im, ax=ax, label="Win rate (%)", shrink=0.8)
    plt.tight_layout()

    filename = game.lower().replace("-", "_").replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"{filename}_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_DIR / f"{filename}_heatmap.png", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()