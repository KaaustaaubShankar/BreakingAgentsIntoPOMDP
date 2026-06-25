import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent


def wilson_interval(k, n, z=1.96):
    """Return (lo, hi) Wilson score CI in percent. k and n must be ints."""
    if k is None or n is None or n == 0:
        return None, None
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    lo = max(0.0, (centre - margin) * 100)
    hi = min(100.0, (centre + margin) * 100)
    return lo, hi


def format_cell(k, n, lo, hi):
    """
    Format a heatmap cell.  Shows the observed fraction and the
    asymmetric Wilson 95 % CI as [lo, hi] so no information is hidden.
    """
    # show observed percent with one decimal and CI bounds with one decimal
    pct = (float(k) / float(n)) * 100.0
    ci_lo = lo or 0.0
    ci_hi = hi or 0.0

    if k == n:
        return f"{k}/{n}\n100.0%\n[{ci_lo:.1f}–100.0]"
    elif k == 0:
        return f"{k}/{n}\n0.0%\n[0.0–{ci_hi:.1f}]"
    else:
        return f"{k}/{n}\n{pct:.1f}%\n[{ci_lo:.1f}–{ci_hi:.1f}]"


# ---------------------------------------------------------------------------
# Data
# vals entries are raw *counts* (k), not percentages, so that back-calculation
# rounding is eliminated.  If you only have percentages, convert them once
# here, cleanly.
# ---------------------------------------------------------------------------

def pct_to_k(pct, n):
    """Convert a percentage + sample size to the nearest integer count."""
    if pct is None or n is None:
        return None
    k = int(round(float(pct) / 100.0 * float(n)))
    return max(0, min(int(n), k))


# Raw percentage inputs — converted to counts once, up front.
_data_pct = {
    "BP35": {
        "rows": ["GPT-5.2 (med)", "GPT-5.2 (none)", "Deepseek (med)", "Deepseek (none)"],
        "cols": ["BASELINE", "WORLD_HARD", "MECHANICS_HARD", "FEEDBACK_HARD"],
        "pcts": [
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
        ],
    },
    "LS20": {
        "rows": ["GPT-5.2 (med)", "GPT-5.2 (none)", "Deepseek (med)", "Deepseek (none)"],
        "cols": ["BASELINE", "WORLD_HARD", "MECHANICS_HARD", "FEEDBACK_HARD"],
        "pcts": [
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
        ],
    },
    "KA59-Simple": {
        "rows": ["GPT-5.2 (med)", "GPT-5.2 (none)", "Deepseek (med)", "Deepseek (none)"],
        "cols": ["BASELINE", "WORLD_HARD", "MECHANICS_HARD", "MECHANICS_HARD\n_NORULES", "FEEDBACK_HARD"],
        "pcts": [
            [ 90,  0,   0, 0,  80],
            [ 90,  0,   0,    0,  80],
            [ 20,  0,   0,    0,  10],
            [ 60,  0,  20,   20,  75],
        ],
        "ns": [
            [10, 10, 10, 10, 10],
            [10, 10, 10,   10, 10],
            [20, 20, 20,   20, 20],
            [20, 20, 20,   20, 20],
        ],
    },
}

# Build data dict with pre-computed counts
data = {}
for game, d in _data_pct.items():
    ks = [
        [pct_to_k(v, n) for v, n in zip(row_pcts, row_ns)]
        for row_pcts, row_ns in zip(d["pcts"], d["ns"])
    ]
    data[game] = {
        "rows": d["rows"],
        "cols": d["cols"],
        "ks":   ks,
        "ns":   d["ns"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

cmap = plt.cm.RdYlGn

for game, d in data.items():
    ncols = len(d["cols"])
    fig, ax = plt.subplots(figsize=(7 if ncols < 5 else 9, 5))

    # Build a float array of observed percentages for the colour map
    pct_matrix = np.full((len(d["rows"]), ncols), np.nan)
    for i, row_ks in enumerate(d["ks"]):
        for j, (k, n) in enumerate(zip(row_ks, d["ns"][i])):
            if k is not None and n is not None:
                pct_matrix[i, j] = k / n * 100

    im = ax.imshow(pct_matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(d["cols"], rotation=30, ha="right", fontsize=9,
                       fontfamily="monospace")
    ax.set_yticks(range(len(d["rows"])))
    ax.set_yticklabels(d["rows"], fontsize=10)
    ax.set_title(game, fontsize=13, fontweight="bold", pad=12)

    for i, row_ks in enumerate(d["ks"]):
        for j, (k, n) in enumerate(zip(row_ks, d["ns"][i])):
            if k is None or n is None:
                # Grey out truly missing cells
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, color="#dddddd", zorder=2
                ))
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=9, color="gray", zorder=3)
            else:
                lo, hi = wilson_interval(k, n)
                label = format_cell(k, n, lo, hi)
                pct = k / n * 100
                # Use white text only in the extreme dark zones (red <30, deep green >70)
                text_color = "white" if pct < 30 or pct > 70 else "black"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=text_color,
                        linespacing=1.4)

    plt.colorbar(im, ax=ax, label="Win rate (%)", shrink=0.8)
    plt.tight_layout()

    stem = game.lower().replace("-", "_").replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"{stem}_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_DIR / f"{stem}_heatmap.png", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

print("Done — heatmaps saved.")