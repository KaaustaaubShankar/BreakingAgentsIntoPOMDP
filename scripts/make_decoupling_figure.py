"""Generate the verbal/behavioral decoupling figure as a PDF and PNG fallback.

The paper uses an inline TikZ version of this chart. This script produces
identical content as a vector PDF (best for inclusion via \includegraphics)
and a PNG (for quick visual checks).

Usage:
    python3 scripts/make_decoupling_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).parents[1]
OUT_DIR = REPO / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("Random",        50,  0.00),
    ("Baseline",      32,  0.84),
    ("World hard",    30,  0.97),
    ("Goal hard",     50,  0.66),
    ("Mech.\\ hard",  42,  0.12),
    ("Feedback hard", 30,  0.67),
    ("OODA-F",        16,  0.44),
]
labels = [c[0].replace("\\\\", "") for c in CONFIGS]
behavioral = [c[2] for c in CONFIGS]
verbal = [None] + [0.0] * 6  # Random has no verbal regex (no LLM)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7.0, 4.5), sharex=True,
    gridspec_kw={"hspace": 0.32},
)

# --- top panel: behavioral ---
x = np.arange(len(labels))
bars_top = ax_top.bar(x, behavioral, color="#1F8a8a", edgecolor="#0f5050",
                     linewidth=0.6, width=0.62)
for bar, val in zip(bars_top, behavioral):
    ax_top.text(bar.get_x() + bar.get_width() / 2, val + 0.025,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
ax_top.set_ylim(0, 1.15)
ax_top.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_top.set_ylabel("avg wall-transfers / trial", fontsize=9)
ax_top.set_title("Behavioral discovery: avg wall-transfer events per trial",
                 fontsize=10, fontweight="bold", loc="left")
ax_top.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.4)
ax_top.set_axisbelow(True)
for s in ("top", "right"):
    ax_top.spines[s].set_visible(False)

# --- bottom panel: verbal ---
verbal_plot = [v if v is not None else 0.0 for v in verbal]
bars_bot = ax_bot.bar(x, verbal_plot, color="#c84040", edgecolor="#7a2222",
                      linewidth=0.6, width=0.62)
# annotate: "--" for random (no LLM), "0" otherwise
labels_for_bars = ["--"] + ["0"] * 6
for bar, txt in zip(bars_bot, labels_for_bars):
    ax_bot.text(bar.get_x() + bar.get_width() / 2, 0.04,
                txt, ha="center", va="bottom", fontsize=8)
ax_bot.set_ylim(0, 1.15)
ax_bot.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax_bot.set_ylabel("verbal regex matches / trial", fontsize=9)
ax_bot.set_title("Verbal discovery: regex matches per trial (always 0)",
                 fontsize=10, fontweight="bold", loc="left")
ax_bot.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.4)
ax_bot.set_axisbelow(True)
for s in ("top", "right"):
    ax_bot.spines[s].set_visible(False)

ax_bot.set_xticks(x)
ax_bot.set_xticklabels([l.replace("Mech.\\ hard", "Mech. hard") for l in labels],
                       rotation=20, ha="right", fontsize=9)

pdf_path = OUT_DIR / "decoupling.pdf"
png_path = OUT_DIR / "decoupling.png"
plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
plt.savefig(png_path, bbox_inches="tight", dpi=200)
print(f"saved: {pdf_path}")
print(f"saved: {png_path}")
