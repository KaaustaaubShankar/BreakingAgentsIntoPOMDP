# Figure specs for the COLM paper (for Kaaus)

Data: `results/figure_data/winrate_ci.csv` — one row per cell with
`env, model, reasoning, condition, k, n, win_pct, ci_lo, ci_hi, note`.
CIs are **Wilson 95%** (already computed; in percent). `note` flags cells whose
win%/n is inconsistent — see caveats.

---

## Figure 1 — Win rates with Wilson 95% error bars (the §5 `% TODO: bar chart`)

**Layout:** 3 subplots side by side, one per environment (`ka59simple`, `bp35`, `ls20`).
- x-axis: conditions in order `baseline, world_hard, mechanics_hard, feedback_hard`
  (add `goal_hard` only if we keep Goal as a 4th axis — see caveat).
- y-axis: win rate 0–100%.
- Within each condition, grouped bars per `model`+`reasoning` (e.g. GPT-5.2 none,
  GPT-5.2 medium, DeepSeek none, DeepSeek medium, …).
- **Error bars = asymmetric Wilson 95%**: for each bar,
  `yerr_lower = win_pct - ci_lo`, `yerr_upper = ci_hi - win_pct`
  (matplotlib: `yerr=[[lower],[upper]]`).
- Add a dashed horizontal line at 0% labeled "random floor (0/50)".
- Suggested: DeepSeek bars in a distinct hue (the n=20 rows), GPT/Grok lighter
  (small-n / qualitative).

**matplotlib sketch:**
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv("results/figure_data/winrate_ci.csv")
df = df[df.note==""]                      # drop inconsistent cells
df["lo"] = df.win_pct - df.ci_lo
df["hi"] = df.ci_hi - df.win_pct
# group/plot per env; bar(...; yerr=[df.lo, df.hi], capsize=3)
```

## Figure 2 — Knockout diagram (the §3 `% TODO: Knockout Diagram`)

A schematic of the POMDP M = (S, A, T, Ω, O, R, H) as a box, with the four
agent-facing information channels drawn as removable "cards" feeding the LLM agent:
- **World → Ω** (observation rendering)
- **Goal → R** (success criterion)
- **Mechanics → T** (transition rules)
- **Feedback** (per-step outcome signal)

Show two states per channel: **EASY** (full card) vs **HARD** (greyed/stripped card).
The point to convey: a single-axis knockout greys exactly one card while the
latent POMDP (S, A, T, Ω, R) is unchanged — so a win-rate drop isolates
dependence on that channel. Matches the formalism in §3 (Problem Setting) and
Table 1 (axis conditions).

---

## Caveats baked into the data (so the figure doesn't mislead)
- **GPT-4.1 / LS20** cells are flagged `INCONSISTENT` (40%/60% are impossible at
  n=2 — likely n=5). They're excluded by the `note==""` filter above; fix the
  underlying n/values before plotting them.
- **DeepSeek KA59-Simple medium** n varies by cell (16/23/14/18) because it pools
  the OpenRouter + direct-API real trials; the CSV has the exact per-cell n.
- **Budget:** DeepSeek ran at the 128-turn KA59-Simple budget; confirm GPT/Grok
  KA59-Simple are also 128 before putting them on the same axes.
- Only DeepSeek (n≥14) cells have tight enough CIs for significance claims; the
  GPT/Grok n=5 bars will show very wide error bars (by design — that's the
  "qualitative" point).
