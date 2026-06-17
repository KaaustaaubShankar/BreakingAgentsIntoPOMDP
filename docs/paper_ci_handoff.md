# Paper CI / error-bar work — handoff (resume tomorrow)

_Session ending 2026-06-17. COLM submission deadline: **June 23, 2026 (AoE)**._

## TL;DR — where we are
- LS20 (env3) deepseek-v4-pro N=20 sweep: **DONE**, committed, dashboard updated (see `results/ls20_real_ablation/`, `dashboard/`).
- CI tooling: `scripts/wilson_cis.py` **rewritten + fixed** (it was pointing at a non-existent CSV). Now reads the canonical `dashboard/jkj results - Detailed_Results.csv`, prints (1) a Wilson 95% CI table for every cell and (2) Fisher's-exact none-vs-medium primary comparisons. Pure stdlib (no scipy) — runs after a plain `git pull`.
- Output captured at `docs/ci_table.txt` (regenerate: `python scripts/wilson_cis.py`).

## Why N=20 matters (the framing change for the COLM paper)
`colm2026_conference.tex` currently says "n=5 pilot… qualitative patterns, not significance." With our N=20 deepseek data that's outdated. The headline reasoning-hurts-mechanics result is now **confirmatory**:

| cell | none | medium | Fisher exact |
|---|---|---|---|
| n=5 (old) | 2/5 (40%) [12,77] | 0/5 (0%) [0,43] | p=0.44 — ns |
| **n=20 (now)** | **9/20 (45%) [26,66]** | **0/20 (0%) [0,16]** | **p=0.0012 \*\*** |

→ The COLM version can report these as real effects, not just "qualitative."

## How CIs / error bars go into the paper (methodology, agreed)
1. Per-cell: **Wilson 95%** (binary win/loss; correct at 0%/100% boundaries). `wilson_cis.py`.
2. Tables: `k/n [lo, hi]` per cell.
3. Figures: win-rate bars with **asymmetric** error bars = Wilson [lo, hi].
4. Condition comparisons (none vs medium, GPT vs Grok): **Fisher's exact** on the 2x2; report p (+ odds ratio). Non-overlapping Wilson CIs is a conservative sufficient signal.
5. Mixed N across cells — report N per cell; designate primary comparisons confirmatory, rest exploratory (or Holm-correct).

## BLOCKERS found today (need fixing before the full CI table is correct)
1. **ka59simple deepseek N=20 not in the CSV.** It exists in `results/ka59simple_real_ablation/ablation_deepseek_*.json` (6 files incl. 2 `*_recovered_*`) but was never added to `dashboard/jkj results - Detailed_Results.csv`. → add those rows (Model `deepseek-v4-pro`, Game `ka59simple`, none+medium x 5 configs).
2. **Inconsistent reasoning labels in ka59simple rows:** `no-R` / `medium-R` / `default-R` instead of `none` / `medium` / `default`. This silently drops ka59simple from the Fisher comparison pairing. → normalize labels in the CSV (or map them in the script).
   - Both of these are part of the [[paper-data-reconciliation-conflicts]] memory — some calls may need Edward (Sheet vs paper.tex mismatches).

## Next steps (recommended order)
1. **Reconcile the CSV** (the two blockers above) → re-run `python scripts/wilson_cis.py > docs/ci_table.txt`. Confirm ka59simple deepseek none-vs-medium now appears with Fisher p.
2. **Flip the COLM framing**: edit `colm2026_conference.tex` §5 (Preliminary Results) + §6 (Discussion) from "qualitative pilot" to CI-backed; insert the Wilson CI table (`paper.tex` already has the table format to copy) and the significant Fisher results.
3. **Error-bar figures**: matplotlib win-rate bars with Wilson asymmetric `yerr`. New script e.g. `scripts/plot_winrate_cis.py` reading the same CSV.
4. **Background**: tighten `colm2026_conference.tex` §3 (POMDP→knockout mapping) and §2 (Related Work — position the verbal/behavioral discovery decoupling against self-report eval literature). `paper.tex` has fuller prose to draw from.

## Files touched this session (all committed/pushed on branch `levi/ka59simple-level-attempts`)
- `scripts/wilson_cis.py` (rewritten), `docs/ci_table.txt` (new)
- `env3/{llm_client,experiment,ablation}.py`, `env4/llm_client.py` (LS20 parity + arc_agi auth fix)
- `ka59_game/llm_client.py`, `scripts/run_real_ablation.py` (DeepSeek provider + pid fix)
- `results/ls20_real_ablation/` (LS20 N=20 data), `dashboard/` (Flask app + CSVs)

## Resume command
```
git pull && python scripts/wilson_cis.py | less
```
