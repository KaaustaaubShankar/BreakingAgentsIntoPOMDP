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

## ka59simple deepseek — DATA IS CONFOUNDED, needs a clean re-run (2026-06-17 update)
Tried to "add ka59simple fully" and found three problems making the existing deepseek ka59simple data NOT paper-usable as-is:
1. **`medium` is ~75% zero-token contaminated.** Team's older OpenRouter run: ~20/26 trials per cell were silent 401/zero-token failures counted as losses (raw `medium::baseline`=4% is an artifact; real ~17% on N≈6). Same silent-failure bug class as the arc_agi/.env one.
2. **Turn-budget mismatch.** deepseek ka59simple ran at **128 turns** (1 level x turns_per_level=128) while gpt/grok + paper.tex use a **32-turn budget**. 4x larger budget → not comparable across models.
3. **Endpoint split.** deepseek `none` = direct DeepSeek API; `medium` = OpenRouter. (LS20 was all-OpenRouter = clean.)
4. (Fixed) **Label inconsistency** `no-R`/`medium-R`/`default-R` — now normalized non-destructively in `wilson_cis.py` on read (ka59simple gpt-5.2 now pairs in Fisher output).

### Corrected plan (CONFIRM budget=32, then launch) — fixes all 3 at once
Re-run deepseek ka59simple **none + medium together, OpenRouter, `--max-turns 32`, N=20, 5 configs**
(baseline/world_hard/mechanics_hard/mechanics_hard_format_only/feedback_hard; no goal_hard).
→ 32-turn budget matches gpt/grok/paper; OpenRouter matches LS20; fresh run kills contamination; ~4x cheaper
than 128 (≈ $15 / a few hours, not $50 / 2 days). Command:
```
for eff in none medium; do for cfg in baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard; do
  python -m scripts.run_real_ablation --env ka59simple --provider openrouter --model deepseek/deepseek-v4-pro \
    --reasoning-effort $eff --configs $cfg --trials 20 --max-turns 32 \
    --input-cost-per-m 0.30 --output-cost-per-m 0.90 > /tmp/ka59s_${eff}_${cfg}.log 2>&1 &
done; done; wait
```
**MONITOR first trials for tok=0 (abort if contaminated).** The wrong-budget (128) medium run launched 2026-06-17 was killed.
After it finishes: add deepseek ka59simple rows to `dashboard/` CSVs (Detailed + Overview), update VM CSVs, regenerate `docs/ci_table.txt`.
NOTE: the gpt/grok ka59simple data may ALSO have zero-token contamination worth auditing (see [[paper-data-reconciliation-conflicts]] — needs Edward's calls).

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
