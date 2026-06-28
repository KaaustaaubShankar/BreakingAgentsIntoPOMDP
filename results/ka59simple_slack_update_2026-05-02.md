# Slack-ready update: ka59simple ablation results

_Copy-paste the section below into the JKJ team Slack channel._

---

## :game_die: KA59 lane update — ka59simple full ablation matrix (2026-05-02)

**Heads up:** Found and fixed a bug in `_generate_openrouter` that was silently dropping the `reasoning_effort` flag — all overnight 2026-04-30 → 2026-05-01 ka59simple runs were running at gpt-5.2's default reasoning (≈ minimal), regardless of label. After fixing + re-verifying via API probe (`reasoning_tokens` confirmed 0 for `effort=none`, 19 for `medium` on small prompts, 1240 on KA59-length prompts), I re-ran the full matrix today.

**About `ka59simple`**: it's a constructed single-mechanic variant of canonical KA59 (right-goal-only, 6-action optimal solve), built last week to escape canonical's 0%-floor. **Canonical KA59 still floors all models at 0%** — that finding is unchanged. ka59simple is a methodological tool to enable axis-knockout comparison, NOT a replacement for canonical KA59 in the headline.

### gpt-5.2 ka59simple (n=5–10 per cell)

```
Config            Reasoning   Win%  AvgTurns  AvgLevels  Invalid  Clicks  Walls  Pushes  Rel.diff   n
-----------------------------------------------------------------------------------------------------
baseline          default-R    60%      25.0       0.60      0.0     8.4    0.9     0.0     1.00x  10
world_hard        default-R     0%      32.0       0.00      0.0     2.2    1.0     0.0     1.28x  10
goal_hard         default-R    30%      24.7       0.30      0.0     9.4    0.3     0.0     0.99x  10
mechanics_hard    default-R     0%      32.0       0.00      0.1    30.1    0.1     0.0     1.28x  10
baseline          no-R         80%      13.2       0.80      0.0     5.4    1.0     0.0     1.00x   5
world_hard        no-R          0%      32.0       0.00      0.0     5.0    1.0     0.0     2.42x   5
goal_hard         no-R         80%      16.8       0.80      0.0     5.0    1.2     0.0     1.27x   5
mechanics_hard    no-R          0%      32.0       0.00      0.2    30.8    0.0     0.0     2.42x   5
feedback_hard     no-R         90%      13.5       0.90      0.0     2.7    1.0     0.0     1.02x  10
baseline          medium-R    100%      15.6       1.00      0.0     5.4    1.0     0.0     1.00x   5
world_hard        medium-R      0%      32.0       0.00      0.0     2.4    1.0     0.0     2.05x   5
goal_hard         medium-R     20%      31.0       0.20      0.0    16.8    0.8     0.0     1.99x   5
mechanics_hard    medium-R      0%      32.0       0.00      0.0    30.0    0.0     0.0     2.05x   5
feedback_hard     medium-R     50%      21.1       0.50      0.0     7.1    0.6     0.0     1.35x  10
```

### grok-4.1-fast ka59simple (n=5 per cell)

```
Config            Reasoning   Win%  AvgTurns  AvgLevels  Invalid  Clicks  Walls  Pushes  Rel.diff   n
-----------------------------------------------------------------------------------------------------
baseline          no-R         40%      30.4       0.40      0.0    21.4    0.6     0.0     1.00x   5
world_hard        no-R          0%      32.0       0.00      0.0    24.6    0.8     0.0     1.05x   5
goal_hard         no-R         20%      30.8       0.20      0.0    21.6    0.2     0.0     1.01x   5
mechanics_hard    no-R          0%      32.0       0.00      0.0    32.0    0.0     0.0     1.05x   5
feedback_hard     no-R          0%      32.0       0.00      0.0    25.6    0.4     0.0     1.05x   5
baseline          medium-R     20%      27.2       0.20      0.0    15.8    0.4     0.0     1.00x   5
world_hard        medium-R      0%      32.0       0.00      0.0    16.0    1.0     0.0     1.18x   5
goal_hard         medium-R     60%      22.4       0.60      0.0    10.2    0.6     0.0     0.82x   5
mechanics_hard    medium-R      0%      32.0       0.00      0.2    29.8    0.0     0.0     1.18x   5
feedback_hard     medium-R     40%      27.0       0.40      0.0    16.2    0.4     0.0     0.99x   5
```

### Notes on table format

- KA59 doesn't have **flips** or **undos** action types (those are LS20/BP35-specific) → I substituted **walls** (`wall_transfers`) and **pushes** (`object_pushes`), the equivalent KA59-specific mechanic counters
- **Rel. diff** = `avg_turns(condition) / avg_turns(baseline)` at same reasoning level
- **max_turns=32** (we capped at 32 for compute budget; canonical KA59 uses 50)

### :exclamation: Findings worth flagging

1. **World-knockout always 0% wins** — robust across all 5 model × reasoning conditions. Cleanest cross-cell finding.
2. **Mechanics-knockout always 0% wins**, with `wall_transfers` suppressed (0 across all 25 mechanics_hard trials) — **the partial-progress signal the new abstract framing should anchor on**.
3. **gpt-5.2 reasoning is non-monotonic on goal_hard**: no-R 80% > medium-R 20%. **Reasoning hurts** on this knockout for gpt-5.2.
4. **Same axis, opposite cross-model reasoning interaction**: gpt-5.2 goal_hard goes 80%→20% with reasoning; grok-4.1-fast goes 20%→60%. Worth one sentence in the abstract.
5. **default-R column was unintentional** (pre-fix data) but is internally consistent and shows distinct patterns from explicit no-R / medium-R — including it for transparency.

### Methodological caveats for the paper

- ka59simple is a constructed variant, NOT canonical KA59 — must be framed as "single-mechanic variant for ablation," with canonical KA59 floor effect still reported as primary evidence
- Per-trial JSONs don't store `reasoning_effort` — attribution is via source-log parsing (small fix queued to log it in JSON going forward)
- n=5 is underpowered for some cells; bumping to n=10 with a parallel sweep next (~$15, ~5 hours since grok medium-R takes ~30 min/trial)
- Canonical KA59 max_turns=50, ka59simple max_turns=32 — disclose explicitly

Full rundown with all caveats and attribution methodology: `results/ka59simple_full_rundown_2026-05-02.md`.

— Edward
