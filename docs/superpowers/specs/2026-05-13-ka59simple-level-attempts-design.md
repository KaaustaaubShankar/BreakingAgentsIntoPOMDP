# KA59simple: 2-attempts-per-level retry protocol + reasoning sweep

**Status:** Draft (pending user sign-off)
**Author:** Edward Lue Chee Lip
**Date:** 2026-05-13
**Driving deadline:** Kaaus's report for Kevin, Sunday 2026-05-17

## Background

The 2026-05-12 JKJ meeting standardized a retry protocol across the three game environments:
each level gets up to **2 attempts** before the run is scored as a failure on that level.
Josh's game (LS20) has built-in multi-lives, so the protocol "harmonizes" by giving Kaaus's
BP35 (env4) and Edward's KA59 (ka59_game) lane the same 2-attempt budget per level.

Kaaus has already shipped this for env4 in commit `efbb91f` ("swapped to two attempts
per level and included reasoning parameter for openrouter call"). This spec ports the
same pattern to `ka59_game/experiment.py` and `scripts/run_real_ablation.py` for the
ka59simple environment.

## Scope

**In scope:**
- Add per-level retry (`LEVEL_ATTEMPTS = 2`) to `ka59_game/experiment.py`, mirroring
  `env4/experiment.py`'s shape.
- Extend `scripts/run_real_ablation.py` to accept multiple reasoning levels in one
  invocation (`--reasoning-effort none medium`), mirroring `env4/ablation.py`.
- Run `openai/gpt-5.2` × `{none, medium}` reasoning × 5 trials × 5 configs on ka59simple.
- Add an `qwen-mlx` provider to `ka59_game/llm_client.py` for Edward's M5 Pro 48 GB
  Apple Silicon machine.

**Explicitly out of scope:**
- Canonical 7-level KA59. Only `--env ka59simple` runs this protocol.
- OODA configs (`mechanics_ooda`, `mechanics_ooda_f`). Not run.
- Per-level retry as a CLI knob. `LEVEL_ATTEMPTS = 2` is a hard-coded protocol constant
  (matches env4).
- Adding canonical KA59 retry support. The OODA forced-reframe code in
  `ka59_game/experiment.py` stays in place but isn't exercised by ka59simple sweeps.

## Reference: env4 retry shape (canonical pattern)

```python
# env4/experiment.py
LEVEL_ATTEMPTS = 2

for level in range(levels_to_play):
    for attempt in range(1, LEVEL_ATTEMPTS + 1):
        for turn in range(turns_per_level):
            ... LLM turn ...
            if level_progressed: break inner two loops
            if frame_data.state.name == "GAME_OVER": break  # try again
        if stop_run or level_progressed: break
        if attempt < LEVEL_ATTEMPTS:
            log "attempt N/M failed, restarting"
            action_history.append("... game over. Restarting same level, keeping failure in context.")
            env.step(RESET)
            continue
        # else: out of attempts → stop_run = True
```

Key invariants this preserves:
1. **One trial = one win/lose outcome.** A trial wins iff the agent completes all levels;
   each level can use up to 2 attempts.
2. **Failure context flows forward.** Between attempts, the agent sees a synthetic
   `action_history` entry saying it failed and is restarting. No clean slate.
3. **Per-attempt counters reset, per-trial counters accumulate.** `position_history`,
   `last_action_name`, and `attempt_turn` reset between attempts; `result.forced_reframes`,
   `result.click_actions`, `result.invalid_actions`, `result.turns` (global) accumulate.
4. **Engine reset is `env.step(RESET)`,** not env recreation. arc_agi expects RESET as a
   GameAction to restart the current level in-place.

## Design

### A. `ka59_game/experiment.py` changes

1. Add module constant `LEVEL_ATTEMPTS = 2` near `STUCK_THRESHOLD` (line ~44).
2. Restructure `run_agent()`'s main loop from `for turn in range(1, max_turns + 1):` to
   a nested `for level → for attempt → for turn` loop, matching env4's diff.
3. Track `global_turn` (cross-attempt cumulative) and `attempt_turn` (resets per attempt).
   Existing `result.turns` should remain the global cumulative count.
4. Reset between attempts: `prev_state`, `last_action_name`, `position_history`,
   `consecutive_same_type_count`, `last_action_type_tracked`, `turn_in_level`.
5. Accumulate across attempts (per-trial totals): `result.forced_reframes`,
   `result.click_actions`, `result.invalid_actions`, `result.moves_blocked`,
   `result.object_pushes`, `result.wall_transfers`.
6. After a `GAME_OVER` or attempt-turn-budget exhaustion:
   - If `attempt < LEVEL_ATTEMPTS`: log a `level_retry` event, append failure summary
     to `action_history`, call `env.step(GameAction.RESET)`, refresh `frame_data`,
     continue the attempt loop.
   - Else: log `game_over` event with "no attempts remaining", set `stop_run = True`,
     break.
7. The status block shown to the LLM each turn gains an `Attempt: {n}/{LEVEL_ATTEMPTS}`
   line, matching env4's `status_lines`. Helps the agent reason about its retry budget.
8. Per-action log entries gain `attempt` and `attempt_turn` fields. Filename scheme
   `run_{tag}_{run_id}.json` stays as-is — the timestamp run_id already disambiguates,
   and the env4 pattern of embedding reasoning_effort in run_id is handled at the
   ablation-script level (see B).

### B. `scripts/run_real_ablation.py` changes

1. `--reasoning-effort` becomes `nargs="+"`, default `["none", "medium"]`, choices match
   env4. Validation: error early on unknown effort values.
2. Outer loop over `reasoning_efforts` wraps the existing config loop. So one invocation
   produces sweeps × configs × trials of per-trial JSONs.
3. Drop `mechanics_ooda` and `mechanics_ooda_f` from `ALL_CONFIGS`. Final 5 configs:
   `baseline, world_hard, goal_hard, mechanics_hard, feedback_hard`.
4. Default `--env` flips to `ka59simple` (was `ka59`). The `ka59` choice stays valid for
   ad-hoc re-runs, but isn't the default any more.
5. Per-trial run_id format gains `_{reasoning_effort}_t{trial}` suffix so per-trial JSONs
   for the two reasoning levels don't collide in the results dir.
6. Sidecar filename includes `{reasoning_effort}` so config-level aggregates split cleanly:
   `sidecar_{provider}_{model}_{timestamp}_{cfg_name}_{reasoning}.json`.

### C. Qwen MLX provider — `ka59_game/llm_client.py` + new `qwen_mlx.py`

1. Add provider branch `qwen-mlx` in `LLMClient.generate()` parallel to `qwen-local`.
2. New top-level module `qwen_mlx.py` mirroring `qwen_local.py`'s structure:
   - Module-level `_MLX_CACHE` keyed by model name.
   - `_load(model_name)` uses `mlx_lm.load()`.
   - `generate(model_name, system_prompt, user_prompt, reasoning_effort)` applies the
     chat template with `enable_thinking=reasoning_effort is not None` (same toggle as
     `qwen_local.py:58`), calls `mlx_lm.generate()` with `max_tokens` from a
     `_MAX_NEW_TOKENS` table matching transformers path.
   - Returns `(content, input_tokens, output_tokens)`. MLX-LM's `generate()` returns
     a `GenerationResponse` with `prompt_tokens` and `generation_tokens` fields.
3. The `<think>...</think>` stripping logic from `qwen_local.py:78-79` ports over verbatim.

**MLX viability caveat:** User flagged uncertainty about whether MLX can handle the
context size / throughput we need for a full ablation sweep. Initial scope is wiring +
smoke test on one cell (1 trial × 1 config). Full sweep only after smoke test confirms
acceptable per-turn latency on M5 Pro 48 GB.

### D. Tonight's run command

After A + B land:

```bash
python3 -m scripts.run_real_ablation \
  --env ka59simple \
  --provider openrouter \
  --model openai/gpt-5.2 \
  --trials 5 \
  --reasoning-effort none medium
```

Expected output:
- 5 configs × 2 reasoning levels × 5 trials = **50 per-trial JSONs** in `results/ka59simple_game/`
- 10 sidecar JSONs in `results/ka59simple_real_ablation/`
- 2 aggregated `ablation_*.json` files (one per reasoning level) in same dir

Per-cell game-play count: 5 trials × up to 2 attempts = **up to 10 game-plays per cell**,
matching meeting standard.

## Risks & gotchas

1. **429 mislabeling (per memory):** OpenRouter rate-limit responses produce
   complete-looking per-trial JSONs that don't surface the failure clearly. Add a
   post-run validator that flags trials with `errors` containing "429" or with
   abnormally short turn counts (< 3 turns and no win → suspect).
2. **Reasoning_effort dropped by handler (per memory, since fixed):** The OpenRouter
   handler in `ka59_game/llm_client.py:312-345` correctly forwards
   `extra_body={"reasoning": {"effort": ...}}` as of 2026-05-01. Before kicking off the
   full sweep, run `scripts/probe_reasoning_effort.py` on `openai/gpt-5.2` to confirm
   `none` returns `reasoning_tokens=0` and `medium` returns non-zero reasoning tokens.
3. **Aggregate sidecar schema confusion (per memory):** Some validators default-to-0
   and misclassify sidecar JSONs as "all failed". The new sidecar filename includes
   `_{reasoning}_` suffix, which is enough to filter them out of per-trial validators
   that glob `run_*.json`. No schema change needed.
4. **env.step(RESET) behavior on ka59simple:** ka59simple is a single-level game built on
   2026-04-30 as a fork. Verify `env.step(GameAction.RESET)` returns a valid `FrameDataRaw`
   for ka59simple before relying on it for retries. If RESET on a single-level game
   transitions to a "game complete" state instead, we need a different reset path
   (e.g., recreating the env). Smoke-test as the first implementation step.
5. **OODA forced-reframe code path:** Stays in place but unused for ka59simple's
   `mechanics_hard` (not OODA). Don't remove it — the code is still exercised by canonical
   ka59 ad-hoc runs.

## Open questions

None. Proceeding on:
- ka59simple only (no canonical KA59)
- 5 configs (no OODA variants)
- LEVEL_ATTEMPTS = 2 hard-coded
- Both `none` and `medium` reasoning sweeps in one invocation
- MLX wired up but treated as a smoke-test side path, not blocking the gpt-5.2 deadline
