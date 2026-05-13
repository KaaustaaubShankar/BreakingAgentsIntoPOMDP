# KA59simple 2-attempts-per-level + reasoning sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Kaaus's `LEVEL_ATTEMPTS=2` retry pattern (env4 commit `efbb91f`) into the ka59_game harness for ka59simple, extend `scripts/run_real_ablation.py` to sweep `none` and `medium` reasoning in one invocation, and wire up an MLX-Qwen provider for the M5 Pro Apple Silicon side path.

**Architecture:** Restructure `run_agent()`'s flat turn loop into nested `level → attempt → turn` matching env4's shape. Reset env state between attempts via `env.step(GameAction.RESET)`; carry failure context forward in `action_history` so the agent sees that it failed and is retrying. Drop OODA configs from the ablation script and flip the default `--env` to `ka59simple`. MLX support is an additive provider branch, runs on the Mac, doesn't affect the Linux pipeline.

**Tech Stack:** Python 3.12, `arc_agi` + `arcengine` (game engine, ships `GameAction.RESET`), OpenAI SDK (OpenRouter routing for gpt-5.2), `mlx_lm` (Apple Silicon only, Task 7+8).

**Spec:** `docs/superpowers/specs/2026-05-13-ka59simple-level-attempts-design.md`

**Verification style:** This is a research codebase with no live pytest suite under `tests/` (only stale .pyc files). The team verifies empirically via smoke-test scripts that run the real env and inspect outputs — matching the memory feedback "verify mechanics empirically." Each task includes a smoke runner where appropriate; no mocked-LLM unit tests.

---

## Task 1: De-risk `env.step(GameAction.RESET)` on ka59simple

**Why first:** Spec risk #4. If RESET on a single-level ka59simple env transitions to "game complete" instead of restarting the level in-place, the whole env4 retry pattern doesn't translate and Task 2 needs a different approach (env recreation between attempts). Catch this before refactoring 200 lines of `experiment.py`.

**Files:**
- Create: `scripts/smoke_test_ka59simple_reset.py`

- [ ] **Step 1: Write the smoke test script**

```python
"""Smoke-test env.step(GameAction.RESET) on ka59simple.

Verifies that RESET on a single-level fork restarts the level in-place
(returns a valid FrameDataRaw with the agent back at the start position)
rather than transitioning to a "game complete" state.

If RESET works: per-spec Task 2 can use env.step(RESET) between attempts.
If RESET fails: Task 2 needs to recreate the env between attempts.

Run: python3 -m scripts.smoke_test_ka59simple_reset
"""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from arcengine import GameAction
from ka59_game.experiment import _make_env
from ka59_game.game_interface import get_structured_state

env = _make_env("ka59simple")
frame_data = env.observation_space
state0 = get_structured_state(env, frame_data, 1, 64)
print("INITIAL state:")
print(f"  player.position = {state0['player']['position']}")
print(f"  player.id       = {state0['player']['id']}")
print(f"  level.current   = {state0['level']['current']}")
print(f"  game_state      = {state0['game_state']}")
print(f"  frame state     = {frame_data.state.name}")
print(f"  levels_completed= {frame_data.levels_completed}")

print("\nTaking 3 MOVE_LEFT actions to shift state...")
for i in range(3):
    frame_data = env.step(GameAction.ACTION3)  # MOVE_LEFT
    if frame_data is None:
        print(f"  step {i+1}: env.step returned None")
        sys.exit(1)

state_mid = get_structured_state(env, frame_data, 4, 64)
print(f"AFTER 3 LEFT moves:")
print(f"  player.position = {state_mid['player']['position']}")
print(f"  frame state     = {frame_data.state.name}")

print("\nCalling env.step(GameAction.RESET)...")
try:
    frame_data_reset = env.step(GameAction.RESET)
except Exception as exc:
    print(f"RESET raised: {type(exc).__name__}: {exc}")
    sys.exit(2)

if frame_data_reset is None:
    print("RESET returned None — recreate-env path needed in Task 2.")
    sys.exit(3)

state_after = get_structured_state(env, frame_data_reset, 1, 64)
print(f"AFTER RESET:")
print(f"  player.position = {state_after['player']['position']}")
print(f"  player.id       = {state_after['player']['id']}")
print(f"  level.current   = {state_after['level']['current']}")
print(f"  game_state      = {state_after['game_state']}")
print(f"  frame state     = {frame_data_reset.state.name}")
print(f"  levels_completed= {frame_data_reset.levels_completed}")

if state_after['player']['position'] == state0['player']['position']:
    print("\nRESET WORKS — env.step(GameAction.RESET) restores initial position.")
    print("Proceed with Task 2 using env.step(RESET) between attempts.")
    sys.exit(0)
else:
    print("\nRESET DID NOT RESTORE initial position. Task 2 needs env recreation.")
    sys.exit(4)
```

- [ ] **Step 2: Run the smoke test**

Run: `python3 -m scripts.smoke_test_ka59simple_reset`

Expected outcomes:
- Exit 0: RESET works; proceed with Task 2 as written.
- Exit 1: env.step returned None on a normal move — env setup is broken, investigate before Task 2.
- Exit 2: RESET raised — likely `GameAction.RESET` isn't supported on this env. **STOP and re-plan Task 2** to use env recreation.
- Exit 3: RESET returned None — same as exit 2, recreate-env needed.
- Exit 4: RESET ran but didn't restore initial position. **STOP and re-plan Task 2** — investigate what RESET actually does.

- [ ] **Step 3: Commit the smoke test**

```bash
git add scripts/smoke_test_ka59simple_reset.py
git commit -m "test(ka59simple): smoke test env.step(RESET) before retry refactor

De-risks spec risk #4 (RESET on single-level ka59simple). Run before
modifying ka59_game/experiment.py to confirm env.step(GameAction.RESET)
restarts the level cleanly rather than transitioning to game-complete."
```

---

## Task 2: Add per-level retry to `ka59_game/experiment.py`

**Files:**
- Modify: `ka59_game/experiment.py` (the `run_agent()` function and module constants)
- Create: `scripts/smoke_test_level_attempts.py`

**Pre-req:** Task 1 exits with code 0 (or you've adjusted this task per the alternate outcome).

- [ ] **Step 1: Add `LEVEL_ATTEMPTS` constant**

Edit `ka59_game/experiment.py` at the constants block (currently around line 42-46). After `STUCK_THRESHOLD = 5`, add:

```python
LEVEL_ATTEMPTS = 2  # protocol constant — agent gets 2 attempts at each level
                    # before the trial scores as failed at that level. Mirrors
                    # env4/experiment.py:LEVEL_ATTEMPTS (commit efbb91f).
```

- [ ] **Step 2: Add `GameAction` import for RESET**

Confirm the top of `ka59_game/experiment.py` already imports `GameAction` from `arcengine` (it does, line 30). No change needed — just verify before Step 3.

- [ ] **Step 3: Restructure `run_agent()`'s main loop into `level → attempt → turn`**

The current shape (around line 273) is:

```python
for turn in range(1, max_turns + 1):
    # ... single flat turn loop, breaks on WIN / GAME_OVER / timeout
```

Replace with the nested shape below. **Locate** the current `for turn in range(1, max_turns + 1):` loop (line ~273) and the per-turn body through to the end of `run_agent()` (the `understanding_prompt` block at the end of the function, line ~520 or so — confirm before editing).

The new loop shape:

```python
global_turn = 0
stop_run = False

# Walk levels one at a time. Each level gets LEVEL_ATTEMPTS shots.
while not stop_run and result.levels_completed < levels_to_play:
    level_start_completed = int(result.levels_completed)
    current_level = level_start_completed + 1
    level_progressed = False

    log({
        "type": "level_start",
        "summary": f"Level {current_level}/{levels_to_play} started.",
        "level": current_level,
    })

    for level_attempt in range(1, LEVEL_ATTEMPTS + 1):
        # Per-attempt state — reset every retry.
        attempt_turn = 0
        prev_state = None
        last_action_name = ""
        position_history = []
        consecutive_same_type_count = 0
        last_action_type_tracked = ""

        action_history.append(
            f"Level {current_level} attempt {level_attempt}/{LEVEL_ATTEMPTS} started."
        )

        while attempt_turn < turns_per_level:
            global_turn += 1
            attempt_turn += 1
            result.turns = global_turn

            curr_state = get_structured_state(env, frame_data, attempt_turn, turns_per_level)

            if prev_state is not None:
                if feedback_level == "EASY":
                    feedback_text = build_feedback_easy(prev_state, curr_state, last_action_name)
                else:
                    feedback_text = FEEDBACK_HARD
                action_history.append(f"Turn {global_turn - 1}: {last_action_name}\n  Result: {feedback_text}")

            # ----- begin: existing per-turn body, indented under the new while -----
            # (stuck-detection, observation block, prompt build, LLM call,
            #  JSON parse, action execution, log() write) — preserve exactly,
            # but everywhere it currently references the variable `turn` use
            # `global_turn` (for cross-attempt cumulative metrics shown to the
            # agent) and `attempt_turn` (for per-attempt budget display).
            # ----- end -----

            # After env.step:
            #   result.levels_completed = frame_data.levels_completed (already set by existing code)
            #   if frame_data.state.name == "WIN" or result.levels_completed > level_start_completed:
            #       level_progressed = True
            #       if result.levels_completed >= levels_to_play or frame_data.state.name == "WIN":
            #           result.won = True
            #           stop_run = True
            #       break  # exit inner turn loop
            #   if frame_data.state.name == "GAME_OVER":
            #       break  # exit inner turn loop, retry the level

        # Inner turn loop ended. Decide next action:
        if stop_run or level_progressed:
            break  # exit attempt loop; next level or end of run

        failure_summary = (
            f"Level {current_level} attempt {level_attempt}/{LEVEL_ATTEMPTS} "
            f"failed after {attempt_turn} turns."
        )
        if level_attempt < LEVEL_ATTEMPTS:
            log({"type": "level_retry", "summary": failure_summary + " Restarting the same level."})
            action_history.append(
                failure_summary
                + " Game over. Restarting the same level and keeping this failure in context."
            )
            frame_data = env.step(GameAction.RESET)
            if frame_data is None:
                result.errors.append(
                    f"Level {current_level} attempt {level_attempt}: env.step(RESET) returned None."
                )
                stop_run = True
                break
            continue

        log({"type": "game_over", "summary": failure_summary + " No attempts remaining."})
        action_history.append(failure_summary + " No attempts remaining.")
        stop_run = True
        break

    if stop_run:
        break

if not result.won and not stop_run and result.levels_completed < levels_to_play:
    log({"type": "timeout", "summary": f"Turn budget exhausted at turn {global_turn}."})
```

When porting the existing per-turn body into the inner `while attempt_turn < turns_per_level:` loop:

1. Replace every reference to `turn` (the old loop variable) with `global_turn` in places that go into `result.errors` / `log()` / `result.turns = ...`. Use `attempt_turn` only when displaying turn budget to the LLM in the status block.
2. The status block already shows `Steps remaining this level`. Add an `Attempt: {level_attempt}/{LEVEL_ATTEMPTS}` line right above it:

```python
status_lines = [
    f"Selected: {curr_state['player']['id']} @ {curr_state['player']['position']}",
    f"Level: {curr_state['level']['current']}/{curr_state['level']['total']}",
    f"Attempt: {level_attempt}/{LEVEL_ATTEMPTS}",
    f"Steps remaining this level: {curr_state['resources']['steps_remaining']}",
]
```

3. Add `attempt` and `attempt_turn` fields to the action log entry (around the current line 417, `log({"type": "action", ...})`):

```python
log({
    "type": "action",
    "summary": f"Turn {global_turn}: {action_name}" + (f" | {reasoning or decide_text or orient_text}" if (reasoning or decide_text or orient_text) else ""),
    "turn": global_turn,
    "level": current_level,
    "attempt": level_attempt,
    "attempt_turn": attempt_turn,
    "action": action_name,
    "target_position": target_position,
    "reasoning": reasoning,
    "state_before": curr_state,
})
```

4. The forced-reframe block (lines ~329-343) stays — but `turn > STUCK_THRESHOLD` becomes `global_turn > STUCK_THRESHOLD`. `result.forced_reframes` continues to accumulate across attempts since it's on `result`.

5. The OODA orient_history append and `discovery_turn` capture (lines ~383-386) stay; use `global_turn` for `discovery_turn`.

6. Counters `result.click_actions`, `result.invalid_actions`, `result.moves_blocked`, `result.object_pushes`, `result.wall_transfers`, `result.max_goals_occupied` all accumulate across attempts (already on `result` — no change).

- [ ] **Step 4: Delete the old flat loop**

Make sure no stale `for turn in range(1, max_turns + 1):` remains. The outer construct is now the `while not stop_run and result.levels_completed < levels_to_play:` loop. Also delete the old `else:` branch on the `for` loop that logged `"timeout"` — replaced by the new post-loop timeout check.

- [ ] **Step 5: Write a no-LLM smoke test that forces retries**

Create `scripts/smoke_test_level_attempts.py`:

```python
"""Smoke-test LEVEL_ATTEMPTS=2 retry path in ka59_game.experiment.run_agent.

Injects a stub LLMClient that always returns MOVE_LEFT (which on ka59simple
either bumps the wall or wastes turns), forces the agent to exhaust its
turn budget on attempt 1, and verifies:
  - history contains a `level_retry` event
  - action log entries have `attempt` and `attempt_turn` fields
  - result.turns counts cumulative across attempts
  - env.step(RESET) was called between attempts (level_retry event proves it)

Run: python3 -m scripts.smoke_test_level_attempts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from ka59_game.experiment import LEVEL_ATTEMPTS, run_agent
from ka59_game.llm_client import LLMClient


class StubLLMClient(LLMClient):
    """Always responds with MOVE_LEFT. Inherits parse_json + usage tracking."""
    def __init__(self):
        super().__init__(provider="openrouter", model="stub", reasoning_effort=None)

    def generate(self, system_prompt, user_prompt):
        return '{"reasoning": "stub: always left", "action": "MOVE_LEFT"}'


assert LEVEL_ATTEMPTS == 2, f"Expected LEVEL_ATTEMPTS=2, got {LEVEL_ATTEMPTS}"

stub = StubLLMClient()
result = run_agent(
    world_level="EASY",
    goal_level="EASY",
    mechanics_level="EASY",
    feedback_level="EASY",
    max_levels=1,
    turns_per_level=4,  # tiny budget — guarantees attempt 1 fails
    verbose=False,
    llm_client=stub,
    env_id="ka59simple",
)

retry_events = [e for e in result.history if e.get("type") == "level_retry"]
action_events = [e for e in result.history if e.get("type") == "action"]

print(f"won={result.won}  turns={result.turns}  levels_completed={result.levels_completed}")
print(f"  retry events    : {len(retry_events)}")
print(f"  action events   : {len(action_events)}")
print(f"  first action ev : {json.dumps(action_events[0], indent=2)[:400] if action_events else '(none)'}")

assert len(retry_events) >= 1, "Expected at least one level_retry event (attempt 1 should fail with 4-turn budget)"
assert all("attempt" in e and "attempt_turn" in e for e in action_events), \
    "Every action event must include `attempt` and `attempt_turn` fields"
assert result.turns > 4, f"Expected global_turn > turns_per_level=4 (multiple attempts); got {result.turns}"
print("\nSMOKE TEST PASSED")
```

- [ ] **Step 6: Run the smoke test**

Run: `python3 -m scripts.smoke_test_level_attempts`

Expected: prints `won=False ... retry events: 1 ... SMOKE TEST PASSED` and exits 0.

If the AssertionError fires on `retry events >= 1`: the inner GAME_OVER / turn-budget-exhausted detection is broken — re-check Step 3 logic.

If AssertionError fires on `attempt/attempt_turn` fields: the action log entry in Step 3.3 wasn't updated — re-check.

- [ ] **Step 7: Commit**

```bash
git add ka59_game/experiment.py scripts/smoke_test_level_attempts.py
git commit -m "feat(ka59): port LEVEL_ATTEMPTS=2 retry pattern from env4

Restructures run_agent()'s flat turn loop into nested level/attempt/turn,
matching env4/experiment.py (commit efbb91f). Between attempts:
  - env.step(GameAction.RESET) restarts the level in-place
  - action_history carries forward a 'failed last attempt' note so the
    agent reasons about its retry budget
  - per-attempt state (position_history, last_action_name) resets
  - per-trial counters (forced_reframes, click_actions) accumulate

Adds scripts/smoke_test_level_attempts.py: no-LLM verification that
forces an attempt-1 failure via 4-turn budget and asserts the retry
loop fires and emits a level_retry event."
```

---

## Task 3: Update `scripts/run_real_ablation.py` for the new sweep shape

**Files:**
- Modify: `scripts/run_real_ablation.py`

- [ ] **Step 1: Drop OODA configs from `ALL_CONFIGS`**

In `scripts/run_real_ablation.py` (around line 37), remove `mechanics_ooda` and `mechanics_ooda_f` entries. Final dict:

```python
ALL_CONFIGS = {
    "baseline":       {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard":     {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard":      {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "feedback_hard":  {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}
```

- [ ] **Step 2: Add `REASONING_EFFORTS` validation list near `ALL_CONFIGS`**

```python
REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]
```

- [ ] **Step 3: Change `run_ablation()` signature to accept a list of efforts**

Current signature:

```python
def run_ablation(
    provider: str,
    model: str,
    n_trials: int,
    verbose: bool = False,
    max_turns: int = 64,
    configs: list | None = None,
    reasoning_effort: str | None = None,
    env_id: str = "ka59",
) -> dict:
```

New signature:

```python
def run_ablation(
    provider: str,
    model: str,
    n_trials: int,
    verbose: bool = False,
    max_turns: int = 64,
    configs: list | None = None,
    reasoning_efforts: list[str] | None = None,
    env_id: str = "ka59simple",
) -> dict:
```

Notes:
- `reasoning_effort` (singular) → `reasoning_efforts` (plural, list).
- Default env flips from `"ka59"` to `"ka59simple"`.
- Validate inside the function:

```python
efforts_to_run = [e.lower().strip() for e in (reasoning_efforts or ["none"])]
invalid = [e for e in efforts_to_run if e not in REASONING_EFFORTS]
if invalid:
    raise ValueError(f"Invalid reasoning_effort value(s): {invalid}. Expected one of {REASONING_EFFORTS}.")
```

- [ ] **Step 4: Wrap the config loop in a reasoning-effort outer loop**

Currently the function body (after `summary: dict = {}` line ~71) has `for cfg_name, cfg in run_configs.items():`. Wrap that whole block:

```python
summary: dict = {}  # key: f"{effort}::{cfg_name}"

for reasoning_effort in efforts_to_run:
    print(f"\n{'#' * 62}")
    print(f"  Reasoning effort: {reasoning_effort}")
    print(f"{'#' * 62}")

    for cfg_name, cfg in run_configs.items():
        # ... existing per-config body, but:
        # - pass `reasoning_effort=reasoning_effort` to run_agent()
        # - summary key becomes f"{reasoning_effort}::{cfg_name}" not just cfg_name
        # - sidecar filename includes _{reasoning_effort} (see Step 6)
        # - per-trial run_id: append _{reasoning_effort} suffix (see Step 7)
```

- [ ] **Step 5: Update the summary key format**

Where the code currently does `summary[cfg_name] = {...}` (line ~164), change to:

```python
summary_key = f"{reasoning_effort}::{cfg_name}"
summary[summary_key] = {
    "reasoning_effort": reasoning_effort,
    "config": cfg_name,
    "win_rate": win_rate,
    # ... rest of fields unchanged ...
}
```

This lets a single `summary` dict hold all reasoning levels × configs, and the print loop at the end can read the `reasoning_effort` field for display.

- [ ] **Step 6: Update sidecar filename**

Around line 187:

```python
sidecar = results_dir / (
    f"sidecar_{provider}_{model.replace('/', '_')}_{timestamp}_"
    f"{cfg_name}_{reasoning_effort}.json"
)
sidecar.write_text(json.dumps({
    "provider": provider, "model": model, "env_id": env_id,
    "config": cfg_name, "reasoning_effort": reasoning_effort,
    "timestamp": timestamp, "n_trials": n_trials,
    **summary[summary_key],
}, indent=2))
```

- [ ] **Step 7: Update the per-trial run_id format**

The current code passes through to `save_result()` in `ka59_game/experiment.py`, which auto-generates a run_id from microsecond timestamp. To match Kaaus's env4 naming, build an explicit run_id here and pass it through.

`scripts/run_real_ablation.py` doesn't currently call `save_result()` directly — `run_agent()` does (it returns a `RunResult` but the per-trial save happens elsewhere). Confirm whether `run_agent` calls `save_result` internally. Reading `ka59_game/experiment.py`, **the per-trial save is NOT auto-called by `run_agent`** — the ablation script writes its own aggregated JSON, and per-trial JSONs are produced only when the ablation script explicitly invokes `save_result()`.

Check whether the current `run_real_ablation.py` calls `save_result()`. Grep:

```bash
grep -n save_result scripts/run_real_ablation.py
```

If it does not, **per-trial JSONs aren't being produced by this script** — they're being produced by a separate path (maybe by `ka59_game/experiment.py` itself, or by older scripts). In that case: skip Step 7 entirely. The aggregated `ablation_*.json` and per-config `sidecar_*.json` are the relevant outputs.

If `save_result` IS called, add `reasoning_effort` to the run_id before passing it in:

```python
run_id = f"{timestamp}_{cfg_name}_{reasoning_effort}_t{i+1}"
save_result(result, run_id=run_id)
```

- [ ] **Step 8: Update the final summary print block**

Around lines 217-227 (the SUMMARY table). Add a `Reasoning` column. Since the summary keys are now `effort::cfg_name`:

```python
print(f"{'Config':<18} {'Reason':<8} {'Wins':>6} {'Win%':>6} {'AvgTurns':>10} {'AvgLevels':>10} {'P.Walls':>8}")
print("-" * 70)
for key, r in summary.items():
    avg_t = f"{r['avg_turns_on_win']}" if r['avg_turns_on_win'] else "n/a"
    print(
        f"{r['config']:<18} {r['reasoning_effort']:<8} "
        f"{r['wins']:>4}/{r['trials']:<2} {r['win_rate']:>5.0%} "
        f"{avg_t:>10} {r['avg_levels_completed']:>10.2f} "
        f"{r['passable_walls_total']:>8}"
    )
```

- [ ] **Step 9: Update CLI argparse to accept multiple efforts and the new default env**

Around lines 232-247:

```python
parser.add_argument("--provider", default="openrouter")  # was: default="xai"
parser.add_argument("--model", default="openai/gpt-5.2")  # was: default="grok-4-1-fast"
parser.add_argument("--env", default="ka59simple", choices=ENV_CHOICES,
                    help="Which env to ablate. Default ka59simple (single-level fork).")
parser.add_argument("--trials", type=int, default=5)
parser.add_argument("--max-turns", type=int, default=64, dest="max_turns")
parser.add_argument("--configs", nargs="+", default=None, choices=list(ALL_CONFIGS.keys()))
parser.add_argument("--reasoning-effort", nargs="+", default=["none", "medium"],
                    choices=REASONING_EFFORTS, dest="reasoning_efforts",
                    help="One or more reasoning effort levels to sweep in one invocation.")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
run_ablation(args.provider, args.model, args.trials, args.verbose, args.max_turns,
             args.configs, args.reasoning_efforts, env_id=args.env)
```

- [ ] **Step 10: Smoke-test the script with 1 trial × baseline × `none` reasoning**

```bash
python3 -m scripts.run_real_ablation \
    --trials 1 \
    --configs baseline \
    --reasoning-effort none \
    --provider openrouter \
    --model openai/gpt-5.2 \
    --max-turns 8
```

Expected outputs:
- One ablation_*.json in `results/ka59simple_real_ablation/`
- One sidecar with filename pattern `sidecar_openrouter_openai_gpt-5.2_*_baseline_none.json`
- Stdout reports `1/1 trials run; win_rate=X%`

Verify the sidecar filename ends in `_baseline_none.json` (not `_baseline.json`). If it doesn't, Step 6 wasn't applied correctly.

- [ ] **Step 11: Commit**

```bash
git add scripts/run_real_ablation.py
git commit -m "feat(ablation): multi-reasoning sweep + drop OODA + default ka59simple

Mirrors env4/ablation.py (commit efbb91f) shape:
- --reasoning-effort accepts multiple values (default: none medium)
- outer reasoning-effort loop wraps the config loop
- sidecar filename includes _{effort} to disambiguate across efforts
  written in the same invocation (same timestamp)
- summary keyed by {effort}::{cfg} so both reasoning levels coexist
  in one aggregated ablation_*.json

Drops mechanics_ooda + mechanics_ooda_f from ALL_CONFIGS (out of scope
per 2026-05-12 meeting). Flips default --env from ka59 to ka59simple."
```

---

## Task 4: Probe gpt-5.2 reasoning differentiation

**Why:** Spec risk #2. Confirm `reasoning_effort=none` and `reasoning_effort=medium` produce different `reasoning_tokens` counts on OpenRouter before kicking off the expensive full sweep. The OpenRouter handler in `ka59_game/llm_client.py:312-345` was fixed on 2026-05-01 to use `extra_body={"reasoning": {"effort": ...}}` — this task confirms the fix held.

**Files:** none modified. Just runs the existing `scripts/probe_reasoning_effort.py`.

- [ ] **Step 1: Run the probe**

```bash
python3 -m scripts.probe_reasoning_effort
```

- [ ] **Step 2: Inspect the output**

Look for `openai/gpt-5.2` rows. Expected behavior per the 2026-05-01 memory:

| effort | expected reasoning_tokens |
|---|---|
| (omitted, default) | ≈15 (minimal) |
| `none` | 0 |
| `medium` | ≈19 |

If `none` returns nonzero reasoning_tokens: the OpenRouter handler isn't forwarding `extra_body` correctly. **STOP and fix `ka59_game/llm_client.py:_generate_openrouter`** before proceeding to Task 5 — otherwise the full sweep will produce mislabeled data again.

- [ ] **Step 3: Record the result inline**

No git artifact needed — this is a verification step. If it passes, proceed. If it fails, the fix becomes its own task and the plan branches.

---

## Task 5: End-to-end smoke test of the gpt-5.2 sweep

**Files:** none. Runs the ablation script with a small slice.

- [ ] **Step 1: Run a 1-trial slice across both reasoning levels**

```bash
python3 -m scripts.run_real_ablation \
    --trials 1 \
    --configs baseline mechanics_hard \
    --reasoning-effort none medium \
    --provider openrouter \
    --model openai/gpt-5.2 \
    --max-turns 16
```

This runs 2 configs × 2 reasoning levels × 1 trial = 4 trial executions. With `max-turns 16` and up to 2 attempts per trial, worst case is 4 × 16 × 2 = 128 LLM calls. At ~$0.0004 per gpt-5.2 medium call ≈ $0.05.

- [ ] **Step 2: Inspect outputs**

Check `results/ka59simple_real_ablation/` for:
- 1 `ablation_openrouter_openai_gpt-5.2_*.json` (aggregate, both reasoning levels inside)
- 4 sidecars: `sidecar_*_baseline_none.json`, `sidecar_*_baseline_medium.json`, `sidecar_*_mechanics_hard_none.json`, `sidecar_*_mechanics_hard_medium.json`

```bash
ls -la results/ka59simple_real_ablation/ | tail -10
```

Open one sidecar and verify it contains `"reasoning_effort": "none"` (or `"medium"`):

```bash
cat results/ka59simple_real_ablation/sidecar_*_baseline_none.json | python3 -m json.tool | head -30
```

Open the aggregate ablation file and verify summary keys are `none::baseline`, `medium::baseline`, etc.

- [ ] **Step 3: Spot-check a per-trial JSON for `attempt` fields**

Find the most recent per-trial JSON (if any were produced — depends on whether Task 3 Step 7 was needed):

```bash
ls -t results/ka59simple_game/run_*.json 2>/dev/null | head -3
```

If present, open one and verify the action log entries have `"attempt"` and `"attempt_turn"` fields:

```bash
python3 -c "
import json, sys
from pathlib import Path
latest = sorted(Path('results/ka59simple_game').glob('run_*.json'))[-1]
data = json.loads(latest.read_text())
action_events = [e for e in data['history'] if e.get('type') == 'action']
print(f'Latest: {latest.name}')
print(f'Action events: {len(action_events)}')
print(f'Sample event keys: {list(action_events[0].keys())[:10] if action_events else \"(none)\"}')
print(f'Has attempt field: {\"attempt\" in action_events[0] if action_events else False}')
"
```

If `Has attempt field: False`: Task 2 Step 3.3 wasn't applied. Go fix.

- [ ] **Step 4: No commit**

This is a verification step. No artifacts to commit. If failures: fix them and re-run.

---

## Task 6: Wire `qwen-mlx` provider into `LLMClient`

**Apple Silicon only — run this task on the M5 Pro Mac, not the Linux dev box.** MLX is not available on Linux.

**Files:**
- Modify: `ka59_game/llm_client.py` (the `generate()` dispatch and a new `_generate_qwen_mlx()` method)

- [ ] **Step 1: Add `qwen-mlx` to the dispatch in `generate()`**

In `ka59_game/llm_client.py` around line 115-134, add a branch:

```python
if self.provider == "qwen-mlx":
    return self._generate_qwen_mlx(system_prompt, user_prompt)
```

…right before the `qwen-local` branch. Update the error message at the end to include `'qwen-mlx'` in the valid-provider list.

- [ ] **Step 2: Add `_generate_qwen_mlx()` method mirroring `_generate_qwen_local()`**

Right below `_generate_qwen_local()` (which ends around line 178):

```python
def _generate_qwen_mlx(self, system_prompt: str, user_prompt: str) -> str:
    """MLX-based local Qwen inference (Apple Silicon only).

    Delegates to top-level qwen_mlx module so the model cache is shared
    across ka59_game / env3 / env4 within one process. Mirrors the
    transformers-based qwen-local provider, but uses mlx_lm instead.
    """
    import sys, pathlib
    repo_root = str(pathlib.Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from qwen_mlx import generate as _qwen_mlx_generate
    text, input_tokens, output_tokens = _qwen_mlx_generate(
        self.model, system_prompt, user_prompt, self.reasoning_effort
    )

    class _U:
        pass

    _U.input_tokens = input_tokens
    _U.output_tokens = output_tokens
    _U.total_tokens = input_tokens + output_tokens

    class _R:
        usage = _U()

    self._record_usage(_R())
    return text
```

- [ ] **Step 3: Commit**

```bash
git add ka59_game/llm_client.py
git commit -m "feat(llm_client): add qwen-mlx provider for Apple Silicon

Parallels the qwen-local provider but routes through mlx_lm via a new
qwen_mlx.py module (next commit). Same reasoning_effort -> Qwen3
enable_thinking mapping pattern. Shared model cache across ka59_game /
env3 / env4 within one process."
```

---

## Task 7: Create `qwen_mlx.py` MLX inference helper

**Apple Silicon only.** Skip on Linux.

**Files:**
- Create: `qwen_mlx.py` (top-level, parallel to `qwen_local.py`)

- [ ] **Step 1: Confirm `mlx-lm` is installed**

Run: `python3 -c "import mlx_lm; print(mlx_lm.__version__)"`

If ImportError: `pip install mlx-lm`. Confirm it imports before proceeding.

- [ ] **Step 2: Write `qwen_mlx.py`**

```python
"""MLX-based Qwen local inference. Apple Silicon (M-series) only.

Parallel to qwen_local.py (transformers-based, CPU/CUDA). Loads MLX
weights once per process via module-level cache keyed by model name.
Maps the harness-wide `reasoning_effort` knob to Qwen 3's
`enable_thinking` chat-template toggle plus a proportional max_tokens
budget.

Model IDs typically come from the `mlx-community/` HuggingFace org,
e.g. `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`.

Imported lazily from ka59_game/llm_client.py's qwen-mlx branch so
non-Apple environments don't hit a hard import error at top-of-file.
"""
from __future__ import annotations

import re
from typing import Any

_MLX_CACHE: dict[str, Any] = {}

# Same shape as qwen_local._MAX_NEW_TOKENS — proportional budget per
# reasoning level. None disables thinking; non-None enables it and
# scales the token budget so the model has room to think.
_MAX_NEW_TOKENS = {
    None: 1024,
    "minimal": 1536,
    "low": 2560,
    "medium": 4608,
    "high": 8704,
}


def _load(model_name: str) -> Any:
    if model_name in _MLX_CACHE:
        return _MLX_CACHE[model_name]
    from mlx_lm import load
    model, tokenizer = load(model_name)
    _MLX_CACHE[model_name] = (model, tokenizer)
    return _MLX_CACHE[model_name]


def generate(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str | None,
) -> tuple[str, int, int]:
    """Run one MLX Qwen generation.

    Returns (visible_text, input_tokens, output_tokens). The Qwen
    <think>...</think> block (when emitted) is stripped before return
    so callers see only the visible reply.
    """
    from mlx_lm import generate as mlx_generate

    model, tokenizer = _load(model_name)
    enable_thinking = reasoning_effort is not None
    max_tokens = _MAX_NEW_TOKENS.get(reasoning_effort, 1024)

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # mlx_lm.generate returns the generated text directly. To get token
    # counts, encode the prompt ourselves and count the generated chars
    # roughly via re-encoding the response.
    input_tokens = len(tokenizer.encode(prompt))
    full_text = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    output_tokens = len(tokenizer.encode(full_text))

    match = re.search(r"</think>\s*", full_text)
    content = full_text[match.end():] if match else full_text
    return content.strip(), int(input_tokens), int(output_tokens)
```

- [ ] **Step 3: Commit**

```bash
git add qwen_mlx.py
git commit -m "feat(qwen-mlx): MLX inference helper for Apple Silicon

Mirrors qwen_local.py (transformers) but uses mlx_lm. Shares the same
reasoning_effort -> enable_thinking mapping and the same
_MAX_NEW_TOKENS budget table. Module-level cache keyed by model name
so one process loading multiple models pays the load cost once each.

Imported lazily from ka59_game/llm_client._generate_qwen_mlx so Linux
environments without mlx-lm don't fail at import time."
```

---

## Task 8: Smoke-test `qwen-mlx` end-to-end

**Apple Silicon only.** Skip on Linux.

**Files:**
- Create: `scripts/smoke_test_qwen_mlx.py`

- [ ] **Step 1: Write the smoke test**

```python
"""Smoke-test qwen-mlx provider via LLMClient.

Loads a small MLX Qwen model, runs one generation, prints the response
and token counts. Sanity check before running a full ablation cell
via qwen-mlx.

Run on Apple Silicon only:
    python3 -m scripts.smoke_test_qwen_mlx
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from ka59_game.llm_client import LLMClient

# Pick a small MLX model that fits on M5 Pro 48 GB comfortably and
# loads quickly for smoke purposes. Swap for the production model
# (Qwen3-30B-A3B-Instruct-2507-4bit) once the smoke test is green.
MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

client = LLMClient(provider="qwen-mlx", model=MODEL, reasoning_effort=None)
t0 = time.time()
reply = client.generate(
    system_prompt="You are a helpful assistant. Respond in one short sentence.",
    user_prompt="What is 17 * 23?",
)
elapsed = time.time() - t0
usage = client.get_usage_summary()

print(f"Model:       {MODEL}")
print(f"Reply:       {reply!r}")
print(f"Latency:     {elapsed:.1f}s")
print(f"In tokens:   {usage['input_tokens']}")
print(f"Out tokens:  {usage['output_tokens']}")

assert "391" in reply, f"Expected '391' in reply (17*23=391), got: {reply!r}"
print("\nSMOKE TEST PASSED")
```

- [ ] **Step 2: Run on the Mac**

```bash
python3 -m scripts.smoke_test_qwen_mlx
```

Expected:
- Reply contains `391`
- Latency < 30s for the small smoke model
- Token counts non-zero

If load fails: check `mlx-lm` is installed and the model name resolves on HuggingFace.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test_qwen_mlx.py
git commit -m "test(qwen-mlx): smoke test for MLX provider on Apple Silicon

Confirms the qwen-mlx provider end-to-end:
LLMClient(provider='qwen-mlx') -> qwen_mlx.generate() -> mlx_lm.

Uses Qwen2.5-3B-Instruct-4bit (small/fast) for the smoke. Swap to
Qwen3-30B-A3B-Instruct-2507-4bit for real ablation runs on M5 Pro 48GB."
```

---

## After all tasks: the actual run

Not part of this plan (no code), but for reference. Once Tasks 1-5 land on the Linux side, kick off the gpt-5.2 sweep:

```bash
python3 -m scripts.run_real_ablation \
    --env ka59simple \
    --provider openrouter \
    --model openai/gpt-5.2 \
    --trials 5 \
    --reasoning-effort none medium
```

Expected: 5 configs × 2 reasoning levels × 5 trials × up to 2 attempts = up to 100 game-plays. ~$5 in OpenRouter spend, ~1-2 hours wall time.
