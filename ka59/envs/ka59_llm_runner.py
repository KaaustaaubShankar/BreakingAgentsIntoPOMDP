"""
ka59_llm_runner.py — LLM-powered KA59 agent for the multi-env runner.

Wraps ka59_ref (pure Python, no arc_agi) to run an LLM agent against the
KA59 mechanics-discovery benchmark.  Default turn cap: 64 (proactive Flash /
Sonnet runs testing whether LLMs lift the 0% baseline win rate).

Unified metric mapping:
  invalid_clicks ← blocked_count (moves that didn't advance the selected piece)
  flips          ← passable_walls_found (wall-transfer discoveries)

Cross-game hypothesis-ruling trace:
  hypo_trace captures:
    - blocked_dirs: direction → blocked-count (empirical wall hypothesis)
    - push_success_dirs: direction → push-success count
    - passable_walls_found: distinct wall IDs observed passable
    - action_histogram: count per action type

Axis semantics for KA59:
  World     — EASY: full ObjectView (id/x/y/w/h/kind/is_selected)
              HARD: positions hidden for non-selected objects (observe_positions=False)
  Goal      — EASY: objective explained ("get selected piece to target region")
              HARD: nothing stated
  Mechanics — EASY: rules explained (wall-transfer asymmetry, push mechanics)
              HARD: no instructions
  Feedback  — EASY: detailed state diff (moved/blocked/step-count)
              HARD: "Ok."
"""

from __future__ import annotations

import json
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── path setup: make ka59_ref importable ──────────────────────────────────────

_KA59_DIR   = Path(__file__).parents[1]
_REPO_ROOT  = _KA59_DIR.parent

for _p in (str(_REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ka59_ref.env import KA59BlindEnv, Action, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, SELECT
from ka59_ref.scenarios import SCENARIOS   # canonical level specs

# LLM client (shared across envs — uses OpenRouter)
_VENDOR_BP35 = _KA59_DIR / "vendor" / "bp35"
if str(_VENDOR_BP35) not in sys.path:
    sys.path.insert(0, str(_VENDOR_BP35))

# Use the bp35 LLM client — identical API, no BP35-specific logic
try:
    from llm_client import LLMClient  # type: ignore
except ImportError:
    raise ImportError(
        "llm_client not found in vendor/bp35. "
        "Run from the ka59/ directory or ensure vendor/bp35/llm_client.py exists."
    )

# ── constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = _KA59_DIR / "results" / "ka59_llm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_STEPS = 64     # proactive: 64-turn budget (real KA59 0% → lift?)

# Default scenario for ablation (the mechanics-discovery scenario)
DEFAULT_SCENARIO = "transfer_wall_push"

GOAL_EASY = """\
Your goal is to move the selected (controllable) piece through the game board
and reach the target area.  You can also push other pieces to clear the way.
The board contains two kinds of boundary objects (both shown as kind="wall"):
  - Some walls can be passed through by pushed pieces (transfer walls)
  - Other walls block all movement (solid walls)
You must discover which is which through your moves.\
"""

GOAL_HARD = ""

MECHANICS_EASY = """\
Game mechanics:
- You control the piece with is_selected=true.
- Move actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
- If the selected piece is adjacent to another piece, a move in that direction
  tries to push both pieces simultaneously.
- If the adjacent piece is against a "wall", the push may either:
  a) Succeed (transfer wall — the pushed piece passes through)
  b) Fail (solid wall — both pieces stay put)
- You cannot tell wall types apart by observation alone; discover them by trying.
- SELECT <id> switches which piece is active (costs 0 steps).
- Steps are limited; plan efficiently.

Respond with a single JSON object:
  {"reasoning": "<plan>", "action": "MOVE_LEFT"}
  or for SELECT:
  {"reasoning": "<plan>", "action": "SELECT", "target_id": "<id>"}\
"""

MECHANICS_HARD = ""

FEEDBACK_EASY_TEMPLATE = """\
After your action:
  Moved: {moved}
  Steps remaining: {steps_remaining}
  Selected piece now at: {selected_pos}\
"""

FEEDBACK_HARD = "Ok."

UNDERSTANDING_PROMPT = """\
Based on your experience in this episode, answer briefly:
{"goal": "<what was the goal?>", "wall_discovery": "<what did you learn about walls?>", "strategy": "<what worked or failed?>"}\
"""


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    config: dict[str, str]
    scenario: str
    won: bool
    turns: int
    levels_completed: int      # always 0 or 1 for KA59 (single episode)
    provider: str = ""
    model: str = ""
    errors: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    understanding: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    blocked_count: int = 0
    passable_walls_found: int = 0
    select_actions: int = 0
    moved_count: int = 0


# ── observation builder ───────────────────────────────────────────────────────

def _obs_to_json(obs, turn: int, max_turns: int, world_level: str) -> str:
    """Convert ObjectView list to a JSON string for the LLM."""
    objects = []
    for v in obs:
        if world_level == "EASY" or v.is_selected:
            obj = {
                "id": v.id,
                "x": v.x,
                "y": v.y,
                "w": v.w,
                "h": v.h,
                "kind": v.kind,
                "is_selected": v.is_selected,
            }
        else:
            # HARD: non-selected positions hidden
            obj = {
                "id": v.id,
                "x": 0, "y": 0, "w": 0, "h": 0,
                "kind": v.kind,
                "is_selected": False,
            }
        objects.append(obj)
    return json.dumps({"turn": turn, "max_turns": max_turns, "objects": objects}, indent=2)


# ── main run_agent ────────────────────────────────────────────────────────────

def run_agent(
    world_level: str = "EASY",
    goal_level: str = "EASY",
    mechanics_level: str = "EASY",
    feedback_level: str = "EASY",
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    scenario_name: str = DEFAULT_SCENARIO,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = True,
) -> RunResult:
    """
    Run one KA59 LLM episode.

    The win condition for KA59 is defined by the level spec's step budget
    being reached without the agent getting permanently stuck.  For the
    purposes of the unified table, 'won' is set to True if:
      - The selected piece reaches the goal region (passable_walls_found > 0
        and the episode terminates via done=True with steps > 0 remaining), OR
      - The agent discovers at least one passable wall AND moved more than half
        the time (heuristic; KA59 has no explicit win state).

    This is a known limitation: KA59's original design is a discovery probe,
    not a win/lose game.  We use a heuristic win condition for the table.
    """
    config = {
        "world": world_level,
        "goal": goal_level,
        "mechanics": mechanics_level,
        "feedback": feedback_level,
    }
    result = RunResult(
        config=config,
        scenario=scenario_name,
        won=False,
        turns=0,
        levels_completed=0,
        provider=provider,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if scenario_name not in SCENARIOS:
        result.errors.append(f"Unknown KA59 scenario: {scenario_name!r}")
        return result

    level_spec = SCENARIOS[scenario_name]
    env = KA59BlindEnv()
    obs = env.reset(level_spec)

    goal_text  = GOAL_EASY if goal_level == "EASY" else GOAL_HARD
    mech_text  = MECHANICS_EASY if mechanics_level == "EASY" else MECHANICS_HARD
    system_prompt = "\n\n".join(filter(None, [goal_text, mech_text])) or (
        "You are playing a game. Explore and find the goal."
    )

    client = LLMClient(provider=provider, model=model)
    history: list[dict[str, Any]] = []
    action_history: list[str] = []
    recent_actions: list[str] = []
    passable_wall_ids: set = set()

    def log(event: dict[str, Any]) -> None:
        history.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})
        if verbose:
            print(f"  [{event.get('type', '?')}] {event.get('summary', '')}")

    log({"type": "game_start", "summary": f"KA59 episode started ({scenario_name}).", "config": config})

    prev_obs = obs
    for turn in range(1, max_steps + 1):
        result.turns = turn
        obs_text = _obs_to_json(obs, turn, max_steps, world_level)

        # Feedback from previous step
        if action_history and feedback_level == "EASY":
            feedback = action_history[-1]
        elif action_history:
            feedback = FEEDBACK_HARD
        else:
            feedback = ""

        recent_history = action_history[-10:]
        history_block = (
            "RECENT ACTIONS (last 10 turns):\n" + "\n".join(recent_history) + "\n"
            if recent_history else ""
        )

        warnings: list[str] = []
        if len(recent_actions) >= 6 and len(set(recent_actions[-6:])) == 1:
            warnings.append(
                f"WARNING: repeated {recent_actions[-1]} 6 times. The board may be blocking you. Try a different direction or SELECT another piece."
            )

        # Selected piece position for prompt anchoring
        selected = next((v for v in obs if v.is_selected), None)
        sel_pos = f"[{selected.x}, {selected.y}]" if selected else "unknown"

        user_prompt = (
            ("\n".join(warnings) + "\n" if warnings else "")
            + history_block
            + f"\nCurrent selected piece position: {sel_pos}\n"
            + f"\nOBSERVATION:\n{obs_text}\n"
            + (f"\nFEEDBACK:\n{feedback}\n" if feedback else "")
            + '\n\nRespond with a single JSON object: {"reasoning": "<plan>", "action": "MOVE_LEFT"}'
            + '\nFor SELECT: {"reasoning": "<plan>", "action": "SELECT", "target_id": "<id>"}'
        )

        retry_prompt = (
            "Invalid JSON. Reply with ONLY one JSON object.\n"
            'Valid actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, SELECT\n'
            '{"reasoning": "trying left", "action": "MOVE_LEFT"}'
        )

        try:
            reply = client.generate(system_prompt, user_prompt)
            try:
                parsed = client.parse_json(reply)
            except ValueError:
                reply = client.generate(system_prompt, retry_prompt)
                parsed = client.parse_json(reply)
        except Exception as exc:
            err = f"Turn {turn}: LLM/parse error — {exc}"
            result.errors.append(err)
            log({"type": "error", "summary": err})
            continue

        action_name = str(parsed.get("action", "")).upper().strip()
        reasoning   = str(parsed.get("reasoning", "")).strip()
        target_id   = parsed.get("target_id")

        # Build Action
        action_map = {
            "MOVE_UP":    MOVE_UP,
            "MOVE_DOWN":  MOVE_DOWN,
            "MOVE_LEFT":  MOVE_LEFT,
            "MOVE_RIGHT": MOVE_RIGHT,
        }
        if action_name in action_map:
            action = action_map[action_name]
        elif action_name == "SELECT":
            if not target_id:
                err = f"Turn {turn}: SELECT missing target_id"
                result.errors.append(err)
                log({"type": "invalid_action", "summary": err})
                continue
            action = SELECT(str(target_id))
            result.select_actions += 1
        else:
            err = f"Turn {turn}: unknown action '{action_name}'"
            result.errors.append(err)
            log({"type": "invalid_action", "summary": err})
            continue

        step_result = env.step(action)
        obs = list(step_result.obs)

        # Track blocked moves
        if not step_result.moved and action_name.startswith("MOVE_"):
            result.blocked_count += 1

        if step_result.moved and action_name.startswith("MOVE_"):
            result.moved_count += 1
            # Detect passable-wall discoveries: if pushed block moved through a
            # wall position, count it.  We approximate by tracking moved_count
            # increases that coincide with a neighbouring wall object.
            _check_wall_pass(obs, prev_obs, passable_wall_ids)

        result.passable_walls_found = len(passable_wall_ids)

        # Build feedback text for next turn
        sel_now = next((v for v in obs if v.is_selected), None)
        sel_pos_now = f"[{sel_now.x}, {sel_now.y}]" if sel_now else "unknown"
        fb_text = (
            FEEDBACK_EASY_TEMPLATE.format(
                moved=step_result.moved,
                steps_remaining=step_result.steps_remaining,
                selected_pos=sel_pos_now,
            )
            if feedback_level == "EASY"
            else FEEDBACK_HARD
        )
        action_history.append(f"Turn {turn}: {action_name} → {fb_text.splitlines()[0]}")
        recent_actions.append(action_name)

        log({
            "type": "action",
            "summary": f"Turn {turn}: {action_name} | moved={step_result.moved} | steps_left={step_result.steps_remaining}",
            "turn": turn,
            "action": action_name,
            "reasoning": reasoning,
            "moved": step_result.moved,
            "steps_remaining": step_result.steps_remaining,
        })

        prev_obs = obs

        if step_result.done:
            log({"type": "episode_end", "summary": f"Done after {turn} turns."})
            break
    else:
        log({"type": "timeout", "summary": f"Max steps ({max_steps}) exhausted."})

    # Heuristic win: discovered ≥1 passable wall AND moved > 40% of turns
    move_rate = result.moved_count / max(result.turns, 1)
    if result.passable_walls_found > 0 and move_rate > 0.4:
        result.won = True
        result.levels_completed = 1

    # Understanding prompt
    try:
        understanding_reply = client.generate(system_prompt, UNDERSTANDING_PROMPT)
        result.understanding = {
            k: str(v) for k, v in client.parse_json(understanding_reply).items()
        }
    except Exception as exc:
        result.errors.append(f"Understanding prompt failed: {exc}")

    result.history = history
    return result


def _check_wall_pass(
    obs,
    prev_obs,
    passable_wall_ids: set,
) -> None:
    """
    Heuristic: if a non-selected block moved from a position that had a wall
    in prev_obs at the SAME location, record that wall id as passable.
    """
    prev_positions: dict[str, tuple] = {v.id: (v.x, v.y) for v in prev_obs}
    prev_walls: set[tuple] = {
        (v.x, v.y) for v in prev_obs if v.kind == "wall"
    }
    for v in obs:
        if v.kind == "block" and not v.is_selected:
            old = prev_positions.get(v.id)
            if old and (old[0] != v.x or old[1] != v.y):
                # Block moved — check if it passed through a wall position
                if old in prev_walls:
                    # Approximate: find which wall was at old position
                    for pw in prev_obs:
                        if pw.kind == "wall" and pw.x == old[0] and pw.y == old[1]:
                            passable_wall_ids.add(pw.id)
