"""
experiment.py — Single-run LLM agent harness for KA59.

Mirrors env4/experiment.py: same RunResult schema, same axis config shape
({world, goal, mechanics, feedback}), same JSON response protocol, so KA59
and BP35 results go into the same ablation table.

KA59-specific differences vs BP35:
- One scenario per run (no level progression). Default scenario is
  `transfer_wall_push` — the core Mechanics contrastive probe.
- Action set: MOVE_{LEFT,RIGHT,UP,DOWN}, SELECT(target_id).
- No clicks, undos, or gravity flips (not part of KA59 mechanics).
- WORLD_HARD is implemented via observe_positions=False on the level spec,
  which the KA59BlindEnv honours natively (non-selected object positions
  collapse to [0, 0]).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .engine import STEP
from .env import KA59BlindEnv, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, SELECT, Action
from .game_interface import build_feedback_easy, get_structured_state
from .llm_client import LLMClient
from .prompts import FEEDBACK_HARD, UNDERSTANDING_PROMPT, build_system_prompt
from .scenarios import SCENARIOS


RESULTS_DIR = Path("results") / "ka59"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Win condition per scenario: ANY pushable object (selected piece or pushable
# block) must reach x >= target_x. This captures KA59's discovery signal —
# e.g. `transfer_wall_push` is only winnable if the agent discovers that
# pushed blocks can cross a wall_transfer boundary. Contrastive unwinnable
# baselines (`solid_wall_push_blocked`, `transfer_wall_direct_block`) are
# intentional — they anchor the ablation table's loss row.
SCENARIO_GOALS: dict[str, int] = {
    "open_move_right": STEP * 3,
    "transfer_wall_direct_block": STEP * 2,
    "transfer_wall_push": STEP * 2,
    "solid_wall_push_blocked": STEP * 2,
    "push_chain": STEP * 2,
    "transfer_wall_push_world_blind": STEP * 2,
}

ACTION_MAP: dict[str, Action] = {
    "MOVE_LEFT": MOVE_LEFT,
    "MOVE_RIGHT": MOVE_RIGHT,
    "MOVE_UP": MOVE_UP,
    "MOVE_DOWN": MOVE_DOWN,
}


@dataclass
class RunResult:
    config: dict[str, str]
    won: bool
    turns: int
    levels_completed: int
    provider: str = ""
    model: str = ""
    errors: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    understanding: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    invalid_actions: int = 0
    select_actions: int = 0
    push_events: int = 0
    moves_blocked: int = 0


def save_result(result: RunResult, run_id: Optional[str] = None) -> Path:
    cfg = result.config
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = (
        f"f{cfg['feedback'][0]}_g{cfg['goal'][0]}_"
        f"m{cfg['mechanics'][0]}_w{cfg['world'][0]}_"
        f"{cfg['scenario']}"
    )
    path = RESULTS_DIR / f"run_{tag}_{run_id}.json"
    payload = {
        "config": result.config,
        "won": result.won,
        "turns": result.turns,
        "levels_completed": result.levels_completed,
        "provider": result.provider,
        "model": result.model,
        "timestamp": result.timestamp,
        "errors": result.errors,
        "understanding": result.understanding,
        "invalid_actions": result.invalid_actions,
        "select_actions": result.select_actions,
        "push_events": result.push_events,
        "moves_blocked": result.moves_blocked,
        "history": result.history,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved -> {path}")
    return path


def _indexed_semantic_grid_text(rows: list[str]) -> str:
    if not rows:
        return "(empty grid)"
    width = len(rows[0])
    tens = "".join(str((x // 10) % 10) if x >= 10 else " " for x in range(width))
    ones = "".join(str(x % 10) for x in range(width))
    out: list[str] = [
        "x-index:",
        f"      {tens}",
        f"      {ones}",
    ]
    for y, row in enumerate(rows):
        out.append(f"y={y:02d}  {row}")
    return "\n".join(out)


def _build_world_easy_observation(
    curr_state: dict[str, Any], turn: int, max_turns: int
) -> str:
    state_no_grid = {k: v for k, v in curr_state.items() if k != "semantic_grid"}
    indexed_grid = _indexed_semantic_grid_text(curr_state["semantic_grid"])
    legend = (
        "SEMANTIC LEGEND:\n"
        "- P=selected piece  B=pushable block  #=wall boundary  .=empty cell\n"
        "COORDINATES:\n"
        "- Positions are [x, y] with x increasing rightward, y increasing downward.\n"
        "- 1 cell in the grid = 1 STEP = 3 pixels. Object positions in `objects` are pixel coordinates."
    )
    return (
        f"CURRENT STATE (turn {turn}/{max_turns}):\n"
        f"{json.dumps(state_no_grid, indent=2)}\n\n"
        f"{legend}\n\n"
        f"SEMANTIC GRID (indexed):\n{indexed_grid}"
    )


def _build_world_hard_observation(
    curr_state: dict[str, Any], turn: int, max_turns: int
) -> str:
    minimal = {
        "player": curr_state["player"],
        "movement": {
            "left_blocked": curr_state["movement"]["left_blocked"],
            "right_blocked": curr_state["movement"]["right_blocked"],
            "up_blocked": curr_state["movement"]["up_blocked"],
            "down_blocked": curr_state["movement"]["down_blocked"],
        },
        "resources": curr_state["resources"],
        "game_state": curr_state["game_state"],
    }
    return (
        f"CURRENT STATE (turn {turn}/{max_turns}, degraded):\n"
        f"{json.dumps(minimal, indent=2)}\n"
        "(Non-player object positions are not available this run. Reason about blocked-flags and action outcomes only.)"
    )


def _count_push_events(prev: dict[str, Any], curr: dict[str, Any]) -> int:
    prev_blocks = {o["id"]: tuple(o["position"]) for o in prev["objects"]["blocks"]}
    curr_blocks = {o["id"]: tuple(o["position"]) for o in curr["objects"]["blocks"]}
    count = 0
    for obj_id, curr_pos in curr_blocks.items():
        prev_pos = prev_blocks.get(obj_id)
        if prev_pos is not None and prev_pos != curr_pos:
            count += 1
    return count


def run_agent(
    scenario: str = "transfer_wall_push",
    world_level: str = "EASY",
    goal_level: str = "EASY",
    mechanics_level: str = "EASY",
    feedback_level: str = "EASY",
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    max_turns: Optional[int] = None,
    verbose: bool = True,
    llm_client: Optional[LLMClient] = None,
) -> RunResult:
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Available: {sorted(SCENARIOS.keys())}"
        )
    if scenario not in SCENARIO_GOALS:
        raise ValueError(f"No target_x configured for scenario {scenario!r}.")

    target_x = SCENARIO_GOALS[scenario]
    spec = dict(SCENARIOS[scenario])
    if world_level == "HARD":
        spec["observe_positions"] = False

    step_budget = int(spec.get("steps", 10))
    turns_budget = max_turns if max_turns is not None else step_budget

    config = {
        "scenario": scenario,
        "world": world_level,
        "goal": goal_level,
        "mechanics": mechanics_level,
        "feedback": feedback_level,
        "target_x": target_x,
    }
    result = RunResult(
        config=config,
        won=False,
        turns=0,
        levels_completed=0,
        provider=provider,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    client = llm_client if llm_client is not None else LLMClient(provider=provider, model=model)
    system_prompt = build_system_prompt(goal_level, mechanics_level)

    env = KA59BlindEnv()
    env.reset(spec)

    history: list[dict[str, Any]] = []
    action_history: list[str] = []
    prev_state: Optional[dict[str, Any]] = None
    last_action_name = ""
    recent_actions: list[str] = []
    history_block = ""

    def log(event: dict[str, Any]) -> None:
        history.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})
        if verbose:
            print(f"  [{event.get('type', '?')}] {event.get('summary', '')}")

    log({"type": "game_start", "summary": f"KA59 scenario={scenario} started.", "config": config})

    for turn in range(1, turns_budget + 1):
        result.turns = turn
        curr_state = get_structured_state(env, scenario, step_budget, target_x)

        if prev_state is not None:
            if feedback_level == "EASY":
                feedback_text = build_feedback_easy(prev_state, curr_state, last_action_name)
            else:
                feedback_text = FEEDBACK_HARD
            action_history.append(f"Turn {turn - 1}: {last_action_name}\n  Result: {feedback_text}")

        if world_level == "HARD":
            obs_block = _build_world_hard_observation(curr_state, turn, turns_budget)
        else:
            obs_block = _build_world_easy_observation(curr_state, turn, turns_budget)

        recent_history = action_history[-10:]
        history_block = (
            "RECENT ACTIONS (last 10 turns):\n" + "\n".join(recent_history) + "\n"
            if recent_history else ""
        )

        status_lines = [
            f"Selected piece: id={curr_state['player']['id']} position={curr_state['player']['position']}",
            f"Target x: {target_x} (win when selected.x >= target_x)",
            f"Steps remaining: {curr_state['resources']['steps_remaining']}",
        ]
        status_block = "STATUS:\n" + "\n".join(status_lines) + "\n"

        warnings: list[str] = []
        if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 1:
            warnings.append(
                f"WARNING: You repeated {recent_actions[-1]} for 5 turns. Try a different action or SELECT."
            )

        user_prompt = (
            ("\n".join(warnings) + "\n" if warnings else "")
            + history_block
            + "\n"
            + status_block
            + "\n"
            + obs_block
            + '\n\nRespond with a single JSON object only: {"reasoning": "<plan>", "action": "MOVE_RIGHT"}'
        )

        retry_prompt = (
            "You forgot valid JSON. Reply with ONLY one JSON object.\n"
            "Examples:\n"
            '{"reasoning": "I should move right", "action": "MOVE_RIGHT"}\n'
            '{"reasoning": "Try controlling the block", "action": "SELECT", "target_id": "pb"}'
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
        reasoning = str(parsed.get("reasoning", "")).strip()
        target_id = parsed.get("target_id")

        env_action: Optional[Action] = None
        if action_name in ACTION_MAP:
            env_action = ACTION_MAP[action_name]
        elif action_name == "SELECT":
            if not isinstance(target_id, str) or not target_id:
                result.invalid_actions += 1
                err = f"Turn {turn}: SELECT missing target_id"
                result.errors.append(err)
                log({"type": "invalid_action", "summary": err, "raw_reply": reply})
                continue
            try:
                env_action = SELECT(target_id)
                result.select_actions += 1
            except Exception as exc:
                result.invalid_actions += 1
                err = f"Turn {turn}: invalid SELECT target {target_id!r} — {exc}"
                result.errors.append(err)
                log({"type": "invalid_action", "summary": err, "raw_reply": reply})
                continue
        else:
            result.invalid_actions += 1
            err = f"Turn {turn}: unknown action '{action_name}'"
            result.errors.append(err)
            log({"type": "invalid_action", "summary": err, "raw_reply": reply})
            continue

        log(
            {
                "type": "action",
                "summary": f"Turn {turn}: {action_name}" + (f" | {reasoning}" if reasoning else ""),
                "turn": turn,
                "action": action_name,
                "target_id": target_id,
                "reasoning": reasoning,
                "state_before": curr_state,
            }
        )

        prev_state = curr_state
        last_action_name = action_name if action_name != "SELECT" else f"SELECT {target_id}"
        recent_actions.append(last_action_name)

        try:
            env.step(env_action)
        except (KeyError, ValueError) as exc:
            result.invalid_actions += 1
            err = f"Turn {turn}: env rejected action {action_name} — {exc}"
            result.errors.append(err)
            log({"type": "env_error", "summary": err})
            continue

        post_state = get_structured_state(env, scenario, step_budget, target_x)
        result.push_events += _count_push_events(curr_state, post_state)
        if (
            action_name in ACTION_MAP
            and curr_state["player"]["position"] == post_state["player"]["position"]
        ):
            result.moves_blocked += 1
        if post_state["game_state"] == "WIN":
            result.won = True
            result.levels_completed = 1
            log({"type": "win", "summary": f"WIN after {turn} turns."})
            break
        if post_state["game_state"] == "GAME_OVER":
            log({"type": "game_over", "summary": f"GAME_OVER after {turn} turns."})
            break
    else:
        log({"type": "timeout", "summary": f"Turn budget ({turns_budget}) exhausted."})

    understanding_prompt_text = (history_block if history_block else "") + "\n" + UNDERSTANDING_PROMPT
    try:
        understanding_reply = client.generate(system_prompt, understanding_prompt_text)
        result.understanding = {
            key: str(value) for key, value in client.parse_json(understanding_reply).items()
        }
        log(
            {
                "type": "understanding",
                "summary": "Agent explained goal and mechanics.",
                "understanding": result.understanding,
            }
        )
    except Exception as exc:
        result.errors.append(f"Understanding prompt failed: {exc}")

    result.history = history
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run KA59 single-run experiment.")
    p.add_argument("--scenario", default="transfer_wall_push", choices=sorted(SCENARIOS.keys()))
    p.add_argument("--provider", default="openrouter", choices=["openrouter"])
    p.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free")
    p.add_argument("--world", default="EASY", choices=["EASY", "HARD"])
    p.add_argument("--goal", default="EASY", choices=["EASY", "HARD"])
    p.add_argument("--mechanics", default="EASY", choices=["EASY", "HARD"])
    p.add_argument("--feedback", default="EASY", choices=["EASY", "HARD"])
    p.add_argument("--max-turns", type=int)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_agent(
        scenario=args.scenario,
        world_level=args.world,
        goal_level=args.goal,
        mechanics_level=args.mechanics,
        feedback_level=args.feedback,
        provider=args.provider,
        model=args.model,
        max_turns=args.max_turns,
        verbose=not args.quiet,
    )
    save_result(result)


if __name__ == "__main__":
    main()
