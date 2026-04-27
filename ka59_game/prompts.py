"""
prompts.py — Axis-separated prompts for the real KA59 game.

Describes the actual game: multi-level progression, goal tiles, MOVE + CLICK
action set. Deliberately does NOT describe the wall_transfer asymmetry — that
is the discovery signal KA59 measures.

Same public API as env4/prompts.py and ka59_ref/prompts.py:
  GOAL_EASY / GOAL_HARD
  MECHANICS_EASY / MECHANICS_HARD
  FEEDBACK_HARD
  UNDERSTANDING_PROMPT
  build_system_prompt(goal_level, mechanics_level)
"""

from __future__ import annotations


GOAL_EASY = """\
Your goal is to complete the KA59 puzzle game by progressing through all of its levels.
Each level presents a grid with obstacles, selectable pieces, and goal tile(s) marked `+` in the semantic grid.
You win a level by getting a pushable object onto the goal tile(s), or by satisfying whatever success condition the engine reports as WIN.
Once the run reports WIN, the whole game is won. Use actions efficiently; the step budget per level is finite.\
"""

GOAL_HARD = ""

MECHANICS_EASY = """\
Game mechanics:
- Positions are `[x, y]` pixel coordinates. x increases rightward, y increases downward.
- The grid uses 3 pixels per cell. Every movement action shifts the currently selected piece by exactly one cell (3 pixels) in the chosen direction.
- Some objects are pushable: if a pushable object is immediately adjacent in the movement direction, a push may occur (the pushable shifts along with the selected piece).
- Some objects are walls. Walls may block movement in various ways. Not all wall-like boundaries behave identically — you must interact with them to discover their behavior.
- A level can contain more than one selectable piece. Use `CLICK` with a `target_position` to switch control to another selectable piece at that position. `target_position` must be given in the SAME coordinates as `player.position` and every entry in `objects` — i.e. the pixel/grid coordinates shown in the state dict, NOT cell indices from the semantic_grid.

Observation each turn:
- `player.position` and `player.id` identify the currently selected piece.
- `movement.{direction}_blocked` is a heuristic hint (True if there is any adjacent object in that direction). It does NOT guarantee an action will fail — try it to learn.
- `objects.{walls,blocks,selectables,goals}` lists every on-screen object's position and size.
- `semantic_grid` is a compact ASCII picture of the current cell-grid: P=selected, S=other selectable, B=block, #=wall, +=goal, .=empty.
- `level.current` / `level.total` show your progress. `resources.steps_remaining` is the per-level budget.

Response protocol:
Each turn, respond with a single JSON object only:
{"reasoning": "<1-2 sentences>", "action": "MOVE_RIGHT"}

For CLICK, include target_position in the SAME coordinates as `player.position`:
{"reasoning": "Switch to the other selectable at [18, 21].", "action": "CLICK", "target_position": [18, 21]}

Valid actions:
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `MOVE_UP`
- `MOVE_DOWN`
- `CLICK` with `"target_position": [x, y]` in grid/pixel coordinates (same system as `player.position` and `objects.*[*].position`)\
"""

MECHANICS_HARD = """\
Respond each turn with a JSON object only:
{"action": "<chosen action>", "target_position": [x, y] if CLICK}\
"""

MECHANICS_OODA_F = """\
Each turn, respond with a JSON object only using this exact structure:
{"observe": "<what changed since last turn>", "orient": "<your current hypothesis about how this world works>", "decide": "<what you will do and why>", "action": "<ACTION_NAME>", "target_position": [x, y] if CLICK}

For your first turn, set observe to your initial reading of the state.
Update your orient hypothesis every turn based on outcomes — if something unexpected happened, revise it.
Valid actions: MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, CLICK

IMPORTANT: When you see a [FORCED REFRAME] notice in the observation, your recent strategy has not made progress. You MUST: (1) discard your current hypothesis entirely, (2) state a new hypothesis based solely on what you have actually observed, (3) choose a completely different action type than what you have been using."""

MECHANICS_OODA = """\
Each turn, respond with a JSON object only using this exact structure:
{"observe": "<what changed since last turn>", "orient": "<your current hypothesis about how this world works>", "decide": "<what you will do and why>", "action": "<ACTION_NAME>", "target_position": [x, y] if CLICK}

For your first turn, set observe to your initial reading of the state.
Update your orient hypothesis every turn based on outcomes — if something unexpected happened, revise it.
Valid actions: MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, CLICK\
"""

FEEDBACK_HARD = "Ok."

UNDERSTANDING_PROMPT = """\
The run has ended. Please reflect on what you experienced and respond with a JSON object only:
{
  "goal_understanding": "<what the goal of the game was>",
  "mechanics_understanding": "<how movement, pushing, walls, and selection worked>"
}\
"""


DISCOVERY_KEYWORDS = [
    "transfer", "pass through", "asymmetr", "different for push",
    "through the wall", "through wall", "push through",
    "wall allow", "wall let", "boundary allow",
    "block crosses", "block pass", "pushed block pass",
]


def check_discovery(orient_text: str) -> bool:
    low = orient_text.lower()
    return any(kw in low for kw in DISCOVERY_KEYWORDS)


def build_system_prompt(goal_level: str, mechanics_level: str) -> str:
    parts: list[str] = []

    goal_text = GOAL_EASY if goal_level == "EASY" else GOAL_HARD
    if goal_text:
        parts.append(goal_text)

    if mechanics_level == "OODA_F":
        parts.append(MECHANICS_OODA_F)
    elif mechanics_level == "OODA":
        parts.append(MECHANICS_OODA)
    elif mechanics_level == "EASY":
        parts.append(MECHANICS_EASY)
    else:
        parts.append(MECHANICS_HARD)
    return "\n\n".join(parts)
