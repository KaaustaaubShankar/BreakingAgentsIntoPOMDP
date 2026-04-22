"""
prompts.py — Axis-separated prompt construction for KA59.

Matches env4/prompts.py API so the same experiment runner can drive both.

Axes (mirroring env4):
  GOAL      — EASY: describe the win condition.  HARD: empty.
  MECHANICS — EASY: describe actions, pushing, walls.  HARD: JSON spec only.
  WORLD     — EASY/HARD handled by observation builder in experiment.py
              (HARD hides non-selected object positions via observe_positions).
  FEEDBACK  — EASY: narrative (game_interface.build_feedback_easy).
              HARD: the constant FEEDBACK_HARD.

Deliberately omitted from MECHANICS_EASY: the specific wall-transfer asymmetry.
That is the discovery signal KA59 measures. We describe pushing and walls
generally; the agent must learn via interaction that some wall boundaries
permit pushed objects to cross while blocking direct movement.
"""

from __future__ import annotations


GOAL_EASY = """\
Your goal is to get ANY pushable object — the selected piece or any pushable block — to reach a target x-coordinate.
The target_x value appears in the structured state under `level.target_x`.
You win the scenario when at least one pushable object's x-position is at or beyond target_x.
This may mean pushing a block toward the target rather than moving the selected piece itself; whichever succeeds first wins the scenario.
The scenario ends in failure if the step budget is exhausted first.
Plan efficient moves; wasted actions drain the budget.\
"""

GOAL_HARD = ""

MECHANICS_EASY = """\
Game mechanics:
- The world is a small pixel grid. Positions are `[x, y]` with x increasing rightward and y increasing downward.
- One STEP = 3 pixels. Every move shifts the selected piece by exactly one STEP in the chosen direction.
- Actions: `MOVE_LEFT`, `MOVE_RIGHT`, `MOVE_UP`, `MOVE_DOWN`, `SELECT`.
- If something pushable is adjacent to the selected piece in the direction of movement, a push may occur: the pushable object shifts one STEP along with the selected piece.
- Boundary objects (walls) may block movement. Not all wall-like boundaries behave the same way — some may behave differently for direct movement versus for objects being pushed against them. You must observe outcomes to learn the rules.
- `SELECT` switches which object is currently controllable. Include `"target_id": "<id>"` when using SELECT.

Observation each turn:
- `player.position` and `player.id` identify the selected piece.
- `movement.{direction}_blocked` is a heuristic hint (True if there is any adjacent object). It does NOT tell you whether the action will succeed — try actions to learn.
- `objects.{walls,blocks,controllables}` lists every object's id, position, size, and selection flag.
- `semantic_grid` is a compact ASCII picture: P=selected, B=block, #=wall, .=empty.
- `resources.steps_remaining` is your remaining action budget.

Response protocol:
Each turn, respond with a single JSON object only:
{"reasoning": "<1-2 sentences>", "action": "MOVE_RIGHT"}

For SELECT, include target_id:
{"reasoning": "I will try controlling the block instead.", "action": "SELECT", "target_id": "pb"}

Valid actions:
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `MOVE_UP`
- `MOVE_DOWN`
- `SELECT` with `"target_id": "<id>"`\
"""

MECHANICS_HARD = """\
Respond each turn with a JSON object only:
{"action": "<your chosen action>", "target_id": "<id if SELECT>"}\
"""

FEEDBACK_HARD = "Ok."

UNDERSTANDING_PROMPT = """\
The scenario has ended. Please reflect on what you experienced and respond with a JSON object only:
{
  "goal_understanding": "<what the goal of the scenario was>",
  "mechanics_understanding": "<how movement, pushing, and walls worked in this world>"
}\
"""


def build_system_prompt(goal_level: str, mechanics_level: str) -> str:
    parts: list[str] = []

    goal_text = GOAL_EASY if goal_level == "EASY" else GOAL_HARD
    if goal_text:
        parts.append(goal_text)

    parts.append(MECHANICS_EASY if mechanics_level == "EASY" else MECHANICS_HARD)
    return "\n\n".join(parts)
