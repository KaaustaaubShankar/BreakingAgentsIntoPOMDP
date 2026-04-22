"""
prompts.py — System prompt construction for BP35.
"""

from __future__ import annotations

GOAL_EASY = """\
Your goal is to finish the current BP35 level and then continue until the game is won.
In each level, you need to reach the goal gem tile marked `+`.
Avoid spike tiles: landing on a spike ends the run.
There is a step budget each level. If you spend too many actions, the level ends in failure.
Complete levels efficiently.\
"""

GOAL_HARD = ""

MECHANICS_EASY = """\
Game mechanics:
- The player acts on a tile grid.
- Primary movement is horizontal only: `MOVE_LEFT` and `MOVE_RIGHT`.
- After a horizontal move, the character automatically continues moving in the current gravity direction until landing on a solid tile, hitting a hazard, or reaching the goal.
- `CLICK` interacts with clickable tiles. For a click action, include `"target_position": [x, y]`.
- `UNDO` reverts the previous action.

Important tile types:
- `+` = goal gem. Reaching it wins the level.
- `u` / `v` = spikes. Landing on them loses immediately.
- `x` = breakable block. Clicking it removes it.
- `o`, `m`, `w` = safe support tiles you can land on.

Planning rules:
1. Think in terms of where a left/right move will make you land after gravity finishes acting.
2. Before clicking, check whether the click changes the path the player will take.
3. Preserve `UNDO` for recovery when you realize a click or move created a bad position.
4. Watch the step budget closely. Fast solutions matter.

Coordinate system:
- Positions are `[x, y]`.
- Moving left decreases `x`. Moving right increases `x`.
- Use the structured state's explicit positions rather than guessing from prose.

Each turn, respond with a single JSON object only:
{"reasoning": "<1-2 sentences>", "action": "MOVE_LEFT"}

Valid actions:
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `CLICK` with `"target_position": [x, y]`
- `UNDO`\
"""

MECHANICS_HARD = """\
Respond each turn with a JSON object only:
{"action": "<your chosen action>", "target_position": [x, y]}\
"""

FEEDBACK_HARD = "Ok."

UNDERSTANDING_PROMPT = """\
The episode has ended. Please reflect on what you experienced and respond with a JSON object only:
{
  "goal_understanding": "<what the goal of the game was>",
  "mechanics_understanding": "<how movement, gravity, clicks, hazards, and undo worked>"
}\
"""


def build_system_prompt(goal_level: str, mechanics_level: str) -> str:
    parts: list[str] = []

    goal_text = GOAL_EASY if goal_level == "EASY" else GOAL_HARD
    if goal_text:
        parts.append(goal_text)

    parts.append(MECHANICS_EASY if mechanics_level == "EASY" else MECHANICS_HARD)
    return "\n\n".join(parts)
