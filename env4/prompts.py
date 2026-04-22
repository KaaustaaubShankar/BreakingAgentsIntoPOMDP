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

Fall-destination preview:
- The structured state includes `movement.left_landing` and `movement.right_landing`: where you will end up (after gravity fully acts) if you move left or right this turn.
- `corridor_scan` shows the landing position for **every column** reachable horizontally without clicking. Any entry with `"passage": true` means landing there would drop you to a different y-level — a new section of the level. **Always move toward a passage column first.**
- If no passage exists in the corridor scan, use CLICK to remove a breakable block that blocks the gravity direction, then re-evaluate.

Rising platform (early levels only):
- The structured state includes `moving_platform` when relevant. It shows how many horizontal moves remain before the rising platform kills you.
- **CLICK actions do NOT advance the platform timer.** Only MOVE_LEFT and MOVE_RIGHT count.
- When `moves_until_kill` is low, stop horizontal bouncing immediately and commit to a forward path.

Planning rules:
1. Check `left_landing` and `right_landing` each turn. If one of them is at a very different `y`, that is a passage — use it.
2. For every MOVE_LEFT or MOVE_RIGHT, confirm `left_landing`/`right_landing` reaches a useful position and has no spikes.
3. Spend clicks to open fall paths (remove breakable blocks that block the gravity direction), not randomly.
4. Preserve `UNDO` for recovery when you realize a click or move created a bad position.
5. Track `moving_platform.moves_until_kill` and prioritize upward progress over exploration when time is short.

Coordinate system:
- Positions are `[x, y]`.
- Moving left decreases `x`. Moving right increases `x`.
- `y` increases downward: `y=0` is the top of the grid, higher `y` is lower on screen.
- Gravity DOWN means the player falls toward **lower** `y` values (toward `y=0`, the top) after each move.
- Gravity UP means the player falls toward **higher** `y` values (toward the bottom) after each move.
- A tile above the player (visually higher) has a **smaller** `y`; a tile below has a **larger** `y`.
- The goal tile is always in the direction of gravity — you fall toward it, not away from it.
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
