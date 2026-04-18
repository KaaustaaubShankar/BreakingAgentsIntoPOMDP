"""
prompts.py — System prompt construction for each axis level combination.

Each axis has two levels matching the research proposal:
  EASY  — full information provided
  HARD  — minimal or no information; agent must infer

Axis: World   → how game state is presented (structured JSON vs raw ASCII)
Axis: Goal    → whether the objective is stated
Axis: Mechanics → whether rules are explained
Axis: Feedback → detail level of post-action feedback
"""

from __future__ import annotations

# ── Goal axis ─────────────────────────────────────────────────────────────────

GOAL_EASY = """\
Your goal is to navigate the grid and complete all levels.
Each level contains one or more goal cells. To activate a goal cell, your carried piece's \
shape, color, and rotation must all match that goal's requirements simultaneously.
Modifier tiles scattered across the grid change your piece's attributes:
  - Shape modifier: cycles the piece's shape
  - Color modifier: cycles the piece's color
  - Rotation modifier: rotates the piece 90° clockwise
Once your piece matches a goal's requirements, step onto the goal cell to complete it.
Complete all goals in a level to advance. Win all levels to complete the game.
Manage your stamina — losing all stamina costs a life, and losing all lives ends the game.\
"""

GOAL_HARD = ""  # Nothing stated — agent must infer that a goal exists and what it is


# ── Mechanics axis ────────────────────────────────────────────────────────────

MECHANICS_EASY = """\
Game mechanics:
- Each move costs 1 stamina.
- Stamina reaching 0 costs 1 life and resets your stamina to its maximum value.
- Losing all lives results in GAME_OVER.
- Walls block movement. The state always includes "blocked_directions" — a list of directions \
you CANNOT move from your current position. NEVER attempt a blocked direction; it wastes a \
turn and stamina with no benefit.
- Stepping on a shape modifier tile cycles your piece to the next shape (shapes 0–5, wraps \
from 5 back to 0).
- Stepping on a color modifier tile cycles your piece to the next color. Color order: \
orange → blue → green → red → orange (wraps).
- Stepping on a rotation modifier tile rotates your piece 90° clockwise. Rotation order: \
0° → 90° → 180° → 270° → 0° (wraps).
- A goal cell activates only when your piece's shape, color, AND rotation all match its \
required values.
- Stepping onto an activated goal completes it.
- Completing all goals in a level advances you to the next level.

Navigation rules — follow these strictly:
1. Check "blocked_directions" every turn before choosing a move. Never pick a blocked direction.
2. If you are stuck (a direction you need is blocked), try a perpendicular direction to find a \
path around the wall rather than retrying the same blocked direction.
3. Plan your route before moving: figure out the sequence of moves needed, then execute it \
step by step.

Tile appearances in the grid (ASCII view):
  Wall             — solid # characters filling the tile. Impassable.
  Floor            — . characters. Walkable.
  Shape changer    — white dots in a small diagonal cluster (#):
                     . # . .
                     . . # #
                     . . # .
  Color changer    — 2×2 block of all four game colors (B=blue, G=green, R=red, O=orange):
                     . B G G .
                     . B # R .
                     . O O R .
  Rotation changer — white/light arrow pattern (# = white, = = light):
                     . . # . .
                     . = # # .
                     . . = . .
  Stamina refill   — hollow yellow ring (Y):
                     Y Y Y
                     Y   Y
                     Y Y Y

Example: if your piece is currently color=blue, rotation=270°, and you need color=blue, \
rotation=0° to match the goal — step on one rotation modifier tile to reach 0°, then \
navigate to the goal. If you need color=green instead, step on one color modifier tile \
(blue → green), then proceed.

Coordinate system: position is [x, y]. MOVE_NORTH decreases y. MOVE_SOUTH increases y. MOVE_EAST increases x. MOVE_WEST decreases x. Use this to plan your route — e.g. to go from [34, 15] to [19, 30] you need to move west (x: 34→19) and south (y: 15→30).

You have a limited turn budget. Plan your route before moving — wasted moves cost you the game.

Each turn, respond with a single JSON object containing your reasoning and action:
{"reasoning": "<1-2 sentences explaining your plan>", "action": "MOVE_NORTH"}
Valid actions: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST\
"""

MECHANICS_HARD = """\
Respond each turn with a JSON object only:
{"action": "<your chosen action>"}\
"""


# ── Feedback (injected at runtime, not in system prompt) ─────────────────────

FEEDBACK_HARD = "Ok."


# ── Post-game understanding prompt ────────────────────────────────────────────

UNDERSTANDING_PROMPT = """\
The episode has ended. Please reflect on what you experienced and respond with a JSON \
object only:
{
  "goal_understanding": "<your explanation of what the goal of the game was>",
  "mechanics_understanding": "<your explanation of how the game mechanics work>"
}\
"""


# ── Builder ───────────────────────────────────────────────────────────────────

def build_system_prompt(goal_level: str, mechanics_level: str) -> str:
    """
    Construct the system prompt from goal and mechanics axis levels.

    Args:
        goal_level:      "EASY" or "HARD"
        mechanics_level: "EASY" or "HARD"

    Returns:
        A single system prompt string.
    """
    parts: list[str] = []

    goal_text = GOAL_EASY if goal_level == "EASY" else GOAL_HARD
    if goal_text:
        parts.append(goal_text)

    mechanics_text = MECHANICS_EASY if mechanics_level == "EASY" else MECHANICS_HARD
    parts.append(mechanics_text)

    return "\n\n".join(parts)
