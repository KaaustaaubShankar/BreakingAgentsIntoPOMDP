"""
Seed the ASI-Evolve cognition store with KA59 domain knowledge.

Run once before starting evolution:
    python experiments/asi_evolve_ka59/init_cognition.py

Intentionally omits the hidden wall_transfer asymmetry — that is the
mechanic we want ASI-Evolve to discover, not receive as a prior.
"""

import sys
from pathlib import Path

ASI_EVOLVE_PATH = Path(__file__).resolve().parents[3] / "ASI-Evolve"
sys.path.insert(0, str(ASI_EVOLVE_PATH))

from cognition.store import CognitionStore

store = CognitionStore(
    storage_dir=str(Path(__file__).parent / "cognition_data")
)

store.add([
    {
        "title": "KA59 Game Overview",
        "content": (
            "KA59 is a multi-level grid puzzle. The agent controls a selected piece "
            "(P in the grid) and must progress through 7 levels. Each level has a goal "
            "tile (+). Winning a level advances to the next; winning level 7 ends the run. "
            "The step budget is 64 turns per level — exceeding it fails that level."
        ),
    },
    {
        "title": "Action Space",
        "content": (
            "Five actions: MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, CLICK. "
            "Movement shifts the selected piece by exactly one cell (3 pixels) per action. "
            "CLICK requires a target_position [x, y] in the same pixel coordinate system "
            "as player.position — use it to switch control to another selectable piece (S). "
            "Return a dict: {'action': 'MOVE_RIGHT'} or {'action': 'CLICK', 'target_position': [x, y]}."
        ),
    },
    {
        "title": "Observation Structure",
        "content": (
            "Each turn you receive a state dict with: semantic_grid (ASCII: P=selected, "
            "S=other selectable, B=block, #=wall, +=goal, .=empty), player.position [x,y], "
            "objects.{selectables, blocks, walls, goals} each a list of {id, position, size}, "
            "movement.{left,right,up,down}_blocked (heuristic hints only — not guaranteed), "
            "resources.steps_remaining, level.{current,total}, game_state. "
            "Positions are pixel-based: x increases rightward, y increases downward, STEP=3px/cell."
        ),
    },
    {
        "title": "Object Types and Interaction",
        "content": (
            "Selectables (P/S): player-controlled pieces. Switch with CLICK. "
            "Blocks (B): pushable objects. If a block is adjacent in your movement direction, "
            "moving into it pushes it one cell in that direction. "
            "Walls (#): obstruct movement — but NOT all walls behave the same. Some boundaries "
            "allow certain interactions while blocking others. Probe them by attempting moves "
            "and observing what happens. Do not assume uniform behavior. "
            "Goals (+): target tiles. Get a pushable object (or selectable) to a goal tile to win."
        ),
    },
    {
        "title": "Pushing and Spatial Reasoning",
        "content": (
            "Blocks can be pushed by moving the selected piece into them. "
            "Chains: if block_A is behind block_B in the push direction, pushing A may also push B. "
            "Strategic implications: position yourself behind a block to push it toward the goal. "
            "Not all push attempts succeed — if something stops the block, the push fails silently. "
            "Track whether moved=True in history to learn which actions actually had effect."
        ),
    },
    {
        "title": "Why Naive Agents Fail",
        "content": (
            "Static agents using fixed action cycles or simple heuristics achieve 0% win rate on KA59. "
            "They consistently hit the 64-step limit without completing level 1. "
            "The failure mode: agents repeat ineffective moves without updating their model of the game. "
            "Successful strategies require: (1) noticing when repeated moves produce no effect, "
            "(2) trying different action sequences to probe the environment, "
            "(3) switching between selectable pieces when stuck, "
            "(4) inferring which push directions are productive from history."
        ),
    },
    {
        "title": "History-Based Adaptation",
        "content": (
            "Use the history parameter to adapt. Key signals: "
            "moved=False for N consecutive steps → you are stuck, try a different action or CLICK. "
            "moved=True after a specific sequence → that sequence is productive, repeat or extend it. "
            "Count how often each action direction produces movement; bias toward high-yield directions. "
            "If you tried MOVE_RIGHT 5 times with no movement, try MOVE_UP or CLICK to a different piece."
        ),
    },
    {
        "title": "Multi-Piece Strategy",
        "content": (
            "Some levels have multiple selectable pieces. Use CLICK to switch control. "
            "Different pieces may have access to different parts of the grid. "
            "If piece A is stuck, switch to piece B and try to create a path. "
            "CLICK target_position must match the piece's exact pixel coordinates from state['objects']['selectables']."
        ),
    },
])

print(f"Cognition store initialized with {len(store.get_all())} entries.")
print(f"Path: {Path(__file__).parent / 'cognition_data'}")
