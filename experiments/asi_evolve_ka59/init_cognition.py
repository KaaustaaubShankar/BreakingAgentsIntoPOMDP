"""
Seed the ASI-Evolve cognition store with KA59 domain knowledge.

Run once before starting evolution:
    cd ~/ASI-Evolve && python ~/jkj-breaking-agents/experiments/asi_evolve_ka59/init_cognition.py

Intentionally omits the hidden wall_transfer asymmetry — that is the
mechanic we want ASI-Evolve to discover, not receive as a prior.
"""

import sys
import importlib.util
from pathlib import Path

ASI_EVOLVE_PATH = Path("~/ASI-Evolve").expanduser()
sys.path.insert(0, str(ASI_EVOLVE_PATH))

# Register ASI-Evolve as the 'Evolve' package (matches their bootstrap in main.py)
_spec = importlib.util.spec_from_file_location(
    "Evolve",
    ASI_EVOLVE_PATH / "__init__.py",
    submodule_search_locations=[str(ASI_EVOLVE_PATH)],
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["Evolve"] = _mod
_spec.loader.exec_module(_mod)

from Evolve.cognition.cognition import Cognition
from Evolve.utils.structures import CognitionItem

COGNITION_DIR = Path(__file__).parent / "cognition_data"

cog = Cognition(storage_dir=str(COGNITION_DIR))
cog.reset()

items = [
    CognitionItem(
        content=(
            "KA59 is a multi-level grid puzzle. The agent controls a selected piece "
            "(P in the grid) and must progress through 7 levels. Each level has goal "
            "tile(s) marked +. Winning a level advances to the next; winning level 7 ends "
            "the run. The step budget is 64 turns per level — exceeding it fails that level."
        ),
        source="KA59 Game Overview",
        metadata={"topic": "overview", "importance": "critical"},
    ),
    CognitionItem(
        content=(
            "Five actions: MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, CLICK. "
            "Movement shifts the selected piece by exactly one cell (3 pixels) per action. "
            "CLICK requires target_position [x, y] in pixel coordinates — use it to switch "
            "control to another selectable piece (S in the grid). "
            "Return {'action': 'MOVE_RIGHT'} or {'action': 'CLICK', 'target_position': [x, y]}."
        ),
        source="Action Space",
        metadata={"topic": "actions", "importance": "critical"},
    ),
    CognitionItem(
        content=(
            "Each turn: state['semantic_grid'] is an ASCII grid (P=selected, S=other selectable, "
            "B=block, #=wall, +=goal, .=empty). state['player']['position'] is [x, y] in pixels. "
            "state['objects'] has selectables/blocks/walls/goals, each with position and size. "
            "state['movement']['right_blocked'] etc are heuristic hints only. "
            "state['resources']['steps_remaining'] is the remaining budget. "
            "state['level']['current'] and ['total'] show progress. "
            "Positions: x increases rightward, y increases downward, STEP=3 pixels per cell."
        ),
        source="Observation Structure",
        metadata={"topic": "observation", "importance": "high"},
    ),
    CognitionItem(
        content=(
            "Selectables (P/S): player-controlled pieces, switch with CLICK. "
            "Blocks (B): pushable objects. Moving into an adjacent block pushes it one cell. "
            "Walls (#): obstruct movement — but NOT all walls behave identically. "
            "Some boundaries allow certain interactions that others block. "
            "You must probe wall behavior through interaction — do not assume uniformity. "
            "Goals (+): target tiles. Get a block or selectable onto a goal tile to win the level."
        ),
        source="Object Types and Interaction",
        metadata={"topic": "objects", "importance": "high"},
    ),
    CognitionItem(
        content=(
            "Blocks are pushed by moving the selected piece into them. "
            "Push chains: if two blocks are lined up, pushing the first may push both. "
            "Not all push attempts succeed — if something blocks the block, the selected piece "
            "also does not move. "
            "Use history[i]['moved'] (True/False) to determine whether each action had effect. "
            "Position yourself behind a block, then push it toward the goal."
        ),
        source="Pushing and Spatial Reasoning",
        metadata={"topic": "pushing", "importance": "high"},
    ),
    CognitionItem(
        content=(
            "Static agents (fixed cycles, simple heuristics) achieve 0% win rate on KA59. "
            "They hit the 64-step limit without completing level 1. "
            "Root cause: they repeat ineffective moves without adapting their strategy. "
            "Successful agents must: detect when moves are failing (moved=False repeatedly), "
            "try alternative action sequences to probe the environment, switch pieces when stuck, "
            "and update their strategy based on what the history reveals about the game's rules."
        ),
        source="Why Naive Agents Fail",
        metadata={"topic": "failure_modes", "importance": "high"},
    ),
    CognitionItem(
        content=(
            "Use history to adapt. Key signals: "
            "moved=False for 3+ consecutive steps means you are stuck — try a different direction or CLICK. "
            "moved=True after a specific action sequence means that sequence is productive — repeat it. "
            "Count successful moves per direction; bias toward high-yield directions. "
            "If MOVE_RIGHT fails 5 times, the path is blocked — try MOVE_UP, MOVE_DOWN, or CLICK to switch piece."
        ),
        source="History-Based Adaptation",
        metadata={"topic": "adaptation", "importance": "high"},
    ),
    CognitionItem(
        content=(
            "Some levels have multiple selectable pieces (S in grid). Use CLICK to switch. "
            "Different pieces may be positioned to reach different parts of the grid. "
            "If your current piece is completely stuck, switch to another piece and try from there. "
            "CLICK target_position must exactly match the pixel coordinates from "
            "state['objects']['selectables'][i]['position'] — not cell indices."
        ),
        source="Multi-Piece Strategy",
        metadata={"topic": "multi_piece", "importance": "medium"},
    ),
]

ids = cog.add_batch(items)
print(f"Cognition store initialized: {len(ids)} entries at {COGNITION_DIR}")
