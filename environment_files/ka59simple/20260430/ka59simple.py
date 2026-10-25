"""ka59simple - KA59 level 1 reduced to one wall-traversal puzzle.

Forks canonical KA59 (environment_files/ka59/38d34dbb/ka59.py) by
runtime-importing its source, then defining ``Ka59Simple`` with a single
level that strips out the second sub-puzzle while preserving the
hidden-mechanic-discovery test the paper measures.

Engine mechanic (verified empirically 2026-05-01):
  The wall (tag 0015qniapgwsvb) is impassable to a sprite *moving as the
  player* via ``loydmqkgjw`` (which checks walls). But when one
  selectable PUSHES another into the wall via the recursive push function
  ``ifoelczjjh``, the pushed sprite passes THROUGH the wall (because
  ``ifoelczjjh`` only checks letterbox + selectables, never walls). This
  is the wall-transfer mechanic: push to teleport across.

Layout (45x45 cell grid, positions in pixel coords with 3px/cell):
  Pusher selectable @ (9, 21)   - canonical's left sel, initial player
  Pushee selectable @ (18, 21)  - canonical's right sel, gets pushed
  Purple wall       @ (24, 12)  - blocks player movement, transparent to push
  Goal box          @ (35, 17)  - canonical's right goal (only goal kept)
  Letterbox         @ (-3, -3)  - platform shape

Optimal win path (6 actions):
  RIGHT  (player 9->12)
  RIGHT  (player 12->15)
  RIGHT  (player blocks at 15; pushee teleports 18->33 via wall transfer)
  CLICK  on pushee at (33, 21)  - switches player to the transferred sel
  RIGHT  (player 33->36)
  UP     (player 36,21 -> 36,18) - fills goal at (35, 17), WIN

What this preserves vs. removes:
  preserves -> the hidden mechanic test (push-into-wall triggers
               transfer), the click-to-switch-selectable transition
               (the failure mode Ben identified in 2026-04-27 meeting),
               and all 4 ablation axes (world/goal/mechanics/feedback).
  removes   -> the second sub-puzzle (filling the LEFT goal at (2, 23))
               which doubles the action count without adding new
               mechanics to discover.

Compare to canonical level 1 (verified solvable in ~12 actions): same
hidden mechanic, half the path length. If a model wins ka59simple but
fails canonical, we isolate the deficit to "compounding multi-stage
plans" rather than "discovering the push-transfer-click sequence."

The canonical source must be locatable. Resolution order:
  1. ENV var KA59_BASE_DIR (set by ka59_game.experiment._make_env)
  2. CWD-relative environment_files/ka59/38d34dbb/ka59.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from arcengine import ARCBaseGame, Camera, Level


def _import_canonical_ka59():
    base = os.environ.get("KA59_BASE_DIR")
    candidates = []
    if base:
        candidates.append(Path(base) / "ka59.py")
    candidates.append(Path("environment_files/ka59/38d34dbb/ka59.py").resolve())

    src_path = next((p for p in candidates if p.is_file()), None)
    if src_path is None:
        raise FileNotFoundError(
            "ka59simple requires the canonical ka59 source. "
            f"Tried: {[str(c) for c in candidates]}. "
            "Set KA59_BASE_DIR to override."
        )

    spec = importlib.util.spec_from_file_location("_ka59_canonical_runtime", str(src_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {src_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ka59 = _import_canonical_ka59()
_sprites = _ka59.sprites
_Ka59 = _ka59.Ka59
_StepCounterDisplay = _ka59.ckawyvsxuv  # canonical's obfuscated name
_BG = _ka59.BACKGROUND_COLOR
_PAD = _ka59.PADDING_COLOR


# Canonical level 1 layout (ka59.py:40936-40947):
#   "0009ouocpihipp"@(2, 23)   left goal  (filled via second sub-puzzle)
#   "0009ouocpihipp"@(35, 17)  right goal (filled via wall-transfer puzzle)
#   "0014ysspdlqsqg"@(24, 12)  purple wall
#   "0021xplppqqmfb"@(9, 21)   left selectable (the pusher)
#   "0021xplppqqmfb"@(18, 21)  right selectable (the pushee)
#   "0028lydaygyjbu"@(-3, -3)  letterbox
#
# Drop: left goal (eliminates the second sub-puzzle so total action count
# halves). Keep: BOTH selectables (need two for the push mechanic to fire
# wall-transfer), right goal (the wall-transfer destination), wall,
# letterbox.
def _build_simple_level() -> Level:
    return Level(
        sprites=[
            _sprites["0009ouocpihipp"].clone().set_position(35, 17),
            _sprites["0014ysspdlqsqg"].clone().set_position(24, 12),
            _sprites["0021xplppqqmfb"].clone().set_position(9, 21),
            _sprites["0021xplppqqmfb"].clone().set_position(18, 21),
            _sprites["0028lydaygyjbu"].clone().set_position(-3, -3),
        ],
        grid_size=(45, 45),
        data={"StepCounter": 100},
    )


class Ka59Simple(_Ka59):
    """KA59 with a single 1-goal level. Bypasses ``_Ka59.__init__`` to swap
    the level list; all step()/movement/win-check methods on ``_Ka59``
    work unchanged because they read ``self.current_level``."""

    def __init__(self) -> None:
        self.urgssjskot = _StepCounterDisplay(0)
        camera = Camera(background=_BG, letter_box=_PAD, interfaces=[self.urgssjskot])
        ARCBaseGame.__init__(
            self,
            game_id="ka59simple",
            levels=[_build_simple_level()],
            camera=camera,
            available_actions=[1, 2, 3, 4, 6],
        )
