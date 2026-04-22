"""
game_interface.py — Structured state + feedback for the real KA59 game.

Extracts agent-facing state from an arc_agi LocalEnvironmentWrapper around
the actual ARC Prize KA59 game (obfuscated source at
environment_files/ka59/<version>/ka59.py).

Dict shape deliberately mirrors env4/game_interface.py and ka59_ref/
game_interface.py so the same experiment / ablation harness works across
all three environments.

Semantic tag map (verified against environment_files/ka59/38d34dbb/ka59.py):
  0022vrxelxosfy  -> selectable (the player-controllable pieces)
  0015qniapgwsvb  -> wall_transfer (blocks direct move; pushed objects pass)
  0029ifoxxfvvvs  -> wall_solid   (blocks everything, incl. pushes)
  0001uqqokjrptk  -> cross        (pushable cross-shaped block)
  0003umnkyodpjp  -> block        (pushable rectangular block)
  0010xzmuziohuf  -> goal         (target tile — observed on level 0)
  sys_click       -> (engine-level) the sprite is clickable
"""

from __future__ import annotations

from typing import Any

from arcengine import FrameDataRaw, GameAction


STEP = 3  # one movement action shifts the selected piece by this many pixels


TAG_SELECTABLE = "0022vrxelxosfy"
TAG_WALL_TRANSFER = "0015qniapgwsvb"
TAG_WALL_SOLID = "0029ifoxxfvvvs"
TAG_CROSS = "0001uqqokjrptk"
TAG_BLOCK = "0003umnkyodpjp"
TAG_GOAL = "0010xzmuziohuf"

PUSHABLE_TAGS = {TAG_CROSS, TAG_BLOCK, TAG_SELECTABLE}
WALL_TAGS = {TAG_WALL_TRANSFER, TAG_WALL_SOLID}

TAG_TO_LABEL: dict[str, str] = {
    TAG_SELECTABLE: "selectable",
    TAG_WALL_TRANSFER: "wall",  # collapsed — the agent can't distinguish visually
    TAG_WALL_SOLID: "wall",
    TAG_CROSS: "block",
    TAG_BLOCK: "block",
    TAG_GOAL: "goal",
}


def _primary_label(tags) -> str:
    for t in tags:
        if t in TAG_TO_LABEL:
            return TAG_TO_LABEL[t]
    return "other"


def _char_for(label: str, is_selected: bool) -> str:
    if is_selected:
        return "P"
    return {
        "selectable": "S",
        "block": "B",
        "wall": "#",
        "goal": "+",
    }.get(label, "?")


ACTION_MAP: dict[str, GameAction] = {
    "MOVE_UP": GameAction.ACTION1,
    "MOVE_DOWN": GameAction.ACTION2,
    "MOVE_LEFT": GameAction.ACTION3,
    "MOVE_RIGHT": GameAction.ACTION4,
    "CLICK": GameAction.ACTION6,
}

DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "left": (-STEP, 0),
    "right": (+STEP, 0),
    "up": (0, -STEP),
    "down": (0, +STEP),
}


def _selected_sprite(game) -> Any:
    """The game tracks its current selectable on `game.prkgpeyexo`."""
    return game.prkgpeyexo


def _sprite_to_dict(sprite, is_selected: bool) -> dict[str, Any]:
    label = _primary_label(sprite.tags)
    return {
        "id": f"{label}_{sprite.x}_{sprite.y}",  # stable per-position id
        "name": sprite.name,
        "position": [int(sprite.x), int(sprite.y)],
        "size": [int(sprite.width), int(sprite.height)],
        "label": label,
        "is_selected": bool(is_selected),
        "collidable": bool(sprite.is_collidable),
    }


def _aabb_intersects(ax: int, ay: int, aw: int, ah: int,
                     bx: int, by: int, bw: int, bh: int) -> bool:
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _adjacent_labels_in_direction(selected, sprites, direction: str) -> list[str]:
    """Return labels of sprites touching the rectangle one STEP away."""
    dx, dy = DIRECTION_DELTAS[direction]
    nx = int(selected.x) + dx
    ny = int(selected.y) + dy
    nw = int(selected.width)
    nh = int(selected.height)
    hits: list[str] = []
    for s in sprites:
        if s is selected:
            continue
        if _aabb_intersects(nx, ny, nw, nh,
                            int(s.x), int(s.y), int(s.width), int(s.height)):
            hits.append(_primary_label(s.tags))
    return hits


def _semantic_grid(
    sprites,
    selected,
    grid_cells_w: int,
    grid_cells_h: int,
) -> list[str]:
    """Render a small cell-grid legend with P=selected, B=block, #=wall, +=goal."""
    rows: list[str] = []
    for cy in range(grid_cells_h):
        row_chars: list[str] = []
        py = cy * STEP
        for cx in range(grid_cells_w):
            px = cx * STEP
            ch = "."
            best_priority = -1
            # Priority so selected wins over goals wins over blocks wins over walls.
            # Also non-collidable goals should not be clobbered by overlapping walls.
            for s in sprites:
                if not _aabb_intersects(px, py, STEP, STEP,
                                        int(s.x), int(s.y),
                                        int(s.width), int(s.height)):
                    continue
                label = _primary_label(s.tags)
                is_sel = s is selected
                priority = (
                    5 if is_sel
                    else 4 if label == "goal"
                    else 3 if label == "block"
                    else 2 if label == "selectable"
                    else 1 if label == "wall"
                    else 0
                )
                if priority > best_priority:
                    ch = _char_for(label, is_sel)
                    best_priority = priority
            row_chars.append(ch)
        rows.append("".join(row_chars))
    return rows


def _partition(sprites, selected) -> dict[str, list[dict[str, Any]]]:
    walls: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    selectables: list[dict[str, Any]] = []
    goals: list[dict[str, Any]] = []
    for s in sprites:
        label = _primary_label(s.tags)
        entry = _sprite_to_dict(s, is_selected=(s is selected))
        if label == "wall":
            walls.append(entry)
        elif label == "block":
            blocks.append(entry)
        elif label == "selectable":
            selectables.append(entry)
        elif label == "goal":
            goals.append(entry)
    return {"walls": walls, "blocks": blocks, "selectables": selectables, "goals": goals}


def _game_state_label(frame_data: FrameDataRaw) -> str:
    name = frame_data.state.name
    # arcengine uses NOT_FINISHED / WIN / GAME_OVER; normalise to env4's vocabulary.
    if name == "NOT_FINISHED":
        return "IN_PROGRESS"
    return name


def get_structured_state(
    env,
    frame_data: FrameDataRaw,
    step_count: int,
    step_budget: int,
) -> dict[str, Any]:
    game = env._game
    level = game.current_level
    sprites = list(level.get_sprites())
    selected = _selected_sprite(game)

    grid_w, grid_h = level.grid_size  # pixel extent
    cells_w = max(1, int(grid_w) // STEP)
    cells_h = max(1, int(grid_h) // STEP)

    partitioned = _partition(sprites, selected)

    movement: dict[str, Any] = {}
    for direction in ("left", "right", "up", "down"):
        labels = _adjacent_labels_in_direction(selected, sprites, direction)
        dx, dy = DIRECTION_DELTAS[direction]
        movement[f"{direction}_target"] = [int(selected.x) + dx, int(selected.y) + dy]
        movement[f"{direction}_blocked"] = len(labels) > 0
        movement[f"{direction}_adjacent_labels"] = labels

    return {
        "player": {
            "id": f"selectable_{int(selected.x)}_{int(selected.y)}",
            "position": [int(selected.x), int(selected.y)],
            "size": [int(selected.width), int(selected.height)],
            "name": selected.name,
        },
        "movement": movement,
        "level": {
            "current": int(game.level_index) + 1,
            "total": int(frame_data.win_levels),
            "grid_size": [int(grid_w), int(grid_h)],
        },
        "resources": {
            "step_budget": int(step_budget),
            "steps_used": int(step_count),
            "steps_remaining": max(0, int(step_budget) - int(step_count)),
        },
        "objects": partitioned,
        "semantic_grid": _semantic_grid(sprites, selected, cells_w, cells_h),
        "game_state": _game_state_label(frame_data),
    }


def build_feedback_easy(
    prev: dict[str, Any],
    curr: dict[str, Any],
    action: str,
) -> str:
    parts: list[str] = [f"Action taken: {action}."]

    prev_pos = prev["player"]["position"]
    curr_pos = curr["player"]["position"]
    if prev_pos != curr_pos:
        parts.append(f"Selected piece moved from {prev_pos} to {curr_pos}.")
    else:
        parts.append("Selected piece did not change position (blocked or action has no movement effect).")

    def _by_id(objs: list[dict[str, Any]]) -> dict[str, list[int]]:
        return {o["id"]: o["position"] for o in objs}

    # Blocks that moved (push signal)
    prev_blocks = _by_id(prev["objects"]["blocks"])
    curr_blocks = _by_id(curr["objects"]["blocks"])
    moved_blocks: list[str] = []
    for bid, new_pos in curr_blocks.items():
        # IDs are position-based so a moved block will have a NEW id not in prev.
        # Detect pushes by looking for blocks that disappeared from prev AND
        # whose label/count remained the same (heuristic).
        if bid not in prev_blocks:
            moved_blocks.append(bid)
    if moved_blocks:
        parts.append(f"A block moved on the board (new block id(s): {moved_blocks}).")

    # Selection change
    if prev["player"]["position"] != curr["player"]["position"]:
        pass  # already reported
    elif prev["player"]["id"] != curr["player"]["id"]:
        parts.append(
            f"Active selection changed from {prev['player']['id']} to {curr['player']['id']}."
        )

    # Level progression
    prev_level = prev["level"]["current"]
    curr_level = curr["level"]["current"]
    if curr_level > prev_level:
        parts.append(f"Advanced to level {curr_level}/{curr['level']['total']}.")

    parts.append(f"Steps remaining: {curr['resources']['steps_remaining']}.")

    gs = curr["game_state"]
    if gs == "WIN":
        parts.append("You reached the overall win state.")
    elif gs == "GAME_OVER":
        parts.append("The run ended in failure.")

    return " ".join(parts)
