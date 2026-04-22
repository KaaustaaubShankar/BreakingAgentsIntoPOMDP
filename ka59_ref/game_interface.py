"""
game_interface.py — Structured state extraction + feedback builder for KA59.

Mirrors env4/game_interface.py shape so KA59 and BP35 runs produce comparable
state dicts and feedback strings, letting a single ablation table span both
environments.
"""

from __future__ import annotations

from typing import Any

from .engine import PUSHABLE_TAGS, STEP
from .env import KA59BlindEnv, ObjectView


_WALL_KINDS = {"wall"}
_BLOCK_KINDS = {"block"}
_CONTROLLABLE_KINDS = {"controllable"}


def _max_pushable_x_true(env: KA59BlindEnv) -> int:
    """True max x-coordinate across all pushable objects, ignoring blinding.

    Used for game_state determination so that WORLD_HARD (which hides
    non-selected positions in the observation) does not also hide the
    win condition from the harness.
    """
    if env._state is None:
        return 0
    best = -1
    for obj in env._state.objects:
        if any(tag in PUSHABLE_TAGS for tag in obj.tags):
            if obj.x > best:
                best = obj.x
    return best


def _view_to_dict(v: ObjectView) -> dict[str, Any]:
    return {
        "id": v.id,
        "position": [v.x, v.y],
        "size": [v.w, v.h],
        "kind": v.kind,
        "is_selected": bool(v.is_selected),
    }


def _partition_objects(views: list[ObjectView]) -> dict[str, list[dict[str, Any]]]:
    walls: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    controllables: list[dict[str, Any]] = []
    for v in views:
        entry = _view_to_dict(v)
        if v.kind in _WALL_KINDS:
            walls.append(entry)
        elif v.kind in _BLOCK_KINDS:
            blocks.append(entry)
        elif v.kind in _CONTROLLABLE_KINDS:
            controllables.append(entry)
    return {"walls": walls, "blocks": blocks, "controllables": controllables}


def _selected_view(views: list[ObjectView]) -> ObjectView:
    for v in views:
        if v.is_selected:
            return v
    raise RuntimeError("No selected object in observation")


def _adjacent_ids(
    selected: ObjectView,
    others: list[ObjectView],
    dx: int,
    dy: int,
) -> list[str]:
    """Return ids of objects whose bounding box touches the selected piece's
    neighbour rectangle one STEP away in (dx, dy). Used as a blocked-heuristic.
    """
    nx = selected.x + dx * STEP
    ny = selected.y + dy * STEP
    nw, nh = selected.w, selected.h
    hits: list[str] = []
    for o in others:
        if o.id == selected.id:
            continue
        # AABB intersection in pixel space (same convention as engine.Obj).
        if nx < o.x + o.w and nx + nw > o.x and ny < o.y + o.h and ny + nh > o.y:
            hits.append(o.id)
    return hits


def _compute_movement(views: list[ObjectView]) -> dict[str, Any]:
    sel = _selected_view(views)
    others = [v for v in views if v.id != sel.id]

    right_ids = _adjacent_ids(sel, others, +1, 0)
    left_ids = _adjacent_ids(sel, others, -1, 0)
    up_ids = _adjacent_ids(sel, others, 0, -1)
    down_ids = _adjacent_ids(sel, others, 0, +1)

    def _blocked(ids: list[str]) -> bool:
        # Any adjacent object is a blocked-heuristic — actual push-through is
        # discoverable only via interaction, which is the point of the probe.
        return len(ids) > 0

    return {
        "left_target": [sel.x - STEP, sel.y],
        "right_target": [sel.x + STEP, sel.y],
        "up_target": [sel.x, sel.y - STEP],
        "down_target": [sel.x, sel.y + STEP],
        "left_blocked": _blocked(left_ids),
        "right_blocked": _blocked(right_ids),
        "up_blocked": _blocked(up_ids),
        "down_blocked": _blocked(down_ids),
        "adjacent_left_ids": left_ids,
        "adjacent_right_ids": right_ids,
        "adjacent_up_ids": up_ids,
        "adjacent_down_ids": down_ids,
    }


def _semantic_grid(views: list[ObjectView]) -> list[str]:
    if not views:
        return []

    positioned = [v for v in views if (v.x, v.y, v.w, v.h) != (0, 0, 0, 0) or v.is_selected]
    if not positioned:
        return []

    max_x = max(v.x + v.w for v in positioned)
    max_y = max(v.y + v.h for v in positioned)
    cells_w = max(1, max_x // STEP)
    cells_h = max(1, max_y // STEP)

    def _char_for(v: ObjectView) -> str:
        if v.is_selected:
            return "P"
        if v.kind == "block":
            return "B"
        if v.kind == "wall":
            return "#"
        return "?"

    rows: list[str] = []
    for cy in range(cells_h):
        row_chars: list[str] = []
        py = cy * STEP
        for cx in range(cells_w):
            px = cx * STEP
            ch = "."
            for v in positioned:
                if v.x <= px < v.x + v.w and v.y <= py < v.y + v.h:
                    ch = _char_for(v)
                    break
            row_chars.append(ch)
        rows.append("".join(row_chars))
    return rows


def get_structured_state(
    env: KA59BlindEnv,
    scenario_name: str,
    step_budget: int,
    target_x: int,
) -> dict[str, Any]:
    """
    Extract a JSON-friendly, env4-compatible state dict from a KA59BlindEnv.

    The dict shape mirrors env4.game_interface.get_structured_state so the
    same experiment / ablation machinery can drive either environment.
    """
    views = env._observe()  # same-package internal access; returns list[ObjectView]
    partitioned = _partition_objects(views)

    selected = _selected_view(views)
    steps_remaining = env._state.steps if env._state is not None else 0
    steps_used = step_budget - steps_remaining

    # Win condition: any pushable object (selected piece or pushable block)
    # reached target_x. We use TRUE engine positions here, not the blinded
    # observation, so WORLD_HARD does not mask the win.
    max_pushable_x_true = _max_pushable_x_true(env)
    if max_pushable_x_true >= target_x:
        game_state = "WIN"
    elif steps_remaining <= 0:
        game_state = "GAME_OVER"
    else:
        game_state = "IN_PROGRESS"

    return {
        "player": {
            "id": selected.id,
            "position": [selected.x, selected.y],
            "size": [selected.w, selected.h],
        },
        "movement": _compute_movement(views),
        "level": {
            "scenario": scenario_name,
            "target_x": target_x,
        },
        "resources": {
            "step_budget": step_budget,
            "steps_used": steps_used,
            "steps_remaining": max(0, steps_remaining),
        },
        "objects": partitioned,
        "semantic_grid": _semantic_grid(views),
        "game_state": game_state,
    }


def build_feedback_easy(
    prev: dict[str, Any],
    curr: dict[str, Any],
    action: str,
) -> str:
    """Narrative feedback describing the effect of `action`. For the KA59
    discovery task, block-position deltas are the key signal — that's how the
    agent learns the wall-transfer asymmetry."""
    parts: list[str] = [f"Action taken: {action}."]

    prev_pos = prev["player"]["position"]
    curr_pos = curr["player"]["position"]
    if prev_pos != curr_pos:
        parts.append(f"Selected piece moved from {prev_pos} to {curr_pos}.")
    else:
        parts.append("Selected piece did not change position (action blocked or inert).")

    def _by_id(objs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {o["id"]: o for o in objs}

    prev_blocks = _by_id(prev["objects"]["blocks"])
    curr_blocks = _by_id(curr["objects"]["blocks"])
    for obj_id, curr_obj in curr_blocks.items():
        prev_obj = prev_blocks.get(obj_id)
        if prev_obj is None:
            continue
        if prev_obj["position"] != curr_obj["position"]:
            parts.append(
                f"Block '{obj_id}' moved from {prev_obj['position']} to {curr_obj['position']}."
            )

    prev_ctrls = _by_id(prev["objects"]["controllables"])
    curr_ctrls = _by_id(curr["objects"]["controllables"])
    prev_sel = next((o["id"] for o in prev_ctrls.values() if o["is_selected"]), None)
    curr_sel = next((o["id"] for o in curr_ctrls.values() if o["is_selected"]), None)
    if prev_sel and curr_sel and prev_sel != curr_sel:
        parts.append(f"Active selection changed from '{prev_sel}' to '{curr_sel}'.")

    curr_steps = curr["resources"]["steps_remaining"]
    parts.append(f"Steps remaining: {curr_steps}.")

    if curr["game_state"] == "WIN":
        parts.append("You reached the goal and won the scenario.")
    elif curr["game_state"] == "GAME_OVER":
        parts.append("Step budget exhausted; the scenario ended without reaching the goal.")

    return " ".join(parts)
