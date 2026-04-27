"""
game_interface.py — Structured state extraction for BP35.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from arcengine import FrameDataRaw

ASCII_MAP: dict[int, str] = {
    5: " ",
    4: ".",
    3: ":",
    2: "+",
    1: "=",
    0: "#",
    6: "M",
    7: "m",
    8: "R",
    9: "B",
    10: "b",
    11: "Y",
    12: "O",
    13: "X",
    14: "G",
    15: "P",
}

NAME_TO_CHAR: dict[str, str] = {
    "qclfkhjnaac": "x",
    "xcjjwqfzjfe": "o",
    "etlsaqqtjvn": "y",
    "lrpkmzabbfa": "g",
    "yuuqpmlxorv": "1",
    "oonshderxef": "2",
    "ubhhgljbnpu": "v",
    "hzusueifitk": "u",
    "aknlbboysnc": "m",
    "jcyhkseuorf": "w",
    "fjlzdjxhant": "+",
    "player_right": "P",
    "player_left": "P",
    "player": "P",
}

NAME_TO_LABEL: dict[str, str] = {
    "qclfkhjnaac": "breakable_block",
    "xcjjwqfzjfe": "solid_block",
    "etlsaqqtjvn": "expansion_tile",
    "lrpkmzabbfa": "gravity_switch",
    "yuuqpmlxorv": "toggle_open",
    "oonshderxef": "toggle_closed",
    "ubhhgljbnpu": "spike_down",
    "hzusueifitk": "spike_up",
    "aknlbboysnc": "moving_support_a",
    "jcyhkseuorf": "moving_support_b",
    "fjlzdjxhant": "goal",
}

CLICKABLE_NAMES = {
    "qclfkhjnaac",
    "etlsaqqtjvn",
    "lrpkmzabbfa",
    "yuuqpmlxorv",
    "oonshderxef",
}

SAFE_LANDING_NAMES = {
    "oonshderxef",
    "aknlbboysnc",
    "jcyhkseuorf",
}

GOAL_NAMES = {"fjlzdjxhant"}
SPIKE_NAMES = {"ubhhgljbnpu", "hzusueifitk"}

ASCII_LEGEND = (
    "P=player  +=goal  x=breakable  y=expansion  g=gravity switch  "
    "1/2=toggle tiles  u/v=spikes  o/m/w=support tiles"
)


def frame_to_ascii(frame: np.ndarray) -> str:
    rows: list[str] = []
    height, width = frame.shape
    for y in range(height):
        rows.append("".join(ASCII_MAP.get(int(frame[y, x]), "?") for x in range(width)))
    return "\n".join(rows)


def frame_to_base64_png(frame: np.ndarray, scale: int = 8) -> str:
    from PIL import Image
    from arc_agi.rendering import COLOR_MAP, frame_to_rgb_array

    rgb = frame_to_rgb_array(0, frame, scale=scale, color_map=COLOR_MAP)
    image = Image.fromarray(rgb.astype("uint8"), "RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _step_budget_for_level(level_index: int) -> int:
    # Real termination for levels 1-3 is the rising moving-support platform, not this counter.
    # For levels 4+, the engine triggers lose() when hbqwwgceeqp reaches 64 or 128.
    if level_index <= 6:
        return 64
    return 128


def _tile_names_at(game: Any, x: int, y: int) -> list[str]:
    return [sprite.name for sprite in game.hdnrlfmyrj.jhzcxkveiw(x, y)]


def _is_blocked(names: list[str]) -> bool:
    if not names:
        return False
    if names == ["oonshderxef"] or names == ["aknlbboysnc"] or names == ["fjlzdjxhant"]:
        return False
    if set(names) == {"aknlbboysnc", "oonshderxef"}:
        return False
    return True


def _semantic_grid(game: Any) -> list[str]:
    level = game.hdnrlfmyrj
    width, height = level.grid_size
    rows: list[str] = []
    player_x, player_y = game.twdpowducb.qumspquyus
    for y in range(height):
        row_chars: list[str] = []
        for x in range(width):
            if (x, y) == (player_x, player_y):
                row_chars.append("P")
                continue
            names = _tile_names_at(game, x, y)
            char = " "
            for name in names:
                if name in NAME_TO_CHAR and NAME_TO_CHAR[name] != "P":
                    char = NAME_TO_CHAR[name]
                    break
            row_chars.append(char)
        rows.append("".join(row_chars))
    return rows


def _fall_destination(game: Any, x: int, y: int) -> list[int]:
    """Return the tile the player would land on after gravity acts from (x, y)."""
    dy = -1 if bool(game.vivnprldht) else 1
    nx, ny = x, y + dy
    width, height = game.hdnrlfmyrj.grid_size
    last_open = [x, y]
    while 0 <= nx < width and 0 <= ny < height:
        names = [s.name for s in game.hdnrlfmyrj.jhzcxkveiw(nx, ny)]
        passable = (
            names == []
            or names == ["oonshderxef"]
            or names == ["aknlbboysnc"]
            or set(names) == {"aknlbboysnc", "oonshderxef"}
        )
        if not passable:
            break
        last_open = [nx, ny]
        ny += dy
    return last_open


def _is_passable_horizontal(game: Any, x: int, y: int) -> bool:
    """True if the player can physically occupy (x, y) — i.e. no solid blocking tile."""
    names = [s.name for s in game.hdnrlfmyrj.jhzcxkveiw(x, y)]
    return _is_blocked(names) is False


def _corridor_scan(game: Any, player_x: int, player_y: int) -> list[dict[str, Any]]:
    """
    Scan every x-column reachable horizontally from the player's current position
    (without clicking) and report where gravity would take the player from each column.
    Columns where landing_y != player_y are highlighted as passages.
    """
    width, _ = game.hdnrlfmyrj.grid_size
    results: list[dict[str, Any]] = []

    # Walk left and right from player until blocked
    for direction in (-1, 1):
        x = player_x + direction
        while 0 <= x < width:
            if not _is_passable_horizontal(game, x, player_y):
                break
            landing = _fall_destination(game, x, player_y)
            entry: dict[str, Any] = {
                "column": x,
                "landing": landing,
                "passage": landing[1] != player_y,
            }
            results.append(entry)
            x += direction

    results.sort(key=lambda e: e["column"])
    return results


def _moving_platform_state(game: Any) -> dict[str, Any] | None:
    """Return the current position and moves-until-kill for the rising platform (levels 1-3 only)."""
    level_num = int(game.qswcochjodb)
    if level_num > 3:
        return None
    try:
        supports = game.hdnrlfmyrj.wwkbcxznzg("aknlbboysnc")
        if not supports:
            return None
        top_y = min(int(s.qumspquyus[1]) for s in supports)
        player_y = int(game.twdpowducb.qumspquyus[1])
        # Platform starts below player (higher y) and rises toward lower y.
        # It moves up 1 tile every 2 horizontal moves. Kill when top_y reaches player_y.
        # Distance to close = top_y - player_y (positive means platform is still below).
        gap = top_y - player_y
        moves_until_kill = max(0, gap * 2)
        return {
            "platform_top_y": top_y,
            "player_y": player_y,
            "tiles_below_player": gap,
            "moves_until_kill": moves_until_kill,
            "warning": (
                "URGENT: rising platform will reach you in {} horizontal moves. "
                "Clicks do not advance the platform. Move upward (toward lower y) or die."
            ).format(moves_until_kill) if moves_until_kill <= 6 else None,
        }
    except Exception:
        return None


def get_structured_state(env: Any, frame_data: FrameDataRaw) -> dict[str, Any]:
    game = env._game.oztjzzyqoek
    player_x, player_y = game.twdpowducb.qumspquyus
    level_num = int(game.qswcochjodb)
    step_count = int(env._game.hbqwwgceeqp)
    step_budget = _step_budget_for_level(level_num)
    gravity = "DOWN" if bool(game.vivnprldht) else "UP"

    clickable_tiles: list[dict[str, Any]] = []
    goal_tiles: list[list[int]] = []
    spike_tiles: list[list[int]] = []
    safe_supports: list[dict[str, Any]] = []

    width, height = game.hdnrlfmyrj.grid_size
    for y in range(height):
        for x in range(width):
            names = _tile_names_at(game, x, y)
            if not names:
                continue
            if any(name in CLICKABLE_NAMES for name in names):
                clickable_tiles.append(
                    {
                        "position": [x, y],
                        "types": [NAME_TO_LABEL[name] for name in names if name in NAME_TO_LABEL],
                    }
                )
            if any(name in GOAL_NAMES for name in names):
                goal_tiles.append([x, y])
            if any(name in SPIKE_NAMES for name in names):
                spike_tiles.append([x, y])
            if any(name in SAFE_LANDING_NAMES for name in names):
                safe_supports.append(
                    {
                        "position": [x, y],
                        "types": [NAME_TO_LABEL[name] for name in names if name in NAME_TO_LABEL],
                    }
                )

    left_names = _tile_names_at(game, player_x - 1, player_y) if player_x > 0 else ["boundary"]
    right_names = _tile_names_at(game, player_x + 1, player_y) if player_x < width - 1 else ["boundary"]

    left_blocked = _is_blocked(left_names)
    right_blocked = _is_blocked(right_names)

    # Where does the player actually land after moving left/right and gravity acts?
    left_landing = (
        _fall_destination(game, player_x - 1, player_y)
        if not left_blocked and player_x > 0
        else None
    )
    right_landing = (
        _fall_destination(game, player_x + 1, player_y)
        if not right_blocked and player_x < width - 1
        else None
    )

    moving_platform = _moving_platform_state(game)
    corridor = _corridor_scan(game, player_x, player_y)

    return {
        "player": {
            "position": [player_x, player_y],
            "gravity": gravity,
            "facing": "RIGHT" if bool(game.ybmkdxbdko) else "LEFT",
        },
        "movement": {
            "left_target": [player_x - 1, player_y] if player_x > 0 else None,
            "right_target": [player_x + 1, player_y] if player_x < width - 1 else None,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked,
            "left_tile_names": left_names,
            "right_tile_names": right_names,
            "left_landing": left_landing,
            "right_landing": right_landing,
        },
        "level": {
            "current": level_num,
            "total": int(frame_data.win_levels),
        },
        "resources": {
            "steps_used": step_count,
            "step_budget": step_budget,
            "steps_remaining": max(0, step_budget - step_count),
        },
        "moving_platform": moving_platform,
        "corridor_scan": corridor,
        "objects": {
            "goals": goal_tiles,
            "spikes": spike_tiles,
            "clickable_tiles": clickable_tiles,
            "safe_supports": safe_supports,
        },
        "semantic_grid": _semantic_grid(game),
        "game_state": frame_data.state.name,
    }


def build_feedback_easy(prev: dict[str, Any], curr: dict[str, Any], action: str) -> str:
    parts: list[str] = [f"Action taken: {action}."]

    if prev["player"]["position"] != curr["player"]["position"]:
        parts.append(
            f"Player moved from {prev['player']['position']} to {curr['player']['position']}."
        )
    else:
        parts.append("Player position did not change.")

    if prev["player"]["gravity"] != curr["player"]["gravity"]:
        parts.append(
            f"Gravity flipped from {prev['player']['gravity']} to {curr['player']['gravity']}."
        )

    prev_steps = prev["resources"]["steps_remaining"]
    curr_steps = curr["resources"]["steps_remaining"]
    if curr_steps != prev_steps:
        parts.append(f"Steps remaining: {curr_steps}.")

    prev_clickables = {
        tuple(item["position"]): tuple(item["types"])
        for item in prev["objects"]["clickable_tiles"]
    }
    curr_clickables = {
        tuple(item["position"]): tuple(item["types"])
        for item in curr["objects"]["clickable_tiles"]
    }
    removed = sorted(pos for pos in prev_clickables if pos not in curr_clickables)
    added = sorted(pos for pos in curr_clickables if pos not in prev_clickables)
    if removed:
        parts.append(f"Clickable tiles removed at: {removed}.")
    if added:
        parts.append(f"Clickable tiles added at: {added}.")

    if curr["game_state"] == "WIN":
        parts.append("You reached the goal and won the level.")
    elif curr["game_state"] == "GAME_OVER":
        parts.append("The run ended in failure.")

    parts.append(
        f"Current gravity is {curr['player']['gravity']}; steps remaining: {curr['resources']['steps_remaining']}."
    )
    return " ".join(parts)
