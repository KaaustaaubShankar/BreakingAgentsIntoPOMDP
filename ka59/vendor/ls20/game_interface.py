"""
game_interface.py — Structured state extraction from the ls20 game environment.

Accesses Ls20 internal attributes to build clean JSON state for the Easy World
axis level. For Hard World, the raw ASCII frame or a rendered PNG image is used.

Key Ls20 internals used:
  gudziatsk           — player sprite (.x, .y)
  aqygnziho           — lives remaining (starts at 3)
  _step_counter_ui    — stamina UI (.current_steps)
  fwckfzsyc           — current shape index (0–5)
  hiaauhahz           — current color index into tnkekoeuk
  cklxociuu           — current rotation index into dhksvilbb
  tnkekoeuk           — list of 4 color ints [12, 9, 14, 8] = orange/blue/green/red
  dhksvilbb           — rotation values [0, 90, 180, 270]
  plrpelhym           — goal position sprites (tag: rjlbuycveu)
  lvrnuajbl           — bool list: which goals are completed
  ldxlnycps           — required shape index per goal
  yjdexjsoa           — required color index per goal
  ehwheiwsk           — required rotation index per goal
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

import numpy as np
from arcengine import FrameDataRaw

# ARC color index → human-readable name (covers all 16 palette slots)
_COLOR_NAMES: dict[int, str] = {
    0: "white", 1: "off_white", 2: "light_gray", 3: "gray", 4: "dark_gray",
    5: "black", 6: "magenta", 7: "light_magenta", 8: "red", 9: "blue",
    10: "light_blue", 11: "yellow", 12: "orange", 13: "maroon",
    14: "green", 15: "purple",
}

# ASCII rendering map: color index → character
ASCII_MAP: dict[int, str] = {
    5: " ", 4: ".", 3: ":", 2: "+", 1: "=", 0: "#",
    6: "M", 7: "m", 8: "R", 9: "B", 10: "b",
    11: "Y", 12: "O", 13: "X", 14: "G", 15: "P",
}

ASCII_LEGEND = (
    "# = wall (impassable — moving into it wastes a turn and your position will not change)  "
    ". = floor (walkable)  B = blue  G = green  Y = yellow  "
    "O = orange  R = red  M = magenta  P = purple  : = dark surface  + = mid surface"
)


def frame_to_ascii(frame: np.ndarray) -> str:
    """Convert a 64×64 numpy frame to a plain ASCII string (no ANSI codes)."""
    rows: list[str] = []
    height, width = frame.shape
    for y in range(height):
        rows.append("".join(ASCII_MAP.get(int(frame[y, x]), "?") for x in range(width)))
    return "\n".join(rows)


def frame_to_base64_png(frame: np.ndarray, scale: int = 8) -> str:
    """
    Render a 64×64 frame as a PNG and return it base64-encoded.

    Used for the visual Hard World axis level — the image is sent directly
    to a vision-capable model as a data URI.

    Args:
        frame: 64×64 numpy array of ARC color indices (0–15).
        scale: Upscaling factor (default 8 → 512×512 output).

    Returns:
        Base64-encoded PNG string (no data-URI prefix).
    """
    from PIL import Image
    from arc_agi.rendering import frame_to_rgb_array, COLOR_MAP

    rgb = frame_to_rgb_array(0, frame, scale=scale, color_map=COLOR_MAP)
    img = Image.fromarray(rgb.astype("uint8"), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def get_structured_state(env: Any, frame_data: FrameDataRaw) -> dict[str, Any]:
    """
    Extract structured game state from the Ls20 game instance.

    Accesses private attributes of the Ls20 object (env._game).
    Returns a clean JSON-serialisable dict for the Easy World axis level.
    """
    game = env._game
    if game is None:
        return {"error": "game not initialised"}

    def color_name(color_int: int) -> str:
        return _COLOR_NAMES.get(int(color_int), f"color_{color_int}")

    # Piece state
    shape_idx: int = int(game.fwckfzsyc)
    color_int: int = int(game.tnkekoeuk[game.hiaauhahz])
    rotation: int = int(game.dhksvilbb[game.cklxociuu])

    # Player position and resources
    lives: int = int(game.aqygnziho)
    stamina: int = max(0, int(game._step_counter_ui.current_steps))

    # Goals
    goals: list[dict[str, Any]] = []
    for i, goal_sprite in enumerate(game.plrpelhym):
        req_color_int = int(game.tnkekoeuk[game.yjdexjsoa[i]])
        req_rotation = int(game.dhksvilbb[game.ehwheiwsk[i]])
        req_shape = int(game.ldxlnycps[i])
        goals.append({
            "id": i,
            "position": [int(goal_sprite.x), int(goal_sprite.y)],
            "completed": bool(game.lvrnuajbl[i]),
            "required_shape": req_shape,
            "required_color": color_name(req_color_int),
            "required_rotation_degrees": req_rotation,
        })

    # Modifier tile positions — each tile changes one attribute of the piece
    def tile_positions(tag: str) -> list[list[int]]:
        return [
            [int(s.x), int(s.y)]
            for s in game.current_level.get_sprites_by_tag(tag)
            if s.is_visible
        ]

    # Stamina refill tiles
    stamina_refills = tile_positions("npxgalaybz")

    # Blocked directions — check adjacent cells in the frame for walls (color index 0)
    px, py = int(game.gudziatsk.x), int(game.gudziatsk.y)
    blocked: list[str] = []
    if frame_data.frame:
        frame = frame_data.frame[-1]
        h, w = frame.shape
        if py > 0     and int(frame[py - 1, px]) == 0: blocked.append("NORTH")
        if py < h - 1 and int(frame[py + 1, px]) == 0: blocked.append("SOUTH")
        if px > 0     and int(frame[py, px - 1]) == 0: blocked.append("WEST")
        if px < w - 1 and int(frame[py, px + 1]) == 0: blocked.append("EAST")

    return {
        "player": {
            "position": [px, py],
            "lives": lives,
            "stamina": stamina,
            "blocked_directions": blocked,
        },
        "piece": {
            "shape": shape_idx,
            "color": color_name(color_int),
            "rotation_degrees": rotation,
        },
        "modifier_tiles": {
            "shape_changer":    tile_positions("ttfwljgohq"),
            "color_changer":    tile_positions("soyhouuebz"),
            "rotation_changer": tile_positions("rhsxkxzdjz"),
            "stamina_refill":   stamina_refills,
        },
        "goals": goals,
        "level": {
            "current": int(frame_data.levels_completed) + 1,
            "total": int(frame_data.win_levels),
        },
        "game_state": frame_data.state.name,
    }


def build_feedback_easy(
    prev: dict[str, Any],
    curr: dict[str, Any],
    action: str,
) -> str:
    """
    Build a detailed, human-readable feedback string for the Easy Feedback axis.
    Compares previous and current structured states to narrate what changed.
    """
    parts: list[str] = [f"Action taken: {action}."]

    # Wall detection
    if prev["player"]["position"] == curr["player"]["position"]:
        parts.append("Blocked by a wall — your position did not change. Try a different direction.")

    # Stamina / lives
    prev_stam = prev["player"]["stamina"]
    curr_stam = curr["player"]["stamina"]
    prev_lives = prev["player"]["lives"]
    curr_lives = curr["player"]["lives"]

    if curr_lives < prev_lives:
        parts.append(
            f"Stamina depleted — lost a life! Lives remaining: {curr_lives}. "
            f"Stamina reset to {curr_stam}."
        )
    else:
        parts.append(f"Stamina: {curr_stam}.")

    # Piece attribute changes
    prev_p, curr_p = prev["piece"], curr["piece"]
    if curr_p["shape"] != prev_p["shape"]:
        parts.append(
            f"Piece shape changed from {prev_p['shape']} → {curr_p['shape']} "
            f"(stepped on a shape modifier tile)."
        )
    if curr_p["color"] != prev_p["color"]:
        parts.append(
            f"Piece color changed from {prev_p['color']} → {curr_p['color']} "
            f"(stepped on a color modifier tile)."
        )
    if curr_p["rotation_degrees"] != prev_p["rotation_degrees"]:
        parts.append(
            f"Piece rotated from {prev_p['rotation_degrees']}° → {curr_p['rotation_degrees']}° "
            f"(stepped on a rotation modifier tile)."
        )

    # Goal completions
    for g_curr, g_prev in zip(curr["goals"], prev["goals"]):
        if g_curr["completed"] and not g_prev["completed"]:
            parts.append(f"Goal {g_curr['id']} completed!")

    # Current goal match status + navigation hints for each incomplete goal
    cp = curr["piece"]
    mods = curr.get("modifier_tiles", {})
    for g in curr["goals"]:
        if not g["completed"]:
            shape_ok = cp["shape"] == g["required_shape"]
            color_ok = cp["color"] == g["required_color"]
            rot_ok = cp["rotation_degrees"] == g["required_rotation_degrees"]
            if shape_ok and color_ok and rot_ok:
                parts.append(
                    f"Goal {g['id']} at {g['position']}: ACTIVATED — your piece matches! "
                    f"Step on it to complete."
                )
            else:
                mismatches = []
                hints = []
                if not shape_ok:
                    mismatches.append(
                        f"shape (have {cp['shape']}, need {g['required_shape']})"
                    )
                    if mods.get("shape_changer"):
                        hints.append(f"shape_changer tiles at {mods['shape_changer']}")
                if not color_ok:
                    mismatches.append(
                        f"color (have {cp['color']}, need {g['required_color']})"
                    )
                    if mods.get("color_changer"):
                        hints.append(f"color_changer tiles at {mods['color_changer']}")
                if not rot_ok:
                    mismatches.append(
                        f"rotation (have {cp['rotation_degrees']}°, "
                        f"need {g['required_rotation_degrees']}°)"
                    )
                    if mods.get("rotation_changer"):
                        hints.append(f"rotation_changer tiles at {mods['rotation_changer']}")
                msg = (
                    f"Goal {g['id']} at {g['position']}: not yet matched — "
                    + ", ".join(mismatches) + "."
                )
                if hints:
                    msg += " To fix: " + "; ".join(hints) + "."
                parts.append(msg)

    return " ".join(parts)
