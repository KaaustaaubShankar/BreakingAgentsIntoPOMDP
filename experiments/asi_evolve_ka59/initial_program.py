"""
Baseline agent for KA59. Cycles through a fixed action sequence.
ASI-Evolve will evolve this into something better.
"""


def agent_step(state, history):
    step = len(history)

    # Try to move toward the goal if we can see one
    goals = state.get("objects", {}).get("goals", [])
    player_pos = state.get("player", {}).get("position", [0, 0])

    if goals:
        gx, gy = goals[0]["position"]
        px, py = player_pos
        dx, dy = gx - px, gy - py
        if abs(dx) >= abs(dy):
            return {"action": "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT"}
        else:
            return {"action": "MOVE_DOWN" if dy > 0 else "MOVE_UP"}

    # Fallback: cycle through moves
    cycle = ["MOVE_RIGHT", "MOVE_RIGHT", "MOVE_DOWN", "MOVE_RIGHT", "MOVE_UP", "MOVE_LEFT"]
    return {"action": cycle[step % len(cycle)]}
