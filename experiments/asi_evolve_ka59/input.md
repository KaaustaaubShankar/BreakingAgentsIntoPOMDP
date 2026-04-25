# KA59 — Hidden-Mechanic Puzzle Agent

## Problem

KA59 is a multi-level grid puzzle game. You must write a Python function `agent_step(state, history)` that plays this game by selecting actions one step at a time.

**Goal:** Progress through as many levels as possible. The game has 7 levels total. Winning the final level ends the run.

## State structure

Each call to `agent_step` receives:

- `state["semantic_grid"]` — ASCII grid snapshot: `P`=selected piece, `S`=other selectable, `B`=block, `#`=wall, `+`=goal, `.`=empty
- `state["player"]["position"]` — `[x, y]` pixel coordinates of the currently selected piece
- `state["player"]["id"]` — identifier of the selected piece
- `state["objects"]["selectables"]` — list of all selectable pieces with their positions
- `state["objects"]["blocks"]` — list of pushable blocks with positions
- `state["objects"]["walls"]` — list of wall objects with positions
- `state["objects"]["goals"]` — list of goal tiles with positions
- `state["movement"]["right_blocked"]` etc — heuristic hints (True if something is adjacent that direction)
- `state["resources"]["steps_remaining"]` — budget left this level
- `state["level"]["current"]` / `state["level"]["total"]` — progress

Positions use pixel coordinates where `STEP=3` pixels per cell. Every movement shifts the piece by exactly one cell (3 pixels).

## Action space

Return a dict with `"action"` set to one of:
- `"MOVE_LEFT"`, `"MOVE_RIGHT"`, `"MOVE_UP"`, `"MOVE_DOWN"`
- `"CLICK"` — must also include `"target_position": [x, y]` in the same pixel coordinate system as `player.position`

## History

`history` is a list of all previous turns in the current run. Each entry is:
```python
{"state": <state dict>, "action": <returned dict>, "moved": bool}
```
`moved` is True if the selected piece actually changed position.

## Initial program (baseline)

```python
def agent_step(state, history):
    step = len(history)
    cycle = ["MOVE_RIGHT", "MOVE_RIGHT", "MOVE_DOWN", "MOVE_RIGHT", "MOVE_UP"]
    return {"action": cycle[step % len(cycle)]}
```

## Scoring

Score = `levels_completed` (0–7). A score of 7 means the game was fully won.
The function is evaluated over 3 independent trials; average score is used.

## Key constraints

- The step budget per level is 64 turns. Exceeding it ends the level with failure.
- Some walls behave differently depending on what tries to cross them. Observe carefully.
- You can switch control to a different selectable piece using CLICK.
- Not all selectables may be useful on every level.

## What good solutions look like

- Efficiently move the selected piece toward goal tiles
- Push blocks strategically — they can interact with walls in non-obvious ways
- Switch pieces when the current one is stuck
- Probe wall types by attempting interactions and observing what happens

Write a `agent_step(state, history)` function in Python that plays as well as possible.
