# KA59 Faithful Simulator & Blinded Environment

Faithful, readable reimplementation of the core movement mechanics from
`ka59.py` (ARC Prize 2026 game source, MIT licence).

This is a **faithful simulator for analysis** — not a solver or benchmark agent.

## Critical Asymmetry

KA59 has two wall types:

| Tag | Constant | Blocks direct move? | Blocks push? |
|-----|----------|---------------------|--------------|
| `0015qniapgwsvb` | `TAG_WALL_TRANSFER` | ✅ yes | ❌ no |
| `0029ifoxxfvvvs` | `TAG_WALL_SOLID`    | ✅ yes | ✅ yes |

This asymmetry means a **pushable object can be pushed through a
`WALL_TRANSFER` boundary that the selected/player piece cannot enter
directly**. The mechanism is sometimes called a "transfer gate" or
"wall-transfer" mechanic.

### Why it works (source mapping)

| Source function | Our name | Wall check |
|----------------|----------|------------|
| `loydmqkgjw`   | `direct_move` | BOTH wall types |
| `ifoelczjjh`   | `push`        | WALL_SOLID only |

## Constants from source

| Obfuscated name | Value | Meaning |
|-----------------|-------|---------|
| `zsqdfmgyjo`    | `3`   | `STEP` — pixels per action |
| `zuhizjvlpo`    | `5`   | `MAX_PUSH_TRACK_STEPS` |

## Module layout

```
ka59_ref/
  engine.py   — faithful simulator (rule-aware, uses internal tag names)
  env.py      — blinded agent-facing environment (hides rules / tag names)
  README.md   — this file
```

### `engine.py` — KA59State
Faithful reimplementation of the source mechanics. Knows about wall tags,
push asymmetry, etc. Used directly only for analysis and test scaffolding.

### `env.py` — KA59BlindEnv
Agent-facing wrapper. The agent interacts via:

```python
from ka59_ref.env import KA59BlindEnv, MOVE_RIGHT, MOVE_LEFT, SELECT

env = KA59BlindEnv()
obs = env.reset(level_spec)      # → list[ObjectView]
result = env.step(MOVE_RIGHT)    # → StepResult
result = env.step(SELECT("b"))   # → StepResult
```

**ObjectView fields:** `id`, `x`, `y`, `w`, `h`, `kind`, `is_selected`  
**kind values:** `"wall"` | `"block"` | `"controllable"` — both wall types map to `"wall"`  
**StepResult fields:** `obs`, `moved`, `steps_remaining`, `done`

**Level spec format** (used by `reset()` to configure the engine — NOT exposed to agent):
```python
{
    "steps": 10,
    "objects": [
        {"id": "sel", "x": 0,  "y": 0, "w": 3, "h": 3, "kind": "selected"},
        {"id": "pb",  "x": 3,  "y": 0, "w": 3, "h": 3, "kind": "block"},
        {"id": "wt",  "x": 6,  "y": 0, "w": 3, "h": 3, "kind": "wall_transfer"},
        {"id": "ws",  "x": 9,  "y": 0, "w": 3, "h": 3, "kind": "wall_solid"},
    ]
}
```

## Running the tests

```bash
# Engine (faithful simulator) tests only:
python3 -m pytest tests/test_ka59_ref.py -v

# Env (blinded agent) tests only:
python3 -m pytest tests/test_ka59_env.py -v

# Both together (56 total):
python3 -m pytest tests/test_ka59_ref.py tests/test_ka59_env.py -v
```

All 56 tests pass, including the critical no-leakage and transition-asymmetry
suites in `TestNoTagLeakage` and `TestAsymmetryViaTransitions`.

## Known limitations / ambiguities

- **Collision model**: uses AABB (bounding-box) collision instead of the
  arcengine pixel-mask collision. For all rectangular solid sprites this is
  equivalent; cross-shaped sprites with transparent pixels would diverge in
  edge cases.
- **Multi-step push tracking**: the source spreads push resolution over
  multiple game ticks (via `lphmmaeepj`/`dgjbrykwhi`). Our `action_move`
  resolves the push atomically in one call, which is sufficient for testing
  movement semantics but not for timing-sensitive rendering behaviour.
- **Identity guard**: `Obj.collides_with(self)` returns False (identity guard),
  mirroring what arcengine almost certainly does internally to prevent a sprite
  from colliding with itself during push resolution.
