# KA59 Reference Simulator

Faithful, readable reimplementation of the core movement mechanics from
`ka59.py` (ARC Prize 2026 game source, MIT licence).

This is a **reference clone for analysis** — not a solver or benchmark agent.

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

## Running the tests

```bash
python3 -m pytest tests/test_ka59_ref.py -v
```

All 23 tests should pass, including the focused asymmetry tests in
`TestPushWallAsymmetry`.

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
