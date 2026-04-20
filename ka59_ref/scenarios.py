"""
KA59 Canonical Scenario Corpus
================================
A small set of named level specs that cover the mechanistically important
situations in KA59.  These are the "paper-useful" reference scenarios —
interpretable enough to reason about, minimal enough to inspect by hand.

Each entry is a plain dict that KA59BlindEnv.reset() accepts directly.
All object IDs are short human-readable strings, never internal tag constants.

Scenario inventory
------------------
open_move_right
    No obstacles.  One selected piece, nothing else.
    Expected behaviour: agent moves freely in any direction.
    Establishes performance ceiling for unobstructed movement.

transfer_wall_direct_block
    Selected piece is directly adjacent to a WALL_TRANSFER tile.
    No pushable block sits between them.
    Expected behaviour: direct move right is blocked immediately;
    agent must rotate to another direction.
    Probes: can agent recover from a wall it cannot cross directly?

transfer_wall_push
    sel → block → WALL_TRANSFER → open space.
    Expected behaviour: first move right succeeds (block is pushed through
    the transfer wall); agent records passable_walls discovery.
    Probes: the core wall-transfer asymmetry via a push scenario.

solid_wall_push_blocked
    sel → block → WALL_SOLID → (no space).
    Expected behaviour: first move right is blocked (block cannot pass
    solid wall); agent records a block in direction 'right'.
    Probes: the same visual layout as transfer_wall_push but with
    fundamentally different mechanics — indistinguishable from observation.

push_chain
    sel → block_a → block_b → open space (no walls).
    Expected behaviour: first move right propagates through the two-block
    chain; both blocks advance; selected piece advances.
    Probes: recursive push propagation.
"""

from __future__ import annotations

STEP: int = 3   # mirror engine.STEP; kept local so this file is self-contained


SCENARIOS: dict[str, dict] = {

    # ------------------------------------------------------------------
    "open_move_right": {
        "steps":   20,
        "objects": [
            {"id": "sel", "x": 0, "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
        ],
    },

    # ------------------------------------------------------------------
    "transfer_wall_direct_block": {
        # Selected is one STEP to the left of a WALL_TRANSFER tile.
        # No pushable block in between → direct move right is immediately blocked.
        "steps":   10,
        "objects": [
            {"id": "sel", "x": 0,    "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
            {"id": "wt",  "x": STEP, "y": 0, "w": STEP, "h": STEP, "kind": "wall_transfer"},
        ],
    },

    # ------------------------------------------------------------------
    "transfer_wall_push": {
        # sel(0) → block(STEP) → WALL_TRANSFER(2*STEP) → open space
        # Push right: block passes through WALL_TRANSFER; selected advances.
        "steps":   10,
        "objects": [
            {"id": "sel", "x": 0,          "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
            {"id": "pb",  "x": STEP,       "y": 0, "w": STEP, "h": STEP, "kind": "block"},
            {"id": "wt",  "x": STEP * 2,   "y": 0, "w": STEP, "h": STEP, "kind": "wall_transfer"},
        ],
    },

    # ------------------------------------------------------------------
    "solid_wall_push_blocked": {
        # sel(0) → block(STEP) → WALL_SOLID(2*STEP)
        # Identical visual layout to transfer_wall_push; different mechanics.
        # Push right: block is stopped by WALL_SOLID; selected does not advance.
        "steps":   10,
        "objects": [
            {"id": "sel", "x": 0,          "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
            {"id": "pb",  "x": STEP,       "y": 0, "w": STEP, "h": STEP, "kind": "block"},
            {"id": "ws",  "x": STEP * 2,   "y": 0, "w": STEP, "h": STEP, "kind": "wall_solid"},
        ],
    },

    # ------------------------------------------------------------------
    "push_chain": {
        # sel(0) → block_a(STEP) → block_b(2*STEP) → open space
        # No walls.  Push right propagates through the full two-block chain.
        "steps":   10,
        "objects": [
            {"id": "sel", "x": 0,          "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
            {"id": "pa",  "x": STEP,       "y": 0, "w": STEP, "h": STEP, "kind": "block"},
            {"id": "pb",  "x": STEP * 2,   "y": 0, "w": STEP, "h": STEP, "kind": "block"},
        ],
    },
}
