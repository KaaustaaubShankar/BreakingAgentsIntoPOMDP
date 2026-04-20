"""
KA59 Canonical Scenario Corpus + Paper-Facing Metadata
========================================================
Two dictionaries exported from this module:

  SCENARIOS      — level specs consumed by KA59BlindEnv.reset()
  SCENARIO_META  — paper-facing metadata keyed by the same names

The metadata maps each scenario to the JKJ paper's four-axis knockout
framing (World / Goal / Mechanics / Feedback), following the design in
MONDAY_EXPERIMENT_MATRIX.md and PROJECT_ARCHAEOLOGY.md:

  primary_axis — which capability axis this scenario primarily stresses
  tags         — tuple of short paper-legible labels for the scenario
  rationale    — one-sentence justification for the paper

All names and tags are human-readable.  No internal engine tag constants
(0015qniapgwsvb etc.) appear anywhere in this module.

Scenario inventory
------------------
open_move_right         baseline — no hidden rules, ceiling for free movement
transfer_wall_direct_block  Mechanics — transfer wall blocks direct move
transfer_wall_push          Mechanics — core asymmetry: push passes through
solid_wall_push_blocked     Mechanics — contrast: visually identical, fully blocking
push_chain                  Mechanics — recursive chain-push dynamics
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

STEP: int = 3   # mirror engine.STEP; kept local so this file is self-contained


# ---------------------------------------------------------------------------
# ScenarioMeta: paper-facing taxonomy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioMeta:
    """
    Paper-facing metadata for one KA59 scenario.

    Fields
    ------
    primary_axis : str
        Which of the four capability axes this scenario primarily stresses.
        One of: "baseline" | "World" | "Goal" | "Mechanics" | "Feedback"
        Mirrors the JKJ proposal / MONDAY_EXPERIMENT_MATRIX framing.

    tags : tuple[str, ...]
        Short paper-legible labels describing the scenario's key properties.
        No engine-internal constants; human-readable only.

    rationale : str
        One-sentence justification for including this scenario.
    """
    primary_axis: str
    tags:         Tuple[str, ...]
    rationale:    str


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


# ---------------------------------------------------------------------------
# SCENARIO_META: paper-facing metadata for each canonical scenario
# ---------------------------------------------------------------------------

SCENARIO_META: dict[str, ScenarioMeta] = {

    "open_move_right": ScenarioMeta(
        primary_axis = "baseline",
        tags         = ("no_obstacles", "ceiling", "free_movement"),
        rationale    = (
            "Control condition with no hidden rules or obstacles; establishes "
            "the performance ceiling against which all other scenarios are compared."
        ),
    ),

    "transfer_wall_direct_block": ScenarioMeta(
        primary_axis = "Mechanics",
        tags         = ("hidden_transition", "direct_block", "wall_probe",
                        "no_push_candidate"),
        rationale    = (
            "Mechanics probe: the transfer wall blocks direct movement; agent "
            "must discover via interaction that this boundary cannot be crossed "
            "directly and must find an alternate path."
        ),
    ),

    "transfer_wall_push": ScenarioMeta(
        primary_axis = "Mechanics",
        tags         = ("hidden_transition", "asymmetric_push", "wall_probe",
                        "wall_transfer", "contrastive_pair"),
        rationale    = (
            "Core Mechanics probe: the wall-transfer asymmetry — a boundary that "
            "blocks direct movement but allows pushed blocks to pass through — is "
            "only discoverable through interaction, not from observation alone."
        ),
    ),

    "solid_wall_push_blocked": ScenarioMeta(
        primary_axis = "Mechanics",
        tags         = ("hidden_transition", "wall_solid", "wall_probe",
                        "fully_blocking", "contrastive_pair"),
        rationale    = (
            "Mechanics contrast condition: visually identical layout to "
            "transfer_wall_push but mechanically fully blocking; paired to "
            "isolate the Mechanics axis in the knockout-matrix design."
        ),
    ),

    "push_chain": ScenarioMeta(
        primary_axis = "Mechanics",
        tags         = ("chain_dynamics", "recursive_push", "multi_block"),
        rationale    = (
            "Mechanics probe for recursive push propagation: tests whether "
            "the agent can leverage a two-block chain as an instrument for "
            "advancement when the direct path is long."
        ),
    ),
}
