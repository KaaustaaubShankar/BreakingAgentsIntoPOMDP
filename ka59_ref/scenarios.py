"""
KA59 Canonical Scenario Corpus + Paper-Facing Metadata + Axis Coverage
========================================================================
Three dictionaries and two dataclasses exported from this module:

  SCENARIOS          — level specs consumed by KA59BlindEnv.reset()
  SCENARIO_META      — paper-facing taxonomy keyed by the same names
  KA59_AXIS_COVERAGE — honest per-axis coverage declaration

Honest axis-coverage framing (see MONDAY_EXPERIMENT_MATRIX.md):
  Mechanics : strong   — KA59's core probe; all main scenarios cover this
  World     : partial  — one probe (position-blind variant) is implemented
  Goal      : none     — no win condition is varied; not yet implemented
  Feedback  : none     — feedback richness is fixed; not yet implemented

This makes explicit that KA59 is NOT a full-axis benchmark today.
Do not oversell coverage to Kaus / Ben / the paper.

Scenario inventory (primary_axis)
----------------------------------
open_move_right                  baseline
transfer_wall_direct_block       Mechanics
transfer_wall_push               Mechanics  (core contrastive pair, with solid)
solid_wall_push_blocked          Mechanics  (core contrastive pair, with transfer)
push_chain                       Mechanics
transfer_wall_push_world_blind   World      (observe_positions=False)
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

    # ------------------------------------------------------------------
    "transfer_wall_push_world_blind": {
        # World-axis probe: same mechanics as transfer_wall_push, but
        # non-selected object positions are hidden from the agent.
        # observe_positions=False → agent cannot detect block displacement
        # → cannot discover passable walls even though the push still works.
        "steps":             10,
        "observe_positions": False,
        "objects": [
            {"id": "sel", "x": 0,        "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
            {"id": "pb",  "x": STEP,     "y": 0, "w": STEP, "h": STEP, "kind": "block"},
            {"id": "wt",  "x": STEP * 2, "y": 0, "w": STEP, "h": STEP, "kind": "wall_transfer"},
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

SCENARIO_META.update({
    "transfer_wall_push_world_blind": ScenarioMeta(
        primary_axis = "World",
        tags         = ("world_probe", "degraded_observation", "no_position_info",
                        "contrastive_with_mechanics"),
        rationale    = (
            "World-axis probe: identical mechanics and object layout to "
            "transfer_wall_push, but non-selected object positions are hidden "
            "(observe_positions=False), preventing position-based discovery even "
            "when the push mechanic succeeds."
        ),
    ),
})


# ---------------------------------------------------------------------------
# AxisCoverage: honest per-axis coverage declaration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisCoverage:
    """
    Honest declaration of how well KA59 covers one capability axis.

    Fields
    ------
    axis           : "World" | "Goal" | "Mechanics" | "Feedback"
    level          : "strong" | "partial" | "none"
    notes          : honest narrative (no internal tag strings)
    is_implemented : True if at least one testable scenario exists for this axis
    """
    axis:           str
    level:          str    # "strong" | "partial" | "none"
    notes:          str
    is_implemented: bool


KA59_AXIS_COVERAGE: dict[str, AxisCoverage] = {

    "Mechanics": AxisCoverage(
        axis           = "Mechanics",
        level          = "strong",
        notes          = (
            "KA59's primary contribution. Five scenarios probe hidden transition "
            "rules: the wall-transfer asymmetry (direct-vs-push collision "
            "difference), push-chain propagation, and direct blocking. "
            "The contrastive pair (transfer_wall_push / solid_wall_push_blocked) "
            "directly implements the Mechanics knockout condition."
        ),
        is_implemented = True,
    ),

    "World": AxisCoverage(
        axis           = "World",
        level          = "partial",
        notes          = (
            "One World-probe scenario is implemented: transfer_wall_push_world_blind "
            "hides non-selected object positions (observe_positions=False), "
            "degrading the agent's spatial observability. Full occlusion or "
            "multi-level observability degradation is deferred to future work."
        ),
        is_implemented = True,
    ),

    "Goal": AxisCoverage(
        axis           = "Goal",
        level          = "none",
        notes          = (
            "KA59 currently evaluates no explicit win condition. Episodes terminate "
            "by stamina exhaustion only. Goal clarity (knowing what configuration "
            "to achieve) is not independently varied. This axis is deferred; "
            "adding a goal-signal scaffold is the natural next step."
        ),
        is_implemented = False,
    ),

    "Feedback": AxisCoverage(
        axis           = "Feedback",
        level          = "none",
        notes          = (
            "Per-step feedback is fixed at binary moved/not-moved (inferred from "
            "position delta). Feedback richness — e.g. richer causal explanation "
            "vs. no feedback at all — is not independently varied. "
            "Degraded-feedback variants are deferred to future work."
        ),
        is_implemented = False,
    ),
}
