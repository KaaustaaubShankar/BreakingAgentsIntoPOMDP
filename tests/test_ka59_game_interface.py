"""
Tests for ka59_ref.game_interface — structured state + feedback builder.

These tests lock the dict shape that ka59_ref/experiment.py (LLM harness) and
env4/experiment.py both consume, so KA59 and BP35 runs can land in the same
ablation table.
"""

from __future__ import annotations

from ka59_ref.env import KA59BlindEnv, MOVE_RIGHT, SELECT
from ka59_ref.game_interface import (
    build_feedback_easy,
    get_structured_state,
)
from ka59_ref.scenarios import SCENARIOS


STEP = 3


def _reset_to(scenario_name: str):
    env = KA59BlindEnv()
    spec = SCENARIOS[scenario_name]
    env.reset(spec)
    return env, spec


def test_structured_state_has_env4_compatible_top_level_keys() -> None:
    env, spec = _reset_to("transfer_wall_push")
    state = get_structured_state(
        env,
        scenario_name="transfer_wall_push",
        step_budget=spec["steps"],
        target_x=STEP * 2,
    )
    assert set(state.keys()) >= {
        "player",
        "movement",
        "level",
        "resources",
        "objects",
        "semantic_grid",
        "game_state",
    }


def test_structured_state_player_reports_selected_position() -> None:
    env, spec = _reset_to("transfer_wall_push")
    state = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)
    assert state["player"]["position"] == [0, 0]
    assert state["player"]["id"] == "sel"


def test_structured_state_objects_partitioned_by_kind() -> None:
    env, spec = _reset_to("transfer_wall_push")
    state = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)
    assert len(state["objects"]["controllables"]) == 1
    assert len(state["objects"]["blocks"]) == 1
    assert len(state["objects"]["walls"]) == 1
    assert state["objects"]["controllables"][0]["id"] == "sel"
    assert state["objects"]["blocks"][0]["id"] == "pb"
    assert state["objects"]["walls"][0]["id"] == "wt"


def test_structured_state_movement_right_shows_adjacent_block() -> None:
    env, spec = _reset_to("transfer_wall_push")
    state = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)
    # Right of selected at x=0 is the pushable block at x=STEP. The heuristic
    # says right_blocked=True (wall or block adjacent); adjacent_right captures
    # the neighbor id so the agent can see what's there.
    assert state["movement"]["right_blocked"] is True
    assert "pb" in state["movement"]["adjacent_right_ids"]


def test_structured_state_world_hard_hides_non_selected_positions() -> None:
    env = KA59BlindEnv()
    spec = dict(SCENARIOS["transfer_wall_push"])
    spec["observe_positions"] = False
    env.reset(spec)
    state = get_structured_state(env, "transfer_wall_push_world_blind", spec["steps"], STEP * 2)
    # Non-selected objects must have position [0, 0] (blinded) while selected
    # keeps its true position.
    assert state["player"]["position"] == [0, 0]  # selected starts at 0,0 anyway
    for obj in state["objects"]["blocks"] + state["objects"]["walls"]:
        assert obj["position"] == [0, 0], f"world-blind leak: {obj}"


def test_structured_state_resources_track_budget() -> None:
    env, spec = _reset_to("transfer_wall_push")
    state = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)
    assert state["resources"]["step_budget"] == spec["steps"]
    assert state["resources"]["steps_remaining"] == spec["steps"]
    assert state["resources"]["steps_used"] == 0

    env.step(MOVE_RIGHT)
    state2 = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)
    assert state2["resources"]["steps_used"] == 1
    assert state2["resources"]["steps_remaining"] == spec["steps"] - 1


def test_structured_state_win_when_pushable_crosses_target() -> None:
    env, spec = _reset_to("transfer_wall_push")
    # MOVE_RIGHT pushes pb through WALL_TRANSFER and advances selected to x>=STEP.
    env.step(MOVE_RIGHT)
    # At least one pushable (selected or pb) is now at x>=STEP → WIN.
    state = get_structured_state(env, "transfer_wall_push", spec["steps"], target_x=STEP)
    assert state["game_state"] == "WIN"
    # No pushable can reach x=30 in this scenario → still in progress.
    state_far = get_structured_state(env, "transfer_wall_push", spec["steps"], target_x=STEP * 10)
    assert state_far["game_state"] == "IN_PROGRESS"


def test_structured_state_uses_true_positions_for_win_even_in_world_hard() -> None:
    """WORLD_HARD blinds the observation but the harness still needs correct
    win/loss determination. get_structured_state must read engine-true positions
    for game_state, not the blinded ObjectView positions."""
    env = KA59BlindEnv()
    spec = dict(SCENARIOS["transfer_wall_push"])
    spec["observe_positions"] = False
    env.reset(spec)
    env.step(MOVE_RIGHT)  # pushes pb through the wall

    state = get_structured_state(env, "transfer_wall_push_world_blind", spec["steps"], target_x=STEP)
    # Observation shows block at [0,0] (blinded) but game_state is still WIN
    # because the harness checks true engine positions.
    for obj in state["objects"]["blocks"]:
        assert obj["position"] == [0, 0], "blinding should still apply to the obs dict"
    assert state["game_state"] == "WIN"


def test_structured_state_no_win_when_wall_solid_blocks_push() -> None:
    """Contrastive scenario: solid_wall_push_blocked is the Mechanics loss
    baseline — no pushable can cross the solid wall, so game_state never WINs
    for any target_x past the wall."""
    env, spec = _reset_to("solid_wall_push_blocked")
    for _ in range(spec["steps"]):
        env.step(MOVE_RIGHT)
    state = get_structured_state(env, "solid_wall_push_blocked", spec["steps"], target_x=STEP * 2)
    assert state["game_state"] != "WIN"


def test_feedback_easy_reports_selected_delta() -> None:
    env, spec = _reset_to("transfer_wall_push")
    prev = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)
    env.step(MOVE_RIGHT)
    curr = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)

    text = build_feedback_easy(prev, curr, "MOVE_RIGHT")
    assert "MOVE_RIGHT" in text
    assert "[0, 0]" in text  # previous position
    assert f"[{STEP}, 0]" in text  # new position


def test_feedback_easy_reports_block_delta_and_wall_transfer_hint() -> None:
    """
    In transfer_wall_push the first MOVE_RIGHT causes the block (pb) to pass
    through the wall_transfer (wt). Feedback should surface the block moving
    so the agent can infer the wall-transfer mechanic.
    """
    env, spec = _reset_to("transfer_wall_push")
    prev = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)
    env.step(MOVE_RIGHT)
    curr = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)

    text = build_feedback_easy(prev, curr, "MOVE_RIGHT")
    # Block must be reported as having moved (that's how the agent learns push).
    assert "pb" in text or "block" in text.lower()


def test_feedback_easy_reports_no_change_when_blocked() -> None:
    env, _ = _reset_to("transfer_wall_direct_block")
    prev = get_structured_state(env, "transfer_wall_direct_block", 10, STEP * 2)
    env.step(MOVE_RIGHT)  # immediately blocked by wall_transfer, no block to push
    curr = get_structured_state(env, "transfer_wall_direct_block", 10, STEP * 2)

    text = build_feedback_easy(prev, curr, "MOVE_RIGHT")
    assert ("did not" in text.lower()) or ("blocked" in text.lower())


def test_feedback_reports_win_on_goal_reached() -> None:
    env, spec = _reset_to("transfer_wall_push")
    prev = get_structured_state(env, "transfer_wall_push", spec["steps"], target_x=STEP)
    env.step(MOVE_RIGHT)
    curr = get_structured_state(env, "transfer_wall_push", spec["steps"], target_x=STEP)
    text = build_feedback_easy(prev, curr, "MOVE_RIGHT")
    assert "goal" in text.lower() or "win" in text.lower()


def test_structured_state_select_changes_player_id() -> None:
    env, spec = _reset_to("transfer_wall_push")
    env.step(SELECT("pb"))
    state = get_structured_state(env, "transfer_wall_push", spec["steps"], STEP * 2)
    assert state["player"]["id"] == "pb"
    # And selected-flag on that object must be true.
    pb_view = next(o for o in state["objects"]["blocks"] if o["id"] == "pb")
    assert pb_view["is_selected"] is True
