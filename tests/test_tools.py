"""Tests for the tool layer: specs and execution. No API calls."""

import json

import pytest

from armbench.agent.tools import ToolExecutor, tool_specs
from armbench.domain import WorldState
from armbench.sim.world import CUBE_HALF_SIZE, CUBE_NAME

# ---------- specs ----------


def test_tool_specs_have_required_fields():
    specs = tool_specs()
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names)), "tool names must be unique"
    for spec in specs:
        assert spec["description"], f"{spec['name']} needs a description: Claude reads it"
        assert spec["input_schema"]["type"] == "object"


def test_move_to_schema_requires_xyz():
    move = next(s for s in tool_specs() if s["name"] == "move_to")
    assert set(move["input_schema"]["required"]) == {"x", "y", "z"}


# ---------- execution ----------


@pytest.fixture
def executor(sim):
    return ToolExecutor(sim)


def test_get_world_state_returns_valid_json(executor):
    result = executor.execute("get_world_state", {})
    assert result.is_error is False
    state = WorldState.model_validate_json(result.content)  # round-trips into our model
    assert CUBE_NAME in state.objects


def test_move_to_succeeds_inside_workspace(executor):
    result = executor.execute("move_to", {"x": 0.5, "y": 0.0, "z": 0.3})
    assert result.is_error is False
    assert json.loads(result.content)["success"] is True


def test_move_outside_workspace_returns_error_not_exception(executor, sim):
    before = sim.get_state().ee_position
    result = executor.execute("move_to", {"x": 2.0, "y": 0.0, "z": 0.3})
    assert result.is_error is True
    assert "Allowed" in result.content  # Claude is told the limits so it can correct itself
    assert sim.get_state().ee_position == before  # and the arm did not move


@pytest.mark.parametrize(
    "bad_input",
    [
        {"x": 0.5, "y": 0.0},  # missing z
        {"x": "far", "y": 0.0, "z": 0.3},  # not a number
        {},  # nothing at all
    ],
)
def test_invalid_input_returns_error(executor, bad_input):
    result = executor.execute("move_to", bad_input)
    assert result.is_error is True
    assert "Invalid input" in result.content


def test_unknown_tool_returns_error(executor):
    result = executor.execute("self_destruct", {})
    assert result.is_error is True


def test_grasp_without_object_is_an_error(executor):
    executor.execute("move_to", {"x": 0.3, "y": -0.3, "z": 0.4})
    assert executor.execute("grasp", {}).is_error is True


def test_pick_via_tools(executor, sim):
    cube = sim.get_state().objects[CUBE_NAME].position
    executor.execute("move_to", {"x": cube.x, "y": cube.y, "z": 0.15})
    executor.execute("move_to", {"x": cube.x, "y": cube.y, "z": CUBE_HALF_SIZE})
    result = executor.execute("grasp", {})
    assert result.is_error is False
    assert sim.get_state().holding == CUBE_NAME


def test_every_call_is_recorded(executor):
    executor.execute("get_world_state", {})
    executor.execute("self_destruct", {})
    assert [c.name for c in executor.calls] == ["get_world_state", "self_destruct"]
    assert executor.calls[1].result.is_error is True
