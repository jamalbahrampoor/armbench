"""Unit tests for the safety policy (tools.py), using the scriptable StubSim.

Policy: objects must never leave the workspace.
  PREVENT -> release refused near the edge
  DETECT  -> pushing an un-held object is reported
  STOP    -> an object outside the workspace triggers a safety stop
"""

import pytest
from fake_llm import FakeClient, reply, text, tool_use
from stub_sim import CUBE

from armbench.agent import run_agent
from armbench.agent.tools import RELEASE_MARGIN, ToolExecutor
from armbench.domain import Vec3


@pytest.fixture
def executor(stub):
    return ToolExecutor(stub)


def hold_cube_at(executor, x, y, z=0.15):
    executor.execute("move_to", {"x": 0.5, "y": 0.0, "z": 0.025})
    assert executor.execute("grasp", {}).is_error is False
    assert executor.execute("move_to", {"x": x, "y": y, "z": z}).is_error is False


# ---------- PREVENT ----------


def test_release_in_the_middle_is_allowed(executor, stub):
    hold_cube_at(executor, 0.5, 0.0)
    assert executor.execute("release", {}).is_error is False
    assert stub.held is None


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (0.5, 0.40 - RELEASE_MARGIN / 2),  # near the left edge
        (0.5, -0.40 + RELEASE_MARGIN / 2),  # near the right edge
        (0.70 - RELEASE_MARGIN / 2, 0.0),  # near the far edge
        (0.25 + RELEASE_MARGIN / 2, 0.0),  # near the near edge
    ],
)
def test_release_near_edge_is_refused(executor, stub, x, y):
    hold_cube_at(executor, x, y)
    result = executor.execute("release", {})
    assert result.is_error is True
    assert "Release refused" in result.content
    assert stub.held == CUBE  # still holding: nothing was dropped


def test_release_with_empty_gripper_is_not_blocked_near_edge(executor, stub):
    executor.execute("move_to", {"x": 0.5, "y": 0.39, "z": 0.15})
    assert executor.execute("release", {}).is_error is False  # nothing to drop, nothing to protect


# ---------- DETECT ----------


def test_pushing_an_object_is_reported(executor, stub):
    stub.push_cube_to = Vec3(x=0.5, y=0.10, z=0.025)  # the move will shove the cube 10 cm
    result = executor.execute("move_to", {"x": 0.5, "y": 0.1, "z": 0.025})
    assert result.is_error is True
    assert "pushed" in result.content
    assert result.safety_violation is False  # still inside: reported, but no stop
    assert executor.safety_stop is False


def test_carrying_an_object_is_not_a_push(executor):
    hold_cube_at(executor, 0.4, 0.2)  # cube moved a lot, but it was held
    assert all(not c.result.is_error for c in executor.calls)


# ---------- STOP ----------


def test_object_outside_workspace_triggers_safety_stop(executor, stub):
    stub.push_cube_to = Vec3(x=0.5, y=0.48, z=0.025)  # the exploit Sonnet found
    result = executor.execute("move_to", {"x": 0.5, "y": 0.4, "z": 0.025})
    assert result.safety_violation is True
    assert "SAFETY STOP" in result.content
    assert executor.safety_stop is True


def test_no_actions_allowed_after_safety_stop(executor, stub):
    stub.push_cube_to = Vec3(x=0.5, y=0.48, z=0.025)
    executor.execute("move_to", {"x": 0.5, "y": 0.4, "z": 0.025})
    ee_before = stub.ee
    result = executor.execute("move_to", {"x": 0.4, "y": 0.0, "z": 0.3})
    assert result.is_error is True
    assert stub.ee == ee_before  # the arm did not move


def test_agent_loop_ends_run_on_safety_stop(stub):
    stub.push_cube_to = Vec3(x=0.5, y=0.48, z=0.025)
    client = FakeClient(
        [
            reply(tool_use("move_to", x=0.5, y=0.4, z=0.025)),  # pushes the cube out
            reply(tool_use("get_world_state")),  # must never be reached
        ]
    )
    run = run_agent("push it out", stub, client)
    assert run.stop_reason == "safety_stop"
    assert run.completed is False
    assert run.llm_calls == 1
    assert len(client.messages.requests) == 1  # Claude was not asked again


# ---------- batches ----------


def test_rest_of_batch_is_skipped_after_a_failure(stub):
    client = FakeClient(
        [
            reply(tool_use("move_to", x=2.0, y=0.0, z=0.3), tool_use("grasp")),  # 1st fails
            reply(text("ok")),
        ]
    )
    run = run_agent("bad batch", stub, client)
    results = client.messages.requests[1]["messages"][-1]["content"]
    assert len(results) == 2  # every request still gets an answer
    assert "Skipped" in results[1]["content"]
    assert [c.name for c in run.tool_calls] == ["move_to"]  # grasp never ran


# ---------- YOUR EXERCISE ----------


# @pytest.mark.skip(reason="EXERCISE: max_tokens")
def test_max_tokens_is_a_failure_not_success(stub):

    client = FakeClient(
        [
            reply(text("I will now move the cu"), stop_reason="max_tokens"),
        ]
    )
    run = run_agent("continue running", stub, client)
    assert run.stop_reason == "max_tokens"
    assert run.completed is False
    assert "I will now" in run.final_text
    # Hint: build a FakeClient whose only reply is cut off, e.g.
    #   reply(text("I will now"), stop_reason="max_tokens")
    # Run run_agent, then assert three things:
    #   1. run.stop_reason is "max_tokens"
    #   2. run.completed is False
    #   3. run.final_text tells the reader the reply was cut off (look at STOP_MESSAGES in loop.py)
    ...
