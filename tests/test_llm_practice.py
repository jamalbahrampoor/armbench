import pytest
from fake_llm import FakeClient, reply, text, tool_use

from armbench.agent import run_agent
from armbench.agent.tools import PUSH_TOLERANCE, ToolExecutor
from armbench.domain import Vec3


@pytest.fixture
def executor(stub):
    return ToolExecutor(stub)


def test_e1_grasp_from_too_far_away(stub, executor):
    executor.execute("move_to", {"x": 0.5, "y": 0.0, "z": 0.275})
    assert executor.execute("grasp", {}).is_error is True
    assert stub.held is None


def test_e2_bad_input_does_not_move_the_robot(stub, executor):
    pos_1 = stub.ee
    result = executor.execute("move_to", {"x": 0.5, "y": 0.0, "z": "high"})
    assert result.is_error is True
    assert "Invalid input for move_to" in result.content
    assert pos_1 == stub.ee


@pytest.mark.parametrize(
    ("y_param", "is_error_result"), [(PUSH_TOLERANCE / 2, False), (PUSH_TOLERANCE * 1.5, True)]
)
def test_e3_boundary_values_on_the_push_back(stub, executor, y_param, is_error_result):

    pos_cube_1 = stub.cube
    stub.push_cube_to = Vec3(x=pos_cube_1.x, y=pos_cube_1.y + y_param, z=pos_cube_1.z)
    result = executor.execute("move_to", {"x": pos_cube_1.x, "y": pos_cube_1.y + y_param, "z": 0.3})
    assert result.is_error is is_error_result


def test_e4_claude_gets_an_error_and_gives_up_politely(stub):
    client = FakeClient([reply(tool_use("grasp")), reply(text("I could not reach the cube"))])
    run = run_agent("grasp now with best effort", stub, client)

    assert run.completed is True
    tool_result = client.messages.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert len(run.tool_calls) == 1
    assert run.tool_calls[0].result.is_error is True


def test_e5_check_what_we_actually_send_to_claude(stub):
    client = FakeClient([reply(text("where should I move to? here looks out !!"))])
    run_agent("move where I say", stub, client)
    tool_result = client.messages.requests[0]
    limit_in_prompt_x_max = stub.workspace.x_max
    assert str(limit_in_prompt_x_max) in tool_result["system"]
    names = {s["name"] for s in tool_result["tools"]}

    assert names == {"get_world_state", "grasp", "move_to", "release"}
