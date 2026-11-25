"""Tests for the agent loop with a FAKE Claude: scripted, free, deterministic.

What we test here is OUR code: does the loop run tools, send results back with the
right ids, stop when Claude is done, and stop when Claude never finishes?
Whether the REAL Claude makes good decisions is a different question, answered by
evals (Sprint 3), not unit tests.
"""

from fake_llm import FakeClient, reply, text, tool_use

from armbench.agent import run_agent
from armbench.sim.world import CUBE_HALF_SIZE, CUBE_NAME


def test_loop_ends_when_claude_stops_asking_for_tools(sim):
    client = FakeClient([reply(text("Nothing to do."))])
    run = run_agent("hello", sim, client)
    assert run.stop_reason == "end_turn"
    assert run.final_text == "Nothing to do."
    assert run.llm_calls == 1
    assert run.tool_calls == []


def test_scripted_pick_and_place_moves_the_real_cube(sim):
    """A fake Claude makes perfect decisions; the REAL sim must carry them out."""
    cube = sim.get_state().objects[CUBE_NAME].position
    goal_x, goal_y = 0.4, 0.25
    client = FakeClient(
        [
            reply(tool_use("get_world_state")),
            reply(tool_use("move_to", x=cube.x, y=cube.y, z=0.15)),
            reply(tool_use("move_to", x=cube.x, y=cube.y, z=CUBE_HALF_SIZE)),
            reply(tool_use("grasp")),
            reply(tool_use("move_to", x=goal_x, y=goal_y, z=0.15)),
            reply(tool_use("release")),
            reply(text("Done: the cube is on the left.")),
        ]
    )

    run = run_agent("put the red cube on the left", sim, client)

    assert run.stop_reason == "end_turn"
    assert [c.name for c in run.tool_calls] == [
        "get_world_state",
        "move_to",
        "move_to",
        "grasp",
        "move_to",
        "release",
    ]
    assert not any(c.result.is_error for c in run.tool_calls)
    final = sim.get_state().objects[CUBE_NAME].position  # ground truth, not Claude's word
    assert abs(final.x - goal_x) < 0.05 and abs(final.y - goal_y) < 0.05


def test_tool_result_is_sent_back_with_matching_id(sim):
    request = tool_use("get_world_state")
    client = FakeClient([reply(request), reply(text("ok"))])
    run_agent("look", sim, client)

    second_call_messages = client.messages.requests[1]["messages"]
    tool_result = second_call_messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == request.id


def test_errors_are_passed_back_to_claude(sim):
    client = FakeClient([reply(tool_use("move_to", x=2.0, y=0.0, z=0.3)), reply(text("Sorry."))])
    run_agent("go far", sim, client)

    tool_result = client.messages.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "outside the workspace" in tool_result["content"]


def test_parallel_tool_calls_all_get_answered(sim):
    client = FakeClient(
        [reply(tool_use("get_world_state"), tool_use("get_world_state")), reply(text("ok"))]
    )
    run_agent("look twice", sim, client)
    results = client.messages.requests[1]["messages"][-1]["content"]
    assert len(results) == 2


def test_step_budget_stops_a_looping_agent(sim):
    """Claude keeps asking for tools forever; the loop must stop anyway."""
    endless = [reply(tool_use("get_world_state")) for _ in range(10)]
    run = run_agent("loop forever", sim, FakeClient(endless), max_steps=3)
    assert run.stop_reason == "max_steps"
    assert run.llm_calls == 3
    assert len(run.tool_calls) == 3


def test_token_usage_is_summed(sim):
    client = FakeClient([reply(tool_use("get_world_state")), reply(text("ok"))])
    run = run_agent("look", sim, client)
    assert run.input_tokens == 200  # 2 calls x 100 in the fake
    assert run.output_tokens == 40
