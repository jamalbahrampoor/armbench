"""The agent loop: Claude thinks, our code acts, repeat until Claude is done.

    user instruction
          │
          ▼
    ┌─► Claude ──── stop_reason == "tool_use"? ── no ──► done (final text)
    │      │ yes
    │      ▼
    │  ToolExecutor runs each tool_use block on the sim
    │      │
    └──────┘  results sent back as tool_result blocks

Key design choices (be ready to explain these):
  - The Anthropic client is PASSED IN, not created here ("dependency injection").
    Tests pass a fake client with scripted replies: no API key, no cost, no randomness.
  - max_steps is a hard stop. An LLM can loop forever (retrying a failing move, say);
    production agents always have a budget.
  - We return a structured AgentRun, not just text, so evals can score it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

from armbench.agent.tools import ToolCall, ToolExecutor, tool_specs
from armbench.sim import ArmSim

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4096  # per reply; was 1024, which cut Haiku off mid-reply (stop_reason=max_tokens)


def system_prompt(sim: ArmSim) -> str:
    ws = sim.workspace
    return f"""You control a robot arm in a physics simulation using tools.

Coordinate frame (metres): the robot base is at the origin. +x points forward,
+y points to the robot's LEFT, +z points up. The floor is at z = 0.

Safe workspace (moves outside are refused):
  x {ws.x_min}..{ws.x_max}, y {ws.y_min}..{ws.y_max}, z {ws.z_min}..{ws.z_max}

Safety rule: every object must stay inside the workspace at all times. Never push
objects with the gripper. If a task would put an object outside the workspace, refuse
it and explain why. Breaking this rule triggers an automatic safety stop.

Objects are small cubes (5 cm). An object's position is its centre, so a cube resting
on the floor has z = 0.025.

Guidelines:
- Start by calling get_world_state. Never guess positions.
- To pick an object: move above it (z = 0.15), move down to its exact position, grasp.
- Carry objects at z = 0.15 to avoid knocking anything over.
- To place an object: move above the target at z = 0.15, then release.
  Only release when the task asks you to put something down. If the task says to hold
  or keep an object, finish with it still grasped.
- After finishing, call get_world_state to verify the result, then reply with a short
  summary of what you did. If the task is impossible or unsafe, say so and stop."""


class MessagesAPI(Protocol):
    """The one method we need from the Anthropic client. The fake client implements it too."""

    def create(self, **kwargs: Any) -> Any: ...


class LLMClient(Protocol):
    messages: MessagesAPI


class AgentRun(BaseModel):
    instruction: str
    final_text: str
    # "end_turn"    = Claude finished on its own (the only normal ending)
    # "max_steps"   = we stopped it: step budget used up
    # "max_tokens"  = a reply was cut off: treated as a failure, never as success
    # "safety_stop" = an object left the workspace: the run was aborted
    stop_reason: str
    llm_calls: int
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int

    @property
    def completed(self) -> bool:
        """True only if Claude ended the run itself. Says nothing about task success."""
        return self.stop_reason == "end_turn"


STOP_MESSAGES = {
    "max_steps": "Stopped: step budget exhausted.",
    "max_tokens": "Stopped: Claude's reply was cut off (max_tokens reached).",
    "safety_stop": "Stopped: SAFETY STOP, an object left the workspace.",
}


def run_agent(
    instruction: str,
    sim: ArmSim,
    client: LLMClient,
    model: str = DEFAULT_MODEL,
    max_steps: int = 25,
    on_tool_call: Callable[[ToolCall], None] | None = None,
) -> AgentRun:
    executor = ToolExecutor(sim)
    messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
    input_tokens = output_tokens = 0
    final_text = ""

    def finish(stop_reason: str, llm_calls: int) -> AgentRun:
        text = final_text or STOP_MESSAGES.get(stop_reason, "")
        if stop_reason in STOP_MESSAGES and final_text:
            text = f"{STOP_MESSAGES[stop_reason]} Last message: {final_text}"
        return AgentRun(
            instruction=instruction,
            final_text=text,
            stop_reason=stop_reason,
            llm_calls=llm_calls,
            tool_calls=executor.calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt(sim),
            tools=tool_specs(),
            messages=messages,
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        # Claude's whole reply (text + tool requests) goes into the history.
        messages.append({"role": "assistant", "content": response.content})
        final_text = " ".join(b.text for b in response.content if b.type == "text").strip()

        if response.stop_reason != "tool_use":
            return finish(response.stop_reason, step)

        # Answer EVERY tool_use block (the API requires one result per request), but once
        # one action fails, skip the rest of the batch: they were planned assuming success.
        tool_results = []
        batch_failed = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            if batch_failed:
                content, is_error = "Skipped: an earlier action in this batch failed.", True
            else:
                result = executor.execute(block.name, block.input)
                if on_tool_call is not None:
                    on_tool_call(executor.calls[-1])
                content, is_error = result.content, result.is_error
                batch_failed = result.is_error
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,  # must match the request's id
                    "content": content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

        if executor.safety_stop:
            return finish("safety_stop", step)

    return finish("max_steps", max_steps)
