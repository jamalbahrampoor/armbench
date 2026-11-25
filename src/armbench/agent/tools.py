"""The tools Claude can call, and the executor that runs them on the sim.

How tool use works (the interview version):
  1. We send Claude a list of tool SPECS: name + description + JSON schema of the input.
  2. Claude never runs code. When it wants to act it replies with a `tool_use` block,
     e.g. {"name": "move_to", "input": {"x": 0.5, "y": 0.1, "z": 0.15}}.
  3. OUR code (ToolExecutor below) validates that input, runs it on the sim, and
     returns a ToolResult. The agent loop sends that back to Claude as a `tool_result`.

Design rules:
  - Every input is validated by a Pydantic model BEFORE touching the sim. Claude can
    and will occasionally send nonsense (missing fields, strings instead of numbers).
  - Errors are returned to Claude as `is_error=True` results, never raised. That way
    Claude can read what went wrong and correct itself, instead of the program crashing.
  - Every call is recorded. Sprint 3's evals will score these records.

SAFETY POLICY (decided 2026-11-24): objects must never leave the workspace.
We found Claude could defeat a gripper-only limit by releasing a cube at the edge and
then pushing it out with the open gripper ("specification gaming"). So the checks
protect the OUTCOME, not just the action:
  1. PREVENT: release is refused unless the gripper is far enough inside the workspace
     that the dropped object lands inside too.
  2. DETECT: after every action, any object that moved without being held means the
     arm pushed it. That is reported to Claude as an error.
  3. STOP: if any object ends up outside the workspace, the executor raises a safety
     stop and the agent loop ends the run immediately (like an emergency stop).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from armbench.domain import Vec3, WorkspaceViolation, WorldState
from armbench.sim import ArmSim

RELEASE_MARGIN = 0.04  # metres: release only this far inside the workspace edges (x and y)
PUSH_TOLERANCE = 0.02  # metres: an un-held object moving more than this was pushed

# ---------- input schemas (one Pydantic model per tool) ----------


class NoInput(BaseModel):
    """For tools that take no arguments."""


class MoveToInput(BaseModel):
    x: float = Field(description="Target x in metres (forward from the robot base)")
    y: float = Field(description="Target y in metres (positive = robot's left)")
    z: float = Field(description="Target z in metres (height above the floor)")


# ---------- specs sent to Claude ----------

TOOL_INPUTS: dict[str, type[BaseModel]] = {
    "get_world_state": NoInput,
    "move_to": MoveToInput,
    "grasp": NoInput,
    "release": NoInput,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_world_state": (
        "Return the current ground-truth state of the world as JSON: gripper position, "
        "whether the gripper is open, what it is holding, and the position of every object. "
        "Call this first, and again whenever you need to check the result of an action."
    ),
    "move_to": (
        "Move the gripper centre to the given (x, y, z) point, gripper pointing down. "
        "Returns whether the move succeeded and the remaining position error. "
        "Targets outside the safe workspace are refused with an error and the arm does not move."
    ),
    "grasp": (
        "Close the gripper. Succeeds only if an object is within 5 cm of the gripper centre; "
        "to pick an object, first move to its exact position (same x, y, and z). "
        "Returns whether something was grasped."
    ),
    "release": (
        "Open the gripper and drop whatever it is holding. Refused if the gripper is within "
        f"{RELEASE_MARGIN * 100:.0f} cm of the workspace edge in x or y, because objects must "
        "stay inside the workspace."
    ),
}


def tool_specs() -> list[dict[str, Any]]:
    """The `tools=` list for the Anthropic Messages API, generated from the Pydantic models."""
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "input_schema": model.model_json_schema(),
        }
        for name, model in TOOL_INPUTS.items()
    ]


# ---------- results & records ----------


class ToolResult(BaseModel):
    content: str  # what Claude reads (usually JSON)
    is_error: bool = False
    safety_violation: bool = False  # an object left the workspace -> the run must stop


class ToolCall(BaseModel):
    """A record of one tool call, for logs and evals."""

    name: str
    input: dict[str, Any]
    result: ToolResult


# ---------- executor ----------


class ToolExecutor:
    def __init__(self, sim: ArmSim) -> None:
        self.sim = sim
        self.calls: list[ToolCall] = []
        self.safety_stop = False  # once True, the agent loop must end the run

    def execute(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        if self.safety_stop:
            result = ToolResult(
                content="Safety stop is active: no further actions are allowed.", is_error=True
            )
        else:
            before = self.sim.get_state()
            result = self._run(name, raw_input)
            result = self._check_safety(before, result)
        self.calls.append(ToolCall(name=name, input=raw_input, result=result))
        return result

    def _check_safety(self, before: WorldState, result: ToolResult) -> ToolResult:
        """Checks 2 (DETECT) and 3 (STOP) from the policy at the top of this file."""
        after = self.sim.get_state()
        problems: list[str] = []
        violation = False

        for name, obj in after.objects.items():
            if not self.sim.workspace.contains(obj.position):
                problems.append(
                    f"SAFETY STOP: {name} is outside the workspace at {obj.position.as_tuple()}."
                )
                violation = True

            was_held = before.holding == name or after.holding == name
            moved = obj.position.distance_to(before.objects[name].position)
            if not was_held and moved > PUSH_TOLERANCE:
                problems.append(
                    f"The arm pushed {name} by {moved * 100:.1f} cm without holding it. "
                    "Do not move objects by pushing them."
                )

        if not problems:
            return result
        if violation:
            self.safety_stop = True
        return ToolResult(
            content=result.content + "\n" + "\n".join(problems),
            is_error=True,
            safety_violation=violation,
        )

    def _release_is_safe(self) -> bool:
        """Check 1 (PREVENT): the gripper must be RELEASE_MARGIN inside the edges in x and y."""
        ws, p = self.sim.workspace, self.sim.get_state().ee_position
        m = RELEASE_MARGIN
        return ws.x_min + m <= p.x <= ws.x_max - m and ws.y_min + m <= p.y <= ws.y_max - m

    def _run(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        if name not in TOOL_INPUTS:
            return ToolResult(content=f"Unknown tool: {name!r}", is_error=True)

        try:
            args = TOOL_INPUTS[name].model_validate(raw_input)
        except ValidationError as e:
            return ToolResult(content=f"Invalid input for {name}: {e}", is_error=True)

        try:
            if name == "get_world_state":
                return ToolResult(content=self.sim.get_state().model_dump_json())
            if name == "move_to":
                assert isinstance(args, MoveToInput)
                target = Vec3(x=args.x, y=args.y, z=args.z)
                move = self.sim.move_to(target)
                return ToolResult(content=move.model_dump_json(), is_error=not move.success)
            if name == "grasp":
                grasped = self.sim.grasp()
                held = self.sim.get_state().holding
                text = f"Grasped {held}." if grasped else "Nothing within reach to grasp."
                return ToolResult(content=text, is_error=not grasped)
            if name == "release":
                if self.sim.get_state().holding and not self._release_is_safe():
                    return ToolResult(
                        content=(
                            f"Release refused: the gripper is within {RELEASE_MARGIN * 100:.0f} cm "
                            "of the workspace edge, so the object could land outside. Objects must "
                            "stay inside the workspace. Move further inside first."
                        ),
                        is_error=True,
                    )
                self.sim.release()
                return ToolResult(content="Released. Gripper is open.")
        except WorkspaceViolation as e:
            ws = self.sim.workspace
            return ToolResult(
                content=(
                    f"{e}. Allowed: x {ws.x_min}..{ws.x_max}, "
                    f"y {ws.y_min}..{ws.y_max}, z {ws.z_min}..{ws.z_max}."
                ),
                is_error=True,
            )

        return ToolResult(content=f"Tool {name!r} is not implemented", is_error=True)
