"""A tiny scriptable stand-in for ArmSim, for testing the SAFETY LOGIC in isolation.

Real physics can't easily put an object in an exact spot on demand. This stub can:
tell it "the next move pushes the cube to (x, y, z)" and the executor's checks can be
tested deterministically. Real-physics versions of the same scenarios live in
test_safety_integration.py.
"""

from __future__ import annotations

from armbench.domain import MoveResult, ObjectState, Vec3, Workspace, WorkspaceViolation, WorldState

CUBE = "red_cube"


class StubSim:
    def __init__(self) -> None:
        self.workspace = Workspace()
        self.ee = Vec3(x=0.5, y=0.0, z=0.3)
        self.cube = Vec3(x=0.5, y=0.0, z=0.025)
        self.held: str | None = None
        self.push_cube_to: Vec3 | None = None  # scripted side effect of the next move_to

    def get_state(self) -> WorldState:
        return WorldState(
            ee_position=self.ee,
            gripper_open=self.held is None,
            holding=self.held,
            objects={CUBE: ObjectState(name=CUBE, position=self.cube)},
            sim_time=0.0,
        )

    def move_to(self, target: Vec3) -> MoveResult:
        if not self.workspace.contains(target):
            raise WorkspaceViolation(f"Target {target.as_tuple()} is outside the workspace")
        self.ee = target
        if self.held:
            self.cube = target
        elif self.push_cube_to is not None:
            self.cube, self.push_cube_to = self.push_cube_to, None
        return MoveResult(success=True, final_error_m=0.0, steps=10)

    def grasp(self) -> bool:
        if self.ee.distance_to(self.cube) <= 0.05:
            self.held = CUBE
            return True
        return False

    def release(self) -> None:
        if self.held:
            self.cube = Vec3(x=self.ee.x, y=self.ee.y, z=0.025)
        self.held = None
