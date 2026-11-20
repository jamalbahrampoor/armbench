"""Domain models: the shared language of the whole project.

These classes know nothing about PyBullet or Claude. The sim produces them,
the agent consumes them, the API serialises them, the evals score them.
Keeping them separate is what lets each layer be tested on its own.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Vec3(BaseModel):
    """A 3D point in metres, in the world frame."""

    model_config = ConfigDict(frozen=True)  # immutable -> safe to share, hashable

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: Vec3) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5


class Workspace(BaseModel):
    """Axis-aligned box the end effector is allowed to move in.

    SAFETY RULE: this is enforced in code (the sim refuses targets outside it).
    Later the AI agent will be told about it in its prompt too, but the prompt
    is a hint; this check is the guarantee.
    """

    model_config = ConfigDict(frozen=True)

    x_min: float = 0.25
    x_max: float = 0.70
    y_min: float = -0.40
    y_max: float = 0.40
    z_min: float = 0.02
    z_max: float = 0.60

    def contains(self, p: Vec3) -> bool:
        return (
            self.x_min <= p.x <= self.x_max
            and self.y_min <= p.y <= self.y_max
            and self.z_min <= p.z <= self.z_max
        )


class ObjectState(BaseModel):
    name: str
    position: Vec3


class WorldState(BaseModel):
    """Ground truth snapshot of the sim. This is what evals will score against."""

    ee_position: Vec3 = Field(description="End-effector (gripper centre) position")
    gripper_open: bool
    holding: str | None = Field(default=None, description="Name of held object, if any")
    objects: dict[str, ObjectState]
    sim_time: float = Field(ge=0.0, description="Seconds of simulated time since reset")


class MoveResult(BaseModel):
    success: bool
    final_error_m: float = Field(ge=0.0, description="Distance from target after the move")
    steps: int = Field(ge=0)
    message: str = ""


class WorkspaceViolation(ValueError):
    """Raised when a command targets a point outside the allowed workspace."""
