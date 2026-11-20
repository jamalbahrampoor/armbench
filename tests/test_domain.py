"""Unit tests for the domain models. No physics, so they run in milliseconds."""

import pytest
from pydantic import ValidationError

from armbench.domain import MoveResult, Vec3, Workspace


def test_distance_between_points():
    assert Vec3(x=0, y=0, z=0).distance_to(Vec3(x=3, y=4, z=0)) == pytest.approx(5.0)


def test_vec3_is_immutable():
    v = Vec3(x=1, y=2, z=3)
    with pytest.raises(ValidationError):
        v.x = 10  # frozen model: mutation is a bug we want to catch early


@pytest.mark.parametrize(
    ("point", "inside"),
    [
        (Vec3(x=0.5, y=0.0, z=0.3), True),  # centre
        (Vec3(x=0.25, y=-0.40, z=0.02), True),  # exact corner: boundary is inclusive
        (Vec3(x=0.80, y=0.0, z=0.3), False),  # too far forward
        (Vec3(x=0.5, y=0.0, z=-0.01), False),  # below the floor
        (Vec3(x=0.5, y=0.5, z=0.3), False),  # too far sideways
    ],
)
def test_workspace_contains(point, inside):
    assert Workspace().contains(point) is inside


def test_move_result_rejects_negative_error():
    with pytest.raises(ValidationError):
        MoveResult(success=True, final_error_m=-0.1, steps=0)
