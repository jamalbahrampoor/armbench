"""Integration tests for the simulation. These run real (headless) physics."""

import pytest

from armbench.domain import Vec3, WorkspaceViolation
from armbench.sim import ArmSim
from armbench.sim.world import CUBE_HALF_SIZE, CUBE_NAME, POSITION_TOLERANCE, TIME_STEP

# ---------- reset & determinism ----------


def test_reset_puts_cube_on_the_floor(sim):
    cube = sim.get_state().objects[CUBE_NAME]
    assert cube.position.z == pytest.approx(CUBE_HALF_SIZE, abs=0.005)


def test_reset_starts_with_open_empty_gripper(sim):
    state = sim.get_state()
    assert state.gripper_open is True
    assert state.holding is None


def test_same_seed_gives_same_world():
    """Determinism: the property every other test silently depends on."""
    with ArmSim() as a, ArmSim() as b:
        pos_a = a.reset(seed=42).objects[CUBE_NAME].position
        pos_b = b.reset(seed=42).objects[CUBE_NAME].position
    assert pos_a == pos_b


def test_different_seeds_move_the_cube():
    with ArmSim() as a, ArmSim() as b:
        pos_a = a.reset(seed=1).objects[CUBE_NAME].position
        pos_b = b.reset(seed=2).objects[CUBE_NAME].position
    assert pos_a.distance_to(pos_b) > 0.001


@pytest.mark.parametrize("seed", range(20))
def test_cube_always_starts_inside_workspace(sim, seed):
    """If a seed put the cube out of reach, evals would blame the AI for an impossible task."""
    cube = sim.reset(seed=seed).objects[CUBE_NAME].position
    assert sim.workspace.contains(cube), f"seed {seed} put the cube at {cube}"


@pytest.mark.parametrize("n_steps", [0, 1, 10, 240])
def test_sim_time_advances_with_steps(sim, n_steps):
    time_before = sim.get_state().sim_time
    sim.step(n_steps)
    time_after = sim.get_state().sim_time
    assert time_after - time_before == pytest.approx(n_steps * TIME_STEP)


# ---------- motion & safety ----------


@pytest.mark.parametrize(
    "target",
    [
        Vec3(x=0.5, y=0.0, z=0.3),
        Vec3(x=0.4, y=0.2, z=0.2),
        Vec3(x=0.6, y=-0.2, z=0.15),
    ],
)
def test_move_reaches_target(sim, target):
    result = sim.move_to(target)
    assert result.success, result.message
    assert sim.get_state().ee_position.distance_to(target) < POSITION_TOLERANCE


def test_move_above_cube_aligns_gripper(sim):
    cube = sim.get_state().objects[CUBE_NAME].position
    assert sim.move_to(Vec3(x=cube.x, y=cube.y, z=0.15)).success  # guard

    gripper = sim.get_state().ee_position
    assert gripper.x == pytest.approx(cube.x, abs=0.01)
    assert gripper.y == pytest.approx(cube.y, abs=0.01)


@pytest.mark.parametrize(
    "target",
    [
        Vec3(x=0.3, y=0.02, z=0.61),  # too high
        Vec3(x=0.5, y=0.2, z=-0.01),  # below the floor
        Vec3(x=0.1, y=0.0, z=0.3),  # too close / behind the workspace
        Vec3(x=0.8, y=0.0, z=0.3),  # too far forward
        Vec3(x=0.3, y=-0.41, z=0.3),  # too far right
        Vec3(x=0.4, y=0.41, z=0.2),  # too far left
    ],
)
def test_move_outside_workspace_is_refused_before_moving(sim, target):
    before = sim.get_state().ee_position
    with pytest.raises(WorkspaceViolation, match="outside the workspace"):
        sim.move_to(target)
    assert sim.get_state().ee_position == before  # the arm did not move at all


# ---------- grasping ----------


def test_grasp_fails_when_far_from_cube(sim):
    far_away = Vec3(x=0.3, y=-0.3, z=0.4)  # fixed point, always inside the workspace
    assert sim.move_to(far_away).success  # guard

    assert sim.grasp() is False
    assert sim.get_state().holding is None


def test_release_drops_cube_to_floor(sim):
    cube = sim.get_state().objects[CUBE_NAME].position
    assert sim.move_to(Vec3(x=cube.x, y=cube.y, z=0.15)).success  # hover
    assert sim.move_to(Vec3(x=cube.x, y=cube.y, z=CUBE_HALF_SIZE)).success  # descend

    assert sim.grasp()  # before: held
    assert sim.get_state().holding == CUBE_NAME

    sim.release()  # after: released and resting on the floor
    state = sim.get_state()
    assert state.holding is None
    assert state.objects[CUBE_NAME].position.z == pytest.approx(CUBE_HALF_SIZE, abs=0.005)


def test_pick_and_place(sim):
    """End-to-end: pick the cube and put it somewhere else."""
    cube = sim.get_state().objects[CUBE_NAME].position
    goal = Vec3(x=0.4, y=0.25, z=CUBE_HALF_SIZE)

    assert sim.move_to(Vec3(x=cube.x, y=cube.y, z=0.15)).success  # above the cube
    assert sim.move_to(Vec3(x=cube.x, y=cube.y, z=CUBE_HALF_SIZE)).success  # down
    assert sim.grasp(), "gripper should be close enough to grasp"
    assert sim.get_state().holding == CUBE_NAME
    assert sim.move_to(Vec3(x=goal.x, y=goal.y, z=0.15)).success  # carry
    sim.release()

    final = sim.get_state().objects[CUBE_NAME].position
    assert final.distance_to(goal) < 0.05  # landed within 5 cm of the goal
    assert sim.get_state().holding is None
