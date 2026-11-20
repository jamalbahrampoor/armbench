"""PyBullet wrapper for a Franka Panda arm with a single cube.

Design rules (you should be able to defend each one in an interview):

1. No global state. We use a BulletClient per ArmSim instance instead of the
   global `pybullet` module, so two sims (e.g. two parallel tests) never
   interfere with each other.
2. Headless by default (p.DIRECT). The GUI is opt-in. Docker, CI and cloud
   machines have no screen.
3. Deterministic. Fixed time step + seeded randomness: same seed in, same
   world out. Without this, a failing test can't be reproduced.
4. Safety in code. Targets outside the Workspace raise WorkspaceViolation
   before the robot moves at all.
5. No AI here. This module must work (and be tested) without an API key.
"""

from __future__ import annotations

import math
import random
import time
from types import TracebackType

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from armbench.domain import (
    MoveResult,
    ObjectState,
    Vec3,
    Workspace,
    WorkspaceViolation,
    WorldState,
)

# --- Panda constants (from franka_panda/panda.urdf in pybullet_data) ---
ARM_JOINTS = list(range(7))  # 7 revolute joints
FINGER_JOINTS = [9, 10]  # 2 prismatic finger joints
EE_LINK = 11  # "panda_grasptarget": point between the fingertips
REST_POSE = [0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.8]
ARM_FORCE = 200.0
FINGER_OPEN = 0.04  # metres per finger
FINGER_FORCE = 20.0

# --- Sim constants ---
TIME_STEP = 1.0 / 240.0
GRASP_RADIUS = 0.05  # gripper must be this close to an object to grasp it
POSITION_TOLERANCE = 0.01  # a move "succeeds" within 1 cm
GRIPPER_DOWN = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

CUBE_NAME = "red_cube"
CUBE_HALF_SIZE = 0.025


class ArmSim:
    """A Panda arm on a floor with one red cube.

    Use it as a context manager so the physics server is always closed:

        with ArmSim() as sim:
            sim.reset(seed=1)
            sim.move_to(Vec3(x=0.5, y=0.0, z=0.3))
    """

    def __init__(self, gui: bool = False, workspace: Workspace | None = None) -> None:
        self.gui = gui
        self.workspace = workspace or Workspace()
        self._client = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._robot: int = -1
        self._objects: dict[str, int] = {}
        self._grasp_constraint: int | None = None
        self._held: str | None = None
        self._gripper_open = True
        self._steps = 0

    # ---------- lifecycle ----------

    def __enter__(self) -> ArmSim:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = None  # type: ignore[assignment]

    def reset(self, seed: int = 0) -> WorldState:
        """Rebuild the world from scratch. Same seed -> identical world."""
        c = self._client
        c.resetSimulation()
        c.setGravity(0, 0, -9.81)
        c.setTimeStep(TIME_STEP)
        c.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        c.loadURDF("plane.urdf")
        self._robot = c.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        for joint, angle in zip(ARM_JOINTS, REST_POSE, strict=True):
            c.resetJointState(self._robot, joint, angle)
        for joint in FINGER_JOINTS:
            c.resetJointState(self._robot, joint, FINGER_OPEN)
        self._command_arm(REST_POSE)
        self._command_fingers(FINGER_OPEN)

        rng = random.Random(seed)  # local RNG: never touch the global random state
        cube_xy = (rng.uniform(0.45, 0.60), rng.uniform(-0.15, 0.15))
        cube = c.loadURDF("cube_small.urdf", [cube_xy[0], cube_xy[1], CUBE_HALF_SIZE])
        c.changeVisualShape(cube, -1, rgbaColor=[0.9, 0.1, 0.1, 1])
        self._objects = {CUBE_NAME: cube}

        self._grasp_constraint = None
        self._held = None
        self._gripper_open = True
        self._steps = 0
        self.step(60)  # let everything settle
        return self.get_state()

    # ---------- observation ----------

    def get_state(self) -> WorldState:
        c = self._client
        ee = c.getLinkState(self._robot, EE_LINK, computeForwardKinematics=True)[4]
        objects = {}
        for name, body in self._objects.items():
            pos, _orn = c.getBasePositionAndOrientation(body)
            objects[name] = ObjectState(name=name, position=Vec3(x=pos[0], y=pos[1], z=pos[2]))
        return WorldState(
            ee_position=Vec3(x=ee[0], y=ee[1], z=ee[2]),
            gripper_open=self._gripper_open,
            holding=self._held,
            objects=objects,
            sim_time=self._steps * TIME_STEP,
        )

    # ---------- actions ----------

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._client.stepSimulation()
            self._steps += 1
            if self.gui:
                time.sleep(TIME_STEP)  # real-time only when a human is watching

    def move_to(self, target: Vec3, max_steps: int = 720) -> MoveResult:
        """Move the gripper centre to `target`, pointing down.

        Raises WorkspaceViolation *before* moving if the target is unsafe.
        """
        if not self.workspace.contains(target):
            raise WorkspaceViolation(f"Target {target.as_tuple()} is outside the workspace")

        joint_targets = self._client.calculateInverseKinematics(
            self._robot,
            EE_LINK,
            target.as_tuple(),
            GRIPPER_DOWN,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )
        self._command_arm(list(joint_targets[:7]))

        steps = 0
        error = self.get_state().ee_position.distance_to(target)
        while steps < max_steps:
            self.step(10)
            steps += 10
            error = self.get_state().ee_position.distance_to(target)
            if error < POSITION_TOLERANCE and self._arm_is_still():
                return MoveResult(success=True, final_error_m=error, steps=steps)
        return MoveResult(
            success=False,
            final_error_m=error,
            steps=steps,
            message="Did not converge (target may be unreachable in this orientation)",
        )

    def grasp(self) -> bool:
        """Close the gripper and attach the nearest object within GRASP_RADIUS.

        SIMPLIFICATION (document this in the README!): instead of simulating
        friction-based grasping, which is slow and fiddly to tune, we "glue"
        the object to the gripper with a fixed constraint. This project tests
        AI planning and verification, not grasp physics.
        """
        self._command_fingers(0.0)
        self._gripper_open = False
        self.step(60)

        ee = self.get_state().ee_position
        for name, body in self._objects.items():
            pos, _ = self._client.getBasePositionAndOrientation(body)
            if ee.distance_to(Vec3(x=pos[0], y=pos[1], z=pos[2])) <= GRASP_RADIUS:
                for finger in FINGER_JOINTS:  # stop fingers fighting the glued cube
                    self._client.setCollisionFilterPair(self._robot, body, finger, -1, 0)
                self._grasp_constraint = self._client.createConstraint(
                    parentBodyUniqueId=self._robot,
                    parentLinkIndex=EE_LINK,
                    childBodyUniqueId=body,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                )
                self._held = name
                return True
        return False

    def release(self) -> None:
        if self._grasp_constraint is not None:
            self._client.removeConstraint(self._grasp_constraint)
            body = self._objects[self._held] if self._held else None
            if body is not None:
                for finger in FINGER_JOINTS:
                    self._client.setCollisionFilterPair(self._robot, body, finger, -1, 1)
        self._grasp_constraint = None
        self._held = None
        self._command_fingers(FINGER_OPEN)
        self._gripper_open = True
        self.step(120)  # let the object drop and settle

    # ---------- internals ----------

    def _command_arm(self, joint_targets: list[float]) -> None:
        self._client.setJointMotorControlArray(
            self._robot,
            ARM_JOINTS,
            p.POSITION_CONTROL,
            targetPositions=joint_targets,
            forces=[ARM_FORCE] * len(ARM_JOINTS),
        )

    def _command_fingers(self, opening: float) -> None:
        self._client.setJointMotorControlArray(
            self._robot,
            FINGER_JOINTS,
            p.POSITION_CONTROL,
            targetPositions=[opening, opening],
            forces=[FINGER_FORCE, FINGER_FORCE],
        )

    def _arm_is_still(self, threshold: float = 0.01) -> bool:
        states = self._client.getJointStates(self._robot, ARM_JOINTS)
        return all(abs(s[1]) < threshold for s in states)  # s[1] = joint velocity
