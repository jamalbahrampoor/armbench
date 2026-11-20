# ArmBench

A robot arm you can command in plain language, with AI evaluation built in.
PyBullet + Franka Panda + Claude.

> Status: Sprint 1 — simulation library + tests.

## Setup (Windows, PowerShell)

```powershell
cd C:\Users\jamal\projects\pybullet_learning\armbench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
python scripts\run_gui.py
```

`pip install -e .` = "editable install": Python imports `armbench` straight from
`src/`, so your edits are live without reinstalling.

## Layout

```
src/armbench/domain.py     Pydantic models: Vec3, Workspace, WorldState, MoveResult
src/armbench/sim/world.py  ArmSim: headless, deterministic PyBullet wrapper
tests/                     pytest: unit (domain) + integration (sim)
scripts/run_gui.py         visual demo
```

## Known simplifications

- **Grasping is "glued"**: a fixed constraint attaches the cube when the gripper
  is within 5 cm. The project is about AI planning and verification, not grasp physics.
