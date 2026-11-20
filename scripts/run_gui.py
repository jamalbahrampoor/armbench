"""Watch the pick-and-place demo in the PyBullet window.

Run from the project root:  python scripts/run_gui.py
"""

import sys

if sys.platform == "win32":  # Windows-only DPI fix; guarded so Linux/Docker/CI still work
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(1)

from armbench.domain import Vec3
from armbench.sim import ArmSim
from armbench.sim.world import CUBE_HALF_SIZE, CUBE_NAME


def main() -> None:
    with ArmSim(gui=True) as sim:
        state = sim.reset(seed=0)
        cube = state.objects[CUBE_NAME].position
        print("Start:", state.model_dump_json(indent=2))

        for target in [
            Vec3(x=cube.x, y=cube.y, z=0.15),
            Vec3(x=cube.x, y=cube.y, z=CUBE_HALF_SIZE),
        ]:
            print("move_to", target.as_tuple(), "->", sim.move_to(target))
        print("grasp ->", sim.grasp())
        print("move_to ->", sim.move_to(Vec3(x=0.4, y=0.25, z=0.15)))
        sim.release()

        print("End:", sim.get_state().model_dump_json(indent=2))
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
