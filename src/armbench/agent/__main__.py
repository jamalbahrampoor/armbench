"""Command-line runner:  python -m armbench.agent "put the red cube on the left"

Options:  --gui (watch it)   --seed N (cube position)   --model NAME
"""

from __future__ import annotations

import argparse
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from armbench.agent.loop import DEFAULT_MODEL, run_agent
from armbench.agent.tools import ToolCall
from armbench.sim import ArmSim


def print_call(call: ToolCall) -> None:
    status = "ERROR" if call.result.is_error else "ok"
    args = ", ".join(f"{k}={v}" for k, v in call.input.items())
    print(f"  -> {call.name}({args})  [{status}]")
    if call.result.is_error:
        print(f"     {call.result.content[:200]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Command the robot arm in plain language.")
    parser.add_argument("instruction", help='e.g. "put the red cube on the left side"')
    parser.add_argument("--gui", action="store_true", help="open the PyBullet window")
    parser.add_argument("--seed", type=int, default=0, help="world seed (cube position)")
    parser.add_argument("--model", default=os.getenv("ARMBENCH_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    load_dotenv()  # reads ANTHROPIC_API_KEY from .env into the environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Put it in a .env file in the project root.")
        return 1

    if args.gui and sys.platform == "win32":
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)

    with ArmSim(gui=args.gui) as sim:
        sim.reset(seed=args.seed)
        print(f"Instruction: {args.instruction!r}   (model {args.model}, seed {args.seed})")
        client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        run = run_agent(args.instruction, sim, client, model=args.model, on_tool_call=print_call)

        print(f"\nClaude: {run.final_text}")
        print(
            f"\nstop={run.stop_reason}  llm_calls={run.llm_calls}  tool_calls={len(run.tool_calls)}"
            f"  tokens in/out={run.input_tokens}/{run.output_tokens}"
        )
        print("\nGround truth after the run:")
        print(sim.get_state().model_dump_json(indent=2))
        if args.gui:
            input("\nPress Enter to close...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
