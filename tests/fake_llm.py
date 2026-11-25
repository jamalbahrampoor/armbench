"""A fake Anthropic client that replays scripted responses.

This is the standard way to unit-test LLM code: replace the non-deterministic,
paid, network-dependent part with a deterministic stand-in, and test YOUR logic
(the loop, the tool execution, the error handling) in isolation.

The fake also records every request, so tests can assert on what was sent to Claude.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

_ids = itertools.count(1)


def text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def tool_use(name: str, **inputs: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=inputs)


def reply(*blocks: SimpleNamespace, stop_reason: str | None = None) -> SimpleNamespace:
    """One Claude response. stop_reason defaults to 'tool_use' if any block requests a
    tool, else 'end_turn'. Pass stop_reason=... to force one (e.g. 'max_tokens')."""
    if stop_reason is None:
        wants_tool = any(b.type == "tool_use" for b in blocks)
        stop_reason = "tool_use" if wants_tool else "end_turn"
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class FakeMessages:
    def __init__(self, script: list[SimpleNamespace]) -> None:
        self._script = iter(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        # Snapshot the messages list: the loop keeps appending to the same list object.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return next(self._script)


class FakeClient:
    def __init__(self, script: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(script)
