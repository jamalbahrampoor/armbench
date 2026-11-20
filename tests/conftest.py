"""Shared pytest fixtures.

A fixture is setup code pytest runs for you and passes into any test that
asks for it by name. `yield` splits setup (before) from teardown (after):
teardown runs even if the test fails, so the physics server is never leaked.
"""

import pytest

from armbench.sim import ArmSim


@pytest.fixture
def sim():
    """A fresh, headless, seeded sim for each test (scope='function').

    Why not scope='session' (one sim shared by all tests)? It would be faster,
    but tests could leak state into each other, so the order they run in would
    change the results. Isolation first; optimise only if the suite gets slow.
    """
    with ArmSim(gui=False) as s:
        s.reset(seed=0)
        yield s
