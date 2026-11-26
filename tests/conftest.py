"""Pytest configuration for the concordia_evals test suite."""

import os

import pytest


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the slow-test switch."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests marked slow unless explicitly enabled."""
    env_val = os.environ.get("RUN_SLOW_TESTS")
    enabled = _truthy(env_val) if env_val is not None else bool(config.getoption("--runslow"))
    if enabled:
        return
    skip_mark = pytest.mark.skip(reason="need --runslow option or RUN_SLOW_TESTS=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_mark)
