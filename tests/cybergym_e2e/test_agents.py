"""Tests for the shared agents layer: registry resolution."""

import pytest
from inspect_ai.agent import Agent

from concordia_evals.agents.registry import (
    AVAILABLE_HARNESSES,
    resolve_agent,
)


def test_react_agent_resolves():
    agent = resolve_agent("react", prompt="You are a tester.")
    assert isinstance(agent, Agent)


def test_inspect_swe_harnesses_resolve():
    pytest.importorskip("inspect_swe")
    for harness in ("claude_code", "codex_cli", "kimi_code"):
        agent = resolve_agent(harness, cwd="/src")
        assert isinstance(agent, Agent)


def test_unknown_harness_raises():
    with pytest.raises(ValueError, match="Unknown harness"):
        resolve_agent("definitely-not-a-harness")


def test_available_harnesses_lists_react_first():
    assert AVAILABLE_HARNESSES[0] == "react"
    assert "kimi_code" in AVAILABLE_HARNESSES
