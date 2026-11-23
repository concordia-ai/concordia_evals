"""Harness registry: map a harness name to an Inspect agent.

Benchmark-agnostic. CLI harnesses come from the inspect_swe package (their
model calls are proxied back to the current Inspect model provider via
sandbox_agent_bridge, so any Inspect model works with any of them);
"react" is Inspect's native agent loop.

Usage:
    agent = resolve_agent("codex_cli", cwd="/src", web_search="disabled")
    # then either pass as the task solver directly, or compose:
    chain(setup_solver(), as_solver(agent))
"""

from typing import Any

from inspect_ai.agent import Agent, react

# Harnesses provided by inspect_swe (attribute names in that package).
INSPECT_SWE_HARNESSES = (
    "claude_code",
    "codex_cli",
    "gemini_cli",
    "kimi_code",
    "opencode",
    "mini_swe_agent",
)

AVAILABLE_HARNESSES = ("react",) + INSPECT_SWE_HARNESSES


def _default_react_agent(**opts: Any) -> Agent:
    """Inspect's native react agent with a bash tool."""
    from inspect_ai.tool import bash

    opts.setdefault("tools", [bash(timeout=1800)])
    return react(**opts)


def resolve_agent(harness: str, **opts: Any) -> Agent:
    """Instantiate the agent for a harness name.

    Args:
        harness: One of AVAILABLE_HARNESSES.
        **opts: Forwarded to the agent factory (e.g. cwd, system_prompt,
            disallowed_tools for claude_code; web_search for codex_cli).

    Returns:
        An Inspect Agent usable as a solver (via as_solver) or standalone.
    """
    if harness == "react":
        return _default_react_agent(**opts)

    if harness in INSPECT_SWE_HARNESSES:
        import inspect_swe

        factory = getattr(inspect_swe, harness)
        agent: Agent = factory(**opts)
        return agent

    raise ValueError(f"Unknown harness: {harness!r}. Available: {', '.join(AVAILABLE_HARNESSES)}")
