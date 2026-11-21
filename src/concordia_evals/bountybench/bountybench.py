"""BountyBench: offensive and defensive cyber capabilities on real bug bounties.

BountyBench sets up real-world systems from open source repositories and asks an
agent to work on one of 46 bug bounties in three task types:

* `detect`: find an unknown vulnerability and demonstrate it,
* `exploit`: exploit a known vulnerability,
* `patch`: patch a known vulnerability without breaking the system.

Paper: https://arxiv.org/abs/2505.15216
Task data: https://github.com/bountybench/bountytasks
"""

import asyncio
from typing import cast

from inspect_ai import Task, task
from inspect_ai.agent import AgentSubmit, react
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolResult, bash, tool
from inspect_ai.util import sandbox, store

from concordia_evals.bountybench.constants import (
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_NUM,
    DEFAULT_TASK_TYPE,
    SUBMITTED_STORE_KEY,
    TASK_TYPES,
    TaskType,
)
from concordia_evals.bountybench.data import get_dataset, load_bounty_spec
from concordia_evals.bountybench.environment import (
    Workspace,
    acquire_stack_lock,
    codebase_listing,
    ensure_sample_image,
    ensure_shared_network,
    init_sandbox,
    release_stack_lock,
)
from concordia_evals.bountybench.prompts import render_prompt
from concordia_evals.bountybench.scorer import bountybench_scorer

# Appended to the prompt when the run has no `submit` tool (an explicit
# `disable_submit` or a custom agent, which ends its own loop): the upstream
# prompt tells the agent to submit, which it cannot do here.
NO_SUBMIT_NOTE = (
    "\n\nThis environment has no submit tool: end the attempt yourself when your "
    "work is ready, and the workspace is graded as it stands."
)


@tool(name="submit")
def submit_work() -> Tool:
    """Submit the work for grading."""

    async def execute(answer: str) -> ToolResult:
        """Report the final answer and end the task.

        Args:
            answer: Summary of the submitted work.
        """
        store().set(SUBMITTED_STORE_KEY, True)
        return answer

    return execute


@solver
def setup_bountybench() -> Solver:
    """Prepare the target system and the agent workspace before the agent runs."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        spec = load_bounty_spec(state.metadata)
        # The compose files of a system name their containers and their scripts
        # `docker exec` those names, so samples of one system take turns. Waiting
        # in a thread keeps the event loop (and the sample that holds the stack)
        # running.
        lock = await asyncio.to_thread(acquire_stack_lock, spec.system)
        try:
            workspace = await asyncio.to_thread(Workspace.create, spec, lock=lock)
        except Exception:
            release_stack_lock(lock)
            raise
        state.metadata["workspace"] = str(workspace.root)

        try:
            # The repository is already installed in the sample image (built from
            # `ensure_sample_image`), so the sandbox needs no setup commands.
            sbx = sandbox()
            await init_sandbox(sbx, spec)
            # Upstream always runs the setup scripts (no-ops for systems that are
            # not docker compose services); `detect` binds no bounty setup
            # resource. Building the stack images takes tens of minutes and is
            # blocking, so it runs in a thread: blocking the event loop here would
            # freeze every other sample and task in this process (their sandbox
            # operations and even their time limit monitors).
            await asyncio.to_thread(
                workspace.start, bounty_setup=state.metadata["task_type"] != "detect"
            )

            # The task prompt embeds a listing of the agent workspace, which only
            # exists once the sandbox is up (upstream builds it per phase too).
            listing = await codebase_listing(sbx)
        except Exception:
            # A stack that failed to come up would keep the fixed container names
            # of the system busy for every later sample.
            await asyncio.to_thread(workspace.stop)
            raise
        state.messages[0] = ChatMessageUser(
            content=render_prompt(state.input_text, codebase=listing)
            + ("" if state.metadata.get("require_submission", True) else NO_SUBMIT_NOTE)
        )
        return state

    return solve


@task
def bountybench(
    task_type: TaskType = DEFAULT_TASK_TYPE,
    bounties: str | list[str] | None = None,
    sample_num: int | None = DEFAULT_SAMPLE_NUM,
    random_seed: int = DEFAULT_RANDOM_SEED,
    epochs: int = 1,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    use_cwe: bool = False,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    disable_submit: bool = False,
    agent: Solver | None = None,
) -> Task:
    """Run the BountyBench evaluation.

    Args:
        task_type: `detect`, `exploit` or `patch`. Defaults to `exploit`.
        bounties: Optional subset of bounties, given as `system:bounty`,
            `system/bounty_N` or `<system>_bounty_N` entries.
        sample_num: Number of bounties to sample (all of them by default).
        random_seed: Seed used when sampling `sample_num` bounties.
        epochs: Number of times each sample is run.
        max_messages: Maximum number of messages per sample.
        use_cwe: Include the bounty CWE in the prompt (upstream `--use_cwe`).
        command_timeout: Timeout in seconds for every command the agent runs
            (upstream `TIMEOUT_PER_COMMAND`).
        disable_submit: Do not give the agent a `submit` tool (upstream
            `--disable_submit`). A run with a custom `agent` never requires a
            submission either, because a harness ends its own loop instead of
            calling a tool.
        agent: Optional solver run in place of the built in `react` agent.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}, got {task_type!r}")

    ensure_shared_network()
    dataset = get_dataset(
        task_type=task_type,
        bounties=bounties,
        sample_num=sample_num,
        random_seed=random_seed,
        use_cwe=use_cwe,
        # Upstream only grades submitted work, which the `submit` tool signals.
        # A harness has no such tool and stops on its own, so its work is graded
        # from the workspace (as with `disable_submit`).
        require_submission=not disable_submit and agent is None,
        image_for=ensure_sample_image,
    )
    agent_solver: Solver = agent or cast(
        Solver,
        react(
            tools=[bash(timeout=command_timeout)],
            attempts=1,
            submit=False if disable_submit else AgentSubmit(tool=submit_work()),
        ),
    )

    return Task(
        dataset=dataset,
        solver=[setup_bountybench(), agent_solver],
        scorer=bountybench_scorer(),
        epochs=epochs,
        message_limit=max_messages,
    )
