"""CyberGym-E2E: end-to-end cybersecurity evaluation for AI agents.

920 real-world vulnerabilities across 139 open-source projects. In e2e mode
the agent receives only the source tree and must discover a vulnerability,
produce a crashing PoC (/output/poc.bin), and write a patch
(/output/fix.patch); in patch-only mode it additionally receives the crash
log and ground-truth PoC and only writes a patch. Scoring validates four
cumulative stages (S1-S4) in fresh containers per stage.

Paper: https://arxiv.org/abs/2606.04460
Upstream: https://github.com/sunblaze-ucb/cybergym-e2e

Example:
    uv run inspect eval src/concordia_evals/cybergym_e2e/cybergym_e2e.py@cybergym_e2e \
        --model openai/gpt-5.2-codex -T harness=codex_cli \
        -T tasks='["curl/arvo_66012"]' --time-limit 14400

Host prerequisite (sanitizer compatibility, from upstream README):
    sudo sysctl -w vm.mmap_rnd_bits=28
"""

import logging
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import as_solver
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver
from inspect_ai.util import SandboxEnvironmentSpec, sandbox

from concordia_evals.agents.lockdown import network_lockdown
from concordia_evals.agents.registry import resolve_agent
from concordia_evals.cybergym_e2e.dataset import (
    COMPOSE_DIR,
    DATA_DIR,
    DEFAULT_BUILD_IMAGE,
    REPO_DIR,
    download_task_data,
    list_tasks,
    load_task_config,
    sync_scripts_repo,
    write_compose_file,
)
from concordia_evals.cybergym_e2e.prompts import get_prompt
from concordia_evals.cybergym_e2e.scorer import cybergym_e2e_scorer
from concordia_evals.cybergym_e2e.setup_solver import setup_workspace

logger = logging.getLogger(__name__)

# Harness-specific agent options matching upstream's anti-web settings
# (run_agent.py: claude --disallowedTools ...; codex web_search = "disabled").
HARNESS_DEFAULT_OPTS: dict[str, dict[str, Any]] = {
    "claude_code": {
        "cwd": "/src",
        "disallowed_tools": [
            "WebFetch",
            "WebSearch",
            "Task",
            "MCPSearch",
            "NotebookEdit",
            "Skill",
            "AskUserQuestion",
        ],
    },
    # model_config pins Codex's prompt/tool set to a classic function-calling
    # slug: inspect_swe otherwise derives a recent GPT slug whose catalog entry
    # is code_mode_only, which non-OpenAI models cannot drive. Override via
    # harness_opts when evaluating actual GPT models.
    "codex_cli": {"cwd": "/src", "web_search": "disabled", "model_config": "gpt-5.2"},
    "kimi_code": {"cwd": "/src"},
    "gemini_cli": {"cwd": "/src"},
    "opencode": {"cwd": "/src"},
    "react": {},
}


@solver
def ground_truth_solver() -> Solver:
    """Write the ground-truth PoC and patch as the agent output.

    Smoke-test harness ("ground_truth"): exercises setup, validation and
    scoring end to end without a model. Expected result is S1-S4 all passed.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sbx = sandbox()
        poc = Path(state.metadata["task_data_dir"]) / "poc.bin"
        patch = Path(state.metadata["script_dir"]) / "patch.diff"
        if state.metadata["mode"] == "e2e":
            await sbx.write_file("/output/poc.bin", poc.read_bytes())
        await sbx.write_file("/output/fix.patch", patch.read_bytes())
        state.output.completion = "ground truth written"
        return state

    return solve


def build_samples(
    *,
    mode: str,
    prompt_style: str,
    tasks: list[str] | None,
    projects: list[str] | None,
    limit: int | None,
    repo_dir: Path,
    data_dir: Path,
    compose_dir: Path,
    validation_compile_timeout: int | None = None,
) -> list[Sample]:
    """Build inspect Samples for the selected tasks (downloading as needed)."""
    all_tasks = list_tasks(repo_dir)
    if tasks:
        wanted = set(tasks)
        unknown = wanted - set(all_tasks)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        selected = [t for t in all_tasks if t in wanted]
    elif projects:
        wanted_projects = set(projects)
        selected = [t for t in all_tasks if t.split("/", 1)[0] in wanted_projects]
    else:
        selected = all_tasks
    if limit is not None:
        selected = selected[:limit]

    samples: list[Sample] = []
    for task_path in selected:
        config = load_task_config(task_path, repo_dir)
        build_image = config.get("build_image", DEFAULT_BUILD_IMAGE)
        repo_to_patch = config.get("repo_to_patch", "")
        pre_patches = config.get("pre_patches") or (
            [config["pre_patch"]] if config.get("pre_patch") else []
        )

        task_data_dir = download_task_data(task_path, data_dir)
        script_dir = repo_dir / "projects" / task_path
        # Images are pulled lazily by docker compose at sandbox start
        # (pull policy "missing"), so we never pre-pull here.

        slug = task_path.replace("/", "_")
        compose_file = write_compose_file(slug, build_image, task_data_dir, script_dir, compose_dir)

        samples.append(
            Sample(
                input=get_prompt(mode, repo_to_patch, prompt_style),  # type: ignore[arg-type]
                id=task_path,
                sandbox=SandboxEnvironmentSpec(type="docker", config=compose_file),
                metadata={
                    "task_path": task_path,
                    "task_id": config.get("task_id", ""),
                    "target_prog": config.get("target_prog", ""),
                    "mode": mode,
                    "prompt_style": prompt_style,
                    "build_image": build_image,
                    "repo_to_patch": repo_to_patch,
                    "immutable_files": config.get("immutable_files", []),
                    "pre_patches": pre_patches,
                    "task_data_dir": str(task_data_dir),
                    "script_dir": str(script_dir),
                    "compose_file": compose_file,
                    "validation_compile_timeout": validation_compile_timeout,
                },
            )
        )
    logger.info("Built %d samples for cybergym_e2e (%s)", len(samples), mode)
    return samples


@task
def cybergym_e2e(
    mode: str = "e2e",
    harness: str = "react",
    harness_opts: dict[str, Any] | None = None,
    prompt_style: str = "iterative",
    tasks: list[str] | None = None,
    projects: list[str] | None = None,
    limit: int | None = None,
    lockdown: bool = True,
    validation_compile_timeout: int | None = None,
    solver: Solver | None = None,
) -> Task:
    """CyberGym-E2E evaluation task.

    Args:
        mode: "e2e" (discover + PoC + patch) or "patch-only" (patch given
            crash log and ground-truth PoC).
        harness: Agent harness name ("react", "claude_code", "codex_cli",
            "kimi_code", "gemini_cli", "opencode"). The model is chosen with
            inspect's --model and proxied into the sandbox via the agent
            bridge, so any harness works with any inspect model provider.
            Special value "ground_truth" writes the ground-truth PoC/patch
            directly (pipeline smoke test, no model needed).
        harness_opts: Extra keyword args for the harness agent factory
            (e.g. {"version": "1.0.0"}).
        prompt_style: "iterative" (agent may run validation itself) or
            "no-test" (agent must not execute builds).
        tasks: Explicit task paths, e.g. ["curl/arvo_66012"].
        projects: Project filter, e.g. ["curl", "mruby"].
        limit: Max number of samples (after other filters).
        lockdown: Disconnect the agent container from all networks after
            setup (the agent bridge still works over loopback/exec).
        validation_compile_timeout: Override the vendored validate.py's
            hardcoded 3600s compile timeout on the deployed in-container copy
            (needed for large projects like ffmpeg on loaded hosts). The
            vendored file stays byte-identical; default None = upstream value.
        solver: Full custom solver chain (overrides the default chain).

    Returns:
        An Inspect Task for CyberGym-E2E.
    """
    if mode not in ("e2e", "patch-only"):
        raise ValueError(f"mode must be 'e2e' or 'patch-only', got {mode!r}")
    if prompt_style not in ("iterative", "no-test"):
        raise ValueError(f"prompt_style must be 'iterative' or 'no-test', got {prompt_style!r}")

    repo_dir = sync_scripts_repo(REPO_DIR)

    if solver is None:
        if harness == "ground_truth":
            solver = chain(setup_workspace(), ground_truth_solver())
        else:
            opts = dict(HARNESS_DEFAULT_OPTS.get(harness, {}))
            opts.update(harness_opts or {})
            agent = resolve_agent(harness, **opts)
            solver = chain(
                setup_workspace(),
                network_lockdown(lockdown),
                as_solver(agent),
            )

    return Task(
        dataset=MemoryDataset(
            samples=build_samples(
                mode=mode,
                prompt_style=prompt_style,
                tasks=tasks,
                projects=projects,
                limit=limit,
                repo_dir=repo_dir,
                data_dir=DATA_DIR,
                compose_dir=COMPOSE_DIR,
                validation_compile_timeout=validation_compile_timeout,
            )
        ),
        solver=solver,
        scorer=cybergym_e2e_scorer(),
    )
