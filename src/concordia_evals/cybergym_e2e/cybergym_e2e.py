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
    (persist across reboots: echo 'vm.mmap_rnd_bits=28' |
     sudo tee /etc/sysctl.d/60-cybergym.conf)
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
    diverse_sample,
    download_task_data,
    list_tasks,
    load_task_config,
    stratified_sample,
    sync_scripts_repo,
    write_compose_file,
)
from concordia_evals.cybergym_e2e.prompts import get_prompt
from concordia_evals.cybergym_e2e.scorer import cybergym_e2e_scorer, cybergym_e2e_scorer_both
from concordia_evals.cybergym_e2e.setup_solver import setup_workspace

logger = logging.getLogger(__name__)

ASLR_SYSCTL_PATH = Path("/proc/sys/vm/mmap_rnd_bits")
ASLR_RECOMMENDED_MAX = 28


def check_aslr_prereq() -> None:
    """Warn if the host ASLR entropy exceeds the sanitizer requirement.

    MSan (and ASan) shadow-memory mappings collide with randomized layouts;
    upstream requires vm.mmap_rnd_bits<=28 (their README prerequisite). We
    can only warn here: the sysctl is host-level and often root-only.
    """
    try:
        value = int(ASLR_SYSCTL_PATH.read_text().strip())
    except Exception:
        logger.debug("Cannot read %s; skipping ASLR check", ASLR_SYSCTL_PATH)
        return
    if value > ASLR_RECOMMENDED_MAX:
        logger.warning(
            "Host vm.mmap_rnd_bits=%d (> %d). MSan-instrumented binaries will "
            "crash probabilistically (shadow-memory/ASLR collisions), making "
            "stage results flaky for MSan tasks. Upstream prerequisite: "
            "sudo sysctl -w vm.mmap_rnd_bits=%d (persist via /etc/sysctl.d/).",
            value,
            ASLR_RECOMMENDED_MAX,
            ASLR_RECOMMENDED_MAX,
        )


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


def sample_modes(mode: str) -> list[str]:
    """Per-task modes to instantiate: "both" runs each task in both modes."""
    if mode == "both":
        return ["e2e", "patch-only"]
    return [mode]


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
    sample_n: int | None = None,
    sample_seed: int = 0,
    sample_method: str = "diverse",
    setup_proxy: str | None = None,
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
    if sample_n is not None and len(selected) > sample_n:
        if sample_method == "diverse":
            selected = diverse_sample(selected, sample_n, sample_seed, repo_dir)
        elif sample_method == "stratified":
            selected = stratified_sample(selected, sample_n, sample_seed)
        else:
            raise ValueError(f"Unknown sample_method: {sample_method!r}")

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

        for sample_mode in sample_modes(mode):
            # mode="both" pairs run concurrently and network_lockdown locates
            # containers by compose-file label: each sample needs its own
            # compose file or lockdown would disconnect both containers.
            compose_slug = f"{slug}-{sample_mode}" if mode == "both" else slug
            compose_file = write_compose_file(
                compose_slug, build_image, task_data_dir, script_dir, compose_dir, setup_proxy
            )
            samples.append(
                Sample(
                    input=get_prompt(sample_mode, repo_to_patch, prompt_style),  # type: ignore[arg-type]
                    id=f"{task_path}#{sample_mode}" if mode == "both" else task_path,
                    sandbox=SandboxEnvironmentSpec(type="docker", config=compose_file),
                    metadata={
                        "task_path": task_path,
                        "task_id": config.get("task_id", ""),
                        "target_prog": config.get("target_prog", ""),
                        "mode": sample_mode,
                        "prompt_style": prompt_style,
                        "build_image": build_image,
                        "repo_to_patch": repo_to_patch,
                        "immutable_files": config.get("immutable_files", []),
                        "pre_patches": pre_patches,
                        "task_data_dir": str(task_data_dir),
                        "script_dir": str(script_dir),
                        "compose_file": compose_file,
                        "validation_compile_timeout": validation_compile_timeout,
                        "setup_proxy": setup_proxy,
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
    sample_n: int | None = None,
    sample_seed: int = 0,
    lockdown: bool = True,
    validation_compile_timeout: int | None = None,
    sample_method: str = "diverse",
    setup_proxy: str | None = None,
    solver: Solver | None = None,
) -> Task:
    """CyberGym-E2E evaluation task.

    Args:
        mode: "e2e" (discover + PoC + patch), "patch-only" (patch given
            crash log and ground-truth PoC), or "both" (each task runs in
            both modes as paired samples; the "success" metrics additionally
            report composite_weighted, the native 0.6*e2e + 0.4*patch-only
            score, so the pipeline reads it from the log directly).
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
        sample_n: If set, draw a subset of this size from the selected tasks
            (reproducible via sample_seed).
        sample_seed: Seed for subset sampling (default 0).
        sample_method: "diverse" (default; distinct projects first, then
            balance source-family/sanitizer coverage) or "stratified"
            (plain project round-robin).
        setup_proxy: HTTP(S) proxy URL used inside containers during setup
            and validation (e.g. "http://hostgw:7890"; "hostgw" resolves to
            the docker host). Prepare-phase git/http fetches can otherwise
            be too slow on restricted networks. Unused when None.
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
    if mode not in ("e2e", "patch-only", "both"):
        raise ValueError(f"mode must be 'e2e', 'patch-only' or 'both', got {mode!r}")
    if prompt_style not in ("iterative", "no-test"):
        raise ValueError(f"prompt_style must be 'iterative' or 'no-test', got {prompt_style!r}")

    check_aslr_prereq()
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
                sample_n=sample_n,
                sample_seed=sample_seed,
                sample_method=sample_method,
                setup_proxy=setup_proxy,
            )
        ),
        solver=solver,
        scorer=cybergym_e2e_scorer_both() if mode == "both" else cybergym_e2e_scorer(),
    )
