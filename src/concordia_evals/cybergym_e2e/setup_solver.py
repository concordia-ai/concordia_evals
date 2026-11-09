"""Workspace setup solver, ported from upstream utils.setup_workspace().

Runs inside the agent sandbox before the agent starts. Mirrors upstream
semantics step for step: image cleanup, validate.py dependencies, source
extraction, sanitized config, .git removal, pre_patches, prepare.sh, and the
/src_backup snapshot that validate.py's restore_src() relies on.

Upstream reference: scripts/utils.py::setup_workspace at the pinned commit.
"""

import logging
from pathlib import Path

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).parent / "vendor"

# Mirrors upstream scripts/install_validate_deps.sh.
INSTALL_VALIDATE_DEPS = """
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq curl >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
. "$HOME/.local/bin/env"
uv python install 3.13 >/dev/null 2>&1
uv venv /scripts/.venv --python 3.13 >/dev/null 2>&1
uv pip install --python /scripts/.venv/bin/python tomli >/dev/null 2>&1
"""

# Mirrors the arvo/cybergym image cleanup in upstream setup_workspace().
ARVO_IMAGE_CLEANUP = """
find /src -mindepth 1 -maxdepth 1 ! -name honggfuzz ! -name libfuzzer ! -name afl ! -name aflplusplus ! -name fuzztest -exec rm -rf {} +
rm -rf /out /work /usr/bin/arvo
find /tmp -mindepth 1 -maxdepth 1 -exec rm -rf {} +
mkdir -p /out /work
"""


def _sanitized_config_toml(repo_to_patch: str, immutable_files: list[str]) -> str:
    """Config visible inside the container; upstream drops everything else."""
    lines = [f'repo_to_patch = "{repo_to_patch}"']
    if immutable_files:
        files = ", ".join(f'"{f}"' for f in immutable_files)
        lines.append(f"immutable_files = [{files}]")
    return "\n".join(lines) + "\n"


async def _exec(cmd: str, description: str, timeout: int) -> None:
    """Run a setup command, raising on failure (setup errors should fail loudly)."""
    result = await sandbox().exec(["bash", "-c", cmd], timeout=timeout)
    if not result.success:
        raise RuntimeError(
            f"setup_workspace: {description} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )


@solver
def setup_workspace() -> Solver:
    """Set up the task workspace inside the agent sandbox."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        mode: str = state.metadata["mode"]
        build_image: str = state.metadata["build_image"]
        repo_to_patch: str = state.metadata["repo_to_patch"]
        immutable_files: list[str] = state.metadata.get("immutable_files", [])
        pre_patches: list[str] = state.metadata.get("pre_patches", [])

        if "arvo" in build_image or "cybergym" in build_image:
            await _exec(ARVO_IMAGE_CLEANUP, "image cleanup", timeout=300)

        await _exec("mkdir -p /config /data /output /scripts", "mkdir", timeout=60)

        await _exec(
            "export DEBIAN_FRONTEND=noninteractive && "
            "apt-get update -qq && apt-get install -y -qq sudo git >/dev/null 2>&1",
            "install sudo/git",
            timeout=900,
        )
        await _exec(INSTALL_VALIDATE_DEPS, "install validate deps", timeout=1200)

        # validate.py is vendored in this package and deployed verbatim.
        validate_py = (VENDOR_DIR / "validate.py").read_bytes()
        await sandbox().write_file("/scripts/validate.py", validate_py)

        await _exec(
            "cp /task_data/src.tgz /src/src.tgz && tar xf /src/src.tgz -C /src && rm /src/src.tgz",
            "extract source",
            timeout=600,
        )

        if mode == "patch-only":
            await _exec(
                "cp /task_data/crash.log /task_data/poc.bin /src/ 2>/dev/null || true; "
                "cp /task_data/poc.bin /data/poc.bin 2>/dev/null || true",
                "copy crash log and PoC",
                timeout=60,
            )

        for script in ("prepare.sh", "compile.sh", "run_poc.sh", "test.sh"):
            await _exec(
                f"cp /task_scripts/{script} /src/{script} 2>/dev/null || true",
                f"copy {script}",
                timeout=60,
            )

        config_toml = _sanitized_config_toml(repo_to_patch, immutable_files)
        await sandbox().write_file("/config/config.toml", config_toml)

        # Drop the target repo's git history: it can contain the fix itself.
        if repo_to_patch:
            await _exec(
                f"rm -rf /src/{repo_to_patch}/.git",
                "remove .git",
                timeout=120,
            )

        pre_patch_repo = f"/src/{repo_to_patch}" if repo_to_patch else "/src"
        for i, pre_patch_name in enumerate(pre_patches):
            applied = False
            for strip_level in (0, 1, 2, 3):
                if strip_level == 0:
                    cmd = f"cd {pre_patch_repo} && git apply /task_scripts/{pre_patch_name}"
                else:
                    cmd = (
                        f"cd {pre_patch_repo} && "
                        f"patch -p{strip_level} < /task_scripts/{pre_patch_name}"
                    )
                result = await sandbox().exec(["bash", "-c", cmd], timeout=120)
                if result.success:
                    applied = True
                    break
            if not applied:
                raise RuntimeError(
                    f"setup_workspace: failed to apply pre_patch {pre_patch_name} "
                    "with strip levels 0-3"
                )
            logger.debug("Applied pre_patch %s (index %d)", pre_patch_name, i)

        await _exec("bash -eux /src/prepare.sh", "prepare.sh", timeout=1800)
        await _exec("cp -a /src /src_backup", "snapshot /src_backup", timeout=600)

        return state

    return solve
