"""Constants for the BountyBench evaluation."""

import os
from typing import Literal

from concordia_evals.constants import CONCORDIA_EVALS_CACHE_PATH

# Upstream task data: https://github.com/bountybench/bountytasks
# Pinned to the commit that bountybench/bountybench pins as a submodule.
BOUNTYTASKS_REPO = "bountybench/bountytasks"
BOUNTYTASKS_COMMIT = "60311f3d9334af493841b769245d9fed4e07cf88"
BOUNTYTASKS_URL = f"https://github.com/{BOUNTYTASKS_REPO}/archive/{BOUNTYTASKS_COMMIT}.tar.gz"

# Target systems keep their code in the cy-suite organisation, one repository per
# system, referenced as a git submodule of bountytasks.
CODEBASE_REPO_ORG = "cy-suite"

TASK_TYPES = ("detect", "exploit", "patch")
TaskType = Literal["detect", "exploit", "patch"]
DEFAULT_TASK_TYPE: TaskType = "exploit"

BOUNTYBENCH_CACHE_PATH = CONCORDIA_EVALS_CACHE_PATH / "bountybench"
BOUNTYTASKS_DIR = BOUNTYBENCH_CACHE_PATH / "bountytasks"
CODEBASE_MIRRORS_DIR = BOUNTYBENCH_CACHE_PATH / "codebases"
SANDBOX_MIRRORS_DIR = BOUNTYBENCH_CACHE_PATH / "sandbox_mirrors"
# Build contexts of the per-sample sandbox images (target repository installed).
SAMPLE_IMAGES_DIR = BOUNTYBENCH_CACHE_PATH / "images"
WORKSPACES_DIR = BOUNTYBENCH_CACHE_PATH / "workspaces"
# One lock per system: the compose files of the bounties name their containers
# (`container_name: mlflow`) and their scripts `docker exec` those names, so two
# samples of the same system cannot run at the same time.
LOCKS_DIR = BOUNTYBENCH_CACHE_PATH / "locks"
SANDBOX_COMPOSE_DIR = BOUNTYBENCH_CACHE_PATH / "compose"

# Shared docker network: every upstream task compose file declares it as an
# external network and the agent container joins it so that target hostnames
# (e.g. `lunary-app:3333`) resolve exactly as they do upstream.
SHARED_NETWORK = "shared_net"

# Base sandbox image, built from the Dockerfile in this directory. Every sample
# gets a derived image with the target repository installed at build time, so
# that nothing is installed (and no network is needed) while the agent works.
AGENT_IMAGE = "inspect-bountybench-agent:latest"
AGENT_IMAGE_NAME = "inspect-bountybench-agent"
AGENT_DOCKERFILE = "Dockerfile"

# Paths inside the agent sandbox, matching the upstream layout so that the
# upstream prompts and verify scripts keep working unchanged.
SANDBOX_APP_DIR = "/app"
SANDBOX_CODEBASE_DIR = f"{SANDBOX_APP_DIR}/codebase"
SANDBOX_EXPLOIT_DIR = f"{SANDBOX_APP_DIR}/exploit_files"
SANDBOX_VERIFY_DIR = f"{SANDBOX_APP_DIR}/verify_files"
SANDBOX_MIRROR_DIR = "/opt/codebase.git"
# Replica of the task directory used by reference exploits and verify scripts.
SANDBOX_GRADING_TREE = "/app/bt"

# Wheelhouse baked into the sandbox image, used as an offline package index by
# the tasks' own grading scripts (`pip install` inside the container).
SANDBOX_WHEELHOUSE = "/opt/wheels"

# Per sample store flag set when the agent submits its work. Upstream only grades
# submitted work (`ExploitAgent`/`PatchAgent`/`DetectAgent` fail on "no_submission").
SUBMITTED_STORE_KEY = "bountybench_submitted"

# Upstream's `run_workflow.sh` passes `--phase_iterations 100` and the detect,
# exploit and patch workflows each have a single root phase, so an agent gets 100
# iterations of one command each. Here one react turn is an assistant message
# plus one tool result per command (about two commands per turn in practice), so
# 150 messages is the same 100 commands. An earlier default of 30 -- about ten
# turns, twenty commands -- ended every sample while it was still reading the
# codebase.
DEFAULT_MAX_MESSAGES = 150
DEFAULT_RANDOM_SEED = 0
DEFAULT_SAMPLE_NUM: int | None = None

# The prompt embeds a `tree` listing of at most this many tokens.
CODEBASE_TREE_MAX_TOKENS = 300
CODEBASE_TREE_IGNORE = "_*|venv|node_modules|.git|__pycache__|.idea|.vscode"

# Command timeout for the commands the agent runs (upstream TIMEOUT_PER_COMMAND).
DEFAULT_COMMAND_TIMEOUT = 120
# Timeout of the image build that installs the target repository
# (upstream PIP_INSTALL_TIMEOUT, raised: some samples compile CPython from source).
INSTALL_TIMEOUT = int(os.environ.get("BOUNTYBENCH_INSTALL_TIMEOUT", "3600"))
# Upstream waits up to 300s for all compose containers to report healthy.
HEALTHCHECK_TIMEOUT = 300

# Build of the image of a target stack (apt and pip of the sample, once per sample).
STACK_BUILD_TIMEOUT = 3600
# Upstream caps invariant test execution at 90 minutes.
INVARIANTS_TIMEOUT = 5400
INVARIANTS_RETRY = 3

DEFAULT_CPU_LIMIT = "4"
DEFAULT_MEMORY_LIMIT = "8g"

# A sample may only change the code under test: the environment of a sample is
# fixed, so a change to how its dependencies are installed or how its image is
# built is rejected instead of fetched and executed by the grading build.
DEPENDENCY_FILES = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "environment.yml",
        "environment.yaml",
        "manifest.in",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "pnpm-workspace.yaml",
        "npm-shrinkwrap.json",
        ".npmrc",
        ".dockerignore",
    }
)
DEPENDENCY_FILE_PREFIXES = ("requirements", "constraints")
DEPENDENCY_FILE_SUFFIXES = (".lock",)
DEPENDENCY_FILE_NAMES = ("dockerfile", "compose.yaml", "compose.yml")
DEPENDENCY_FILE_NAME_PREFIXES = ("dockerfile.", "docker-compose")

# Samples whose only known fix changes how the system is installed, which the
# rule above forbids. They stay in the task types where the reference solution is
# applied by the harness (`exploit`, `detect`) and are skipped in `patch`.
# Systems excluded from every task type because preparing them costs more than
# the samples are worth: `langchain` installs two whole test environments into
# its image (measured 2026-11-16: 12.3GB and over an hour of build time against
# 3-5GB and a few minutes for every other system).
UNAVAILABLE_SYSTEMS: dict[str, str] = {
    "langchain": "its image takes over an hour to build and reaches 12.3GB",
}

UNAVAILABLE_SAMPLES: dict[str, dict[str, str]] = {
    "patch": {
        "fastapi_bounty_0": (
            "its reference fix raises the python-multipart lower bound "
            "(0.0.5 -> 0.0.7), and a sample may not change dependencies"
        )
    }
}
