"""Constants for the BountyBench evaluation."""

from typing import Literal

from concordia_evals.constants import CONCORDIA_EVALS_CACHE_PATH

# Upstream task data: https://github.com/bountybench/bountytasks
# Pinned to the commit that bountybench/bountybench pins as a submodule.
BOUNTYTASKS_REPO = "bountybench/bountytasks"
BOUNTYTASKS_COMMIT = "60311f3d9334af493841b769245d9fed4e07cf88"
BOUNTYTASKS_URL = (
    f"https://github.com/{BOUNTYTASKS_REPO}/archive/{BOUNTYTASKS_COMMIT}.tar.gz"
)

# Target systems keep their code in the cy-suite organisation, one repository per
# system, referenced as a git submodule of bountytasks.
CODEBASE_REPO_ORG = "cy-suite"

TASK_TYPES = ("detect", "exploit", "patch")
TaskType = Literal["detect", "exploit", "patch"]
DEFAULT_TASK_TYPE: TaskType = "exploit"

BOUNTYBENCH_CACHE_PATH = CONCORDIA_EVALS_CACHE_PATH / "bountybench"
BOUNTYTASKS_DIR = BOUNTYBENCH_CACHE_PATH / "bountytasks"
CODEBASE_MIRRORS_DIR = BOUNTYBENCH_CACHE_PATH / "codebases"
WORKSPACES_DIR = BOUNTYBENCH_CACHE_PATH / "workspaces"
SANDBOX_COMPOSE_DIR = BOUNTYBENCH_CACHE_PATH / "compose"

# Shared docker network: every upstream task compose file declares it as an
# external network and the agent container joins it so that target hostnames
# (e.g. `lunary-app:3333`) resolve exactly as they do upstream.
SHARED_NETWORK = "shared_net"

AGENT_IMAGE = "inspect-bountybench-agent:latest"
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

# Per sample store flag set when the agent submits its work. Upstream only grades
# submitted work (`ExploitAgent`/`PatchAgent`/`DetectAgent` fail on "no_submission").
SUBMITTED_STORE_KEY = "bountybench_submitted"

DEFAULT_MAX_MESSAGES = 30
DEFAULT_RANDOM_SEED = 0
DEFAULT_SAMPLE_NUM: int | None = None

# The prompt embeds a `tree` listing of at most this many tokens.
CODEBASE_TREE_MAX_TOKENS = 300
CODEBASE_TREE_IGNORE = "_*|venv|node_modules|.git|__pycache__|.idea|.vscode"

# Command timeout for the commands the agent runs (upstream TIMEOUT_PER_COMMAND).
DEFAULT_COMMAND_TIMEOUT = 120
# Timeout used by our own setup commands inside the agent sandbox.
SANDBOX_COMMAND_TIMEOUT = 600
# Upstream edits the repository in place at setup (PIP_INSTALL_TIMEOUT).
INSTALL_TIMEOUT = 1200
# Upstream waits up to 300s for all compose containers to report healthy.
HEALTHCHECK_TIMEOUT = 300
# Upstream caps invariant test execution at 90 minutes.
INVARIANTS_TIMEOUT = 5400
INVARIANTS_RETRY = 3

DEFAULT_CPU_LIMIT = "4"
DEFAULT_MEMORY_LIMIT = "8g"
