"""Dataset access for CyberGym-E2E.

Two upstream artifacts are needed per task, both cached under the platform
user cache dir:

1. Task scripts from the GitHub repo (pinned commit): config.toml, prepare.sh,
   compile.sh, run_poc.sh, test.sh, patch.diff, plus project-level project.toml.
   Downloaded once as a tarball and extracted.
2. Task data from the HuggingFace dataset (pinned revision): src.tgz, poc.bin,
   crash.log. Downloaded lazily per task (the full dataset is ~160 GB).
"""

import logging
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from textwrap import dedent
from typing import Any

from platformdirs import user_cache_dir

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

CACHE_DIR = Path(user_cache_dir("concordia-evals")) / "cybergym-e2e"
REPO_DIR = CACHE_DIR / "repo"
DATA_DIR = CACHE_DIR / "data"
COMPOSE_DIR = CACHE_DIR / "compose_files"

GITHUB_COMMIT = "d0ae0f02e76ebc084389bf21e755f4b41ac9765a"
GITHUB_TARBALL_URL = f"https://codeload.github.com/sunblaze-ucb/cybergym-e2e/tar.gz/{GITHUB_COMMIT}"
HF_REPO_ID = "sunblaze-ucb/cybergym-e2e"
HF_REVISION = "3aee406e4e915e32527fea16de7f002630fa8c76"

# Upstream default (scripts/run_agent.py --default-build-image).
DEFAULT_BUILD_IMAGE = (
    "gcr.io/oss-fuzz-base/base-builder"
    "@sha256:8eda74a11e800aead5a041ee479a65b33dab3150d6e89e5694e2b6eb27be98fc"
)

TASK_DATA_FILES = ("src.tgz", "poc.bin", "crash.log")


def sync_scripts_repo(repo_dir: Path = REPO_DIR, force: bool = False) -> Path:
    """Download and extract the pinned upstream scripts repo.

    Args:
        repo_dir: Target directory for the extracted repo.
        force: Re-download even if already present.

    Returns:
        Path to the extracted repo root (contains projects/).
    """
    marker = repo_dir / ".commit"
    if (
        not force
        and (repo_dir / "projects").is_dir()
        and marker.exists()
        and marker.read_text().strip() == GITHUB_COMMIT
    ):
        return repo_dir

    logger.info("Downloading cybergym-e2e scripts repo at %s ...", GITHUB_COMMIT[:8])
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(GITHUB_TARBALL_URL, tmp_path)
        with tempfile.TemporaryDirectory() as extract_dir:
            with tarfile.open(tmp_path) as tar:
                tar.extractall(extract_dir, filter="data")
            # The tarball has a single top-level directory named cybergym-e2e-<commit>.
            top = next(Path(extract_dir).iterdir())
            for item in top.iterdir():
                shutil.move(str(item), repo_dir / item.name)
    finally:
        tmp_path.unlink(missing_ok=True)

    marker.write_text(GITHUB_COMMIT + "\n")
    return repo_dir


def list_tasks(repo_dir: Path = REPO_DIR) -> list[str]:
    """List all task paths (e.g. 'curl/arvo_66012') available in the scripts repo."""
    projects_dir = repo_dir / "projects"
    tasks = []
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        for task_dir in sorted(project_dir.iterdir()):
            if task_dir.is_dir() and (task_dir / "config.toml").exists():
                tasks.append(f"{project_dir.name}/{task_dir.name}")
    return tasks


def stratified_sample(tasks: list[str], n: int, seed: int) -> list[str]:
    """Pick n tasks with project-level diversity, reproducibly.

    Round-robin across projects (shuffled task pool per project): the first
    pass takes one task per project, later passes add depth. Same seed and
    task list always give the same subset.
    """
    import random

    rng = random.Random(seed)
    pools: list[list[str]] = []
    by_project: dict[str, list[str]] = {}
    for task in tasks:
        by_project.setdefault(task.split("/", 1)[0], []).append(task)
    for project in sorted(by_project):
        pool = sorted(by_project[project])
        rng.shuffle(pool)
        pools.append(pool)

    picked: list[str] = []
    while len(picked) < n:
        added = False
        for pool in pools:
            if pool:
                picked.append(pool.pop())
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break
    return sorted(picked)


def _task_attributes(task_path: str, repo_dir: Path) -> tuple[str, str, str, str]:
    """Extract (project, language, source_family, sanitizer) for a task.

    language comes from project.toml; source family from the task dir name
    prefix (arvo_ / oss-fuzz_); sanitizer from run_poc.sh's SANITIZER= line.
    """
    import re

    project, name = task_path.split("/", 1)
    config = load_task_config(task_path, repo_dir)
    language = config.get("language", "unknown")
    if name.startswith("arvo_"):
        source = "arvo"
    elif name.startswith("oss-fuzz_"):
        source = "oss-fuzz"
    else:
        source = "other"
    sanitizer = "unknown"
    run_poc = repo_dir / "projects" / task_path / "run_poc.sh"
    if run_poc.exists():
        match = re.search(r"SANITIZER=(\w+)", run_poc.read_text())
        if match:
            sanitizer = match.group(1)
    return project, language, source, sanitizer


def diverse_sample(tasks: list[str], n: int, seed: int, repo_dir: Path = REPO_DIR) -> list[str]:
    """Pick n tasks maximizing coverage: distinct projects first, then balance attributes.

    Within each project prefer the task whose (source, sanitizer) combination is
    currently least represented. Reproducible via seed.
    """
    import random

    rng = random.Random(seed)
    attrs = {t: _task_attributes(t, repo_dir) for t in tasks}

    by_project: dict[str, list[str]] = {}
    for task in tasks:
        by_project.setdefault(task.split("/", 1)[0], []).append(task)
    projects = sorted(by_project)
    rng.shuffle(projects)

    picked: list[str] = []
    combo_count: dict[tuple[str, str], int] = {}
    while len(picked) < n:
        added = False
        for project in projects:
            pool = [t for t in by_project[project] if t not in picked]
            if not pool:
                continue
            # Within this project, prefer the task whose (source, sanitizer)
            # combo is globally least represented so far; ties: seeded order.
            rng.shuffle(pool)
            best = min(
                pool,
                key=lambda t: combo_count.get((attrs[t][2], attrs[t][3]), 0),
            )
            picked.append(best)
            combo = (attrs[best][2], attrs[best][3])
            combo_count[combo] = combo_count.get(combo, 0) + 1
            added = True
            if len(picked) >= n:
                break
        if not added:
            break
    return sorted(picked)


def load_task_config(task_path: str, repo_dir: Path = REPO_DIR) -> dict[str, Any]:
    """Load merged config (project.toml overlaid with task config.toml)."""
    script_dir = repo_dir / "projects" / task_path
    if not script_dir.is_dir():
        raise ValueError(f"Unknown task: {task_path}")
    config: dict[str, Any] = {}
    project_toml = script_dir.parent / "project.toml"
    if project_toml.exists():
        config.update(tomllib.loads(project_toml.read_text()))
    config.update(tomllib.loads((script_dir / "config.toml").read_text()))
    return config


def download_task_data(
    task_path: str,
    data_dir: Path = DATA_DIR,
    required: bool = True,
) -> Path:
    """Download src.tgz / poc.bin / crash.log for one task from HuggingFace.

    Args:
        task_path: Task path like 'curl/arvo_66012'.
        data_dir: Root directory for downloaded data.
        required: If True, raise when a file is missing on the Hub. crash.log
            is always treated as optional in e2e mode by callers.

    Returns:
        Path to the task's local data directory.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    task_data_dir = data_dir / "projects" / task_path
    for filename in TASK_DATA_FILES:
        local = task_data_dir / filename
        if local.exists():
            continue
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"projects/{task_path}/{filename}",
                local_dir=data_dir,
                repo_type="dataset",
                revision=HF_REVISION,
            )
        except EntryNotFoundError:
            if required and filename != "crash.log":
                raise
            logger.debug("Optional file %s missing for %s", filename, task_path)
    return task_data_dir


def ensure_image(build_image: str) -> None:
    """Pull the task's Docker image if not present locally (on-demand pull)."""
    import docker
    from docker.errors import ImageNotFound

    client = docker.from_env()
    try:
        client.images.get(build_image)
    except ImageNotFound:
        logger.info("Pulling image %s ...", build_image)
        if "@sha256:" in build_image:
            client.images.pull(build_image)
        else:
            repo, tag = build_image.rsplit(":", 1)
            client.images.pull(repo, tag=tag)


def write_compose_file(
    sample_slug: str,
    build_image: str,
    task_data_dir: Path,
    script_dir: Path,
    compose_dir: Path = COMPOSE_DIR,
) -> str:
    """Write a per-sample Docker Compose file and return its path.

    The task data and scripts are mounted read-only; the setup solver copies
    what it needs into the container's writable layer.
    """
    compose_dir.mkdir(parents=True, exist_ok=True)
    compose_file = compose_dir / f"{sample_slug}.yaml"
    compose_file.write_text(
        dedent(
            f"""\
            services:
              default:
                image: {build_image}
                platform: linux/amd64
                volumes:
                  - {task_data_dir}:/task_data:ro
                  - {script_dir}:/task_scripts:ro
                working_dir: /src
                # MSan/ASan fuzzer binaries re-exec via
                # personality(ADDR_NO_RANDOMIZE) when their shadow-memory mmap
                # collides; Docker's default seccomp profile blocks the combined
                # personality value, killing ~50% of MSan binary runs (meson's
                # compiler sanity check trips on this). Verified 2026-11-10.
                security_opt:
                  - seccomp=unconfined
                # Make ffmpeg's FATE sample rsync fail fast instead of hanging
                # on slow links (samples are unused: upstream test.sh drops
                # SAMPLES=, and prepare.sh already made the fetch non-fatal).
                extra_hosts:
                  - "fate-suite.ffmpeg.org:127.0.0.1"
                entrypoint:
                  - /bin/sh
                  - -c
                  - exec tail -f /dev/null
            """
        )
    )
    return str(compose_file)
