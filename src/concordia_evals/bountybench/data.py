"""Task data loading for BountyBench.

The task inputs live in the `bountybench/bountytasks` repository (one directory
per target system, one directory per bounty). The repository is downloaded at
the pinned commit into the concordia_evals cache and read from there, so the
dataset itself is never vendored into this repository.
"""

import json
import logging
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.util import SandboxEnvironmentSpec

from concordia_evals.bountybench.constants import (
    AGENT_IMAGE,
    BOUNTYTASKS_COMMIT,
    BOUNTYTASKS_DIR,
    BOUNTYTASKS_URL,
    CODEBASE_MIRRORS_DIR,
    DEFAULT_CPU_LIMIT,
    DEFAULT_MEMORY_LIMIT,
    SANDBOX_COMPOSE_DIR,
    SANDBOX_MIRROR_DIR,
    SANDBOX_MIRRORS_DIR,
    SANDBOX_WHEELHOUSE,
    SHARED_NETWORK,
    UNAVAILABLE_SAMPLES,
)
from concordia_evals.bountybench.prompts import get_initial_prompt
from concordia_evals.bountybench.writeup import read_writeup

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class BountySpec:
    """Everything needed to build a sample for one (system, bounty) pair."""

    system: str
    bounty_number: str
    system_dir: Path
    bounty_dir: Path
    repo_metadata: dict[str, Any]
    bounty_metadata: dict[str, Any]

    @property
    def sample_id(self) -> str:
        return f"{self.system}_bounty_{self.bounty_number}"

    @property
    def vulnerable_commit(self) -> str:
        commit = self.bounty_metadata.get("vulnerable_commit")
        if not commit:
            raise ValueError(f"Missing vulnerable_commit for {self.sample_id}")
        return str(commit)

    @property
    def target_host(self) -> str:
        return str(self.repo_metadata.get("target_host") or "")

    @property
    def bounty_setup_hosts(self) -> list[str]:
        return list(self.bounty_metadata.get("bounty_setup_hosts") or [])

    @property
    def is_service_task(self) -> bool:
        """Whether the system runs as docker compose services."""
        return bool(self.target_host or self.bounty_setup_hosts)

    @property
    def invariant_thresholds(self) -> dict[str, float]:
        """Thresholds a patch has to keep, the way the scorer merges them."""
        thresholds = dict(self.repo_metadata.get("invariant_thresholds") or {})
        thresholds.update(self.bounty_metadata.get("invariant_thresholds") or {})
        return thresholds

    def writeup(self) -> str:
        writeup_path = self.bounty_dir / "writeup" / "writeup.html"
        if not writeup_path.exists():
            return ""
        return read_writeup(
            writeup_path.read_text(encoding="utf-8"),
            self.bounty_metadata.get("bounty_link"),
        )

    def info(self, task_type: str) -> str:
        repo_info = str(self.repo_metadata.get("info") or "")
        bounty_info = str(self.bounty_metadata.get("info") or "")
        if task_type == "exploit":
            return str(self.bounty_metadata.get("exploit_info") or "") or repo_info
        if task_type == "detect":
            return f"{repo_info}\n{bounty_info}"
        return repo_info


def _download(url: str, destination: Path) -> None:
    import requests

    logger.info(f"Downloading {url}")
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()
    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def ensure_bountytasks(cache_dir: Path = BOUNTYTASKS_DIR) -> Path:
    """Download and extract the pinned bountytasks snapshot (idempotent)."""
    marker = cache_dir / ".commit"
    if marker.exists() and marker.read_text().strip() == BOUNTYTASKS_COMMIT:
        return cache_dir

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=cache_dir.parent) as tmp:
        archive = Path(tmp) / "bountytasks.tar.gz"
        _download(BOUNTYTASKS_URL, archive)
        with tarfile.open(archive) as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(path=tmp, filter="data")
            else:  # pragma: no cover - Python < 3.12 has no extraction filters
                tar.extractall(path=tmp)
        extracted = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if len(extracted) != 1:
            raise RuntimeError(f"Unexpected bountytasks archive layout: {extracted}")
        shutil.move(str(extracted[0]), str(cache_dir))

    marker.write_text(BOUNTYTASKS_COMMIT + "\n")
    return cache_dir


def compatible_bounties(bounty_metadata: dict[str, Any]) -> list[str]:
    """Bounties whose reference patch also blocks this bounty's exploit.

    Upstream renamed the field from `compatible_patches` to `compatible_exploits`
    between task data versions, so both are accepted.
    """
    values = bounty_metadata.get("compatible_exploits") or bounty_metadata.get("compatible_patches")
    return [str(item) for item in values or []]


def list_bounty_specs(root: Path) -> list[BountySpec]:
    """Enumerate every (system, bounty) pair of a bountytasks snapshot."""
    specs: list[BountySpec] = []
    for system_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metadata_path = system_dir / "metadata.json"
        bounties_dir = system_dir / "bounties"
        if not metadata_path.exists() or not bounties_dir.is_dir():
            continue
        repo_metadata = json.loads(metadata_path.read_text())
        for bounty_dir in sorted(bounties_dir.iterdir()):
            bounty_metadata_path = bounty_dir / "bounty_metadata.json"
            if not bounty_metadata_path.exists():
                continue
            specs.append(
                BountySpec(
                    system=system_dir.name,
                    bounty_number=bounty_dir.name.removeprefix("bounty_"),
                    system_dir=system_dir,
                    bounty_dir=bounty_dir,
                    repo_metadata=repo_metadata,
                    bounty_metadata=json.loads(bounty_metadata_path.read_text()),
                )
            )
    return specs


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")


def ensure_codebase_mirror(spec: BountySpec, cache_dir: Path = CODEBASE_MIRRORS_DIR) -> Path:
    """Clone (once per system) the target repository as a bare mirror.

    The mirror is mounted read-only into the agent sandbox and is also the
    source for the per-sample checkouts used when building/running the target.
    """
    mirror = cache_dir / f"{spec.system}.git"
    ready = mirror / "HEAD"
    if ready.exists():
        return mirror

    url = f"https://github.com/{_codebase_org()}/{spec.system}.git"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning target codebase {url}")
    shutil.rmtree(mirror, ignore_errors=True)
    _run_git(["clone", "--mirror", url, str(mirror)])
    return mirror


def system_bounty_commits(spec: BountySpec) -> list[str]:
    """Return the vulnerable commits of every bounty of the system."""
    commits: list[str] = []
    for metadata_file in sorted((spec.system_dir / "bounties").glob("*/bounty_metadata.json")):
        commit = json.loads(metadata_file.read_text()).get("vulnerable_commit")
        if commit and commit not in commits:
            commits.append(commit)
    if spec.vulnerable_commit not in commits:
        commits.insert(0, spec.vulnerable_commit)
    return commits


def ensure_sandbox_mirror(spec: BountySpec, cache_dir: Path = SANDBOX_MIRRORS_DIR) -> Path:
    """Build the codebase mirror that is mounted into the agent sandbox.

    The upstream repositories are full GitHub mirrors: their later commits and
    pull request refs contain the fixes for the bounties, so a plain clone would
    hand the answer to the agent. The sandbox mirror keeps only the vulnerable
    commits of the system's bounties (with their ancestors and the tags that
    point at them), which is all the checkouts of the tasks need.
    """
    commits = system_bounty_commits(spec)
    mirror = cache_dir / f"{spec.system}.git"
    ready = mirror / ".commits"
    if ready.exists() and ready.read_text().split() == commits:
        return mirror

    source = ensure_codebase_mirror(spec)
    missing = _missing_commits(source, commits)
    if missing:
        logger.info(f"Fetching {len(missing)} commits of {spec.system} into the cache")
        _run_git(["fetch", "--quiet", "--tags", "origin"], cwd=source)

    shutil.rmtree(mirror, ignore_errors=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["init", "--bare", "--quiet", str(mirror)])
    tags = _reachable_tags(source, commits)
    for index, commit in enumerate(commits):
        # Fetch into a neutral namespace first: commits are referenced by tag
        # name in some tasks and a tag object cannot be written to a branch.
        staging = f"refs/bb/{index}"
        _run_git(
            ["fetch", "--quiet", "--no-tags", str(source), f"{commit}:{staging}"],
            cwd=mirror,
        )
        branch = f"refs/heads/bb-{index}"
        _run_git(
            ["update-ref", branch, _rev_parse(mirror, f"{staging}^{{commit}}")],
            cwd=mirror,
        )
        _run_git(["update-ref", "-d", staging], cwd=mirror)
        if index == 0:
            _run_git(["symbolic-ref", "HEAD", branch], cwd=mirror)
    for tag in tags:
        _run_git(
            [
                "fetch",
                "--quiet",
                "--no-tags",
                str(source),
                f"refs/tags/{tag}:refs/tags/{tag}",
            ],
            cwd=mirror,
        )
    # The vulnerable commits are referenced by name in some tasks (`v3.19.0`).
    for commit in commits:
        _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=mirror)
    ready.write_text("\n".join(commits) + "\n")
    return mirror


def _rev_parse(mirror: Path, rev: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", rev],
        cwd=mirror,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse {rev} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _missing_commits(mirror: Path, commits: list[str]) -> list[str]:
    missing: list[str] = []
    for commit in commits:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
            cwd=mirror,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            missing.append(commit)
    return missing


def _reachable_tags(mirror: Path, commits: list[str]) -> list[str]:
    """Tags that point at one of `commits` or at one of its ancestors."""
    tags: set[str] = set()
    for commit in commits:
        result = subprocess.run(
            ["git", "tag", "--merged", commit],
            cwd=mirror,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            tags.update(result.stdout.split())
    return sorted(tags)


def _codebase_org() -> str:
    from concordia_evals.bountybench.constants import CODEBASE_REPO_ORG

    return CODEBASE_REPO_ORG


def checkout_codebase(mirror: Path, commit: str, destination: Path) -> None:
    """Materialise a working copy of `mirror` at `commit`."""
    if destination.exists():
        shutil.rmtree(destination)
    _run_git(["clone", "--quiet", str(mirror), str(destination)])
    _run_git(["checkout", "--quiet", "--force", commit], cwd=destination)


def sandbox_compose_file(
    spec: BountySpec,
    image: str = AGENT_IMAGE,
    cache_dir: Path = SANDBOX_COMPOSE_DIR,
) -> Path:
    """Write (once per sample) the compose file for the agent sandbox.

    The sandbox runs on the internal shared network only (so the agent needs no
    network access at run time) and its image has the target repository
    installed at build time, plus the pruned codebase mirror for service tasks.
    """
    compose_path = cache_dir / f"{spec.system}_bounty_{spec.bounty_number}.yaml"
    # Service tasks clone their codebase from the pruned mirror at run time; the
    # image of a local task already carries the checkout.
    volumes = ""
    if spec.is_service_task:
        mirror = ensure_sandbox_mirror(spec)
        volumes = f"""    volumes:
      - {mirror.as_posix()}:{SANDBOX_MIRROR_DIR}:ro
"""
    compose_path.parent.mkdir(parents=True, exist_ok=True)
    compose_path.write_text(
        f"""services:
  default:
    image: {image}
    # built locally by `ensure_sample_image`, never pulled
    x-local: true
    init: true
    command: ["tail", "-f", "/dev/null"]
    working_dir: /app
    environment:
      # offline package index baked into the image: the tasks' own grading
      # scripts install their test dependencies from it
      PIP_NO_INDEX: "1"
      PIP_FIND_LINKS: {SANDBOX_WHEELHOUSE}
      PIP_DISABLE_PIP_VERSION_CHECK: "1"
    cpus: "{DEFAULT_CPU_LIMIT}"
    mem_limit: {DEFAULT_MEMORY_LIMIT}
{volumes}    networks:
      - {SHARED_NETWORK}

networks:
  {SHARED_NETWORK}:
    external: true
"""
    )
    return compose_path


def record_to_sample(
    spec: BountySpec,
    task_type: str,
    use_cwe: bool = False,
    require_submission: bool = True,
    image: str = AGENT_IMAGE,
) -> Sample:
    """Build an Inspect sample for one bounty and task type."""
    bounty_setup_hosts = spec.bounty_setup_hosts
    prompt = get_initial_prompt(
        task_type,
        target_host=spec.target_host,
        bounty_setup_hosts="\n".join(bounty_setup_hosts),
        task_description=spec.writeup(),
        info=spec.info(task_type),
        use_cwe=use_cwe,
        cwe=str(spec.bounty_metadata.get("CWE") or ""),
    )
    compose = sandbox_compose_file(spec, image=image)
    metadata = {
        "system": spec.system,
        "bounty_number": spec.bounty_number,
        "task_type": task_type,
        "require_submission": require_submission,
        "target_host": spec.target_host,
        "bounty_setup_hosts": bounty_setup_hosts,
        "cwe": spec.bounty_metadata.get("CWE"),
        "cve": spec.bounty_metadata.get("CVE"),
        "severity": spec.bounty_metadata.get("severity"),
        "bounty_link": spec.bounty_metadata.get("bounty_link"),
        "disclosure_bounty": spec.bounty_metadata.get("disclosure_bounty"),
        "patch_bounty": spec.bounty_metadata.get("patch_bounty"),
        "vulnerable_commit": spec.vulnerable_commit,
        "compatible_bounties": compatible_bounties(spec.bounty_metadata),
        "bountytasks_system_dir": str(spec.system_dir),
        "bountytasks_bounty_dir": str(spec.bounty_dir),
    }
    return Sample(
        id=spec.sample_id,
        input=prompt,
        metadata=metadata,
        sandbox=SandboxEnvironmentSpec("docker", str(compose)),
    )


def get_dataset(
    task_type: str,
    bounties: str | list[str] | None = None,
    sample_num: int | None = None,
    random_seed: int = 0,
    use_cwe: bool = False,
    require_submission: bool = True,
    image_for: Callable[[BountySpec], str] | None = None,
) -> Dataset:
    """Build the sample set for a task type.

    Args:
        task_type: One of `detect`, `exploit` or `patch`.
        bounties: Optional filter, either a list of sample ids
            (`"lunary_bounty_0"`) or `(system, bounty)` pairs such as
            `"lunary:0"`, or a single one of these.
        sample_num: Number of samples to keep (random subset, seeded).
        random_seed: Seed for the subset sampling.
        use_cwe: Pass the bounty CWE to the agent (upstream `--use_cwe`).
        require_submission: Only grade work the agent submitted.
        image_for: Resolve the sandbox image of a sample (built with the target
            repository installed).
    """
    root = ensure_bountytasks(BOUNTYTASKS_DIR)
    specs = list_bounty_specs(root)

    if bounties is not None:
        selected = [bounties] if isinstance(bounties, str) else list(bounties)
        wanted = {_normalise_bounty_key(item) for item in selected}
        available = {_bounty_key(spec) for spec in specs}
        missing = wanted - available
        if missing:
            raise ValueError(f"Unknown bounties: {sorted(missing)}")
        specs = [spec for spec in specs if _bounty_key(spec) in wanted]

    unavailable = UNAVAILABLE_SAMPLES.get(task_type, {})
    if bounties is not None:
        refused = [
            f"{spec.sample_id}: {unavailable[spec.sample_id]}"
            for spec in specs
            if spec.sample_id in unavailable
        ]
        if refused:
            raise ValueError(f"Not available for {task_type}: {refused}")
    specs = [spec for spec in specs if spec.sample_id not in unavailable]

    if task_type == "patch":
        # a patch is graded on the invariants it has to keep passing, so a system
        # that ships none cannot be graded at all: `imaginairy` and
        # `neural-compressor` define no `invariant_thresholds` and their scripts
        # report "no repo invariants for this system"
        specs = [spec for spec in specs if spec.invariant_thresholds]

    if sample_num is not None:
        if sample_num > len(specs):
            raise ValueError(
                f"sample_num={sample_num} exceeds the number of available bounties ({len(specs)})"
            )
        specs = random.Random(random_seed).sample(specs, sample_num)
        specs.sort(key=lambda spec: spec.sample_id)

    samples = [
        record_to_sample(
            spec,
            task_type,
            use_cwe=use_cwe,
            require_submission=require_submission,
            image=image_for(spec) if image_for else AGENT_IMAGE,
        )
        for spec in specs
    ]
    return MemoryDataset(samples=samples, name=f"bountybench_{task_type}")


def _bounty_key(spec: BountySpec) -> tuple[str, str]:
    return (spec.system.lower(), f"bounty_{spec.bounty_number}")


def load_bounty_spec(metadata: dict[str, Any]) -> BountySpec:
    """Rebuild the `BountySpec` of a sample from its metadata."""
    system_dir = Path(str(metadata["bountytasks_system_dir"]))
    bounty_dir = Path(str(metadata["bountytasks_bounty_dir"]))
    return BountySpec(
        system=str(metadata["system"]),
        bounty_number=str(metadata["bounty_number"]),
        system_dir=system_dir,
        bounty_dir=bounty_dir,
        repo_metadata=json.loads((system_dir / "metadata.json").read_text()),
        bounty_metadata=json.loads((bounty_dir / "bounty_metadata.json").read_text()),
    )


def _normalise_bounty_key(value: str) -> tuple[str, str]:
    """Accept `system:bounty`, `system/bounty_N` or `<sample_id>` forms."""
    text = value.strip()
    if ":" in text:
        system, bounty = text.split(":", 1)
    elif "/" in text:
        system, bounty = text.split("/", 1)
    else:
        marker = "_bounty_"
        if marker not in text:
            raise ValueError(
                f"Cannot parse bounty {value!r}; expected 'system:bounty', "
                f"'system/bounty_N' or '<system>_bounty_N'"
            )
        system, bounty = text.rsplit(marker, 1)
        bounty = f"bounty_{bounty}"
    bounty = bounty if bounty.startswith("bounty_") else f"bounty_{bounty}"
    return (system.lower(), bounty)
