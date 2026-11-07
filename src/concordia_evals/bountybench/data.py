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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.util import SandboxEnvironmentSpec

from concordia_evals.bountybench.constants import (
    AGENT_DOCKERFILE,
    AGENT_IMAGE,
    BOUNTYTASKS_COMMIT,
    BOUNTYTASKS_DIR,
    BOUNTYTASKS_URL,
    CODEBASE_MIRRORS_DIR,
    DEFAULT_CPU_LIMIT,
    DEFAULT_MEMORY_LIMIT,
    SANDBOX_COMPOSE_DIR,
    SANDBOX_MIRROR_DIR,
    SHARED_NETWORK,
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
    values = bounty_metadata.get("compatible_exploits") or bounty_metadata.get(
        "compatible_patches"
    )
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
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")


def ensure_codebase_mirror(
    spec: BountySpec, cache_dir: Path = CODEBASE_MIRRORS_DIR
) -> Path:
    """Clone (once per system) the target repository as a bare mirror.

    The mirror is mounted read-only into the agent sandbox and is also the
    source for the per-sample checkouts used when building/running the target.
    """
    mirror = cache_dir / f"{spec.system}.git"
    if mirror.exists():
        return mirror

    url = f"https://github.com/{_codebase_org()}/{spec.system}.git"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning target codebase {url}")
    _run_git(["clone", "--mirror", url, str(mirror)])
    return mirror


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
    spec: BountySpec, cache_dir: Path = SANDBOX_COMPOSE_DIR
) -> Path:
    """Write (once per system) the compose file for the agent sandbox."""
    compose_path = cache_dir / f"{spec.system}.yaml"
    mirror = ensure_codebase_mirror(spec)
    if compose_path.exists() and mirror.as_posix() in compose_path.read_text():
        return compose_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    compose_path.write_text(
        f"""services:
  default:
    build:
      context: {PACKAGE_DIR.as_posix()}
      dockerfile: {AGENT_DOCKERFILE}
    image: {AGENT_IMAGE}
    init: true
    command: ["tail", "-f", "/dev/null"]
    working_dir: /app
    cpus: "{DEFAULT_CPU_LIMIT}"
    mem_limit: {DEFAULT_MEMORY_LIMIT}
    volumes:
      - {mirror.as_posix()}:{SANDBOX_MIRROR_DIR}:ro
    networks:
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
    compose = sandbox_compose_file(spec)
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
        "codebase_mirror": str(ensure_codebase_mirror(spec)),
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

    if sample_num is not None:
        if sample_num > len(specs):
            raise ValueError(
                f"sample_num={sample_num} exceeds the number of available "
                f"bounties ({len(specs)})"
            )
        specs = random.Random(random_seed).sample(specs, sample_num)
        specs.sort(key=lambda spec: spec.sample_id)

    samples = [
        record_to_sample(
            spec, task_type, use_cwe=use_cwe, require_submission=require_submission
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
