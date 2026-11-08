"""Host and sandbox side environment handling for BountyBench.

Upstream runs each task against a per-task copy of the bountytasks layout: setup
scripts bring up the target system with `docker compose`, the agent works in a
Kali container on the shared docker network, and the grading scripts
(`verify.sh`, invariant tests) run on the host against the published ports and
container names of the target stack.

This module keeps that layout:

* `Workspace` owns the host side working copy and drives the target containers.
* the `sbx_*` helpers prepare and drive the agent container, which is the
  Inspect docker sandbox (`/app` mirrors the upstream agent workspace).
"""

import atexit
import base64
import io
import logging
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Callable

from inspect_ai.util import ExecResult, SandboxEnvironment

from concordia_evals.bountybench.constants import (
    AGENT_DOCKERFILE,
    AGENT_IMAGE,
    AGENT_IMAGE_NAME,
    CODEBASE_TREE_IGNORE,
    CODEBASE_TREE_MAX_TOKENS,
    HEALTHCHECK_TIMEOUT,
    INSTALL_TIMEOUT,
    SAMPLE_IMAGES_DIR,
    SANDBOX_APP_DIR,
    SANDBOX_CODEBASE_DIR,
    SANDBOX_EXPLOIT_DIR,
    SANDBOX_MIRROR_DIR,
    SANDBOX_VERIFY_DIR,
    SHARED_NETWORK,
    WORKSPACES_DIR,
)
from concordia_evals.bountybench.data import (
    BountySpec,
    checkout_codebase,
    ensure_codebase_mirror,
)

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent

# Fields of a `git diff --numstat` line: added, removed, path.
NUMSTAT_FIELDS = 3

# "Container <name> (Started|Healthy)" lines printed by `docker compose up`.
CONTAINER_PATTERN = re.compile(r"Container\s+([^\s]+)\s+(Started|Healthy)")

# Identity used for the throwaway git repositories inside the sandbox.
GIT_IDENTITY = "-c user.email=bountybench@example.com -c user.name=bountybench"

# Workspaces whose containers may still be running; torn down on interpreter
# exit so that an interrupted evaluation does not leave a target stack behind.
_LIVE_WORKSPACES: dict[str, "Workspace"] = {}


def run_host(
    command: list[str], cwd: Path | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command on the evaluator host."""
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ensure_shared_network(name: str = SHARED_NETWORK) -> None:
    """Create the upstream shared docker network as an internal network.

    The agent container is attached to this network only, so an internal network
    is what keeps the sandbox off the internet. Target systems keep the networks
    their compose files declare, and reach the internet through those as upstream.
    """
    existing = run_host(["docker", "network", "ls", "--format", "{{.Name}}"])
    if existing.returncode != 0:
        raise RuntimeError(f"Docker is not available: {existing.stderr.strip()}")
    if name in existing.stdout.split():
        internal = run_host(
            ["docker", "network", "inspect", "--format", "{{.Internal}}", name]
        )
        if internal.stdout.strip() == "true":
            return
        removed = run_host(["docker", "network", "rm", name])
        if removed.returncode != 0:
            raise RuntimeError(
                f"docker network {name} exists without --internal and is in use, "
                f"so the agent sandbox could reach the internet: "
                f"{removed.stderr.strip()}. Stop the containers attached to it "
                f"and run again."
            )
    created = run_host(["docker", "network", "create", "--internal", name])
    if created.returncode != 0:
        raise RuntimeError(f"Unable to create docker network {name}")


# Default docker bridge: the only way to the internet used by this evaluation.
EGRESS_NETWORK = "bridge"


async def set_sandbox_egress(sbx: SandboxEnvironment, enabled: bool) -> None:
    """Attach or detach the sandbox to the egress network.

    The sandbox is only attached to the internal shared network, so it has no
    internet access while the agent works. The dataset's grading scripts install
    their own dependencies, so they are run inside such a window.
    """
    hostname = await sh(sbx, "hostname")
    container = hostname.stdout.strip()
    if not container:
        logger.warning("Unable to resolve the sandbox container for network access")
        return
    action = "connect" if enabled else "disconnect"
    args = ["docker", "network", action, EGRESS_NETWORK, container]
    if not enabled:
        args.insert(3, "-f")
    result = run_host(args)
    # `connect` fails when the container is already attached and `disconnect`
    # when it is not, which is the state we want in both cases.
    expected = ["already exists", "is not connected"]
    if result.returncode != 0 and not any(m in result.stderr for m in expected):
        logger.warning(
            "docker network %s %s failed: %s", action, EGRESS_NETWORK, result.stderr
        )


def image_exists(tag: str) -> bool:
    return run_host(["docker", "image", "inspect", tag]).returncode == 0


def ensure_agent_image(tag: str = AGENT_IMAGE) -> None:
    """Build the base sandbox image (Kali + toolchain) if it is missing."""
    if image_exists(tag):
        return
    logger.info(f"Building the agent image {tag}")
    built = run_host(
        [
            "docker",
            "build",
            "--tag",
            tag,
            "--file",
            str(PACKAGE_DIR / AGENT_DOCKERFILE),
            str(PACKAGE_DIR),
        ],
        timeout=INSTALL_TIMEOUT,
    )
    if built.returncode != 0:
        raise RuntimeError(f"Unable to build {tag}: {built.stderr.strip()}")


def sample_image_tag(spec: BountySpec) -> str:
    """Image of one sample: the agent image plus the installed target repository."""
    return f"{AGENT_IMAGE_NAME}-{spec.system}-{spec.bounty_number}:latest"


def install_script(
    codebase: Path, install_command: str | None, is_python: bool | None
) -> str | None:
    """Command that installs the repository, as upstream's Kali env does.

    Returns None for repositories that are not installed (node), as upstream.
    """
    if is_python is None:
        python_repo = (codebase / "setup.py").exists() or (
            codebase / "pyproject.toml"
        ).exists()
    else:
        python_repo = is_python
    if python_repo:
        return install_command or "pip install -e ."
    if not (codebase / "CMakeLists.txt").exists():
        return None
    return install_command or "make install"


def ensure_sample_image(spec: BountySpec, cache_dir: Path = SAMPLE_IMAGES_DIR) -> str:
    """Build (once) the sandbox image of a sample, with the repository installed.

    Upstream installs the repository inside the running Kali container, which
    needs the internet at that point. Doing it while building the image lets the
    sandbox run without any external network access.
    """
    context = cache_dir / f"{spec.system}_bounty_{spec.bounty_number}"
    codebase = context / "codebase"
    checkout_codebase(ensure_codebase_mirror(spec), spec.vulnerable_commit, codebase)
    command = install_script(
        codebase,
        spec.repo_metadata.get("install_command"),
        spec.repo_metadata.get("is_python"),
    )
    if command is None:
        # Nothing to install: the base image is the sample image.
        ensure_agent_image()
        return AGENT_IMAGE

    tag = sample_image_tag(spec)
    if image_exists(tag):
        return tag

    ensure_agent_image()
    context.mkdir(parents=True, exist_ok=True)
    dockerfile = context / AGENT_DOCKERFILE
    dockerfile.write_text(
        f"""# Generated by concordia_evals.bountybench: {spec.sample_id}
FROM {AGENT_IMAGE}
RUN --mount=type=bind,source=codebase,target={SANDBOX_CODEBASE_DIR},rw \
    cd {SANDBOX_CODEBASE_DIR} && {command}
"""
    )
    logger.info(f"Building the sandbox image {tag} ({spec.sample_id})")
    built = run_host(
        [
            "docker",
            "build",
            "--tag",
            tag,
            "--file",
            str(dockerfile),
            str(context),
        ],
        timeout=INSTALL_TIMEOUT,
    )
    if built.returncode != 0:
        raise RuntimeError(
            f"Unable to build the sandbox image of {spec.sample_id}: "
            f"{built.stderr.strip()[-2_000:]}"
        )
    return tag


class Workspace:
    """Host side working copy for one sample.

    The layout matches an upstream task directory: the system files (Dockerfile,
    compose file, setup and invariant scripts) sit next to `codebase/` and
    `bounties/bounty_N/`.
    """

    def __init__(self, spec: BountySpec, root: Path) -> None:
        self.spec = spec
        self.root = root
        self.containers: list[str] = []

    @classmethod
    def create(cls, spec: BountySpec, cache_dir: Path = WORKSPACES_DIR) -> "Workspace":
        cache_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{spec.sample_id}-", dir=cache_dir))
        workspace = cls(spec=spec, root=root)
        workspace.prepare()
        _LIVE_WORKSPACES[str(root)] = workspace
        return workspace

    @classmethod
    def attach(cls, spec: BountySpec, root: str) -> "Workspace":
        """Reattach to a workspace created by the setup solver."""
        workspace = _LIVE_WORKSPACES.get(root)
        if workspace is None:
            raise RuntimeError(f"No live workspace at {root}")
        return workspace

    @classmethod
    def lookup(cls, root: Path) -> "Workspace | None":
        """Look a live workspace up by directory."""
        return _LIVE_WORKSPACES.get(str(root))

    # ------------------------------------------------------------------ layout
    @property
    def codebase(self) -> Path:
        return self.root / "codebase"

    @property
    def bounty_dir(self) -> Path:
        return self.root / "bounties" / f"bounty_{self.spec.bounty_number}"

    @property
    def verify_files_dir(self) -> Path:
        return self.bounty_dir / "verify_files"

    @property
    def reference_exploit_dir(self) -> Path:
        return self.bounty_dir / "exploit_files"

    @property
    def has_compose_file(self) -> bool:
        return (self.root / "docker-compose.yml").exists() or (
            self.root / "docker-compose.yaml"
        ).exists()

    def prepare(self) -> None:
        """Copy system files and check out the codebase at the vulnerable commit."""
        shutil.copytree(
            self.spec.system_dir,
            self.root,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("codebase"),
        )
        # Keep only the graded bounty so that sibling reports do not leak into
        # the workspace and invariant scripts cannot pick up the wrong bounty.
        bounties_dir = self.root / "bounties"
        for sibling in bounties_dir.iterdir():
            if sibling.name != f"bounty_{self.spec.bounty_number}":
                shutil.rmtree(sibling, ignore_errors=True)
        clone_codebase(
            ensure_codebase_mirror(self.spec),
            self.spec.vulnerable_commit,
            self.codebase,
        )

    # ------------------------------------------------------------ target stack
    def start(self, bounty_setup: bool = True) -> None:
        """Bring up the target system the way upstream's setup resources do.

        Upstream binds a bounty setup resource for `exploit`/`patch` phases only;
        `detect` runs with the repository setup alone.
        """
        self._run_setup_script(self.root, "setup_repo_env.sh")
        if bounty_setup:
            self._run_setup_script(
                self.bounty_dir / "setup_files", "setup_bounty_env.sh"
            )
        self.wait_until_healthy()

    def _run_setup_script(self, directory: Path, script: str) -> None:
        script_path = directory / script
        if not script_path.exists():
            return
        result = run_host(["bash", script], cwd=directory, timeout=3_600)
        logger.debug("%s output:\n%s\n%s", script, result.stdout, result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"{script} failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        if not result.stdout:
            return
        output = result.stdout + result.stderr
        for name in sorted(
            {match.group(1) for match in CONTAINER_PATTERN.finditer(output)}
        ):
            if name not in self.containers:
                self.containers.append(name)

    def wait_until_healthy(
        self, timeout: int = HEALTHCHECK_TIMEOUT, check_interval: int = 2
    ) -> None:
        """Wait until every container started by the setup scripts is healthy.

        Ported from upstream `BaseSetupResource.wait_until_all_containers_healthy`:
        a container reporting anything other than `healthy` or `starting` fails
        the setup.
        """
        pending = list(self.containers)
        if not pending:
            return
        start = time.time()
        while pending:
            name = pending[0]
            status = _container_health(name)
            if status == "healthy":
                pending.pop(0)
            elif status != "starting":
                raise RuntimeError(
                    f"Container '{name}' has unexpected health status: {status}"
                )
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Not all containers became healthy within {timeout} seconds"
                )
            time.sleep(check_interval)

    def stop(self) -> None:
        """Tear down the target stack and remove the working copy."""
        _LIVE_WORKSPACES.pop(str(self.root), None)
        if self.root.exists() and self.has_compose_file:
            run_host(["docker", "compose", "down", "-v"], cwd=self.root, timeout=900)
        shutil.rmtree(self.root, ignore_errors=True)

    def restart(self, bounty_setup: bool = True) -> None:
        """Restart the target system after the codebase changed."""
        if not self.has_compose_file:
            # Without compose the setup scripts are the whole environment.
            self.start(bounty_setup=bounty_setup)
            return
        run_host(["docker", "compose", "down", "-v"], cwd=self.root, timeout=900)
        self.start(bounty_setup=bounty_setup)

    # --------------------------------------------------------------- codebase
    def reset_codebase(self) -> None:
        """Restore the working copy to the vulnerable commit."""
        self.checkout(self.spec.vulnerable_commit)

    def checkout(self, commit: str) -> None:
        for args in (["checkout", "--force", commit], ["clean", "-fdq"]):
            result = run_host(["git", *args], cwd=self.codebase)
            if result.returncode != 0:
                raise RuntimeError(
                    f"git {' '.join(args)} failed: {result.stderr.strip()}"
                )

    def copy_patch_file(self, source: Path, destination: Path) -> None:
        """Copy one file of a reference patch into the working copy.

        Upstream `apply_patch_to_bounty` copies whole replacement files instead of
        applying a diff, so this mirrors it exactly.
        """
        if not source.exists():
            raise RuntimeError(f"Patch file not found: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        run_host(["git", "add", str(destination)], cwd=self.codebase)

    def run_git(
        self, args: list[str], cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run git in the workspace."""
        return run_host(["git", *args], cwd=cwd or self.root)

    def apply_diff(self, diff: str) -> bool:
        """Apply the agent's patch (a `git diff`) to the working copy."""
        if not diff.strip():
            return False
        patch_path = self.root / "agent_patch.diff"
        patch_path.write_text(diff)
        for extra in (["--3way"], ["--reject"]):
            result = run_host(
                ["git", "apply", "--whitespace=nowarn", *extra, str(patch_path)],
                cwd=self.codebase,
            )
            if result.returncode == 0:
                return True
        logger.warning("Failed to apply the agent patch: %s", result.stderr)
        return False

    # ------------------------------------------------------------ subprocesses
    def run_script(
        self, directory: Path, script: str, timeout: int | None = None
    ) -> ExecResult[str]:
        """Run one of the task's shell scripts on the host."""
        result = run_host(["bash", script], cwd=directory, timeout=timeout)
        return ExecResult(
            success=result.returncode == 0,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def run_container_script(
    workspace_root: Path,
    work_dir: str,
    script: str = "exploit.sh",
    image: str = AGENT_IMAGE,
    network: str = SHARED_NETWORK,
    timeout: int = 900,
) -> ExecResult[str]:
    """Run a task script in a container with the workspace mounted at `/app`.

    Upstream runs the reference exploits this way (`docker run
    cybench/bountyagent`), so files they write land in the task directory on the
    host and the host side `verify.sh` can see them.
    """
    result = run_host(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            network,
            "-v",
            f"{workspace_root}:/app",
            "-w",
            f"/app/{work_dir}",
            image,
            "bash",
            script,
        ],
        timeout=timeout,
    )
    return ExecResult(
        success=result.returncode == 0,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def clone_codebase(mirror: Path, commit: str, destination: Path) -> None:
    """Clone `mirror` and check out `commit` into `destination`."""
    if destination.exists():
        shutil.rmtree(destination)
    clone = run_host(["git", "clone", "--quiet", str(mirror), str(destination)])
    if clone.returncode != 0:
        raise RuntimeError(f"git clone failed: {clone.stderr.strip()}")
    checkout = run_host(
        ["git", "checkout", "--quiet", "--force", commit], cwd=destination
    )
    if checkout.returncode != 0:
        raise RuntimeError(f"git checkout failed: {checkout.stderr.strip()}")


def _container_health(name: str) -> str:
    """Raw health status of a container (empty when it defines no healthcheck)."""
    result = run_host(
        ["docker", "inspect", "--format={{json .State.Health.Status}}", name]
    )
    return result.stdout.strip().strip("'\"").strip()


# ---------------------------------------------------------------------- sandbox


async def sh(
    sbx: SandboxEnvironment,
    command: str,
    cwd: str | None = None,
    timeout: int | None = None,
) -> ExecResult[str]:
    """Run a shell command inside the agent sandbox."""
    return await sbx.exec(["bash", "-lc", command], cwd=cwd, timeout=timeout)


def tree_filter(
    exclude: set[str] = set(),
    exclude_paths: set[str] = set(),
    skip_hidden: bool = False,
) -> Callable[[tarfile.TarInfo], tarfile.TarInfo | None]:
    """Build the tar filter used to copy host trees into the sandbox.

    `exclude` matches top level entries, `exclude_paths` matches relative
    directories at any depth and `skip_hidden` drops any dot file.
    """

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if parts and parts[0] in exclude:
            return None
        relative = Path(*parts).as_posix() if parts else "."
        if any(
            relative == path or relative.startswith(f"{path}/")
            for path in exclude_paths
        ):
            return None
        if skip_hidden and any(part.startswith(".") and part != "." for part in parts):
            return None
        return info

    return _filter


async def upload_tree(
    sbx: SandboxEnvironment,
    local_dir: Path,
    remote_dir: str,
    exclude: set[str] = set(),
    exclude_paths: set[str] = set(),
    skip_hidden: bool = False,
) -> None:
    """Copy a host directory into the sandbox (tar piped through stdin).

    `exclude` matches top level entries, `exclude_paths` matches relative
    directories at any depth.
    """
    buffer = io.BytesIO()
    _filter = tree_filter(exclude, exclude_paths, skip_hidden)
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(local_dir, arcname=".", filter=_filter)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    result = await sbx.exec(
        [
            "bash",
            "-lc",
            f"mkdir -p {remote_dir} && base64 -d | tar xzf - -C {remote_dir}",
        ],
        input=payload,
        timeout=600,
    )
    if not result.success:
        raise RuntimeError(f"Failed to upload {local_dir}: {result.stderr}")


async def download_tree(
    sbx: SandboxEnvironment, remote_dir: str, local_dir: Path, exclude: set[str] = set()
) -> None:
    """Copy a sandbox directory to the host.

    Excludes are expanded in the shell so that large directories (the codebase)
    are never serialised.
    """
    excludes = " ".join(f"--exclude=./{name}" for name in sorted(exclude))
    result = await sh(
        sbx, f"tar czf - {excludes} -C {remote_dir} . | base64 -w 0", timeout=600
    )
    if not result.success:
        raise RuntimeError(f"Failed to read {remote_dir}: {result.stderr}")
    local_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(result.stdout))) as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(path=local_dir, filter="data")
        else:  # pragma: no cover - Python < 3.12 has no extraction filters
            root = local_dir.resolve()
            for member in tar.getmembers():
                if not _is_safe_member(root, member):
                    raise RuntimeError(f"Refusing to extract {member.name}")
            tar.extractall(path=local_dir)


def _is_safe_member(root: Path, member: tarfile.TarInfo) -> bool:
    """Reject archive members that would escape `root` (`..`, absolute, links)."""
    target = (root / member.name).resolve()
    if not target.is_relative_to(root):
        return False
    if member.issym() or member.islnk():
        link = (target.parent / member.linkname).resolve()
        if not link.is_relative_to(root):
            return False
    return True


async def run_script_in_sandbox(
    sbx: SandboxEnvironment,
    directory: str,
    script: str,
    timeout: int | None = None,
) -> ExecResult[str]:
    """Run one of the task's shell scripts inside the sandbox."""
    return await sh(sbx, f"bash {script}", cwd=directory, timeout=timeout)


async def init_sandbox(sbx: SandboxEnvironment, spec: BountySpec) -> None:
    """Create the agent workspace inside the sandbox.

    Mirrors upstream `InitFilesResource`: a codebase checkout on a `dev` branch,
    an empty `exploit_files` directory and a copy of the bounty `verify_files`.
    """
    clone = await sh(
        sbx,
        " && ".join(
            [
                f"mkdir -p {SANDBOX_APP_DIR}",
                f"rm -rf {SANDBOX_CODEBASE_DIR} {SANDBOX_VERIFY_DIR}",
                f"git clone --quiet {SANDBOX_MIRROR_DIR} {SANDBOX_CODEBASE_DIR}",
                f"git -C {SANDBOX_CODEBASE_DIR} checkout --quiet --force "
                f"{spec.vulnerable_commit}",
                f"git -C {SANDBOX_CODEBASE_DIR} checkout --quiet -B dev",
                f"mkdir -p {SANDBOX_EXPLOIT_DIR}",
            ]
        ),
        timeout=1_800,
    )
    if not clone.success:
        raise RuntimeError(f"Failed to prepare the agent workspace: {clone.stderr}")
    await upload_tree(
        sbx, spec.bounty_dir / "verify_files", SANDBOX_VERIFY_DIR, skip_hidden=True
    )
    await sh(sbx, f"chmod -R a+x {SANDBOX_VERIFY_DIR} || true")
    # Upstream tracks the workspace with a git repository (ignoring the codebase)
    # and uses it to drop the side effects of each exploit run.
    await sh(
        sbx,
        " && ".join(
            [
                f"printf '*.log\\n.DS_Store\\nnode_modules/\\ncodebase\\n' > "
                f"{SANDBOX_APP_DIR}/.gitignore",
                f"git init --quiet {SANDBOX_APP_DIR}",
                f"git -C {SANDBOX_APP_DIR} branch -m main",
                f"git -C {SANDBOX_APP_DIR} add .",
                f"git -C {SANDBOX_APP_DIR} {GIT_IDENTITY} commit --quiet -m 'Initial commit'",
            ]
        ),
    )


async def snapshot_workspace(sbx: SandboxEnvironment) -> None:
    """Commit the current workspace state (upstream `git_commit(tmp_dir)`)."""
    await sh(
        sbx,
        f"git add -A && git {GIT_IDENTITY} commit --quiet -m snapshot || true",
        cwd=SANDBOX_APP_DIR,
    )


async def clean_workspace(sbx: SandboxEnvironment) -> None:
    """Drop the side effects of an exploit run (upstream `git_clean(tmp_dir)`)."""
    await sh(
        sbx,
        "git clean -fdq ; git checkout --quiet -- .",
        cwd=SANDBOX_APP_DIR,
    )


async def codebase_listing(sbx: SandboxEnvironment) -> str:
    """`tree` listing of the agent workspace, as upstream embeds in the prompt."""
    tree_depths = (4, 3, 2)
    depth = tree_depths[-1]
    tokens = 0
    structure = ""
    for candidate in tree_depths:
        depth = candidate
        result = await sh(
            sbx, f"tree -L {candidate} -I '{CODEBASE_TREE_IGNORE}'", cwd=SANDBOX_APP_DIR
        )
        structure = result.stdout
        tokens = len(structure.split())
        if tokens < CODEBASE_TREE_MAX_TOKENS:
            break
        if candidate == tree_depths[-1]:
            structure = (await sh(sbx, "ls -la", cwd=SANDBOX_APP_DIR)).stdout
    command = (
        f"tree -L {depth} -I '{CODEBASE_TREE_IGNORE}'"
        if tokens < CODEBASE_TREE_MAX_TOKENS
        else "ls -la"
    )
    return f"$ {command}\n{structure}"


async def sandbox_git_diff(sbx: SandboxEnvironment) -> str:
    """Diff of the agent's changes to its codebase copy (upstream `git_diff`)."""
    await sh(sbx, "git add -A", cwd=SANDBOX_CODEBASE_DIR)
    numstat = await sh(sbx, "git diff --cached --numstat", cwd=SANDBOX_CODEBASE_DIR)
    binary = {
        parts[2]
        for line in numstat.stdout.splitlines()
        if (parts := line.split("\t"))
        and len(parts) == NUMSTAT_FIELDS
        and parts[0] == "-"
    }
    pathspecs = " ".join(f":(exclude){path}" for path in sorted(binary))
    diff = await sh(sbx, f"git diff --cached -- {pathspecs}", cwd=SANDBOX_CODEBASE_DIR)
    return diff.stdout


async def reset_sandbox_codebase(sbx: SandboxEnvironment, commit: str) -> None:
    """Restore the sandbox codebase copy to `commit`."""
    await sh(
        sbx,
        f"git checkout --force {commit} && git clean -fdq",
        cwd=SANDBOX_CODEBASE_DIR,
    )


@atexit.register
def _stop_live_workspaces() -> None:
    for workspace in list(_LIVE_WORKSPACES.values()):
        try:
            workspace.stop()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to stop workspace on exit", exc_info=True)
