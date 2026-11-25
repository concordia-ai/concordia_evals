"""Host and sandbox side environment handling for BountyBench.

Upstream runs each task against a per-task copy of the bountytasks layout: setup
scripts bring up the target system with `docker compose`, the agent works in a
Kali container on the shared docker network, and the grading scripts
(`verify.sh`, invariant tests) run on the host against the published ports and
container names of the target stack.

This module keeps that layout:

* `Workspace` owns the host side working copy and drives the target containers.
* the sample's own shell scripts (setup, invariants, verify) run in a second
  container, `Workspace.run_script`, so that what they install cannot touch the
  evaluator's python environment while they still reach the target stack.
* the `sbx_*` helpers prepare and drive the agent container, which is the
  Inspect docker sandbox (`/app` mirrors the upstream agent workspace).
"""

import asyncio
import atexit
import base64
import fcntl
import hashlib
import io
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.util import ExecResult, SandboxEnvironment

from concordia_evals.bountybench.constants import (
    AGENT_DOCKERFILE,
    AGENT_IMAGE,
    AGENT_IMAGE_NAME,
    CODEBASE_TREE_IGNORE,
    CODEBASE_TREE_MAX_TOKENS,
    HEALTHCHECK_TIMEOUT,
    INSTALL_TIMEOUT,
    LOCKS_DIR,
    NLTK_DATA_DIR,
    SAMPLE_IMAGES_DIR,
    SANDBOX_APP_DIR,
    SANDBOX_CODEBASE_DIR,
    SANDBOX_COMPOSE_DIR,
    SANDBOX_EXPLOIT_DIR,
    SANDBOX_MIRROR_DIR,
    SANDBOX_VERIFY_DIR,
    SANDBOX_WHEELHOUSE,
    SHARED_NETWORK,
    STACK_BUILD_TIMEOUT,
    TIKTOKEN_CACHE_DIR,
    WORKSPACES_DIR,
)
from concordia_evals.bountybench.data import (
    BountySpec,
    checkout_codebase,
    ensure_codebase_mirror,
    ensure_sandbox_mirror,
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


# BuildKit does not forward the proxy of the client environment to `RUN`, and some
# builds reach hosts no mirror carries (`pypi.org`, `huggingface.co`). The evaluation
# runs with the proxy in its environment, so pass it explicitly.
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY")


def proxy_build_args() -> list[str]:
    args: list[str] = []
    for name in PROXY_VARIABLES:
        value = os.environ.get(name)
        if value:
            args += ["--build-arg", f"{name}={value}"]
    return args


def run_host(
    command: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command on the evaluator host.

    输出按文本读、但不严格解码：赏金的 shell 脚本会把目标返回的原始字节直接打到 stdout
    （`InvokeAI` 的参考 exploit 请求头带 `Accept-Encoding: gzip` 却没加 `curl --compressed`），
    判定只看退出码，解码失败不该变成 `UnicodeDecodeError`——它会从 scorer 上抛，把整个 task 中止。
    """
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
        env={**os.environ, **env} if env else None,
    )


async def acquire_stack_lock(system: str, poll_interval: float = 0.5) -> int:
    """Hold the target stack of one system for the caller.

    The compose files of the bounties name their containers (`container_name:
    mlflow`, `lunary-postgres`) and their scripts `docker exec` those names, so
    two samples of the same system would fight over one container. The lock is a
    file, released by the kernel if the process dies.

    `flock` 本身是不可中断的阻塞调用：放进线程里等锁时，样本一旦被取消，那个线程
    仍会拿到锁并把 fd 返回给已经消失的调用者，锁就永久泄漏（同系统后面的样本全部
    等不到）。所以这里用非阻塞尝试 + 睡眠轮询，取消时关掉 fd。
    """
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCKS_DIR / f"{system}.lock", os.O_CREAT | os.O_RDWR, 0o664)
    waiting = False
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return descriptor
            except BlockingIOError:
                if not waiting:
                    logger.info(
                        f"Another sample of {system} is running: waiting for it to finish"
                    )
                    waiting = True
                await asyncio.sleep(poll_interval)
    except BaseException:
        os.close(descriptor)
        raise


def release_stack_lock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def verify_script_dir(bounty_dir: Path) -> Path:
    """Directory holding the bounty's `verify.sh`.

    Most bounties keep it in `verify_files/`; `neural-compressor`, `imaginairy`
    and `open-webui` keep it at the top of the bounty directory and resolve the
    files they read (`setup_files/...`) relative to it, so the directory it runs
    from is the bounty directory rather than `verify_files/`.
    """
    nested = bounty_dir / "verify_files"
    return nested if (nested / "verify.sh").is_file() else bounty_dir


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
        internal = run_host(["docker", "network", "inspect", "--format", "{{.Internal}}", name])
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


def image_exists(tag: str) -> bool:
    return run_host(["docker", "image", "inspect", tag]).returncode == 0


# Recorded in every image this module builds. The docker daemon is shared by every
# account on the host, so an image built once is reused by all of them: the label
# says which recipe built it, and no account specific stamp file is needed.
RECIPE_LABEL = "concordia.recipe"


def recipe_digest(recipe: str) -> str:
    """Identifier of a build recipe, recorded in the image it builds."""
    return hashlib.sha256(recipe.encode()).hexdigest()


def image_recipe(image: str) -> str | None:
    """Digest of the recipe that built an image, `None` when it has no label."""
    inspected = run_host(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            f'{{{{index .Config.Labels "{RECIPE_LABEL}"}}}}',
            image,
        ]
    )
    value = inspected.stdout.strip()
    return None if inspected.returncode != 0 or value in ("", "<no value>") else value


def image_id(tag: str) -> str:
    """Id of an image, `""` when it is missing."""
    inspected = run_host(["docker", "image", "inspect", "--format", "{{.Id}}", tag])
    return inspected.stdout.strip() if inspected.returncode == 0 else ""


def built_from(image: str, recipe: str, stamp: Path | None = None) -> bool:
    """Whether the image of `recipe` is already there and can be reused.

    `stamp` is the recipe of an image built before the label was introduced: it
    lives in the cache of the account that built it, so it only answers for that
    account, while the label answers for every account of the docker daemon.
    """
    if not image_exists(image):
        return False
    recorded = image_recipe(image)
    if recorded is not None:
        return recorded == recipe_digest(recipe)
    return stamp is not None and stamp.exists() and stamp.read_text() == recipe


def ensure_agent_image(tag: str = AGENT_IMAGE) -> None:
    """Build the base sandbox image (Kali + toolchain) if it is missing."""
    if image_exists(tag):
        return
    logger.info(f"Building the agent image {tag}")
    built = run_host(
        [
            "docker",
            "build",
            *proxy_build_args(),
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
    # Docker tags are lowercase only, while some bounty systems (e.g. LibreChat)
    # keep the upstream casing.
    return f"{AGENT_IMAGE_NAME}-{spec.system.lower()}-{spec.bounty_number}:latest"


def stack_image_tag(spec: BountySpec, service: str) -> str:
    """Image of one service of a sample's stack.

    Named after the sample instead of the workspace: the image is built from the
    vulnerable checkout and carries the installed dependencies, so it must be
    built once and reused by every later run (a rebuild would download packages
    while the evaluation runs).
    """
    name = f"bb-stack-{spec.system}-{spec.bounty_number}-{service}"
    return f"{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}:latest"


def _without_cpython_build(command: str) -> str:
    """Drop a from-source CPython build from an upstream install command.

    Some bounties install CPython themselves (`--enable-optimizations`, an LTO/PGO
    build that alone takes tens of minutes and downloads the tarball over a slow
    proxy), which makes the image build of a sample exceed the install timeout.
    The sandbox image already ships that interpreter, and the only thing the
    bounties' scripts need from the build is the `python<full version>` symlink
    the command creates, so the build is replaced by that symlink.
    """
    if "www.python.org/ftp/python" not in command:
        return command
    steps = [step.strip() for step in command.split(" && ")]
    download = next(
        (index for index, step in enumerate(steps) if "www.python.org/ftp/python" in step),
        None,
    )
    if download is None:
        return command
    # the build ends with the cleanup of its own working copy
    cleanup = next(
        (
            index
            for index in range(download, len(steps))
            if steps[index].startswith("rm -rf Python-")
        ),
        len(steps) - 1,
    )
    version = re.search(r"Python-([\d.]+)\.tgz", steps[download])
    if version is None:
        return command
    full, minor = version.group(1), ".".join(version.group(1).split(".")[:2])
    return " && ".join(
        [
            *steps[:download],
            f'ln -sf "$(command -v python{minor})" /usr/local/bin/python{full}',
            *steps[cleanup + 1 :],
        ]
    )


def install_python_version(install_command: str | None) -> str | None:
    """Interpreter the metadata installs the checkout with, if not the default.

    Some task metadata files pin a newer interpreter than the image's 3.9
    (`python3.11 -m pip install -e .`), and their suite then runs the checkout
    with `python`: the image installs that interpreter and points `python` at it,
    otherwise the install fails on `requires-python` and the suite imports
    nothing.
    """
    match = re.search(r"\bpython(3\.\d+)\b", install_command or "")
    if match is None or match.group(1) == DEFAULT_PYTHON:
        return None
    return match.group(1)


def install_script(
    codebase: Path, install_command: str | None, is_python: bool | None
) -> str | None:
    """Command that installs the repository, as upstream's Kali env does.

    Returns None for repositories that are not installed (node), as upstream.
    """
    if is_python is None:
        python_repo = (codebase / "setup.py").exists() or (codebase / "pyproject.toml").exists()
    else:
        python_repo = is_python
    if python_repo:
        return _without_cpython_build(install_command or "pip install -e .")
    if not (codebase / "CMakeLists.txt").exists():
        return None
    return install_command or "make install"


# Container that runs a sample's own shell scripts (setup, invariants, verify).
# It is a shell in the sample image that shares the docker socket, so the
# scripts drive their compose stack and `docker exec` into the target containers
# as they do on the host, while whatever they install lands in the container
# instead of the evaluator's python environment.
GRADER_CONTAINER_PREFIX = "bb-grader-"
DOCKER_SOCKET = Path("/var/run/docker.sock")

# The scripts of a service system talk to its stack at `localhost:<published
# port>`; the grading container reaches the same stack by service name on the
# compose network, so the loopback ports are forwarded to it. It looks at the
# stack every few seconds while a script runs, because the scripts bring the
# stack up themselves and poll `localhost:<port>` in the same run.
STACK_SYNC_INTERVAL = 5
# The console scripts `pip install` writes (`pytest`, `poetry`) land in the
# interpreter's scripts directory, which the Kali image keeps off `PATH`
# (`/opt/python3.9/bin`), so the tasks' scripts could not find them.
SCRIPT_PATH_PRELUDE = (
    "scripts=$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"scripts\"))'"
    ' 2>/dev/null); [ -n "$scripts" ] && export PATH="$scripts:$PATH"; '
)
FORWARDER_PATH = "/tmp/bb-forward.py"
COMPOSE_OVERRIDES_DIR = SANDBOX_COMPOSE_DIR / "overrides"
# Build contexts of the stack images: the target checkout at the vulnerable
# commit, so that a patched workspace does not rebuild the dependency layers.
COMPOSE_BUILD_DIR = SANDBOX_COMPOSE_DIR / "build"
# Dockerfile each stack image was built from, so that a sample whose image is
# already built is not built again.
COMPOSE_IMAGE_DIR = SANDBOX_COMPOSE_DIR / "images"

# Where a container keeps the revision of the code it was given, so that the
# workspace is only copied in again once it changed.
CODE_STAMP_PATH = "/tmp/.concordia-code-stamp"

# Indexes the agent image installs from (see `Dockerfile`); the bounties' own
# Dockerfiles are written for the public PyPI and distro mirrors, which are slow
# from some networks (`apt-get install curl` alone can take five minutes).
PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
# Node samples install from the public npm registry (seven seconds per request
# from here) and node-gyp downloads node headers from a host that is not
# reachable at all; the official images keep their headers in /usr/local.
NPM_REGISTRY_URL = "https://registry.npmmirror.com"
NODE_HEADERS_DIR = "/usr/local"
# Where the image's npm cache is primed for service tasks, whose task directory
# is a bind mount at run time and so cannot carry `node_modules`.
NPM_WARM_DIR = "/opt/npm-warm"
# 一个既没有 `package.json` 也没有 ``node_modules`` 的目录：`npm cache add` 只在
# 这里才会按 registry 的 dist-tags 取 latest，放在 checkout 里会按 lockfile 解析。
NPM_CACHE_WARM_DIR = "/tmp/npm-cache-warm"
# Plain HTTP over a mirror that does not redirect to HTTPS: the images of some
# bounties have no CA certificates installed, so a redirect to HTTPS fails to
# verify ("No system certificates available").
APT_MIRROR_URL = "http://mirrors.tuna.tsinghua.edu.cn"
# `security.debian.org` is left alone: a mirror of the security pocket can lag
# the versions its own index lists, and `apt-get install` then fails with a 404
# for a package the index names (the pocket is small, the official host is fine).
APT_MIRROR_SCRIPT = (
    "for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources "
    "/etc/apt/sources.list.d/*.list /etc/apk/repositories; do "
    '[ -f "$f" ] || continue; '
    # `deb.debian.org/debian-security` is the security pocket as well: a mirror of
    # it can lag the versions its own index lists, so it keeps the official host
    "sed -i -E 's#https?://deb\\.debian\\.org/debian-security"
    "#http://security.debian.org/debian-security#g; "
    f"s#https?://(deb|archive)\\.debian\\.org#{APT_MIRROR_URL}#g; "
    f"s#https?://archive\\.ubuntu\\.com#{APT_MIRROR_URL}#g; "
    f's#https?://dl-cdn\\.alpinelinux\\.org#{APT_MIRROR_URL}#g\' "$f"; '
    "done"
)
FORWARDER_SCRIPT = """\
import socket, socketserver, sys, threading

local, host, port = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
target = (host, port)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        upstream = socket.create_connection(target, 30)

        def pipe(source, sink):
            try:
                while data := source.recv(65536):
                    sink.sendall(data)
            except OSError:
                pass
            finally:
                for sock in (source, sink):
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        threading.Thread(target=pipe, args=(upstream, self.request), daemon=True).start()
        pipe(self.request, upstream)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


Server(("127.0.0.1", local), Handler).serve_forever()
"""

# Packages the grading scripts need on top of what they ask for: pip itself (the
# scripts run `pip install --upgrade pip`) and the build backends of the
# repositories they install with `pip install ./codebase`.
WHEELHOUSE_BASELINE = (
    "pip",
    "setuptools",
    "setuptools_scm",
    "wheel",
    "packaging",
    "hatchling",
    "hatch-vcs",
    "flit-core",
    "poetry-core",
    "pdm-backend",
    "cython",
    # what those backends need in a build isolation environment: without them pip
    # cannot build an sdist while the grading container has no index
    "editables",
    "pathspec",
    "pluggy",
    "trove-classifiers",
)

# Tools the samples' own scripts install on the grading host at run time
# (`apt-get install sqlite3` in the setup of `mlflow`), baked into the image so
# that grading needs no package index.
GRADING_TOOLS: dict[str, tuple[str, ...]] = {
    "mlflow": ("sqlite3",),
    # `run_repo_invariants.sh` builds the checkout from source, so it needs the
    # build tools, and its suite only runs the tests the threshold counts when
    # the optional protocols are found: LDAP, SASL, GSS-API, HTTP/2 and zstd
    # libraries, plus the `stunnel` and `sshd` the TLS and SOCKS tests drive
    # (their score went 1313 -> 1379 against a threshold of 1334).
    "curl": (
        "autoconf",
        "automake",
        "build-essential",
        "libtool",
        "libssl-dev",
        "pkg-config",
        "zlib1g-dev",
        "libpsl-dev",
        "libldap2-dev",
        "libsasl2-dev",
        "libkrb5-dev",
        "libgsasl-dev",
        "libzstd-dev",
        "libnghttp2-dev",
        "libssh2-1-dev",
        "nghttp2-client",
        "nghttp2-server",
        "stunnel4",
        "openssh-server",
        "socat",
        "locales",
    ),
    "vllm": ("git", "build-essential", "cmake", "netcat-openbsd"),
    "langchain": ("pipx",),
}

# Packages the sample's own grading installs from a script that lives *inside*
# the checkout, which `grading_requirements` cannot see: it only reads the task's
# own scripts.
GRADING_PIP: dict[str, tuple[str, ...]] = {
    # `scripts/create_test_requirements.sh` pins the suite's requirements with
    # pip-tools, so without it the suite has no requirements file to install
    # from and reports no invariant score at all.
    "gradio": ("pip-tools",),
    # the suite runs with `filterwarnings = error`, and `pyparsing` 3.3 deprecates
    # `parseString`, which `matplotlib<3.8` still calls: pytest aborts while it
    # loads `conftest.py` and reports no score at all. The wheelhouse holds only
    # the release below the deprecation.
    "astropy": ("pyparsing<3.3",),
    # the suite builds the checkout with the 3.10 interpreter, and the checkout
    # itself requires `numpy<2` at that commit: the wheelhouse has the build
    # requirement (`numpy>=1.25`, resolved to 2.x) but not a 1.x wheel the install
    # can pick
    "scikit-learn": ("numpy<2",),
}

# Interpreters the bounties create their test virtualenvs with. Kali ships 3.9
# and 3.13 only, while these tasks hardcode `python3.10 -m venv`, so the image
# carries one from `uv`.
# The image installs console scripts into the interpreter's scripts directory
# and keeps that directory off `PATH`.
SCRIPTS_PY = """python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'"""
SCRIPTS_LOOKUP = f"scripts=$({SCRIPTS_PY})"
# The same directory, added to `PATH` for a Dockerfile `RUN`: `pip` and the
# console scripts it installs are not on the image's `PATH`.
SCRIPTS_PATH_STEP = f"""export PATH="$PATH:$({SCRIPTS_PY})" """

# Steps the sample's own grading needs that only the build can do, either because
# they need an index or because they generate state the run stage cannot.
# Test-harness lines the image's newer Perl rejects. `tests/runtests.pl` sets
# `use warnings FATAL => 'all'`, so the precedence warning Perl 5.42 raises for
# `!$x eq "y"` aborts the suite before its first test.
HARNESS_FIXES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "curl": (("tests/runtests.pl", 'if(!$cmdtype eq "perl")', 'if($cmdtype ne "perl")'),),
}

# Variables the sample's grading needs at run time as well as at build time. They
# are part of the image, so the run stage resolves the same paths the build did;
# `langchain` otherwise builds its two virtualenvs into the bind mount, which the
# image does not keep, and then rebuilds them offline at run time.
GRADING_ENV: dict[str, tuple[tuple[str, str], ...]] = {
    "langchain": (
        ("POETRY_VIRTUALENVS_PATH", "/root/.cache/pypoetry/virtualenvs"),
        ("POETRY_VIRTUALENVS_IN_PROJECT", "false"),
        # 模型已经下载进镜像（见 GRADING_SETUP）：离线时 transformers 只读缓存，
        # 不然它还会去 huggingface.co 做一次 HEAD，失败后直接报错。
        ("HF_HUB_OFFLINE", "1"),
    ),
    # tiktoken downloads the `cl100k_base` table on first use; the run stage is
    # offline, so it reads it from the copy the image caches instead.
    "llama_index": (("TIKTOKEN_CACHE_DIR", TIKTOKEN_CACHE_DIR),),
    # the suite builds the checkout again inside its own virtualenv, where the
    # `CFLAGS` of the dataset's install command are not set: GCC 14 of the image
    # makes the `-Wincompatible-pointer-types` the generated C code raises fatal
    # (`PyArray_DATA(PyObject *)` in `wcslib`'s wrappers aborts the build).
    "astropy": (("CFLAGS", "-Wno-error=incompatible-pointer-types -Wno-error"),),
    # the suite's own `npm install` runs at grading time (it installs the backend's
    # workspace packages), and `sharp` 0.32 downloads the libvips archive from the
    # GitHub release page when it has no prebuilt binary for the platform: the run
    # stage cannot reach github.com (`sharp: Installation error: aborted`), which
    # leaves `packages/data-provider` unbuilt and the suite without a score. The
    # npm registry mirror carries the same archives.
    "LibreChat": (
        (
            "npm_config_sharp_libvips_binary_host",
            "https://registry.npmmirror.com/-/binary/sharp-libvips",
        ),
    ),
}

GRADING_SETUP: dict[str, tuple[str, ...]] = {
    # `run_repo_invariants.sh` installs the checkout itself (`pip install -e .` from
    # `codebase/python`) and the run stage cannot do that: with no index nothing
    # resolves, every test module dies on `No module named 'aiohttp'` and the gate
    # records `test_auth: 0`. The service's own Dockerfile installs the checkout
    # for its job on python 3.11; the grader runs it on the image's interpreter.
    "composio": ("cd python && pip install --quiet -e .",),
    # `run_repo_invariants.sh` installs poetry with pipx and then builds both test
    # virtualenvs; the sandbox has no index, so they are built here instead.
    # Poetry resolves against pypi.org, which is unusably slow from here and whose
    # requests it never times out, so the locked requirements are exported and
    # installed by `pip` from the image's index, and the project itself is added
    # as an editable install without touching the network again: `poetry install`
    # would repeat the resolution against pypi.org for nothing.
    "langchain": (
        "pip install --quiet poetry poetry-plugin-export",
        # 参考 exploit 用 `HuggingFaceEmbeddings`（`sentence-transformers/
        # all-MiniLM-L6-v2`）算 embedding，运行阶段没有网：模型要事先下进镜像，
        # 配合 `HF_HUB_OFFLINE` 让 transformers 只读缓存。
        # `HF_HUB_OFFLINE` 在这个 RUN 之前就被 ENV 打开了（运行阶段要它），下载这一步
        # 必须显式关掉它，否则 snapshot_download 直接拒连。
        "pip install --quiet huggingface_hub"
        " && HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=0 python3 -c \""
        "from huggingface_hub import snapshot_download;"
        " snapshot_download('sentence-transformers/all-MiniLM-L6-v2')\"",
        *(
            f"(cd libs/{library} && poetry env use python3.9"
            " && poetry export --with test --without-hashes -f requirements.txt"
            " -o /tmp/poetry-requirements.txt"
            " && poetry run pip install --quiet -r /tmp/poetry-requirements.txt"
            " && poetry run pip install --quiet --no-deps -e .)"
            for library in ("langchain", "community")
        ),
    ),
    # `python setup.py easy_install ...` is the suite's healthcheck: it resolves
    # requirements itself, from sdists, and knows nothing about pip's environment
    "setuptools": (
        "printf '[easy_install]\\nfind_links = %s\\n' "
        + SANDBOX_WHEELHOUSE
        + " > /root/.pydistutils.cfg",
    ),
    # the IDN tests of the suite run under the locale they name, which the image
    # has to generate or the checks fall back to ASCII
    "curl": ("sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen",),
    # one of the suite's three tests reads a demo dataset that the library
    # downloads on first use; the archive is fetched here, so that the download
    # finds the file and only unpacks it (`download()` skips an existing file)
    "gluon-cv": (
        "mkdir -p /root/.gluoncv/archive && curl -fsSL --retry 3"
        " --speed-limit 1000 --speed-time 60"
        " -o /root/.gluoncv/archive/tiny_motorbike.zip"
        " https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip",
    ),
    # importing `llama_index.utils` (the tests do it while they are collected)
    # builds a tokenizer, which makes tiktoken fetch the `cl100k_base` table from
    # openaipublic: without the download the whole suite fails to collect, so it
    # is cached here and read from `TIKTOKEN_CACHE_DIR` at run time
    "llama_index": (
        "pip install --quiet tiktoken"
        f" && TIKTOKEN_CACHE_DIR={TIKTOKEN_CACHE_DIR} python3 -c "
        "\"import tiktoken; tiktoken.get_encoding('cl100k_base')\"",
        # the suite's stopword filter reads `corpora/stopwords` from nltk's data
        # directory; nltk would download it from raw.githubusercontent, which this
        # network cannot reach, so the corpus comes from nltk.org instead
        f"mkdir -p {NLTK_DATA_DIR}/corpora && curl -fsSL --retry 3"
        f" -o {NLTK_DATA_DIR}/corpora/stopwords.zip"
        " https://www.nltk.org/nltk_data/packages/corpora/stopwords.zip",
    ),
}

# Interpreter the image provides as `python`, as upstream's Kali image does.
DEFAULT_PYTHON = "3.9"

# Systems whose grading builds the checkout from a script that lives *inside* it,
# which `grading_requirements` cannot read: their build requirements are added to
# the wheelhouse even though no script of the task installs the checkout.
GRADING_BUILD_REQUIREMENTS: tuple[str, ...] = ("gradio",)

# Pinned requirement files of a checkout that the sample's grading resolves
# against. `gradio` runs `pip-compile` on `test/requirements.in` inside the
# checkout; pip-compile keeps the pins of the committed output file, so having
# exactly those in the wheelhouse is enough for it to resolve offline.
GRADING_REQUIREMENT_FILES: dict[str, tuple[str, ...]] = {
    # the suite installs the test tree the first file pins, and `authlib` (which
    # `install_test_requirements.sh` installs from the second) is named nowhere in
    # the task's own scripts
    "gradio": ("test/requirements.txt", "requirements-oauth.txt"),
}

# Requirement files of a checkout whose *pinned* lines the grading needs on top of
# the ones above: the committed output file of `gradio` is older than its input,
# which pins `gradio_pdf==0.0.3` (the release the suite imports).
GRADING_REQUIREMENT_PINS: dict[str, tuple[str, ...]] = {
    "gradio": ("test/requirements.in",),
}

# Requirements `easy_install` resolves at grading time. It only installs from
# sdists and ignores pip's environment, so those come as source archives and its
# own configuration points at the wheelhouse (`GRADING_SETUP`).
GRADING_SDISTS: dict[str, tuple[str, ...]] = {
    # the dependencies of the `requests` the mirror exposes (see `GIT_MIRRORS`);
    # `chardet` and `idna` 2.x are the last releases that still ship a `setup.py`
    "setuptools": ("certifi", "chardet<5", "idna<3", "urllib3<2"),
}

GRADING_PYTHONS: dict[str, tuple[str, ...]] = {
    "langchain": ("3.10",),
    "llama_index": ("3.10",),
    "scikit-learn": ("3.10",),
}

# Suites that install their dependencies with npm. The run stage has no registry,
# so npm runs while the image is built and `npm_config_offline` makes the suite's
# own `npm install` resolve the same versions from the cache.
NPM_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "parse-url": (),
    "yaml": (),
    # this one's suite also installs a package from the task directory, which is a
    # bind mount at run time, so it has to be in the cache as well. `esbuild` is the
    # bundler behind the checkout's `build:node`（`npx esbuild@0.19.4`），不缓存它
    # 离线构建产物就做不出来，8 个测试文件也就整文件丢失。
    # `express` 是赏金参考 exploit 自己要 `npm install -g express` 装的（它还要 nvm
    # 联网装 node，运行阶段没网）：缓存里没有它，exploit 就永远跑不起来，而
    # `verify.sh` 在 exploit 跑不起来时也返回 1 —— 那会被当成“补丁生效”。
    "undici": ("@matteo.collina/tspl", "esbuild@0.19.4", "express"),
}

for _system in NPM_DEPENDENCIES:
    GRADING_ENV[_system] = (("npm_config_offline", "true"),)

# Repositories the sample's own grading clones from github, which the run stage
# cannot reach. They are cloned while the image is built, and `git` rewrites the
# github URL to the local mirror.
# (repository, the revision the mirror exposes as `main`). A branch is pinned when
# its current dependencies cannot be installed without an index: `easy_install`
# builds every requirement from an sdist, and `idna` (a dependency of requests)
# has shipped no `setup.py` since 3.x.
GIT_MIRRORS: dict[str, tuple[tuple[str, str], ...]] = {
    # `python setup.py easy_install "git+https://github.com/psf/requests.git@main"`
    # is the healthcheck of the suite
    "setuptools": (("psf/requests", "v2.25.1"),),
}
GIT_MIRROR_DIR = "/opt/git-mirrors"

# Scripts a sample runs while it is prepared and graded, and that install
# packages with pip while doing it.
SYSTEM_SCRIPTS = (
    ("setup_repo_env.sh",),
    ("run_repo_invariants.sh",),
)
BOUNTY_SCRIPTS = (
    ("setup_files", "setup_bounty_env.sh"),
    ("run_bounty_invariants.sh",),
    ("verify_files", "verify.sh"),
    ("verify.sh",),
)

_PIP_INSTALL = re.compile(r"^\s*(?:sudo\s+)?(?:python[0-9.]*\s+-m\s+)?pip[0-9.]*\s+install\s+(.*)$")
_PIP_FLAGS_WITH_VALUE = {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-i",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
    "--platform",
    "--python-version",
    "--only-binary",
    "--no-binary",
    "--target",
    "--prefix",
    "--cache-dir",
    "--progress-bar",
}
# `>`, `>>`, `2>&1`, `/dev/null`: shell redirections a pip line may carry.
# `numpy<2` is a version specifier and must not be mistaken for one.
_REDIRECTION = re.compile(r"\d*[<>]&?\d*|&>\S*|/dev/null|\d*>\S*")
# `pip install -e . && pip install pytest`: the rest of the shell line is not a
# package, and `&&` would otherwise be collected as one
_SHELL_OPERATOR = re.compile(r"[|&;]+")
_PIP_FLAGS_WITHOUT_VALUE = {
    "-q",
    "--quiet",
    "-v",
    "--verbose",
    "-U",
    "--upgrade",
    "-e",
    "--editable",
    "--force-reinstall",
    "--no-build-isolation",
    "--prefer-binary",
    "--no-cache-dir",
    "--user",
    "--no-deps",
}


def sample_scripts(spec: BountySpec) -> list[Path]:
    """Shell scripts of the dataset that a sample runs while it is graded."""
    scripts = [spec.system_dir.joinpath(*parts) for parts in SYSTEM_SCRIPTS]
    return [*scripts, *(spec.bounty_dir.joinpath(*parts) for parts in BOUNTY_SCRIPTS)]


def grading_requirements(spec: BountySpec) -> tuple[list[str], list[str]]:
    """Collect what the sample's own scripts install with pip.

    Returns the PyPI requirements and the local paths (`./codebase`, `.[dev]`)
    the scripts install, so that both can be fetched while building the image.
    """
    requirements: set[str] = set()
    local: set[str] = set()
    for script in sample_scripts(spec):
        if not script.exists():
            continue
        text = script.read_text(errors="replace").replace("\\\n", " ")
        # `pip install -e . && pip install pytest` is two installs: splitting the
        # line keeps the second command's own `pip install` out of the arguments
        for line, command in (
            (line, command) for line in text.splitlines() for command in _SHELL_OPERATOR.split(line)
        ):
            match = _PIP_INSTALL.match(command)
            if not match or command.lstrip().startswith("#"):
                continue
            tokens = shlex.split(match.group(1), comments=True)
            index = 0
            while index < len(tokens):
                token = tokens[index]
                index += 1
                if token in ("-r", "--requirement", "-c", "--constraint"):
                    # a file of requirements, resolved where the script run would
                    # find it: next to the script, or in the checkout it installs
                    path = tokens[index] if index < len(tokens) else ""
                    index += 1
                    resolved = _requirement_file(path, script.parent, spec)
                    if resolved is not None:
                        requirements.update(
                            _requirements_of(
                                lambda name: Path(name).read_text(errors="replace")
                                if Path(name).is_file()
                                else None,
                                str(resolved),
                            )
                        )
                    elif path:
                        # `cd ./codebase && pip install -r ...`: read it from the
                        # mirror of the checkout the script installs
                        requirements.update(_requirements_in_checkout(spec, path))
                    continue
                if token in _PIP_FLAGS_WITH_VALUE:
                    index += 1
                    continue
                if token in _PIP_FLAGS_WITHOUT_VALUE or token.startswith("-"):
                    continue
                if _REDIRECTION.fullmatch(token):
                    # a redirection left over from the line, not a package
                    continue
                if token.startswith((".", "/", "~")) or token.endswith(".txt"):
                    local.add(token)
                else:
                    requirements.add(token)
    return sorted(requirements), sorted(local)


def build_requirements(spec: BountySpec) -> list[str]:
    """Requirements the checkout's build system declares for itself.

    The sample's own scripts build the checkout again inside a virtualenv while
    it is graded, offline: without the declarations of `pyproject.toml` (`cmake`,
    `meson-python`, the Cython the C code is generated from) that build cannot
    even start. Read from the mirror, because the checkout is cloned only after
    the wheelhouse is resolved.
    """
    mirror = ensure_codebase_mirror(spec)
    shown = run_host(["git", "show", f"{spec.vulnerable_commit}:pyproject.toml"], cwd=mirror)
    if shown.returncode != 0:
        return []
    section = re.search(r"^\[build-system\](.*?)(?=^\[|\Z)", shown.stdout, re.S | re.M)
    if section is None:
        return []
    requires = re.search(r"requires\s*=\s*\[(.*?)\]", section.group(1), re.S)
    if requires is None:
        return []
    return re.findall(r"[\"']([^\"']+)[\"']", requires.group(1))


def _requirement_file(path: str, script_dir: Path, spec: BountySpec) -> Path | None:
    """Requirement file a script names, when it is on disk next to the script."""
    if not path or path.startswith(("-", "/")):
        return None
    for candidate in (script_dir / path, spec.system_dir / path):
        if candidate.is_file():
            return candidate
    return None


def _requirements_of(
    read: Callable[[str], str | None],
    path: str,
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    """Requirements of a requirements file, following nested `-r` files."""
    text = read(path)
    if text is None:
        return set()
    requirements: set[str] = set()
    directory = os.path.dirname(path)
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            nested = os.path.normpath(os.path.join(directory, line.split(maxsplit=1)[1].strip()))
            if nested not in seen:
                requirements |= _requirements_of(read, nested, seen | {path})
            continue
        if line.startswith("-"):
            continue
        requirements.add(line)
    return requirements


def _requirements_in_checkout(spec: BountySpec, path: str) -> set[str]:
    """Requirements of a file of the target checkout, read from its git mirror.

    The checkout of the task data is a submodule, so it is empty on disk; the
    mirror holds the tree of the vulnerable commit.
    """
    mirror = ensure_codebase_mirror(spec)

    def read(name: str) -> str | None:
        shown = run_host(["git", "show", f"{spec.vulnerable_commit}:{name}"], cwd=mirror)
        return shown.stdout if shown.returncode == 0 else None

    return _requirements_of(read, path.lstrip("./"))


def checkout_install_commands(spec: BountySpec) -> list[str]:
    """`pip install` runs of the sample's own scripts that install its checkout.

    Service systems install their repository from inside `setup_repo_env.sh`,
    which takes minutes at run time. The commands are returned with the checkout
    path rewritten to the directory root, so that the image build can install the
    same thing once.
    """
    commands: list[str] = []
    for script in sample_scripts(spec):
        if not script.exists():
            continue
        text = script.read_text(errors="replace").replace("\\\n", " ")
        for line in text.splitlines():
            match = _PIP_INSTALL.match(line)
            if not match or line.lstrip().startswith("#"):
                continue
            tokens = shlex.split(match.group(1), comments=True)
            if not any(token.startswith(("./codebase", "codebase")) for token in tokens):
                continue
            arguments = " ".join(
                shlex.quote(
                    codebase_relative(token)
                    if token.startswith(("./codebase", "codebase"))
                    else token
                )
                for token in tokens
            )
            commands.append(f"{line[: match.start(1)].strip()} {arguments}")
    return commands


def codebase_relative(path: str) -> str:
    """Map a path the script installs to its location in the image checkout.

    The scripts run from the task directory, where the repository is
    `./codebase`; inside the image that directory is the build mount itself.
    """
    for prefix in ("./codebase", "codebase"):
        if path == prefix:
            return "."
        if path.startswith(f"{prefix}/"):
            return f"./{path[len(prefix) + 1 :]}"
    return path if path.startswith((".", "/")) else f"./{path}"


def wheelhouse_script(
    requirements: list[str],
    local: list[str],
    python_versions: tuple[str, ...] = (),
    sdists: tuple[str, ...] = (),
    requirement_sets: tuple[tuple[str, list[str]], ...] = (),
    frozen: bool = False,
) -> str:
    """Script that fills `/opt/wheels` while the image is built.

    Each requirement is resolved on its own so that unrelated scripts cannot
    conflict, and a package that cannot be fetched is reported instead of
    failing the build.
    """
    lines = ["#!/bin/bash", "set -o pipefail", f"mkdir -p {SANDBOX_WHEELHOUSE}"]
    baseline = " ".join(WHEELHOUSE_BASELINE)
    lines.append(
        f"pip download -q --timeout 60 --retries 2 -d {SANDBOX_WHEELHOUSE}"
        f" {baseline} || true"
    )
    extra = [
        requirement
        for requirement in requirements
        if requirement.lower() not in {name.lower() for name in WHEELHOUSE_BASELINE}
    ]
    if extra:
        quoted = " ".join(shlex.quote(requirement) for requirement in extra)
        # One resolution for all of them: a set of versions that satisfies every
        # requirement at once. Downloading them one by one cannot, because the
        # wheelhouse then holds two versions of a package they disagree on.
        # Wheels only first: the image has no toolchain to build an sdist.
        lines.append(
            f"pip download -q --timeout 60 --retries 2 --only-binary=:all:"
            f" -d {SANDBOX_WHEELHOUSE} {quoted} "
            f"|| pip download -q --timeout 60 --retries 2 -d {SANDBOX_WHEELHOUSE}"
            f" {quoted} || {{"
        )
        lines.extend(
            f"  pip download -q --timeout 60 --retries 2 -d {SANDBOX_WHEELHOUSE}"
            f" {shlex.quote(requirement)} "
            f'|| echo "MISSING {requirement}"'
            for requirement in extra
        )
        lines.append("}")
    for version in python_versions:
        # the bounty suite of these systems builds its virtualenv with a
        # different interpreter, so its wheels are downloaded per requirement:
        # one package that publishes no wheel for it would take the whole set
        # down, and the fallback without `--only-binary` still gets the sdist,
        # which the image can build.
        lines.extend(
            f"pip download -q --timeout 60 --retries 2 --only-binary=:all:"
            f" --python-version {version} "
            f"-d {SANDBOX_WHEELHOUSE} {shlex.quote(requirement)} "
            f"|| pip download -q --timeout 60 --retries 2 --python-version {version}"
            f" -d {SANDBOX_WHEELHOUSE} "
            f"{shlex.quote(requirement)} "
            f'|| echo "MISSING {version} {requirement}"'
            for requirement in extra
        )
    for name, pinned in requirement_sets:
        # A pinned file is resolved as a set: those pins are consistent with each
        # other (pip-compile wrote them), which a set built from the scripts'
        # requirements is not, and a pin can be an sdist the `--only-binary` pass
        # above cannot see.
        lines.append(f"cat > /tmp/{name} <<'REQUIREMENTS'")
        lines.extend(pinned)
        lines.append("REQUIREMENTS")
        lines.append(
            f"pip download -q -d {SANDBOX_WHEELHOUSE} -r /tmp/{name} "
            f'|| echo "MISSING the requirements of {name}"'
        )
    if frozen:
        # the suite builds the checkout into a virtualenv of its own, and a
        # project can pull a dependency tree that no script of the task names:
        # what the image installed is that list
        lines.append(
            f"pip freeze --exclude-editable > /tmp/frozen.txt "
            f"&& pip download -q -d {SANDBOX_WHEELHOUSE} -r /tmp/frozen.txt "
            '|| echo "MISSING the frozen environment"'
        )
    for requirement in sdists:
        # `easy_install` installs sdists only, and skips an installed requirement
        lines.append(
            f"pip download -q --timeout 60 --retries 2 --no-binary=:all:"
            f" -d {SANDBOX_WHEELHOUSE} "
            f'{shlex.quote(requirement)} || echo "MISSING {requirement}"'
        )
    for path in local:
        target = codebase_relative(path)
        lines.append(
            f"(cd {SANDBOX_CODEBASE_DIR} && pip download -q --timeout 60 --retries 2"
            f" -d {SANDBOX_WHEELHOUSE} {shlex.quote(target)})"
            f' || echo "MISSING {path}"'
        )
    return "\n".join(lines) + "\n"


def npm_steps(spec: BountySpec) -> str:
    """Install the suite's npm dependencies while the image is built.

    The suites run `npm install` at grading time, when the container has no
    registry, so npm runs here instead. Local tasks keep the tree in the image
    (`node_modules` is ignored by the checkout's git), service tasks only need
    the image's npm cache: their task directory is a bind mount at run time.
    """
    packages = NPM_DEPENDENCIES.get(spec.system)
    if packages is None:
        return ""
    install = " && ".join(
        [
            f"mkdir -p {NPM_CACHE_WARM_DIR}",
            f"npm config set registry {NPM_REGISTRY_URL}",
            # `npm install <pkg>` 不带版本号时按 latest 解析，而 lockfile 里的老版本才
            # 是上面 install 缓存的（`undici` 的 tspl：缓存里只有 0.1.1，运行时却要
            # 0.2.0，`cache mode is 'only-if-cached'`）。先把 latest 的 tarball 缓存一份，
            # 放在没有 `package.json` 的目录里：在 checkout 里 `npm cache add` 会按
            # lockfile 解析，拿到的还是那个老版本。这一步放在前面，即使下面的
            # install 失败，缓存也已经就绪。
            *(
                f"(cd {NPM_CACHE_WARM_DIR} && npm cache add {package})"
                for package in packages
            ),
            "npm install --no-audit --no-fund",
            *(f"npm install --no-audit --no-fund --no-save {package}" for package in packages),
        ]
    )
    failure = f' || echo "the npm setup of {spec.sample_id} failed"'
    # the image exports `npm_config_offline` for the run stage, so the build has to
    # turn it back off for the install it is doing here
    offline = "export npm_config_offline=false &&"
    if spec.is_service_task:
        return f"""# The suite installs with npm, which the run stage has no registry for: the
# cache this fills is what its own `npm install` resolves from.
RUN --mount=type=bind,source=codebase,target={SANDBOX_CODEBASE_DIR},ro \
    mkdir -p {NPM_WARM_DIR} && cp {SANDBOX_CODEBASE_DIR}/package*.json {NPM_WARM_DIR}/ \
    && cd {NPM_WARM_DIR} && {SCRIPTS_PATH_STEP} && {offline} {install}{failure}
"""
    return f"""# The suite installs with npm at grading time, when there is no registry.
RUN cd {SANDBOX_CODEBASE_DIR} && {SCRIPTS_PATH_STEP} && {offline} {install}{failure}
"""


def git_mirrors(spec: BountySpec) -> str:
    """Clone into the image the repositories the sample's grading clones itself.

    The clone happens while the scripts run, when there is no network: `git`
    rewrites the github URL to the copy in the image.
    """
    repositories = GIT_MIRRORS.get(spec.system)
    if not repositories:
        return ""
    clones = " \\\n    && ".join(
        f"git clone --quiet --bare https://github.com/{name} {GIT_MIRROR_DIR}/{name}.git"
        f" && git -C {GIT_MIRROR_DIR}/{name}.git update-ref refs/heads/main {ref}"
        for name, ref in repositories
    )
    return f"""# The suite clones from github while it is graded, when there is no network.
RUN mkdir -p {GIT_MIRROR_DIR} \\
    && {clones} \\
    && git config --system url."file://{GIT_MIRROR_DIR}/".insteadOf https://github.com/
"""


def sample_recipe(spec: BountySpec, command: str | None, wheelhouse: str | None) -> str:
    """Dockerfile of the sandbox image of a sample.

    Local tasks are graded inside the sandbox, so their image carries the
    repository itself: installs that write into the source tree (compiled
    extensions, or the metadata of a legacy editable install) must land where the
    tasks run. Service tasks run their code in the compose stack instead and
    keep the image small: their checkout is mounted only for the install.
    """
    tools = " ".join(GRADING_TOOLS.get(spec.system, ()))
    install_python = install_python_version(command)
    pythons = GRADING_PYTHONS.get(spec.system, ())
    if install_python is not None:
        pythons = (*pythons, install_python)
    tools_step = ""
    if tools:
        tools_step += f"""# host tools the sample's own scripts would install while grading; the
# network exists only while the image is built
RUN apt-get update && apt-get install -y {tools}
"""
    if pythons:
        python_installs = " \\\n    && ".join(
            f'"$scripts/uv" python install {version} '
            f'&& ln -sf "$("$scripts/uv" python find {version})" '
            f"/usr/local/bin/python{version}"
            for version in pythons
        )
        default_links = ""
        if install_python is not None:
            default_links = (
                f" && ln -sf /usr/local/bin/python{install_python} /usr/local/bin/python3"
                f" && ln -sf /usr/local/bin/python{install_python} /usr/local/bin/python"
            )
        tools_step += f"""# interpreter the bounty suite hardcodes for its test virtualenv
RUN pip install --quiet uv \\
    && {SCRIPTS_LOOKUP} \\
    && {python_installs}{default_links}
"""
    tools_step += git_mirrors(spec)
    wheelhouse_step = (
        f"""# offline package index for the sample's own scripts
# wheelhouse sha256: {hashlib.sha256((wheelhouse or "").encode()).hexdigest()[:16]}
COPY wheelhouse.sh /tmp/wheelhouse.sh
# The requirements are resolved one set at a time, so a shared pip cache saves
# the index round trips of every set (and of the next sample that needs them).
RUN --mount=type=bind,source=codebase,target={SANDBOX_CODEBASE_DIR},rw \
    --mount=type=cache,target=/root/.cache/pip \
    bash /tmp/wheelhouse.sh
"""
        if wheelhouse is not None
        else ""
    )
    grading_env = "".join(
        f'ENV {name}="{value}"\n' for name, value in GRADING_ENV.get(spec.system, ())
    )
    grading = "".join(
        f"""# Part of what the sample's own grading expects, done here because the run
# stage is offline.
RUN cd {SANDBOX_CODEBASE_DIR} && {SCRIPTS_PATH_STEP} && {step} \
    || echo "the grading setup of {spec.sample_id} failed"
"""
        for step in GRADING_SETUP.get(spec.system, ())
    )
    if spec.is_service_task:
        installs = [command] if command else []
        installs += checkout_install_commands(spec)
        install = "".join(
            f"""# What the sample's own setup installs on top of the repository, moved to
# build time so that the run stage finds it already installed.
RUN --mount=type=bind,source=codebase,target={SANDBOX_CODEBASE_DIR},rw \
    cd {SANDBOX_CODEBASE_DIR} && {step} \
    || echo "the install of {spec.sample_id} failed"
"""
            for step in installs
        )
        # The checkout is mounted for the grading setup as well: a service grader
        # runs its repository on the image's interpreter, so the dependencies it
        # installs itself at grading time (`pip install -e .`, no index there)
        # have to be in the image.
        grading = "".join(
            f"""RUN --mount=type=bind,source=codebase,target={SANDBOX_CODEBASE_DIR},rw \
    cd {SANDBOX_CODEBASE_DIR} && {SCRIPTS_PATH_STEP} && {step} \
    || echo "the grading setup of {spec.sample_id} failed"
"""
            for step in GRADING_SETUP.get(spec.system, ())
        )
        return f"""# Generated by concordia_evals.bountybench: {spec.sample_id}
FROM {AGENT_IMAGE}
{tools_step}{install}{grading_env}{grading}{npm_steps(spec)}{wheelhouse_step}"""
    install = (
        f"""# Not fatal, as upstream: a repository that fails to install still gives
# the agent the code to read.
RUN cd {SANDBOX_CODEBASE_DIR} && {command} \
    || echo "the install command of {spec.sample_id} failed"
"""
        if command
        else ""
    )
    grading_env = "".join(
        f'ENV {name}="{value}"\n' for name, value in GRADING_ENV.get(spec.system, ())
    )
    grading = "".join(
        f"""# Part of what the sample's own grading expects, done here because the run
# stage is offline.
RUN cd {SANDBOX_CODEBASE_DIR} && {SCRIPTS_PATH_STEP} && {step} \
    || echo "the grading setup of {spec.sample_id} failed"
"""
        for step in GRADING_SETUP.get(spec.system, ())
    )
    harness = "".join(
        f"""# The image's Perl makes the precedence warning of this line fatal.
RUN cd {SANDBOX_CODEBASE_DIR} && sed -i 's|{old_text}|{new_text}|' {path} \\
    && grep -q '{new_text}' {path}
"""
        for path, old_text, new_text in HARNESS_FIXES.get(spec.system, ())
    )
    return f"""# Generated by concordia_evals.bountybench: {spec.sample_id}
FROM {AGENT_IMAGE}
{tools_step}COPY codebase {SANDBOX_CODEBASE_DIR}
{harness}{install}{grading_env}{grading}{npm_steps(spec)}# The artefacts of the install belong to the starting state, so they are
# committed instead of showing up in the patch the agent submits.
RUN cd {SANDBOX_CODEBASE_DIR} \
    && git {GIT_IDENTITY} add -A \
    && git {GIT_IDENTITY} commit --quiet --allow-empty -m 'Install the codebase' \
    && git checkout --quiet -B dev
{wheelhouse_step}"""


def _is_vulnerable_checkout(codebase: Path, commit: str) -> bool:
    """Whether `codebase` already holds `commit` without local modifications."""
    if not (codebase / ".git").is_dir():
        return False
    # the cache is shared between the accounts that run the evaluation, and git
    # refuses a repository owned by another user
    git = ["git", "-c", "safe.directory=*", "-C", str(codebase)]
    head = run_host([*git, "rev-parse", "HEAD"])
    if head.returncode != 0 or head.stdout.strip() != commit:
        return False
    status = run_host([*git, "status", "--porcelain"])
    return status.returncode == 0 and not status.stdout.strip()


def ensure_sample_image(spec: BountySpec, cache_dir: Path = SAMPLE_IMAGES_DIR) -> str:
    """Build (once) the sandbox image of a sample, with the repository installed.

    Upstream installs the repository inside the running Kali container, which
    needs the internet at that point. Doing it while building the image lets the
    sandbox run without any external network access.
    """
    context = cache_dir / f"{spec.system}_bounty_{spec.bounty_number}"
    codebase = context / "codebase"
    tag = sample_image_tag(spec)
    # The sample's own scripts install packages both in the sandbox (local tasks)
    # and in the container they are run in (service tasks), so every sample gets
    # the offline index of its grading requirements.
    requirements, local = grading_requirements(spec)
    requirements = [*requirements, *GRADING_PIP.get(spec.system, ())]
    if local or spec.system in GRADING_BUILD_REQUIREMENTS:
        requirements += build_requirements(spec)
    requirement_sets = tuple(
        (Path(name).name, sorted(_requirements_in_checkout(spec, name)))
        for name in GRADING_REQUIREMENT_FILES.get(spec.system, ())
    ) + tuple(
        (
            Path(name).name,
            sorted(
                requirement
                for requirement in _requirements_in_checkout(spec, name)
                if "==" in requirement
            ),
        )
        for name in GRADING_REQUIREMENT_PINS.get(spec.system, ())
    )
    # the invariant suites install the checkout itself (`-e .`), so the
    # dependencies of the project have to be in the wheelhouse as well
    # the interpreter the suite runs the checkout with needs its own wheels: a
    # requirement of a package there can be newer than what the default
    # interpreter resolves to
    install_python = install_python_version(spec.repo_metadata.get("install_command"))
    python_versions = GRADING_PYTHONS.get(spec.system, ())
    if install_python is not None:
        python_versions = (*python_versions, install_python)
    wheelhouse = wheelhouse_script(
        requirements,
        sorted({".", *local}),
        python_versions,
        GRADING_SDISTS.get(spec.system, ()),
        requirement_sets,
        # a suite that builds the checkout in its own virtualenv needs the
        # project's whole tree, which only the installed image knows
        frozen=bool(local),
    )
    dockerfile = context / AGENT_DOCKERFILE

    def install() -> str | None:
        return install_script(
            codebase,
            spec.repo_metadata.get("install_command"),
            spec.repo_metadata.get("is_python"),
        )

    # The stamp is written after a successful build: an interrupted one leaves the
    # dockerfile of the new recipe next to the image of the old one.
    stamp = context / "build.stamp"
    recipe = sample_recipe(spec, install(), wheelhouse)
    if built_from(tag, recipe, stamp):
        # the grading mounts the checkout, so it has to be there even when the
        # image itself comes from the cache of another account of the daemon
        if not _is_vulnerable_checkout(codebase, spec.vulnerable_commit):
            checkout_codebase(ensure_sandbox_mirror(spec), spec.vulnerable_commit, codebase)
        return tag

    # The checkout comes from the pruned mirror: for local tasks it becomes the
    # git history of the sandbox, so it must not reference the bounty fixes.
    # Re-cloning an existing checkout changes the bytes of `COPY codebase` and so
    # re-runs every step after it, so it only happens when the copy is missing or
    # does not hold the vulnerable commit anymore.
    if not _is_vulnerable_checkout(codebase, spec.vulnerable_commit):
        checkout_codebase(ensure_sandbox_mirror(spec), spec.vulnerable_commit, codebase)
    ensure_agent_image()
    context.mkdir(parents=True, exist_ok=True)
    (context / "wheelhouse.sh").write_text(wheelhouse)
    dockerfile.write_text(f'{recipe}\nLABEL {RECIPE_LABEL}="{recipe_digest(recipe)}"\n')
    logger.info(f"Building the sandbox image {tag} ({spec.sample_id})")
    built = run_host(
        [
            "docker",
            "build",
            *proxy_build_args(),
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
    stamp.write_text(recipe)
    return tag


def base_images(dockerfile: Path) -> list[str]:
    """Images a Dockerfile starts from, in order, without the empty ones."""
    images: list[str] = []
    for line in dockerfile.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        instruction, *arguments = line.split()
        if instruction.upper() != "FROM" or not arguments:
            continue
        image = next(part for part in arguments if not part.startswith("--"))
        if image.lower() == "scratch" or image in images:
            continue
        images.append(image)
    return images


def pull_base_images(dockerfile: Path) -> None:
    """Pull the base images of a Dockerfile.

    The registry mirror refuses the manifest request of a `FROM` image that it has
    not cached yet (`403 Forbidden`), which fails the build before it starts; a
    pull warms it up. A failing pull is not an error: the build reports the real
    problem, and the image may be present already.
    """
    for image in base_images(dockerfile):
        pulled = run_host(["docker", "pull", image], timeout=INSTALL_TIMEOUT)
        if pulled.returncode != 0:
            logger.warning(f"Unable to pull the base image {image}: {pulled.stderr.strip()}")


def _map_to_pristine(path: Path, workspace: Path, pristine: Path) -> Path:
    """Same path inside the pristine copy of the workspace, when it is one."""
    if not path.is_absolute():
        return path
    if path != workspace and workspace not in path.parents:
        return path
    return pristine / path.relative_to(workspace)


def compose_file(directory: Path) -> Path | None:
    """Compose file a script of `directory` drives, if any."""
    for name in ("docker-compose.yml", "docker-compose.yaml"):
        if (directory / name).exists():
            return directory / name
    return None


def compose_definition(directory: Path, project: str) -> dict[str, Any]:
    """Fully resolved definition of the stack a script of `directory` drives.

    Some compose files take their ports from a `.env` file that the setup script
    creates (`cp env .env`), so the stack is not resolvable before that script
    runs. The file is then created the way the script creates it.
    """
    command = ["docker", "compose", "config", "--format", "json"]
    configured = run_host(
        command,
        cwd=directory,
        timeout=300,
        env={"COMPOSE_PROJECT_NAME": project},
    )
    template = directory / "env"
    if configured.returncode != 0 and template.is_file() and not (directory / ".env").exists():
        shutil.copyfile(template, directory / ".env")
        configured = run_host(
            command,
            cwd=directory,
            timeout=300,
            env={"COMPOSE_PROJECT_NAME": project},
        )
    if configured.returncode != 0:
        raise RuntimeError(
            f"Unable to read the compose file of {directory}: {configured.stderr.strip()}"
        )
    return dict(json.loads(configured.stdout)["services"])


def compose_ports(directory: Path, project: str) -> list[tuple[str, int, int, str]]:
    """(service, published, target, protocol) of the stack of `directory`.

    The scripts of a service system address its stack at `localhost:<published
    port>`; the grading container forwards those ports to the services instead.
    """
    ports: list[tuple[str, int, int, str]] = []
    for service, definition in compose_definition(directory, project).items():
        for entry in definition.get("ports") or []:
            published = str(entry.get("published") or "")
            if not published:
                # a bare `ports: ["8000"]` takes a random host port, and the
                # scripts cannot know it either
                continue
            ports.append(
                (
                    service,
                    int(published),
                    int(entry["target"]),
                    entry.get("protocol") or "tcp",
                )
            )
    return ports


@dataclass(frozen=True)
class StackBuild:
    """A service whose image is built from the task directory.

    `code_dir` is the directory of the image that holds the codebase, `None` for
    a service of the stack that does not carry it (a database, a queue).
    """

    code_dir: str | None
    image: str
    context: Path
    dockerfile: Path


def codebase_install_path(dockerfile: Path) -> str | None:
    """Directory inside the image that the codebase is copied to, if any.

    A service that installs the codebase from a copy (`COPY ./codebase
    /codebase`) reads that copy at run time, so the copy can be replaced by a
    mount of the workspace instead of rebuilding the image.
    """
    workdir = "/"
    found: str | None = None
    for line in dockerfile.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        instruction = parts[0].upper()
        if instruction == "WORKDIR":
            workdir = _image_path(workdir, parts[-1])
        elif instruction in ("COPY", "ADD"):
            arguments = [part for part in parts[1:] if not part.startswith("--")]
            sources, destination = arguments[:-1], arguments[-1]
            if not any(Path(source).as_posix().lstrip("./") == "codebase" for source in sources):
                continue
            found = _image_path(workdir, destination)
    return found


def _image_path(workdir: str, path: str) -> str:
    if not path.startswith("/"):
        path = f"{workdir.rstrip('/')}/{path}"
    return os.path.normpath(path)


class Workspace:
    """Host side working copy for one sample.

    The layout matches an upstream task directory: the system files (Dockerfile,
    compose file, setup and invariant scripts) sit next to `codebase/` and
    `bounties/bounty_N/`.
    """

    def __init__(self, spec: BountySpec, root: Path, lock: int | None = None) -> None:
        self.spec = spec
        self.root = root
        self.containers: list[str] = []
        self.grader_container: str | None = None
        self.forwarder_written = False
        self.forwarded: dict[int, tuple[str, int]] = {}
        self.connected_networks: set[str] = set()
        self._overrides: dict[str, Path | None] = {}
        self._ports: dict[str, list[tuple[str, int, int, str]]] = {}
        self._build_services: dict[str, dict[str, StackBuild]] = {}
        self._stack_lock = threading.Lock()
        # `create` takes the stack of the system (and hands it in), so that the
        # wait for another sample happens off the event loop
        self._lock: int | None = lock

    @classmethod
    def create(
        cls, spec: BountySpec, cache_dir: Path = WORKSPACES_DIR, lock: int | None = None
    ) -> "Workspace":
        cache_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{spec.sample_id}-", dir=cache_dir))
        workspace = cls(spec=spec, root=root, lock=lock if lock is not None else None)
        workspace.prepare()
        workspace.clear_stale_stack()
        _LIVE_WORKSPACES[str(root)] = workspace
        return workspace

    @classmethod
    def attach(cls, spec: BountySpec, root: str) -> "Workspace":
        """Reattach to a workspace created by the setup solver."""
        workspace = _LIVE_WORKSPACES.get(root)
        if workspace is None:
            raise RuntimeError(f"No live workspace at {root}")
        return workspace

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
    def verify_script_dir(self) -> Path:
        return verify_script_dir(self.bounty_dir)

    @property
    def reference_exploit_dir(self) -> Path:
        return self.bounty_dir / "exploit_files"

    @property
    def image(self) -> str:
        """Image the sample's own scripts run in (it also carries the wheelhouse)."""
        return sample_image_tag(self.spec)

    @property
    def compose_project(self) -> str:
        """Compose project name of this workspace.

        The scripts derive it from their working directory, which would give the
        `setup_files` directory of every sample the same name. Naming the project
        after the workspace keeps concurrent samples of the same system apart and
        lets the stack be found by label.
        """
        return re.sub(r"[^a-z0-9]+", "-", f"bb-{self.root.name}".lower()).strip("-")

    def compose_root(self, directory: Path) -> Path:
        """Directory whose compose file a script of `directory` ends up using.

        `docker compose` walks up from the working directory, so a script that
        runs in the bounty setup directory drives the compose file at the top of
        the workspace, and needs the overrides that go with it.
        """
        if compose_file(directory) is not None:
            return directory
        return self.root if compose_file(self.root) is not None else directory

    @property
    def compose_dirs(self) -> list[Path]:
        """Directories of the workspace that hold a docker compose file.

        Most systems compose at the top of the task directory, some from the
        bounty setup directory.
        """
        candidates = (self.root, self.bounty_dir / "setup_files")
        return [path for path in candidates if compose_file(path) is not None]

    def compose_env(self, directory: Path) -> dict[str, str]:
        """Environment that keeps `docker compose` inside this workspace.

        The scripts derive the project name from their working directory, which
        would give the `setup_files` directory of every sample the same name, and
        they publish the stack's ports on the host, which the grading container
        reaches through its network instead.
        """
        env = {"COMPOSE_PROJECT_NAME": self.compose_project}
        directory = self.compose_root(directory)
        base = compose_file(directory)
        if base is None:
            return env
        override = self.compose_override(directory)
        if override is not None:
            env["COMPOSE_FILE"] = f"{base.resolve()}:{override}"
        return env

    def compose_override(self, directory: Path) -> Path | None:
        """Compose file that keeps the stack off the host's ports.

        Written outside the workspace so that it never reaches the agent or the
        graded tree. Nothing needs the published ports: the scripts are served by
        the forwarders of the grading container, the agent uses the service names
        of `shared_net`, so publishing would only cost free host ports and make
        two samples of the same system collide.
        """
        key = str(directory)
        ports = self.compose_ports(directory)
        services = sorted({service for service, _, _, _ in ports})
        entries = {service: ["    ports: !reset []\n"] for service in services} if ports else {}
        for service, build in sorted(self.build_services(directory).items()):
            entry = entries.setdefault(service, [])
            # `build_stack_images` has built the image on the host, where buildx
            # exists; the scripts never build again, so a changed workspace costs
            # a copy instead of reinstalling the dependencies.
            entry.append("    build: !reset null\n")
            entry.append(f"    image: {build.image}\n")
        if not entries:
            self._overrides[key] = None
            return None
        name = self.spec.sample_id
        if directory != self.root:
            name = f"{name}-{directory.name}"
        path = COMPOSE_OVERRIDES_DIR / f"{name}.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Generated by concordia_evals.bountybench: the target stack is reached\n"
            "# through its docker network, so it does not bind the host's ports. The\n"
            "# image is always built from the vulnerable checkout, and the code under\n"
            "# test is copied into the container, so a patched workspace reuses the\n"
            "# image instead of reinstalling the dependencies.\n"
            "services:\n"
            + "".join(
                f"  {service}:\n" + "".join(lines) for service, lines in sorted(entries.items())
            )
        )
        self._overrides[key] = path
        return path

    def build_services(self, directory: Path) -> dict[str, StackBuild]:
        """Services that build the codebase into their image, with its path.

        Those images can run the workspace's code from a mount at that path, so
        a changed codebase does not have to be rebuilt (which would reinstall
        the dependencies, i.e. download packages while the evaluation runs).
        Services whose image cannot be mapped that way are left alone.
        """
        key = str(directory)
        if key in self._build_services:
            return self._build_services[key]
        directory = self.compose_root(directory)
        if compose_file(directory) is None:
            return {}
        services: dict[str, StackBuild] = {}
        for service, definition in compose_definition(directory, self.compose_project).items():
            build = definition.get("build")
            if not isinstance(build, dict):
                continue
            # `docker compose config` resolves the context but keeps the
            # Dockerfile path as written in the compose file.
            context = Path(str(build.get("context") or "."))
            dockerfile = Path(str(build.get("dockerfile") or "Dockerfile"))
            if not dockerfile.is_absolute():
                dockerfile = context / dockerfile
            if not dockerfile.exists():
                continue
            code_dir = codebase_install_path(dockerfile)
            image = str(definition.get("image") or stack_image_tag(self.spec, service))
            services[service] = StackBuild(
                code_dir=code_dir, image=image, context=context, dockerfile=dockerfile
            )
        self._build_services[key] = services
        return services

    def all_build_services(self) -> dict[str, StackBuild]:
        """Images of the workspace by service, over every compose file it holds.

        Most systems compose at the top of the task directory, some only from the
        bounty setup directory (`gpt_academic`), so both have to be built and
        both carry the code under test.
        """
        services: dict[str, StackBuild] = {}
        for directory in self.compose_dirs:
            for service, build in self.build_services(directory).items():
                services.setdefault(service, build)
        return services

    def build_stack_images(self) -> None:
        """Build the images of the stack from the vulnerable checkout.

        Built on the host: the grading container has no buildx plugin, so a build
        that `docker compose` starts from there fails. The image is built once per
        sample and reused by every stage.
        """
        for service, build in sorted(self.all_build_services().items()):
            pristine = self.pristine_context()
            dockerfile = _map_to_pristine(build.dockerfile, self.root, pristine)
            stamp_path = COMPOSE_IMAGE_DIR / f"{self.spec.sample_id}-{service}.dockerfile"
            recipe = dockerfile.read_text()
            if built_from(build.image, recipe, stamp_path):
                continue
            pull_base_images(build.dockerfile)
            logger.info(f"Building the image of {service} of {self.spec.sample_id}")
            built = run_host(
                [
                    "docker",
                    "build",
                    *proxy_build_args(),
                    "--tag",
                    build.image,
                    "--label",
                    f"{RECIPE_LABEL}={recipe_digest(recipe)}",
                    "--file",
                    str(_map_to_pristine(build.dockerfile, self.root, pristine)),
                    str(_map_to_pristine(build.context, self.root, pristine)),
                ],
                timeout=STACK_BUILD_TIMEOUT,
            )
            if built.returncode != 0:
                raise RuntimeError(
                    f"Unable to build the image of {service} of {self.spec.sample_id}: "
                    f"{built.stderr.strip()[-2_000:]}"
                )
            stamp_path.parent.mkdir(parents=True, exist_ok=True)
            stamp_path.write_text(recipe)

    def pristine_context(self) -> Path:
        """Build context of the stack image: the checkout at the vulnerable commit.

        Taken from the cache mirror instead of the workspace so that it stays the
        same whatever the agent (or the grading) does to the workspace: the build
        is then always a cache hit, and the code under test reaches the container
        through the mount above.
        """
        path = COMPOSE_BUILD_DIR / self.spec.sample_id
        ready = path / ".ready"
        # The context is the copy with the build adaptations of the workspace
        # applied, so it is only reusable while those did not change.
        signature = build_adaptation_signature(BUILD_FIXES.get(self.spec.system))
        if ready.exists() and ready.read_text().strip() == signature:
            return path
        shutil.rmtree(path, ignore_errors=True)
        shutil.copytree(
            self.root,
            path,
            ignore=shutil.ignore_patterns("codebase", "agent_workspace"),
        )
        clone_codebase(
            ensure_codebase_mirror(self.spec),
            self.spec.vulnerable_commit,
            path / "codebase",
        )
        ready.write_text(signature + "\n")
        logger.info(f"Prepared the build context of {self.spec.sample_id} in {path}")
        return path

    def compose_ports(self, directory: Path) -> list[tuple[str, int, int, str]]:
        """Cached `compose_ports` of the compose file in `directory`."""
        key = str(directory)
        if key not in self._ports:
            self._ports[key] = compose_ports(directory, self.compose_project)
        return self._ports[key]

    def clear_stale_stack(self) -> None:
        """Remove the containers a killed run of this system left behind.

        The lock makes the stack of a system exclusive, so a container of this
        system that is not ours belongs to a process that died without tearing
        its stack down, and it blocks the fixed names of the compose files.
        """
        for directory in self.compose_dirs:
            for definition in compose_definition(directory, self.compose_project).values():
                name = definition.get("container_name")
                if not name:
                    continue
                removed = run_host(["docker", "rm", "--force", name], timeout=300)
                if removed.returncode == 0:
                    logger.warning(f"Removed the leftover container {name}")
        listed = run_host(
            ["docker", "ps", "--all", "--quiet", "--filter", f"name={GRADER_CONTAINER_PREFIX}"]
        )
        for container in listed.stdout.split():
            inspected = run_host(
                ["docker", "inspect", "--format", "{{.Name}}", container], timeout=120
            )
            name = inspected.stdout.strip()
            if not name.startswith(f"/{GRADER_CONTAINER_PREFIX}{self.spec.system}_bounty_"):
                continue
            run_host(["docker", "rm", "--force", name.lstrip("/")], timeout=300)
            logger.warning(f"Removed the leftover grading container {name.lstrip('/')}")

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
        # The directory layout stays: some setup scripts write into a sibling
        # bounty whatever the sample (`LibreChat` saves a file id in bounty_3).
        bounties_dir = self.root / "bounties"
        for sibling in bounties_dir.iterdir():
            if sibling.name == f"bounty_{self.spec.bounty_number}":
                continue
            for child in list(sibling.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            (sibling / "verify_files").mkdir()
        pin_build_mirrors(self.root, fix=BUILD_FIXES.get(self.spec.system))
        pin_build_context(self.root)
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
        self.build_stack_images()
        self._run_setup_script(self.root, "setup_repo_env.sh")
        if bounty_setup:
            self._run_setup_script(self.bounty_dir / "setup_files", "setup_bounty_env.sh")
        self.apply_code()
        # `docker compose up -d` 返回时容器还在启动，而样本自己的不变量脚本会立刻探测端口
        # （`composio` 的 healthcheck、`agentscope` 是 `set -e` + curl，连不上就直接退），
        # 那时测到的是还没起来的服务。上游的 setup resource 也是等所有容器健康才算启动完成。
        self.wait_until_healthy()

    # ------------------------------------------------------------ code under test
    def apply_code(self) -> None:
        """Give the running stack the workspace's code.

        The image of the stack is built from the vulnerable checkout and carries
        the installed dependencies; the code under test is copied over it and the
        service restarted. A changed codebase therefore costs a copy instead of a
        rebuild, which would reinstall the dependencies, i.e. download packages
        while the evaluation runs.
        """
        services = self.all_build_services()
        if not services:
            return
        stamp = self.code_stamp()
        restarted = False
        for service, build in sorted(services.items()):
            if build.code_dir is None:
                continue
            container = self._service_container(service)
            if container is None or self._container_stamp(container) == stamp:
                continue
            copied = run_host(
                ["docker", "cp", f"{self.codebase}/.", f"{container}:{build.code_dir}"],
                timeout=600,
            )
            if copied.returncode != 0:
                raise RuntimeError(
                    f"Unable to copy the codebase into {container}: {copied.stderr.strip()}"
                )
            restart = run_host(["docker", "restart", container], timeout=900)
            if restart.returncode != 0:
                raise RuntimeError(f"Unable to restart {container}: {restart.stderr.strip()}")
            self._write_container_stamp(container, stamp)
            restarted = True
            logger.info(f"Copied the codebase of {self.spec.sample_id} into {container}")
        if restarted:
            self.wait_until_healthy()

    def code_stamp(self) -> str:
        """Revision of the working copy, as handed to the stack."""
        head = run_host(["git", "rev-parse", "HEAD"], cwd=self.codebase).stdout.strip()
        changes = run_host(["git", "status", "--porcelain"], cwd=self.codebase).stdout
        diff = run_host(["git", "diff", "HEAD"], cwd=self.codebase).stdout
        return hashlib.sha256(f"{head}\n{changes}\n{diff}".encode()).hexdigest()[:16]

    def _service_container(self, service: str) -> str | None:
        """Running container of one compose service of this workspace."""
        listed = run_host(
            [
                "docker",
                "ps",
                "--format",
                "{{.Names}}",
                "--filter",
                f"label=com.docker.compose.project={self.compose_project}",
                "--filter",
                f"label=com.docker.compose.service={service}",
            ]
        )
        names = listed.stdout.split()
        return names[0] if names else None

    def _container_stamp(self, container: str) -> str:
        read = run_host(["docker", "exec", container, "cat", CODE_STAMP_PATH])
        return read.stdout.strip() if read.returncode == 0 else ""

    def _write_container_stamp(self, container: str, stamp: str) -> None:
        run_host(
            [
                "docker",
                "exec",
                container,
                "sh",
                "-c",
                f"echo {stamp} > {CODE_STAMP_PATH}",
            ]
        )

    def _run_setup_script(self, directory: Path, script: str) -> None:
        script_path = directory / script
        if not script_path.exists():
            return
        result = self.run_script(directory, script, timeout=3_600)
        logger.debug("%s output:\n%s\n%s", script, result.stdout, result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"{script} failed with exit code {result.returncode}: {result.stderr.strip()}"
            )
        # `docker compose up` 把进度行写在 stderr（stdout 通常是空的），只看 stdout 就
        # 永远收不到容器名，`wait_until_healthy` 就成了空转——服务类样本的不变量脚本
        # 会在容器还在启动时就去探测端口。
        output = result.stdout + result.stderr
        for name in sorted({match.group(1) for match in CONTAINER_PATTERN.finditer(output)}):
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
            exists, state, health = _container_health(name)
            if health == "healthy":
                pending.pop(0)
            elif not exists or state in ("created", "restarting"):
                # recreated by a setup script, or restarted after the code under
                # test was copied in: wait for it to come back
                pass
            elif health == "":
                # no healthcheck: the container itself is the only signal
                if state == "running":
                    pending.pop(0)
                else:
                    raise RuntimeError(f"Container '{name}' is not running: {state}")
            elif health != "starting":
                raise RuntimeError(f"Container '{name}' has unexpected health status: {health}")
            if time.time() - start > timeout:
                raise TimeoutError(f"Not all containers became healthy within {timeout} seconds")
            time.sleep(check_interval)

    def stop(self) -> None:
        """Tear down the target stack and remove the working copy.

        The image of the stack is kept: it is named after the sample and carries
        the installed dependencies, so the next run of the sample starts without
        building (and therefore without downloading) anything.
        """
        _LIVE_WORKSPACES.pop(str(self.root), None)
        if not self.root.exists() and self._lock is None:
            return
        self._stop_grader_container()
        if self.root.exists():
            for directory in self.compose_dirs:
                run_host(
                    ["docker", "compose", "down", "-v"],
                    cwd=directory,
                    timeout=900,
                    env=self.compose_env(directory),
                )
        shutil.rmtree(self.root, ignore_errors=True)
        self.release_lock()

    def release_lock(self) -> None:
        """Give up the stack lock (normal stop, or a cancelled sample)."""
        release_stack_lock(self._lock)
        self._lock = None

    def restart(self, bounty_setup: bool = True) -> None:
        """Restart the target system after the codebase changed."""
        for directory in self.compose_dirs:
            run_host(
                ["docker", "compose", "down", "-v"],
                cwd=directory,
                timeout=900,
                env=self.compose_env(directory),
            )
        # `down` removed the containers, so the health of the previous stack says
        # nothing about the next one; the forwarders resolve the service names
        # again on every connection and keep working.
        self.containers.clear()
        self.start(bounty_setup=bounty_setup)

    # --------------------------------------------------------------- codebase
    def reset_codebase(self) -> None:
        """Restore the working copy to the vulnerable commit."""
        self.checkout(self.spec.vulnerable_commit)

    def checkout(self, commit: str) -> None:
        for args in (["checkout", "--force", commit], ["clean", "-fdq"]):
            result = run_host(["git", *args], cwd=self.codebase)
            if result.returncode != 0:
                raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")

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

    def run_git(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        """Run git in the workspace."""
        return run_host(["git", *args], cwd=cwd or self.root)

    def apply_diff(self, diff: str) -> bool:
        """Apply the agent's patch (a `git diff`) to the working copy.

        The same fallbacks as upstream (`git_patch_utils.apply_patch`): a plain
        apply, then a three way merge, then a partial apply.
        """
        if not diff.strip():
            return False
        patch_path = self.root / "agent_patch.diff"
        patch_path.write_text(diff)
        for extra in ([], ["--3way"], ["--reject"]):
            result = run_host(
                ["git", "apply", "--whitespace=nowarn", *extra, str(patch_path)],
                cwd=self.codebase,
            )
            if result.returncode == 0:
                return True
        logger.warning("Failed to apply the agent patch: %s", result.stderr)
        # `--reject` applies the hunks it could and leaves `.rej` files behind.
        self.reset_codebase()
        return False

    # ------------------------------------------------------------ subprocesses
    def run_script(
        self, directory: Path, script: str, timeout: int | None = None
    ) -> ExecResult[str]:
        """Run one of the task's shell scripts in the grading container."""
        return self._exec_in_grader(directory, script_command(script), timeout)

    def run_command(
        self, directory: Path, command: str, timeout: int | None = None
    ) -> ExecResult[str]:
        """Run a shell command in the grading container (not one of the scripts)."""
        return self._exec_in_grader(directory, f"exec bash -lc {shlex.quote(command)}", timeout)

    def _exec_in_grader(
        self, directory: Path, command: str, timeout: int | None = None
    ) -> ExecResult[str]:
        self.apply_code()
        container = self._ensure_grader_container()
        self._expose_stack(container)
        stop = threading.Event()
        sync = threading.Thread(target=self._sync_stack, args=(container, stop), daemon=True)
        sync.start()
        try:
            environment: list[str] = []
            for name, value in self.compose_env(directory).items():
                environment += ["--env", f"{name}={value}"]
            result = run_host(
                [
                    "docker",
                    "exec",
                    *environment,
                    "-w",
                    str(directory),
                    container,
                    "bash",
                    "-c",
                    command,
                ],
                timeout=timeout,
            )
        finally:
            stop.set()
            sync.join(timeout=60)
            self._restore_ownership()
        return ExecResult(
            success=result.returncode == 0,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _ensure_grader_container(self) -> str:
        """Start (once) the container the sample's own scripts run in.

        The scripts drive `docker compose` and `docker exec` against the target
        stack, so the container gets the docker socket and the workspace mounted
        at its host path, which keeps the relative paths of the scripts the same
        as upstream's. The stack itself is reached through `_expose_stack`.
        """
        if self.grader_container is not None:
            return self.grader_container
        name = f"{GRADER_CONTAINER_PREFIX}{self.root.name}"
        # a container left behind by an interrupted run still holds the workspace
        run_host(["docker", "rm", "--force", name])
        # the compose override and the build context live outside the workspace,
        # which keeps them away from the agent and the graded tree; the scripts
        # have to see both
        COMPOSE_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
        COMPOSE_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        result = run_host(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                name,
                "--volume",
                f"{self.root}:{self.root}",
                # the image installs the checkout with `pip install -e .` from
                # `/app/codebase`, a mount that exists only while it is built, and
                # the suite imports the project: the mount is repeated here so
                # that the import resolves to the patched checkout
                "--volume",
                f"{self.codebase}:{SANDBOX_CODEBASE_DIR}",
                "--volume",
                f"{COMPOSE_OVERRIDES_DIR}:{COMPOSE_OVERRIDES_DIR}",
                "--volume",
                f"{COMPOSE_BUILD_DIR}:{COMPOSE_BUILD_DIR}:ro",
                "--volume",
                f"{DOCKER_SOCKET}:{DOCKER_SOCKET}",
                "--env",
                "PIP_NO_INDEX=1",
                "--env",
                f"PIP_FIND_LINKS={SANDBOX_WHEELHOUSE}",
                "--env",
                "PIP_DISABLE_PIP_VERSION_CHECK=1",
                self.image,
                "tail",
                "-f",
                "/dev/null",
            ],
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Unable to start the grading container of {self.spec.sample_id}: "
                f"{result.stderr.strip()}"
            )
        self.grader_container = name
        return name

    def _sync_stack(self, container: str, stop: threading.Event) -> None:
        """Keep the view of the stack current while one of the scripts runs."""
        while not stop.wait(STACK_SYNC_INTERVAL):
            try:
                self._expose_stack(container)
            except Exception:
                logger.exception("Unable to expose the target stack")

    def _expose_stack(self, container: str) -> None:
        """Give the scripts the target stack the way they expect it.

        They reach the target containers by name on the compose network and talk
        to their published ports on `localhost`, so the grading container joins
        the networks of the stack and forwards those ports to the services.
        """
        with self._stack_lock:
            self._expose_stack_once(container)

    def _expose_stack_once(self, container: str) -> None:
        for directory in self.compose_dirs:
            for service, published, target, protocol in self.compose_ports(directory):
                self._forward(container, service, published, target, protocol)
        listed = run_host(
            [
                "docker",
                "ps",
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={self.compose_project}",
            ],
            timeout=120,
        )
        containers = listed.stdout.split()
        if not containers:
            return
        inspected = run_host(
            ["docker", "inspect", "--format", "{{json .}}", *containers], timeout=300
        )
        for line in inspected.stdout.splitlines():
            networks = json.loads(line)["NetworkSettings"]["Networks"]
            for network in networks:
                if network in self.connected_networks:
                    continue
                attached = run_host(
                    ["docker", "network", "connect", network, container], timeout=120
                )
                if attached.returncode != 0:
                    raise RuntimeError(
                        f"Unable to attach the grading container to {network}: "
                        f"{attached.stderr.strip()}"
                    )
                self.connected_networks.add(network)

    def _forward(
        self, container: str, service: str, published: int, target: int, protocol: str
    ) -> None:
        """Serve one published port of the stack on the loopback of the container."""
        if protocol != "tcp":
            logger.warning(
                "Not forwarding %s/%s of %s: only tcp is forwarded",
                published,
                protocol,
                service,
            )
            return
        if published in self.forwarded:
            return
        if not self.forwarder_written:
            written = run_host(
                [
                    "docker",
                    "exec",
                    container,
                    "bash",
                    "-c",
                    f"cat > {FORWARDER_PATH} <<'PY'\n{FORWARDER_SCRIPT}PY\n",
                ],
                timeout=120,
            )
            if written.returncode != 0:
                raise RuntimeError(
                    f"Unable to install the forwarder in the grading container: "
                    f"{written.stderr.strip()}"
                )
            self.forwarder_written = True
        started = run_host(
            [
                "docker",
                "exec",
                "--detach",
                container,
                "python3",
                FORWARDER_PATH,
                str(published),
                service,
                str(target),
            ],
            timeout=120,
        )
        if started.returncode != 0:
            raise RuntimeError(
                f"Unable to forward localhost:{published} to {service}:{target}: "
                f"{started.stderr.strip()}"
            )
        self.forwarded[published] = (service, target)

    def _restore_ownership(self) -> None:
        """Hand the files the scripts wrote back to the evaluator user.

        The scripts run as root (some of them install system packages) while the
        working copy belongs to the evaluator, whose later git commands must keep
        working.
        """
        if self.grader_container is None:
            return
        result = run_host(
            [
                "docker",
                "exec",
                self.grader_container,
                "chown",
                "-R",
                f"{os.getuid()}:{os.getgid()}",
                str(self.root),
            ],
            timeout=900,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to hand %s back to the evaluator user: %s",
                self.root,
                result.stderr.strip(),
            )

    def _stop_grader_container(self) -> None:
        if self.grader_container is None:
            return
        run_host(["docker", "rm", "--force", self.grader_container], timeout=300)
        self.grader_container = None
        self.forwarder_written = False
        self.forwarded.clear()
        self.connected_networks.clear()


@dataclass(frozen=True)
class BuildFix:
    """Adaptations for a bounty whose own `Dockerfile` no longer builds.

    `constraints` pin a package that the Dockerfile installs unpinned and whose
    current release broke it; `replace` rewrites text of the `Dockerfile`s (a
    package host that stopped serving the distribution the sample pins, or a
    package the current base image renamed); `env` sets build variables the
    Dockerfile does not know it needs.
    `apt_conf` is written to the image before its first `apt-get update`, for the
    package manager options those replacements need.
    """

    constraints: tuple[str, ...] = ()
    replace: tuple[tuple[str, str], ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    apt_hosts: tuple[tuple[str, str], ...] = ()
    # (substring of a RUN line) -> refresh the package lists before that line:
    # the base image bakes lists that name versions no mirror serves any more
    apt_update_before: tuple[str, ...] = ()
    apt_conf: tuple[str, ...] = ()
    # the archive the sample now installs from forces https, and the base image
    # of some bounties ships no CA certificates
    apt_ca_certificates: bool = False


BUILD_CONSTRAINTS_PATH = "/opt/build-constraints.txt"
APT_CONF_PATH = "/etc/apt/apt.conf.d/99build-fix"

# The Debian archive keeps every package it ever served, at the day it served it:
# the versions the base images of the samples name are still there, which is what
# makes an image built against a package that was superseded buildable again.
SECURITY_ARCHIVE = "http://snapshot.debian.org/archive/debian-security/20260201T000000Z"

# Only for the samples that do not build against today's package indexes. The
# datasets' `Dockerfile`s are unchanged upstream, so they resolve to the current
# release of an unpinned dependency (or to a repository that has been dropped),
# and the image of the target stack cannot be built at all.
BUILD_FIXES: dict[str, BuildFix] = {
    # `victim.py` 用 `agentscope.studio.init(...)` 起服务，而那条导入链需要 `tqdm` 与
    # `chainlit`：两者既不在 `setup.py` 的 `minimal_requires` 里、也不在任何 extra 里，
    # `pip install -e .` 之后 `import agentscope.studio` 直接 ModuleNotFoundError。
    # victim 容器于是永远起不来（healthcheck 一直 unhealthy），而样本的
    # `run_repo_invariants.sh` 是 `set -e` + curl 探活：连不上就直接退，一条分数都不写。
    "agentscope": BuildFix(
        replace=(("RUN pip install -e .", "RUN pip install -e . && pip install tqdm chainlit"),),
    ),
    "gradio": BuildFix(constraints=("huggingface_hub<0.30",)),
    # `onnxruntime==1.16.3`, which this checkout pins, ships a wheel whose
    # extension asks for an executable stack; Debian trixie's glibc 2.41 refuses to
    # load it (`cannot enable executable stack as shared object requires`), so
    # `invokeai-web` dies on import and nothing of the sample runs (its `detect`
    # attempt failed with a dead server). The flag is an artefact of the way the
    # wheel was built — the code does not run from the stack — and clearing it is
    # enough. The later checkout of `InvokeAI_bounty_1` pins 1.19.2, which does not
    # need it.
    "InvokeAI": BuildFix(
        replace=(
            (
                'RUN pip install -e ".[test]"',
                'RUN pip install -e ".[test]" && pip install --quiet patchelf'
                " && patchelf --clear-execstack"
                " /usr/local/lib/python3.11/site-packages/onnxruntime/capi"
                "/onnxruntime_pybind11_state*.so",
            ),
        )
    ),
    "pytorch-lightning": BuildFix(constraints=("setuptools<81",)),
    # `requirements.txt` pins `protobuf==3.20` and leaves `pyautogen` unpinned,
    # while the current release of it needs protobuf 5
    "gpt_academic": BuildFix(constraints=("pyautogen<0.3",)),
    # `lunary` builds its database image from `postgres:15-bullseye`: that image
    # bakes package lists naming `bullseye-security` versions Debian has since
    # removed from its pool, and `bullseye-pgdg` moved to the archive (the
    # `groonga` source package writes that repository into the image itself)
    "lunary": BuildFix(
        replace=(("apt.postgresql.org", "apt-archive.postgresql.org"),),
        apt_hosts=(
            ("apt.postgresql.org", "apt-archive.postgresql.org"),
            ("http://security.debian.org/debian-security", SECURITY_ARCHIVE),
        ),
        apt_conf=(
            # the archive serves the state of that day, so its `Release` file is
            # always older than the validity window Debian gave it
            'Acquire::Check-Valid-Until "false";',
            # the archive is far away and slow: a single dropped connection
            # would otherwise fail the whole build
            'Acquire::Retries "5";',
            'Acquire::http::Timeout "60";',
        ),
        apt_ca_certificates=True,
        apt_update_before=("apt install",),
    ),
    # the `Dockerfile` of the bounty installs two packages under their bookworm
    # names, and its unpinned base image now resolves to trixie
    "open-webui": BuildFix(
        replace=(
            ("libgl1-mesa-glx", "libgl1"),
            ("libglib2.0-0", "libglib2.0-0t64"),
            # 4GB of packages: an interrupted or repeated build would download
            # them again without a cache
            (
                "RUN pip install -e /codebase",
                "RUN --mount=type=cache,target=/root/.cache/pip pip install -e /codebase",
            ),
        ),
        # `npm install` of the frontend pulls `onnxruntime-node`, whose install
        # script downloads the CUDA binaries from the GitHub release page on every
        # Linux x64 machine; the asset it asks for does not exist (`404`), so the
        # whole `pip install -e` fails. The package documents `skip` for exactly
        # this, and the CPU binaries it needs ship in the package (this also keeps
        # github.com out of the build).
        env=(("ONNXRUNTIME_NODE_INSTALL_CUDA", "skip"),),
    ),
}


def _apt_host_rewrite(hosts: tuple[tuple[str, str], ...]) -> str | None:
    """Dockerfile line that replaces package hosts in the sources of the image."""
    if not hosts:
        return None
    expressions = "; ".join(f"s#{old}#{new}#g" for old, new in hosts)
    return (
        "RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/*; do "
        '[ -f "$f" ] || continue; '
        f"sed -i -E '{expressions}' \"$f\"; done"
    )


def build_adaptation_signature(fix: BuildFix | None) -> str:
    """Fingerprint of the build adaptations, to tell a reusable context from a stale one."""
    settings = [
        PIP_INDEX_URL,
        APT_MIRROR_URL,
        APT_MIRROR_SCRIPT,
        NPM_REGISTRY_URL,
        NODE_HEADERS_DIR,
        APT_CONF_PATH,
        repr(fix),
    ]
    return hashlib.sha256("\n".join(settings).encode()).hexdigest()[:16]


def pin_build_mirrors(
    directory: Path,
    index_url: str = PIP_INDEX_URL,
    apt_script: str = APT_MIRROR_SCRIPT,
    fix: BuildFix | None = None,
) -> None:
    """Point the package managers of the bounties' Dockerfiles at local mirrors.

    They install from the public PyPI and the distro mirrors, which are slow
    enough from some networks to push a single target build into tens of minutes
    (and every code change rebuilds those layers). Only the workspace copy is
    rewritten; the dataset stays untouched.
    """
    pins: list[str] = []
    if fix is not None and fix.env:
        pins.extend(f"ENV {name}={value}" for name, value in fix.env)
    if fix is not None and fix.constraints:
        pins.append(f"ENV PIP_CONSTRAINT={BUILD_CONSTRAINTS_PATH}")
        pinned = " ".join(f'"{line}"' for line in fix.constraints)
        pins.append(f"RUN printf '%s\\n' {pinned} > {BUILD_CONSTRAINTS_PATH}")
    if fix is not None and fix.apt_conf:
        configured = " ".join(f"'{line}'" for line in fix.apt_conf)
        pins.append(f"RUN printf '%s\\n' {configured} > {APT_CONF_PATH}")
    for dockerfile in sorted(directory.rglob("Dockerfile*")):
        if not dockerfile.is_file():
            continue
        if dockerfile.relative_to(directory).parts[0] == "codebase":
            # the code under test keeps its own build files
            continue
        text = dockerfile.read_text()
        for host, replacement in fix.replace if fix is not None else ():
            text = text.replace(host, replacement)
        apt_rewrite = _apt_host_rewrite(fix.apt_hosts if fix is not None else ())
        trailing_pins: list[str] = []
        if fix is not None and fix.apt_ca_certificates:
            # the image may ship a repository that is already gone, so the
            # rewrite has to come before this update as well
            rewritten_update = (
                f"{apt_rewrite.removeprefix('RUN ')} && " if apt_rewrite is not None else ""
            )
            trailing_pins.append(
                f"RUN {rewritten_update}apt-get update && apt-get install -y ca-certificates"
            )
        lines = text.splitlines()
        rewritten: list[str] = []
        for line in lines:
            if apt_rewrite is not None and "apt-get update" in line:
                # a package of the image installs a repository of its own
                rewritten.append(apt_rewrite)
            if (
                fix is not None
                and "apt-get update" not in line
                and any(needle in line for needle in fix.apt_update_before)
            ):
                rewritten.append("RUN apt-get update")
            rewritten.append(line)
            if line.strip().upper().startswith("FROM "):
                rewritten.extend(pins)
                rewritten.append(f"ENV PIP_INDEX_URL={index_url}")
                rewritten.append(f"ENV npm_config_registry={NPM_REGISTRY_URL}")
                rewritten.append(f"ENV npm_config_nodedir={NODE_HEADERS_DIR}")
                rewritten.append(f"RUN {apt_script}")
                rewritten.extend(trailing_pins)
        rewritten_text = "\n".join(rewritten) + "\n"
        if rewritten_text != dockerfile.read_text():
            dockerfile.write_text(rewritten_text)


# Files of a fresh checkout that differ between two clones of the same commit
# (the index records modification times, the reflog records when the clone ran).
# They are part of the build context, so without this every workspace would miss
# the build cache and re-run the dependency install, which downloads packages
# during the evaluation.
BUILD_CONTEXT_VOLATILE = ("codebase/.git/index", "codebase/.git/logs", "agent_workspace")


def pin_build_context(directory: Path) -> None:
    """Keep the volatile checkout files out of the build context."""
    (directory / ".dockerignore").write_text("\n".join(BUILD_CONTEXT_VOLATILE) + "\n")


def clone_codebase(mirror: Path, commit: str, destination: Path) -> None:
    """Clone `mirror` and check out `commit` into `destination`."""
    if destination.exists():
        shutil.rmtree(destination)
    clone = run_host(["git", "clone", "--quiet", str(mirror), str(destination)])
    if clone.returncode != 0:
        raise RuntimeError(f"git clone failed: {clone.stderr.strip()}")
    checkout = run_host(["git", "checkout", "--quiet", "--force", commit], cwd=destination)
    if checkout.returncode != 0:
        raise RuntimeError(f"git checkout failed: {checkout.stderr.strip()}")


def _container_health(name: str) -> tuple[bool, str, str]:
    """(running, state, health) of a container.

    `health` is empty while the container has no healthcheck, and also for a
    moment after a restart, before the first probe has run.
    """
    result = run_host(
        [
            "docker",
            "inspect",
            "--format={{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}",
            name,
        ]
    )
    if result.returncode != 0:
        return False, "", ""
    state, _, health = result.stdout.strip().partition("|")
    return True, state, health.strip()


# ---------------------------------------------------------------------- sandbox


async def sh(
    sbx: SandboxEnvironment,
    command: str,
    cwd: str | None = None,
    timeout: int | None = None,
) -> ExecResult[str]:
    """Run a shell command inside the agent sandbox."""
    return await sbx.exec(["bash", "-lc", command], cwd=cwd, timeout=timeout)


def script_command(script: str) -> str:
    """A shell command that runs one of the task's scripts the way upstream does."""
    return f"{SCRIPT_PATH_PRELUDE}exec bash {shlex.quote(script)}"


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
        if any(relative == path or relative.startswith(f"{path}/") for path in exclude_paths):
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
    result = await sh(sbx, f"tar czf - {excludes} -C {remote_dir} . | base64 -w 0", timeout=600)
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
    return await sh(
        sbx, f"bash -c {shlex.quote(script_command(script))}", cwd=directory, timeout=timeout
    )


async def init_sandbox(sbx: SandboxEnvironment, spec: BountySpec) -> None:
    """Create the agent workspace inside the sandbox.

    Mirrors upstream `InitFilesResource`: a codebase checkout on a `dev` branch,
    an empty `exploit_files` directory and a copy of the bounty `verify_files`.

    The image of a local task already carries its checkout; service tasks clone
    it from the pruned mirror the compose file mounts.
    """
    commands = [
        f"mkdir -p {SANDBOX_APP_DIR}",
        f"rm -rf {SANDBOX_VERIFY_DIR}",
    ]
    if spec.is_service_task:
        commands.extend(
            [
                f"rm -rf {SANDBOX_CODEBASE_DIR}",
                f"git clone --quiet {SANDBOX_MIRROR_DIR} {SANDBOX_CODEBASE_DIR}",
                f"git -C {SANDBOX_CODEBASE_DIR} checkout --quiet --force {spec.vulnerable_commit}",
            ]
        )
    commands.extend(
        [
            # creates `dev` on the first run and resets it afterwards
            f"git -C {SANDBOX_CODEBASE_DIR} checkout --quiet -B dev",
            f"mkdir -p {SANDBOX_EXPLOIT_DIR}",
        ]
    )
    clone = await sh(sbx, " && ".join(commands), timeout=1_800)
    if not clone.success:
        raise RuntimeError(f"Failed to prepare the agent workspace: {clone.stderr}")
    # a few tasks (`neural-compressor`, `imaginairy`, `open-webui`) ship no
    # `verify_files`, their `verify.sh` needs nothing next to it
    verify_files = spec.bounty_dir / "verify_files"
    if verify_files.is_dir():
        await upload_tree(sbx, verify_files, SANDBOX_VERIFY_DIR, skip_hidden=True)
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


async def sandbox_git_diff(sbx: SandboxEnvironment, base: str) -> str:
    """Diff of the agent's changes against `base` (upstream `git_diff`).

    Measured against the commit the sandbox started at rather than `HEAD`: agents
    commit their work, and `HEAD` would report their changes as no change at all.
    """
    await sh(sbx, "git add -A", cwd=SANDBOX_CODEBASE_DIR)
    numstat = await sh(sbx, f"git diff --cached {base} --numstat", cwd=SANDBOX_CODEBASE_DIR)
    binary = {
        parts[2]
        for line in numstat.stdout.splitlines()
        if (parts := line.split("\t")) and len(parts) == NUMSTAT_FIELDS and parts[0] == "-"
    }
    pathspecs = " ".join(f":(exclude){path}" for path in sorted(binary))
    diff = await sh(sbx, f"git diff --cached {base} -- {pathspecs}", cwd=SANDBOX_CODEBASE_DIR)
    return diff.stdout


@atexit.register
def _stop_live_workspaces() -> None:
    for workspace in list(_LIVE_WORKSPACES.values()):
        try:
            workspace.stop()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to stop workspace on exit", exc_info=True)
