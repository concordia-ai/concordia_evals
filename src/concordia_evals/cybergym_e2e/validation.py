"""Stage-by-stage validation orchestration, ported from upstream.

Mirrors scripts/run_agent.py::run_final_validation: for every validation
stage a FRESH container is started from the task's build image (avoiding
cross-stage sanitizer contamination), the vendored validate.py is run inside
with --only-stage, and the JSON result is read back.

Containers are started with the docker SDK on the default bridge network
(unrestricted, as upstream does: prepare.sh may need network). The agent
sandbox is never used for validation.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from concordia_evals.cybergym_e2e.setup_solver import (
    APT_BYPASS_PROXY,
    ARVO_IMAGE_CLEANUP,
    INSTALL_VALIDATE_DEPS,
    INSTALL_VALIDATE_DEPS_ATTEMPTS,
    VENDOR_DIR,
)

logger = logging.getLogger(__name__)

StageStatus = Literal["passed", "failed", "skipped", "error"]
VALIDATION_TIMEOUT = 7200  # upstream exec_run timeout per stage


def _docker(
    args: list[str], timeout: int = 120, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr[-1000:]}")
    return result


def _exec(
    container_id: str,
    cmd: str,
    timeout: int = 1200,
    check: bool = True,
) -> tuple[int, str, str]:
    """Docker exec with timeout wrapper, like upstream utils.exec_run."""
    result = subprocess.run(
        ["docker", "exec", container_id, "timeout", str(timeout), "bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=timeout + 60,
        errors="replace",
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"exec in {container_id[:12]} failed ({result.returncode}): "
            f"{cmd[:200]}\nstdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
        )
    return result.returncode, result.stdout, result.stderr


def _start_validation_container(
    build_image: str,
    task_data_dir: Path,
    script_dir: Path,
    setup_proxy: str | None = None,
) -> str:
    args = [
        "run",
        "-d",
        "--rm",
        # MSan/ASan binaries need personality(ADDR_NO_RANDOMIZE) for their
        # shadow-mmap collision recovery; default seccomp blocks it.
        "--security-opt",
        "seccomp=unconfined",
        # ffmpeg-family tasks rsync FATE samples from fate-suite.ffmpeg.org
        # during compile (rsync:// port 873); the transfer hangs on slow
        # links and `|| true` only helps on failure, not on hangs. The
        # samples are provably unused (upstream's own test.sh drops
        # SAMPLES=), so fail the fetch fast via localhost.
        "--add-host",
        "fate-suite.ffmpeg.org:127.0.0.1",
    ]
    if setup_proxy is not None:
        args += [
            "--add-host",
            "hostgw:host-gateway",
            "-e",
            f"HTTP_PROXY={setup_proxy}",
            "-e",
            f"HTTPS_PROXY={setup_proxy}",
            "-e",
            f"http_proxy={setup_proxy}",
            "-e",
            f"https_proxy={setup_proxy}",
            "-e",
            "NO_PROXY=127.0.0.1,localhost",
            "-e",
            "no_proxy=127.0.0.1,localhost",
        ]
    args += [
        "-v",
        f"{task_data_dir}:/task_data:ro",
        "-v",
        f"{script_dir}:/task_scripts:ro",
        "-w",
        "/src",
        build_image,
        "sleep",
        "infinity",
    ]
    result = _docker(args)
    return result.stdout.strip()


def _setup_validation_container(
    container_id: str,
    *,
    build_image: str,
    mode: str,
    repo_to_patch: str,
    immutable_files: list[str],
    pre_patches: list[str],
    compile_timeout: int | None = None,
) -> None:
    """Set up a fresh validation container (upstream setup_workspace, run_prepare=False)."""
    from concordia_evals.cybergym_e2e.setup_solver import _sanitized_config_toml

    if "arvo" in build_image or "cybergym" in build_image:
        _exec(container_id, ARVO_IMAGE_CLEANUP, timeout=300)

    _exec(container_id, "mkdir -p /config /data /output /scripts")
    _exec(container_id, APT_BYPASS_PROXY)
    _exec(
        container_id,
        "export DEBIAN_FRONTEND=noninteractive && "
        "apt-get update -qq && apt-get install -y -qq sudo git >/dev/null 2>&1",
        timeout=900,
    )
    last_err: Exception | None = None
    for attempt in range(INSTALL_VALIDATE_DEPS_ATTEMPTS):
        try:
            _exec(container_id, INSTALL_VALIDATE_DEPS, timeout=1200)
            last_err = None
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "install validate deps attempt %d/%d failed in %s: %s",
                attempt + 1,
                INSTALL_VALIDATE_DEPS_ATTEMPTS,
                container_id[:12],
                e,
            )
    if last_err is not None:
        raise last_err

    _docker(["cp", str(VENDOR_DIR / "validate.py"), f"{container_id}:/scripts/validate.py"])
    if compile_timeout is not None:
        # Documented, opt-in deviation: the vendored validate.py hardcodes a
        # 3600s compile timeout (exactly 4 occurrences, all compile.sh calls).
        # Large projects (e.g. ffmpeg with MSan) exceed this on loaded hosts.
        # The vendored file stays byte-identical; only the deployed copy is
        # adjusted, and only when the caller explicitly sets a timeout.
        code, out, _ = _exec(
            container_id,
            f"sed -i 's/timeout=3600/timeout={compile_timeout}/g' /scripts/validate.py "
            "&& grep -c 'timeout=3600' /scripts/validate.py || true",
            check=False,
        )
        if code != 0:
            raise RuntimeError("Failed to adjust validate.py compile timeout")
        remaining = out.strip()
        if remaining != "0":
            raise RuntimeError(
                f"Unexpected validate.py state after timeout adjustment: "
                f"{remaining} occurrences of timeout=3600 remain"
            )
    _exec(
        container_id,
        "cp /task_data/src.tgz /src/src.tgz && tar xf /src/src.tgz -C /src && rm /src/src.tgz",
        timeout=600,
    )
    if mode == "patch-only":
        _exec(
            container_id,
            "cp /task_data/crash.log /task_data/poc.bin /src/ 2>/dev/null || true",
        )
    for script in ("prepare.sh", "compile.sh", "run_poc.sh", "test.sh"):
        _exec(
            container_id,
            f"cp /task_scripts/{script} /src/{script} 2>/dev/null || true",
        )

    config_toml = _sanitized_config_toml(repo_to_patch, immutable_files)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp:
        tmp.write(config_toml)
        tmp_path = tmp.name
    try:
        _docker(["cp", tmp_path, f"{container_id}:/config/config.toml"])
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if repo_to_patch:
        _exec(container_id, f"rm -rf /src/{repo_to_patch}/.git")

    pre_patch_repo = f"/src/{repo_to_patch}" if repo_to_patch else "/src"
    for pre_patch_name in pre_patches:
        applied = False
        for strip_level in (0, 1, 2, 3):
            if strip_level == 0:
                cmd = f"cd {pre_patch_repo} && git apply /task_scripts/{pre_patch_name}"
            else:
                cmd = (
                    f"cd {pre_patch_repo} && patch -p{strip_level} < /task_scripts/{pre_patch_name}"
                )
            code, _, _ = _exec(container_id, cmd, check=False)
            if code == 0:
                applied = True
                break
        if not applied:
            raise RuntimeError(f"Failed to apply pre_patch {pre_patch_name} with strip levels 0-3")

    # Ground truth PoC always goes to /data for validation.
    _exec(container_id, "cp /task_data/poc.bin /data/poc.bin 2>/dev/null || true")


def run_stage_validation(
    *,
    stage: int,
    build_image: str,
    mode: str,
    repo_to_patch: str,
    immutable_files: list[str],
    pre_patches: list[str],
    task_data_dir: Path,
    script_dir: Path,
    poc_path: Path | None,
    patch_path: Path | None,
    compile_timeout: int | None = None,
    setup_proxy: str | None = None,
) -> StageStatus:
    """Run one validation stage in a fresh container; return its status."""
    container_id: str | None = None
    try:
        container_id = _start_validation_container(
            build_image, task_data_dir, script_dir, setup_proxy
        )
        _setup_validation_container(
            container_id,
            build_image=build_image,
            mode=mode,
            repo_to_patch=repo_to_patch,
            immutable_files=immutable_files,
            pre_patches=pre_patches,
            compile_timeout=compile_timeout,
        )

        if poc_path is not None and stage in (1, 2):
            _docker(["cp", str(poc_path), f"{container_id}:/output/poc.bin"])
        if patch_path is not None:
            _docker(["cp", str(patch_path), f"{container_id}:/output/fix.patch"])

        cmd = (
            "/scripts/.venv/bin/python /scripts/validate.py "
            "--src-dir /src --config-dir /config --data-dir /data "
            "--json-output /output/validation_results.json "
            f"--only-stage {stage} --run-prepare"
        )
        if poc_path is not None and stage in (1, 2):
            cmd += " --poc-file /output/poc.bin"
        if patch_path is not None:
            cmd += " --patch-file /output/fix.patch"

        # The outer exec timeout must exceed the in-container compile timeout
        # (validate.py's own deadline) plus prepare/run overhead.
        exec_timeout = (
            max(VALIDATION_TIMEOUT, compile_timeout + 1800)
            if compile_timeout is not None
            else VALIDATION_TIMEOUT
        )
        code, stdout, stderr = _exec(container_id, cmd, timeout=exec_timeout, check=False)
        if code != 0:
            logger.warning(
                "Stage %d validate.py exited %d:\n%s\n%s",
                stage,
                code,
                stdout[-2000:],
                stderr[-2000:],
            )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = _docker(
                ["cp", f"{container_id}:/output/validation_results.json", tmp_path],
                check=False,
            )
            if result.returncode != 0:
                return "error"
            with open(tmp_path) as f:
                stage_results: dict[str, Any] = json.load(f)
            status = stage_results.get(f"stage{stage}")
            return status if status in ("passed", "failed", "skipped") else "error"
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        logger.exception("Stage %d validation error", stage)
        return "error"
    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                check=False,
                capture_output=True,
            )


def run_full_validation(
    *,
    mode: str,
    build_image: str,
    repo_to_patch: str,
    immutable_files: list[str],
    pre_patches: list[str],
    task_data_dir: Path,
    script_dir: Path,
    poc_path: Path | None,
    patch_path: Path | None,
    compile_timeout: int | None = None,
    setup_proxy: str | None = None,
) -> dict[str, StageStatus | None]:
    """Run all applicable stages with upstream's cumulative skip logic."""
    results: dict[str, StageStatus | None] = {
        "stage1": None,
        "stage2": None,
        "stage3": None,
        "stage4": None,
    }
    stages = [1, 2, 3, 4] if mode == "e2e" else [3, 4]
    # Upstream skip rules: S2 needs S1; S3 needs S2 (e2e only); S4 needs S3.
    depends_on: dict[int, str | None] = {
        2: "stage1",
        3: "stage2" if mode == "e2e" else None,
        4: "stage3",
    }
    for stage in stages:
        dependency = depends_on.get(stage)
        if dependency is not None and results.get(dependency) != "passed":
            results[f"stage{stage}"] = "skipped"
            continue
        results[f"stage{stage}"] = run_stage_validation(
            stage=stage,
            build_image=build_image,
            mode=mode,
            repo_to_patch=repo_to_patch,
            immutable_files=immutable_files,
            pre_patches=pre_patches,
            task_data_dir=task_data_dir,
            script_dir=script_dir,
            poc_path=poc_path,
            patch_path=patch_path,
            compile_timeout=compile_timeout,
            setup_proxy=setup_proxy,
        )
    return results
