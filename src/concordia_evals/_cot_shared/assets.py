"""Fetch and verify pinned upstream benchmark repositories into a local cache."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from filelock import FileLock
from platformdirs import user_cache_dir

_DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 120
_LOCK_TIMEOUT_SECONDS: Final[int] = 120
_TRUTHY_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes"})

ASSET_SPECS: Final[dict[str, dict[str, str]]] = {
    "cot-control": {
        "repository": "https://github.com/YuehHanChen/CoTControl.git",
        "commit": "5d78aeffe0152ba087c2d31cd07712d029c64785",
        "archive_url": (
            "https://codeload.github.com/YuehHanChen/CoTControl/tar.gz/"
            "5d78aeffe0152ba087c2d31cd07712d029c64785"
        ),
        "sha256": "64041b7385f4234cd3cda4ed02749209405b353faa29ea6afbcb0f592c8c56af",
    },
    "openai-monitorability": {
        "repository": "https://github.com/openai/monitorability-evals.git",
        "commit": "d14f181a0623f7949fdb2d8c3f15874115fa44ba",
        "archive_url": (
            "https://codeload.github.com/openai/monitorability-evals/tar.gz/"
            "d14f181a0623f7949fdb2d8c3f15874115fa44ba"
        ),
        "sha256": "d1e40622187cf25dc52b8dac68bc228e8bb10eef3fc6384a42719bbaafc6d5bc",
    },
}


class AssetIntegrityError(ValueError):
    """Raised when an upstream archive or cached file fails digest verification."""


def specifications() -> dict[str, dict[str, str]]:
    """Return pinned upstream asset specifications."""
    return {name: dict(spec) for name, spec in ASSET_SPECS.items()}


def cache_root() -> Path:
    """Return the root directory used for cached upstream assets."""
    raw_env = os.environ.get("CONCORDIA_EVALS_CACHE_DIR", "").strip()
    raw = raw_env if raw_env else user_cache_dir("concordia-evals")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        raise ValueError("CONCORDIA_EVALS_CACHE_DIR must be an absolute path")
    return root


def asset_path(name: str) -> Path:
    """Return the expected cache directory for a pinned asset name."""
    spec = ASSET_SPECS[name]
    return cache_root() / f"{name}-{spec['commit']}"


def sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def offline_mode_enabled() -> bool:
    """Return whether offline mode is requested via environment variables."""
    return any(
        os.environ.get(var, "").strip().lower() in _TRUTHY_ENV_VALUES
        for var in ("CONCORDIA_EVALS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    )


@contextlib.contextmanager
def hf_offline_env() -> Iterator[None]:
    """Propagate ``CONCORDIA_EVALS_OFFLINE`` into HF env vars within a block."""
    if not offline_mode_enabled():
        yield
        return
    keys = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = "1"
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _inside(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise AssetIntegrityError(f"Unsafe archive path: {relative}")
    target = root.joinpath(*path.parts)
    if root.resolve() not in target.resolve().parents:
        raise AssetIntegrityError(f"Archive path escapes its cache: {relative}")
    return target


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        non_empty_parts = [
            PurePosixPath(m.name).parts for m in members if PurePosixPath(m.name).parts
        ]
        first_parts = {parts[0] for parts in non_empty_parts}
        has_root_file = any(len(PurePosixPath(m.name).parts) == 1 and m.isfile() for m in members)
        strip_root = len(first_parts) == 1 and not has_root_file

        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise AssetIntegrityError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise AssetIntegrityError(f"Archive links are not supported: {member.name}")
            relative = PurePosixPath(*path.parts[1:]) if strip_root else path
            if not relative.parts or str(relative) == ".":
                continue
            target = _inside(destination, str(relative))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = source.extractfile(member)
                if handle is None:
                    raise AssetIntegrityError(f"Archive member is unreadable: {member.name}")
                with handle, target.open("wb") as output:
                    shutil.copyfileobj(handle, output)


def _verify(root: Path, spec: dict[str, str]) -> None:
    marker = root / ".asset-manifest.json"
    if not marker.is_file():
        raise AssetIntegrityError(f"Incomplete asset cache: {root}")
    try:
        raw_manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetIntegrityError(f"Corrupt asset manifest: {marker}") from exc
    if not isinstance(raw_manifest, dict):
        raise AssetIntegrityError(f"Invalid asset manifest structure: {marker}")
    manifest = cast(dict[str, Any], raw_manifest)
    if manifest.get("archive_sha256") != spec["sha256"]:
        raise AssetIntegrityError("Cached asset belongs to another archive")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise AssetIntegrityError(f"Empty or invalid asset manifest: {marker}")

    files = cast(dict[str, Any], raw_files)
    expected_paths = {PurePosixPath(str(rel)).as_posix() for rel in files}
    actual_paths: set[str] = set()
    for disk_path in root.rglob("*"):
        if not disk_path.is_file():
            continue
        rel = disk_path.relative_to(root)
        rel_posix = rel.as_posix()
        if rel_posix == ".asset-manifest.json":
            continue
        if rel.parts and rel.parts[0] == ".cache":
            continue
        actual_paths.add(rel_posix)

    unexpected = sorted(actual_paths - expected_paths)
    if unexpected:
        raise AssetIntegrityError(
            f"Unexpected file(s) in cached asset {root}: {','.join(unexpected[:5])}"
        )

    for relative, expected in files.items():
        if not isinstance(expected, str):
            raise AssetIntegrityError(f"Invalid digest entry for {relative}")
        path = _inside(root, str(relative))
        if not path.is_file() or sha256(path) != expected:
            raise AssetIntegrityError(f"Cached asset changed: {relative}")


def ensure_asset(name: str, archive_path: Path | None = None) -> Path:
    """Download, verify, and cache a pinned upstream asset archive.

    Args:
        name: Key in ``ASSET_SPECS`` (``"cot-control"`` or ``"openai-monitorability"``).
        archive_path: Optional local tarball path for air-gapped initialization.

    Returns:
        Path to the verified extracted asset directory.

    Raises:
        FileNotFoundError: If the asset is missing and offline mode is enabled.
        AssetIntegrityError: If the downloaded archive or cached files fail SHA-256 verification.
    """
    spec = ASSET_SPECS[name]
    destination = asset_path(name)
    parent = cache_root()
    parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{destination}.lock", timeout=_LOCK_TIMEOUT_SECONDS):
        if destination.exists():
            _verify(destination, spec)
            return destination
        if archive_path is None and offline_mode_enabled():
            raise FileNotFoundError(
                f"Asset {name} is not cached at {destination}; "
                "prepare assets before enabling offline mode."
            )
        with tempfile.TemporaryDirectory(prefix=".asset-", dir=parent) as temporary:
            temp = Path(temporary)
            archive = archive_path
            if archive is None:
                archive = temp / "source.tar.gz"
                request = urllib.request.Request(
                    spec["archive_url"],
                    headers={"User-Agent": "concordia-evals/0.1"},
                )
                with (
                    urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as source,
                    archive.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target)
            if sha256(archive) != spec["sha256"]:
                raise AssetIntegrityError(f"Archive checksum mismatch: {name}")
            staged = temp / "source"
            staged.mkdir()
            _extract_tar(archive, staged)
            manifest = {
                path.relative_to(staged).as_posix(): sha256(path)
                for path in sorted(staged.rglob("*"))
                if path.is_file()
            }
            (staged / ".asset-manifest.json").write_text(
                json.dumps({"archive_sha256": spec["sha256"], "files": manifest}, indent=2) + "\n",
                encoding="utf-8",
            )
            staged.rename(destination)
    return destination
