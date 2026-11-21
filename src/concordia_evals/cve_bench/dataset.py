"""CVEBench 的挑战数据：`challenge.yml` → Inspect 样本，以及 compose 文件净化。

数据随包发布在 `cve_bench/data/`：

* `challenges/<CVE>/` —— 每个挑战的 `challenge.yml`、`compose.yml`、`.env`、
  `target/`（镜像构建上下文）等；
* `docker/`、`evaluations/`、`metadata/`、`sandboxes/` —— compose 文件用相对路径
  引用它们（`../../docker/compose-target.yml`、`../evaluations` 等），所以目录层级
  必须保持不变。

与 upstream（cve-bench `1a52fd1`）的差别：只保留 docker sandbox，K8s 的
`helm-values.yaml`/`k8s/` 与 nvd 数据不再随包发布。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.util import SandboxEnvironmentSpec

from concordia_evals.constants import CONCORDIA_EVALS_CACHE_PATH

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CHALLENGES_DIR = DATA_DIR / "challenges"
METADATA_DIR = DATA_DIR / "metadata"

CHALLENGE_INFO_FILENAME = "challenge.yml"
COMPOSE_YAML_FILENAME = "compose.yml"

# upstream 的 docker 挑战清单（29 个 K8s 支持 + 1 个仅 docker 支持的 prestashop），
# 顺序与 upstream 一致，所以日志里的样本顺序不变。
DEFAULT_CHALLENGES = (
    "CVE-2024-34716",
    "CVE-2023-37999",
    "CVE-2023-51483",
    "CVE-2024-22120",
    "CVE-2024-2359",
    "CVE-2024-2771",
    "CVE-2024-30542",
    "CVE-2024-31611",
    "CVE-2024-32167",
    "CVE-2024-3234",
    "CVE-2024-32511",
    "CVE-2024-32980",
    "CVE-2024-32986",
    "CVE-2024-34070",
    "CVE-2024-3408",
    "CVE-2024-34359",
    "CVE-2024-3495",
    "CVE-2024-35187",
    "CVE-2024-3552",
    "CVE-2024-36412",
    "CVE-2024-36779",
    "CVE-2024-37388",
    "CVE-2024-37831",
    "CVE-2024-37849",
    "CVE-2024-4223",
    "CVE-2024-4323",
    "CVE-2024-4442",
    "CVE-2024-4443",
    "CVE-2024-5084",
    "CVE-2024-5452",
)

# upstream 在默认清单里排除的 CVE（原因同 upstream）。
UNSUPPORTED_CHALLENGES = {
    "CVE-2024-36858": "Docker image is not public.",
    "CVE-2024-32964": "Target container start failed.",
}

# 净化后的 compose 文件（含 .env）落到缓存目录，不写仓库工作区。
SANITIZED_COMPOSE_DIR = CONCORDIA_EVALS_CACHE_PATH / "cve_bench" / "compose"

# 会让 agent（在目标机上拿到 RCE 之后）控制宿主 docker daemon、或篡改跨样本
# 共享文件的宿主挂载。
DOCKER_SOCK_MOUNTS = ("/var/run/docker.sock", "/run/docker.sock")

# compose 里 service 下 `volumes:` 键的缩进至少是 2（顶层键在 0 缩进）
SERVICE_VOLUME_INDENT = 2


@dataclass(frozen=True)
class Variant:
    """`challenge.yml` 里的一个提示词变体。"""

    prompt: str
    # 键是容器内目标路径，值是宿主机源路径或文件内容（与 upstream 相同）。
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChallengeInfo:
    """`challenge.yml` 的内容。"""

    name: str
    variants: dict[str, Variant]
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def export_data_env() -> None:
    """把本包的数据目录导出成 compose 文件引用的环境变量。

    `docker/compose-wp.yml` 的构建上下文用 `${CVEBENCH_CHALLENGE_DIR}/${CVE}/target`，
    `docker/compose-target.yml` 的 metadata 上下文默认是 `../metadata`（相对路径，
    这里显式给出绝对路径）。upstream 由调用方（inspect_evals）设置这两个变量，移植后
    数据就在包内，直接指向包内目录。
    """
    os.environ["CVEBENCH_CHALLENGE_DIR"] = str(CHALLENGES_DIR)
    os.environ["CVEBENCH_METADATA_DIR"] = str(METADATA_DIR)


def build_dataset(challenges: str | list[str] | None = None) -> Dataset:
    """构造挑战样本。

    `challenges` 为 None 时用 upstream 的 docker 挑战清单（排除已知不能跑的
    CVE，顺序也与 upstream 一致）；否则按传入的名字/目录构造。
    """
    export_data_env()
    roots = _challenge_roots(challenges)
    challenge_dirs = list(_find_challenge_dirs(roots))
    if not challenge_dirs:
        raise ValueError(f"No challenges found in {roots}")
    return MemoryDataset(
        samples=[
            sample
            for challenge_dir in challenge_dirs
            for sample in _samples(challenge_dir)
        ]
    )


def filter_by_variant(dataset: Dataset, variants: str | list[str]) -> Dataset:
    """只保留指定变体的样本（`zero_day` / `one_day`）。"""
    wanted = {variants} if isinstance(variants, str) else set(variants)
    return dataset.filter(
        lambda sample: sample.metadata is not None
        and sample.metadata["variant"] in wanted
    )


def _challenge_roots(challenges: str | list[str] | None) -> list[Path]:
    """把挑战名/路径解析成要扫描的目录列表。"""
    if challenges is None:
        return [CHALLENGES_DIR / name for name in DEFAULT_CHALLENGES]
    names = [challenges] if isinstance(challenges, str) else list(challenges)
    roots = []
    for name in names:
        path = Path(name)
        roots.append(path if path.is_absolute() else (CHALLENGES_DIR / path).resolve())
    return roots


def _find_challenge_dirs(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(f"Could not find directory: '{root}'")
        yield from (path.parent for path in sorted(root.rglob(CHALLENGE_INFO_FILENAME)))


def _samples(challenge_dir: Path) -> Iterator[Sample]:
    info = load_challenge_info(challenge_dir)
    files = _absolute_paths(info.files, challenge_dir)
    sandbox = _sandbox_spec(challenge_dir)
    for variant_name, variant in info.variants.items():
        yield Sample(
            id=f"{info.name}-{variant_name}",
            input=variant.prompt,
            files=files | _absolute_paths(variant.files, challenge_dir),
            metadata={
                "variant": variant_name,
                "challenge_metadata": info.metadata,
                "variant_metadata": variant.metadata,
            },
            sandbox=sandbox,
        )


def load_challenge_info(challenge_dir: Path) -> ChallengeInfo:
    """读一个挑战目录的 `challenge.yml`。"""
    yaml_path = challenge_dir / CHALLENGE_INFO_FILENAME
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not data:
        raise ValueError(f"Empty YAML file: '{yaml_path}'")
    return ChallengeInfo(
        name=data["name"],
        files=data.get("files") or {},
        variants={
            name: Variant(
                prompt=variant["prompt"],
                files=variant.get("files") or {},
                metadata=variant.get("metadata") or {},
            )
            for name, variant in data["variants"].items()
        },
        metadata=data.get("metadata") or {},
    )


def _absolute_paths(files: dict[str, str], base: Path) -> dict[str, str]:
    return {key: _absolute_path(value, base) for key, value in files.items()}


def _absolute_path(path_or_content: str, base: Path) -> str:
    if Path(path_or_content).is_absolute():
        return path_or_content
    path = base / path_or_content
    return str(path.resolve()) if path.exists() else path_or_content


def _sandbox_spec(challenge_dir: Path) -> SandboxEnvironmentSpec:
    sanitized = _sanitize_compose(challenge_dir)
    compose = sanitized or challenge_dir / COMPOSE_YAML_FILENAME
    return SandboxEnvironmentSpec("docker", str(compose))


def _resolve_relative_ref(value: str, base: Path) -> str:
    """把相对路径（或 `${VAR:-相对路径}` 的默认值）解析成基于 `base` 的绝对路径。

    `${VAR:-...}` 表达式本身保持不变，用户给的环境变量仍然优先，只改默认值。
    """
    ref = re.fullmatch(r"\$\{([^:}]+):-(.+?)\}\s*", value)
    if ref:
        var, default = ref.group(1), ref.group(2).strip()
        if default.startswith(("./", "../")) or default == ".":
            default = str((base / default).resolve())
        return f"${{{var}:-{default}}}"
    if value.startswith(("./", "../")) or value == ".":
        return str((base / value).resolve())
    return value


def _sanitize_compose(challenge_dir: Path) -> Path | None:
    """为带危险宿主挂载的挑战生成净化后的 compose 文件；否则返回 None。

    upstream 的部分挑战继承了危险挂载：

    * CVE-2024-2359 挂了宿主 docker socket，目标机拿到 RCE 就能控制宿主 docker
      daemon（该应用只在本机 subprocess 执行代码，用不到 docker）；
    * CVE-2024-32167/34340/36675/37849 把宿主机上的挑战文件以读写方式挂进容器，
      被攻陷的 db/apache/nginx 服务能篡改跨样本共享的文件。

    净化后的文件：丢掉 docker.sock 挂载、其余宿主绑定挂载改只读、把 include/
    extends/build context/volumes 里的相对引用解析成绝对路径（这样文件可以放在
    别处），并把挑战的 `.env` 一起复制过去（docker compose 自动加载同目录 `.env`）。

    按纯文本处理：cve-bench 的 compose 里有未加引号的 `${VAR:-default}`，PyYAML
    解析不了（只有 docker compose 的 go-yaml 支持），所以不做 YAML 往返。
    """
    compose_file = challenge_dir / COMPOSE_YAML_FILENAME
    if not compose_file.exists():
        return None

    changed = False
    out_lines: list[str] = []
    for line in compose_file.read_text(encoding="utf-8").splitlines():
        volume = re.match(r"^( *-\s+)((?:\.{1,2}/|/)[^:]+):(.+)$", line)
        if volume:
            indent, source, rest = volume.group(1), volume.group(2), volume.group(3)
            if any(mount in source for mount in DOCKER_SOCK_MOUNTS):
                changed = True
                continue
            if source.startswith(("./", "../")):
                source = str((challenge_dir / source).resolve())
            parts = rest.split(":")
            mode = parts[1] if len(parts) > 1 else None
            if mode is None or "ro" not in mode.split(","):
                rest = f"{parts[0]}:ro"
                changed = True
            out_lines.append(f"{indent}{source}:{rest}")
            continue
        out_lines.append(line)

    # 没有危险挂载的挑战继续用原地的那份 compose
    if not changed:
        return None

    # 删掉 docker.sock 条目后变空的 service 级 `volumes:` 键（compose 要求它是列表）
    index = 0
    while index < len(out_lines):
        header = re.match(r"^(\s*)volumes:\s*$", out_lines[index])
        if header and len(header.group(1)) >= SERVICE_VOLUME_INDENT:
            indent = len(header.group(1))
            cursor = index + 1
            has_item = False
            while cursor < len(out_lines):
                following = out_lines[cursor]
                if len(following) - len(following.lstrip()) < indent:
                    break
                if re.match(r"^\s+-\s+", following):
                    has_item = True
                    break
                cursor += 1
            if not has_item:
                del out_lines[index]
                continue
        index += 1

    # 把相对引用解析成绝对路径（`${VAR:-...}` 只改默认值），使文件可搬走
    for index, line in enumerate(out_lines):
        ref = re.match(
            r"^( *)(-\s+)?(path|project_directory|file|context):\s*(\S+)\s*$", line
        )
        if ref:
            indent, dash, key, value = ref.groups()
            out_lines[index] = (
                f"{indent}{dash or ''}{key}: {_resolve_relative_ref(value, challenge_dir)}"
            )

    safe_dir = SANITIZED_COMPOSE_DIR / challenge_dir.name
    safe_dir.mkdir(parents=True, exist_ok=True)
    out = safe_dir / COMPOSE_YAML_FILENAME
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    env_file = challenge_dir / ".env"
    if env_file.exists():
        shutil.copy2(env_file, safe_dir / ".env")
    return out
