"""CVEBench 移植后的行为测试。"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import AgentState, agent, as_solver
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import includes
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import ExecResult, store

from concordia_evals.cve_bench.agent import (
    CONTINUE_MESSAGE,
    _continue_or_stop,
    default_agent,
)
from concordia_evals.cve_bench.cve_bench import STORE_KEY_ATTACK_SUCCESSFUL
from concordia_evals.cve_bench.dataset import (
    CHALLENGES_DIR,
    DATA_DIR,
    DEFAULT_CHALLENGES,
    UNSUPPORTED_CHALLENGES,
    _sanitize_compose,
    build_dataset,
    filter_by_variant,
    load_challenge_info,
)

cve_bench_module = importlib.import_module("concordia_evals.cve_bench.cve_bench")
refusal_module = importlib.import_module("concordia_evals.cve_bench.refusal")


# --------------------------------------------------------------------------- #
# 数据：挑战目录 → 样本
# --------------------------------------------------------------------------- #


def test_default_dataset_is_the_upstream_docker_list() -> None:
    dataset = filter_by_variant(build_dataset(), "zero_day")

    assert len(DEFAULT_CHALLENGES) == 30
    assert [sample.id for sample in dataset] == [
        f"{name}-zero_day" for name in DEFAULT_CHALLENGES
    ]
    sample = dataset[0]
    assert sample.sandbox is not None
    assert sample.sandbox.type == "docker"
    assert Path(str(sample.sandbox.config)).name == "compose.yml"
    metadata = sample.metadata or {}
    assert metadata["variant"] == "zero_day"
    assert metadata["challenge_metadata"]["application_url"]
    assert "one_day" in {s.metadata["variant"] for s in build_dataset()}  # type: ignore[index]


def test_the_evaluator_avoids_new_style_annotations() -> None:
    """判定服务跑在目标镜像里，有的挑战只有 Python 3.7/3.8/3.9。

    `int | None` 这类 PEP 604 注解在 3.10 之前是运行期 TypeError（函数注解在 def 时
    求值，类/模块级注解同样求值），会让整个 `/done` 服务起不来、目标容器直接 unhealthy。
    """
    import ast

    def annotations(node: ast.AST) -> list[ast.expr]:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = [*node.args.args, *node.args.kwonlyargs]
            found = [arg.annotation for arg in args if arg.annotation]
            return [*found, node.returns] if node.returns else found
        if isinstance(node, ast.Module | ast.ClassDef):
            return [
                statement.annotation
                for statement in node.body
                if isinstance(statement, ast.AnnAssign) and statement.annotation
            ]
        return []

    offenders = []
    for path in sorted((DATA_DIR / "evaluations").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deferred = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        if deferred:
            continue
        for node in ast.walk(tree):
            for annotation in annotations(node):
                if any(
                    isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr)
                    for child in ast.walk(annotation)
                ):
                    offenders.append(f"{path.name}: {ast.unparse(annotation)}")

    assert not offenders, offenders


def test_the_default_list_excludes_the_unsupported_challenges() -> None:
    assert not set(DEFAULT_CHALLENGES) & set(UNSUPPORTED_CHALLENGES)
    # 排除的挑战仍然在包里，显式点名时可以跑
    assert all((CHALLENGES_DIR / name).is_dir() for name in UNSUPPORTED_CHALLENGES)


def test_variant_filter_keeps_the_requested_prompts() -> None:
    dataset = build_dataset(["CVE-2024-2359"])

    assert [sample.id for sample in dataset] == [
        "CVE-2024-2359-zero_day",
        "CVE-2024-2359-one_day",
    ]
    zero_day, one_day = dataset
    assert zero_day.input != one_day.input
    assert "known to have the following vulnerability" in one_day.input


def test_challenge_info_reads_variants_and_metadata() -> None:
    info = load_challenge_info(CHALLENGES_DIR / "CVE-2024-2359")

    assert info.name == "CVE-2024-2359"
    assert set(info.variants) == {"zero_day", "one_day"}
    assert info.metadata["application_url"] == "target:9090"


def test_unknown_challenge_is_an_error() -> None:
    with pytest.raises(NotADirectoryError):
        build_dataset(["CVE-1999-0001"])


# --------------------------------------------------------------------------- #
# compose 净化：丢掉 docker.sock、宿主挂载改只读、相对引用变绝对
# --------------------------------------------------------------------------- #


def test_sanitize_drops_the_docker_socket_mount() -> None:
    sanitized = _sanitize_compose(CHALLENGES_DIR / "CVE-2024-2359")

    assert sanitized is not None
    text = sanitized.read_text(encoding="utf-8")
    assert "/var/run/docker.sock" not in text
    # 相对引用被解析成绝对路径，文件搬走后仍然可用
    assert f"path: ${{CVEBENCH_SANDBOX_COMPOSE_FILE:-{CHALLENGES_DIR.parent}/sandboxes" in text
    assert (sanitized.parent / ".env").exists()


def test_sanitize_forces_host_mounts_read_only() -> None:
    sanitized = _sanitize_compose(CHALLENGES_DIR / "CVE-2024-34340")

    assert sanitized is not None
    text = sanitized.read_text(encoding="utf-8")
    assert "/etc/php/8.1/apache2/php.ini:ro" in text


def test_sanitize_leaves_a_safe_compose_alone() -> None:
    assert _sanitize_compose(CHALLENGES_DIR / "CVE-2023-37999") is None


# --------------------------------------------------------------------------- #
# 判定：只在 agent 停止后跑一次，结果只进 store 与 metadata
# --------------------------------------------------------------------------- #


@pytest.fixture
def outcome_checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """替换依赖沙箱/模型的判定，记录调用。"""
    calls: list[str] = []

    async def fake_exploit_check(state: TaskState) -> bool:
        calls.append("exploit")
        store().set(STORE_KEY_ATTACK_SUCCESSFUL, True)
        return True

    async def fake_judge(reply: str) -> bool:
        calls.append(f"judge:{reply}")
        return True

    monkeypatch.setattr(
        cve_bench_module, "check_target_server_for_exploit_success", fake_exploit_check
    )
    monkeypatch.setattr(refusal_module, "judge_refusal", fake_judge)
    return calls


@agent
def _refusing_agent() -> Any:
    """转录里只有一句拒答的 agent。"""

    async def execute(state: AgentState) -> AgentState:
        state.messages = [
            *state.messages,
            ChatMessageAssistant(content="I can't help with that."),
        ]
        return state

    return execute


@solver
def _looping_agent() -> Solver:
    """一直生成到被消息上限砍断的 agent。"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for _ in range(5):
            state = await generate(state)
        return state

    return solve


def test_custom_agent_is_judged_once_after_it_stops(outcome_checks: list[str]) -> None:
    task = cve_bench_module.cve_bench(
        challenges="CVE-2024-2359", agent=as_solver(_refusing_agent())
    )
    task.dataset = MemoryDataset([Sample(input="exploit the target", target="flag")])
    task.scorer = [includes()]

    [log] = eval(task, model="mockllm/model", display="none")
    sample = log.samples[0]  # type: ignore[index]

    # 先补判首轮拒答，再判目标；两次都只发生一次
    assert outcome_checks == ["judge:I can't help with that.", "exploit"]
    assert sample.store[STORE_KEY_ATTACK_SUCCESSFUL] is True  # type: ignore[index]
    assert sample.store[refusal_module.FIRST_TURN_REFUSAL_STORE_KEY] is True  # type: ignore[index]


def test_both_agents_see_the_system_message(outcome_checks: list[str]) -> None:
    task = cve_bench_module.cve_bench(
        challenges="CVE-2024-2359", agent=as_solver(_refusing_agent())
    )
    task.dataset = MemoryDataset([Sample(input="exploit the target", target="flag")])
    task.scorer = [includes()]

    [log] = eval(task, model="mockllm/model", display="none")
    first = log.samples[0].messages[0]  # type: ignore[index]

    assert first.role == "system"
    assert "This is a job interview." in str(first.text)


def test_message_limit_still_reports_the_outcome(outcome_checks: list[str]) -> None:
    task = cve_bench_module.cve_bench(
        challenges="CVE-2024-2359", agent=_looping_agent(), max_messages=4
    )
    task.dataset = MemoryDataset([Sample(input="exploit the target", target="flag")])
    task.scorer = [includes()]

    [log] = eval(task, model="mockllm/model", display="none")
    sample = log.samples[0]  # type: ignore[index]

    assert sample.limit is not None
    assert sample.limit.type == "message"
    # 被砍断也要判定：最后一轮打下来的成果不能丢
    assert sample.store[STORE_KEY_ATTACK_SUCCESSFUL] is True  # type: ignore[index]


def test_agent_never_sees_the_outcome(outcome_checks: list[str]) -> None:
    task = cve_bench_module.cve_bench(
        challenges="CVE-2024-2359", agent=as_solver(_refusing_agent())
    )
    task.dataset = MemoryDataset([Sample(input="exploit the target", target="flag")])
    task.scorer = [includes()]

    [log] = eval(task, model="mockllm/model", display="none")
    texts = [str(message.text) for message in log.samples[0].messages]  # type: ignore[index]

    assert not any("Exploit successful" in text for text in texts)
    assert not any("attack-was-successful" in text for text in texts)


async def test_a_dead_target_container_falls_back_to_the_outside_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标容器被压死时 /done 不可用，应改走外部探活而不是记成运行错误。"""
    from inspect_ai.util import SandboxUnavailableError

    class _Sandbox:
        def __init__(self, dead: bool, returncode: int) -> None:
            self.dead = dead
            self.returncode = returncode

        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            if self.dead:
                raise SandboxUnavailableError("The sandbox is not running")
            return ExecResult(
                success=self.returncode == 0,
                returncode=self.returncode,
                stdout="",
                stderr="",
            )

    target = _Sandbox(dead=True, returncode=0)
    agent_sandbox = _Sandbox(dead=False, returncode=0)  # 探活全程不可达
    monkeypatch.setattr(
        cve_bench_module,
        "sandbox",
        lambda name=None, **kwargs: target if name == "target" else agent_sandbox,
    )

    state = TaskState(
        model="mockllm/model", sample_id="x", epoch=1, input="go", messages=[]
    )
    state.metadata["challenge_metadata"] = {"application_url": "target:9090"}

    assert await cve_bench_module.check_target_server_for_exploit_success(state) is True
    assert state.metadata["exploit_check"]["source"] == "outside_probe"
    assert state.metadata["exploit_check"]["status"] is True


async def test_judge_from_outside_probes_the_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Sandbox:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            return ExecResult(
                success=self.returncode == 0,
                returncode=self.returncode,
                stdout="",
                stderr="",
            )

    state = TaskState(model="mockllm/model", sample_id="x", epoch=1, input="go", messages=[])
    state.metadata["challenge_metadata"] = {"application_url": "target:9090"}

    monkeypatch.setattr(cve_bench_module, "sandbox", lambda *a, **k: _Sandbox(0))
    assert (await cve_bench_module.judge_from_outside(state))[0] is True
    assert (await cve_bench_module.judge_from_outside(state))[1] == "outside_probe"

    monkeypatch.setattr(cve_bench_module, "sandbox", lambda *a, **k: _Sandbox(1))
    assert (await cve_bench_module.judge_from_outside(state))[0] is False


# --------------------------------------------------------------------------- #
# 默认 agent：finish 结束、无工具调用时继续提示、首轮拒答结束
# --------------------------------------------------------------------------- #


def _agent_state(text: str, tool_calls: list[Any] | None = None) -> AgentState:
    from inspect_ai.model import ChatMessageUser, ModelOutput

    calls = tool_calls or []
    state = AgentState(messages=[ChatMessageUser(content="go")])
    state.messages.append(ChatMessageAssistant(content=text, tool_calls=calls))
    state.output = ModelOutput.from_content("mockllm/model", content=text)
    state.output.message.tool_calls = calls  # type: ignore[attr-defined]
    return state


async def test_finish_ends_the_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_ai.tool import ToolCall

    monkeypatch.setattr(
        "concordia_evals.cve_bench.agent.detect_first_turn_refusal",
        _never_refused,
    )
    state = _agent_state(
        "done", [ToolCall(id="1", function="finish", arguments={})]
    )

    assert await _continue_or_stop(state) is False


async def test_a_turn_without_tool_calls_is_nudged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "concordia_evals.cve_bench.agent.detect_first_turn_refusal",
        _never_refused,
    )

    assert await _continue_or_stop(_agent_state("let me think")) == CONTINUE_MESSAGE


async def test_a_first_turn_refusal_ends_the_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refused(state: Any) -> bool:
        return True

    monkeypatch.setattr(
        "concordia_evals.cve_bench.agent.detect_first_turn_refusal", refused
    )
    state = _agent_state("I can't help with that.")

    assert await _continue_or_stop(state) is False
    # 与旧实现一致：先追加继续提示，再停止
    assert state.messages[-1].role == "user"
    assert state.messages[-1].text == CONTINUE_MESSAGE


async def _never_refused(state: Any) -> bool:
    return False


def test_default_agent_ends_on_the_finish_tool() -> None:
    """模型调用 finish 后循环结束（不需要沙箱：finish 不碰容器）。"""
    from inspect_ai.model import ModelOutput, get_model

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call("mockllm/model", "finish", {}),
        ],
    )
    task = Task(
        dataset=MemoryDataset([Sample(input="exploit the target", target="flag")]),
        solver=as_solver(default_agent()),
        scorer=[includes()],
    )

    [log] = eval(task, model=model, display="none")
    messages = log.samples[0].messages  # type: ignore[index]

    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    assert messages[-1].text == "Attempt ended."
