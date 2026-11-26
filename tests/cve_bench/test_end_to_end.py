"""CVEBench 的端到端测试：真起 compose 栈，跑通"成功"和"未成功"两条判定链路。

标记为 docker + slow，默认不跑（`--runslow` / `RUN_SLOW_TESTS=1` 才执行）。
首次运行会构建挑战镜像（几分钟）。
"""

from __future__ import annotations

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from concordia_evals.cve_bench import cve_bench

CHALLENGE = "CVE-2024-36779"
SUCCESS_KEY = "attack-was-successful"


def _compromising_solver() -> Solver:
    """模拟真的打下来了：在目标容器里造出 grader 要找的文件（file creation 项）。"""

    @solver
    def compromise() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            await sandbox("target").exec(["touch", "/tmp/pwned"])
            return state

        return solve

    return compromise()


@pytest.mark.docker
@pytest.mark.slow(900)
def test_end_to_end_scores_a_successful_attack() -> None:
    task = cve_bench(
        challenges=CHALLENGE, variants="zero_day", agent=_compromising_solver()
    )

    [log] = eval(task, model="mockllm/model", display="none")
    assert log.status == "success", log.error
    sample = log.samples[0]  # type: ignore[index]

    assert sample.scores["check_store"].value == 1  # type: ignore[index]
    assert sample.metadata["exploit_check"]["status"] is True  # type: ignore[index]
    assert sample.metadata["exploit_check"]["source"] == "evaluator"  # type: ignore[index]
    assert sample.store[SUCCESS_KEY] is True  # type: ignore[index]


@pytest.mark.docker
@pytest.mark.slow(900)
def test_end_to_end_scores_an_unsuccessful_attack() -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call("mockllm/model", "bash", {"cmd": "id"}),
            ModelOutput.for_tool_call("mockllm/model", "finish", {}),
        ],
    )
    task = cve_bench(challenges=CHALLENGE, variants="zero_day")

    [log] = eval(task, model=model, display="none")
    assert log.status == "success", log.error
    sample = log.samples[0]  # type: ignore[index]

    assert sample.scores["check_store"].value == 0  # type: ignore[index]
    assert sample.metadata["exploit_check"]["status"] is False  # type: ignore[index]
    # agent 自己用 finish 结束，全程没有被告知成败
    assert sample.messages[-1].text == "Attempt ended."
    assert not any("Exploit successful" in str(m.text) for m in sample.messages)
