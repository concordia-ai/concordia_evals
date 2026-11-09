"""Scoring-semantics tests: cumulative skip logic and success computation."""

from pathlib import Path
from typing import Any

from concordia_evals.cybergym_e2e import validation
from concordia_evals.cybergym_e2e.setup_solver import _sanitized_config_toml

BASE_KWARGS: dict[str, Any] = {
    "build_image": "example/img:1",
    "repo_to_patch": "alpha",
    "immutable_files": [],
    "pre_patches": [],
    "task_data_dir": Path("/data"),
    "script_dir": Path("/scripts"),
    "poc_path": Path("/tmp/poc.bin"),
    "patch_path": Path("/tmp/fix.patch"),
}


def _run_with_fake_stages(monkeypatch, mode: str, statuses: dict[int, str]):
    """Run run_full_validation with stage executions replaced by a fake."""
    executed: list[int] = []

    def fake_stage(*, stage: int, **kwargs: Any) -> str:
        executed.append(stage)
        return statuses[stage]

    monkeypatch.setattr(validation, "run_stage_validation", fake_stage)
    results = validation.run_full_validation(mode=mode, **BASE_KWARGS)
    return results, executed


def test_e2e_all_pass(monkeypatch):
    results, executed = _run_with_fake_stages(
        monkeypatch, "e2e", {1: "passed", 2: "passed", 3: "passed", 4: "passed"}
    )
    assert executed == [1, 2, 3, 4]
    assert results == {
        "stage1": "passed",
        "stage2": "passed",
        "stage3": "passed",
        "stage4": "passed",
    }


def test_e2e_stage1_failure_skips_rest(monkeypatch):
    results, executed = _run_with_fake_stages(
        monkeypatch, "e2e", {1: "failed", 2: "passed", 3: "passed", 4: "passed"}
    )
    assert executed == [1]
    assert results["stage1"] == "failed"
    assert results["stage2"] == "skipped"
    assert results["stage3"] == "skipped"
    assert results["stage4"] == "skipped"


def test_e2e_stage2_failure_skips_3_and_4(monkeypatch):
    results, executed = _run_with_fake_stages(
        monkeypatch, "e2e", {1: "passed", 2: "failed", 3: "passed", 4: "passed"}
    )
    assert executed == [1, 2]
    assert results["stage3"] == "skipped"
    assert results["stage4"] == "skipped"


def test_e2e_stage3_failure_skips_4(monkeypatch):
    results, executed = _run_with_fake_stages(
        monkeypatch, "e2e", {1: "passed", 2: "passed", 3: "failed", 4: "passed"}
    )
    assert executed == [1, 2, 3]
    assert results["stage4"] == "skipped"


def test_patch_only_runs_3_and_4(monkeypatch):
    results, executed = _run_with_fake_stages(monkeypatch, "patch-only", {3: "passed", 4: "passed"})
    assert executed == [3, 4]
    assert results["stage1"] is None
    assert results["stage2"] is None


def test_patch_only_stage3_failure_skips_4(monkeypatch):
    results, executed = _run_with_fake_stages(monkeypatch, "patch-only", {3: "failed", 4: "passed"})
    assert executed == [3]
    assert results["stage4"] == "skipped"


def test_sanitized_config_toml():
    toml = _sanitized_config_toml("curl", ["src/"])
    assert 'repo_to_patch = "curl"' in toml
    assert 'immutable_files = ["src/"]' in toml
    # nothing else may leak into the container-visible config
    assert "task_id" not in toml
    assert "build_image" not in toml
    assert _sanitized_config_toml("curl", []) == 'repo_to_patch = "curl"\n'
