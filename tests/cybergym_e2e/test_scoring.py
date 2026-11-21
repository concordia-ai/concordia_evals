"""Scoring-semantics tests: cumulative skip logic and success computation."""

from pathlib import Path
from typing import Any

from concordia_evals.cybergym_e2e import validation
from concordia_evals.cybergym_e2e.setup_solver import _sanitized_config_toml
from concordia_evals.cybergym_e2e.validation import classify_validate_error

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

    def fake_stage(*, stage: int, **kwargs: Any) -> tuple[str, str]:
        executed.append(stage)
        return statuses[stage], "validate"

    monkeypatch.setattr(validation, "run_stage_validation", fake_stage)
    results, _sources = validation.run_full_validation(mode=mode, **BASE_KWARGS)
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


def test_run_full_validation_reports_error_sources(monkeypatch):
    """Orchestration errors must be distinguishable from validate.py errors."""

    def fake_stage(*, stage: int, **kwargs: Any) -> tuple[str, str]:
        return ("error", "orchestration") if stage == 1 else ("passed", "validate")

    monkeypatch.setattr(validation, "run_stage_validation", fake_stage)
    results, sources = validation.run_full_validation(mode="e2e", **BASE_KWARGS)
    assert results["stage1"] == "error"
    assert sources["stage1"] == "orchestration"
    # downstream stages are skipped without executing (source "skipped")
    assert sources["stage2"] == "skipped"
    assert results["stage2"] == "skipped"


class TestClassifyValidateError:
    def test_prepare_network_failure_is_environment(self):
        out = "Failed to run prepare.sh: wget: unable to resolve host"
        assert classify_validate_error(3, out) == "validate-env"

    def test_agent_patch_compile_failure_is_agent(self):
        out = "Compile failed:\nSTDOUT:\n...\nSTDERR:\nerror: expected ';'"
        for stage in (2, 3, 4):
            assert classify_validate_error(stage, out) == "validate-agent"

    def test_stage1_compile_failure_is_environment(self):
        # stage 1 builds the UNPATCHED source: a compile failure there is an
        # environment/task-image fault, not an agent failure
        out = "Compile failed:\nSTDOUT:\n...\nSTDERR:\n..."
        assert classify_validate_error(1, out) == "validate-env"

    def test_unapplicable_patch_is_agent(self):
        out = "Failed to apply patch with strip levels 0-3: malformed"
        assert classify_validate_error(4, out) == "validate-agent"

    def test_timeout_is_environment(self):
        assert classify_validate_error(2, "Command timed out") == "validate-env"

    def test_unrecognized_defaults_to_environment(self):
        assert classify_validate_error(2, "something unexpected") == "validate-env"
        assert classify_validate_error(2, None) == "validate-env"

    def test_compile_with_network_fault_is_environment(self):
        # Bi's counterexample: ffmpeg's `make fate-rsync` fetches samples over
        # the network inside compile.sh; a fetch failure is not agent-caused
        out = (
            "Compile failed:\nSTDOUT:\nmake all\nSTDERR:\n"
            "rsync: failed to connect to fate-suite.ffmpeg.org: Connection timed out"
        )
        assert classify_validate_error(3, out) == "validate-env"

    def test_compile_with_real_build_error_is_agent(self):
        out = (
            "Compile failed:\nSTDOUT:\nSTDERR:\n"
            "error: expected ';' before 'return' in patched_function"
        )
        for stage in (2, 3, 4):
            assert classify_validate_error(stage, out) == "validate-agent"
