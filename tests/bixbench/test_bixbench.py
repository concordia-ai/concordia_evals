"""Tests for BixBench Inspect implementation."""

import string
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState, generate

from concordia_evals.bixbench.bixbench import (
    CELL_TIMEOUT_SECONDS,
    MAX_OUTPUT_CHARS_PER_CELL,
    MAX_TOTAL_DIFF_CHARS,
    RUNNER_SCRIPT_TEMPLATE,
    _get_compose_file,
    _prepare_capsule_data,
    bixbench,
    bixbench_scorer,
    extract_answer,
    grade_range,
    parse_range,
)


@pytest.mark.dataset_download
def test_prepare_capsule_data(tmp_path: Path) -> None:
    """Test that capsule data is downloaded and prepared correctly."""
    question = {
        "question_id": "bix-1-q1",
        "data_folder": "CapsuleFolder-33b801bb-9b47-4a0a-9314-05325c82fde7.zip",
    }
    capsule_dir = _prepare_capsule_data(question, data_dir=tmp_path)

    assert capsule_dir.exists()
    assert any(capsule_dir.iterdir())

    # Data folder contents should be moved to root
    assert (capsule_dir / "Issy_ASXL1_blood_featureCounts_GeneTable_final.txt").exists()

    # Notebook folder should be removed
    assert not any(p for p in capsule_dir.rglob("*") if p.is_dir() and "Notebook" in p.name)

    # Original .ipynb files from the zip should be removed
    assert not any(capsule_dir.glob("*.ipynb"))


def test_compose_file_isolates_workspace(tmp_path: Path) -> None:
    """Test that the generated compose file mounts the capsule read-only.

    It also copies it into a fresh in-container /workspace on startup.
    """
    capsule_dir = tmp_path / "CapsuleFolder-xxx"
    capsule_dir.mkdir()
    (capsule_dir / "data.csv").write_text("data")

    compose_path = _get_compose_file(
        question_id="bix-test-q1",
        capsule_dir=capsule_dir,
        docker_image_name="bixbench-inspect-env:latest",
    )

    compose_text = Path(compose_path).read_text()

    # Capsule is mounted read-only as a template, not as /workspace.
    assert "/workspace_template:ro" in compose_text
    assert "/workspace:rw" not in compose_text

    # Container startup clears /workspace and copies the template in fresh.
    assert "rm -rf /workspace" in compose_text
    assert "cp -a /workspace_template/. /workspace/" in compose_text

    # The image's default entrypoint is overridden so the workspace-setup
    # command actually runs.
    assert "entrypoint:" in compose_text
    assert "/bin/sh" in compose_text
    assert "-c" in compose_text

    # Working directory is the fresh in-container /workspace.
    assert "working_dir: /workspace" in compose_text


def test_runner_script_compiles() -> None:
    """Test that the persistent-kernel runner template substitutes and compiles."""
    runner_script = string.Template(RUNNER_SCRIPT_TEMPLATE).substitute(
        MAX_PER_CELL=str(MAX_OUTPUT_CHARS_PER_CELL),
        MAX_TOTAL=str(MAX_TOTAL_DIFF_CHARS),
        PER_CELL_TIMEOUT=str(CELL_TIMEOUT_SECONDS),
    )
    compile(runner_script, "<runner>", "exec")


@pytest.mark.dataset_download
def test_bixbench_task_creation() -> None:
    """Test that the bixbench task can be created with a limit."""
    task = bixbench(limit=2)
    assert len(task.dataset) == 2

    sample = task.dataset[0]
    assert sample.id is not None
    assert sample.target is not None
    assert sample.metadata is not None
    assert "question" in sample.metadata
    assert "ideal" in sample.metadata
    assert sample.sandbox is not None
    assert sample.sandbox.type == "docker"


def test_parse_range_accepts_dataset_formats() -> None:
    """Reference ranges are parsed in the formats the dataset actually uses."""
    assert parse_range("(0.74,0.77)") == (0.74, 0.77)
    assert parse_range("(0.6 , 0.7)") == (0.6, 0.7)
    assert parse_range("(24,26)") == (24.0, 26.0)


def test_parse_range_rejects_non_ranges() -> None:
    """Answers that are not a bound pair are not treated as ranges."""
    assert parse_range("0.05") is None
    assert parse_range("35%") is None
    assert parse_range("(1.0,2.0,3.0)") is None


def test_extract_answer_prefers_answer_tag() -> None:
    """The graded value comes from the answer tag, not the surrounding prose."""
    assert extract_answer("Some reasoning.\n<answer> 0.755 </answer>") == "0.755"
    assert extract_answer("<answer>1</answer> and <answer>2</answer>") == "2"
    assert extract_answer("no tags here") == "no tags here"


def test_grade_range_uses_inclusive_bounds() -> None:
    """Values on or inside the bounds are correct; outside is incorrect."""
    assert grade_range("(0.74,0.77)", "0.74") is True
    assert grade_range("(0.74,0.77)", "0.77") is True
    assert grade_range("(0.74,0.77)", "0.755") is True
    assert grade_range("(0.74,0.77)", "0.9") is False


def test_grade_range_parses_numbers_out_of_prose() -> None:
    """An answer that states its value in prose is still graded numerically."""
    assert grade_range("(0.74,0.77)", "roughly 0.75") is True
    assert grade_range("(0.74,0.77)", "about 1,200 counts") is False


def test_grade_range_defers_to_judge_without_a_number() -> None:
    """A non-numeric answer returns None so the LLM judge decides."""
    assert grade_range("(0.74,0.77)", "no idea") is None
    assert grade_range("(0.74,0.77)", "") is None


@pytest.mark.anyio
async def test_scorer_grades_range_without_calling_judge() -> None:
    """range_verifier samples are graded deterministically, with no judge call."""
    scorer_fn = bixbench_scorer(judge_model="mockllm/model")
    state = TaskState(
        model="mockllm/model",
        sample_id="bix-1-q1",
        epoch=1,
        input="q",
        messages=[],
        output=ModelOutput.from_content("mockllm/model", "<answer>0.755</answer>"),
        metadata={"question": "What fraction?", "eval_mode": "range_verifier"},
    )

    score = await scorer_fn(state, Target("(0.74,0.77)"))
    assert score.value["accuracy"] == 1
    assert "inside the bounds" in (score.explanation or "")

    state.output = ModelOutput.from_content("mockllm/model", "<answer>0.9</answer>")
    score = await scorer_fn(state, Target("(0.74,0.77)"))
    assert score.value["accuracy"] == 0


@pytest.mark.anyio
async def test_scorer_str_verifier_matches_normalized_string() -> None:
    """str_verifier samples compare normalized strings before judging."""
    scorer_fn = bixbench_scorer(judge_model="mockllm/model")
    state = TaskState(
        model="mockllm/model",
        sample_id="bix-1-q1",
        epoch=1,
        input="q",
        messages=[],
        output=ModelOutput.from_content("mockllm/model", "<answer>1.9E-05</answer>"),
        metadata={"question": "What p-value?", "eval_mode": "str_verifier"},
    )

    score = await scorer_fn(state, Target("1.9E-05"))
    assert score.value["accuracy"] == 1


@pytest.mark.dataset_download
def test_custom_solver_keeps_environment_setup() -> None:
    """A custom solver must not drop environment initialization."""
    task = bixbench(solver=generate(), limit=1)
    assert task.setup is not None
    assert task.dataset[0].metadata["eval_mode"] in {
        "llm_verifier",
        "str_verifier",
        "range_verifier",
    }
