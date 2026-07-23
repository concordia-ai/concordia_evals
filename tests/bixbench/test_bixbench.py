"""Tests for BixBench Inspect implementation."""

import string
from pathlib import Path

import pytest

from concordia_evals.bixbench.bixbench import (
    CELL_TIMEOUT_SECONDS,
    MAX_OUTPUT_CHARS_PER_CELL,
    MAX_TOTAL_DIFF_CHARS,
    RUNNER_SCRIPT_TEMPLATE,
    _get_compose_file,
    _prepare_capsule_data,
    bixbench,
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
