"""Dataset module tests: config merge, task listing, compose generation."""

from pathlib import Path

import pytest

from concordia_evals.cybergym_e2e.dataset import (
    DEFAULT_BUILD_IMAGE,
    list_tasks,
    load_task_config,
    write_compose_file,
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A minimal fake scripts repo with two projects / three tasks."""
    repo = tmp_path / "repo"
    # project alpha: project-level build_image, two tasks
    proj = repo / "projects" / "alpha"
    (proj / "t1").mkdir(parents=True)
    (proj / "project.toml").write_text(
        'name = "alpha"\n'
        'repo_to_patch = "alpha"\n'
        'immutable_files = ["extras/"]\n'
        'build_image = "example/alpha:1"\n'
    )
    (proj / "t1" / "config.toml").write_text('task_id = "arvo:1"\n')
    (proj / "t2").mkdir()
    (proj / "t2" / "config.toml").write_text(
        'task_id = "arvo:2"\nbuild_image = "example/alpha-task:2"\n'
    )
    # project beta: no project build_image, one task with pre_patches
    proj_b = repo / "projects" / "beta"
    (proj_b / "v1").mkdir(parents=True)
    (proj_b / "project.toml").write_text('name = "beta"\nrepo_to_patch = "beta"\n')
    (proj_b / "v1" / "config.toml").write_text(
        'task_id = "arvo:3"\npre_patches = ["a.diff", "b.diff"]\n'
    )
    return repo


def test_list_tasks(fake_repo: Path):
    assert list_tasks(fake_repo) == ["alpha/t1", "alpha/t2", "beta/v1"]


def test_load_task_config_merges_project_and_task(fake_repo: Path):
    config = load_task_config("alpha/t1", fake_repo)
    assert config["name"] == "alpha"
    assert config["task_id"] == "arvo:1"
    assert config["repo_to_patch"] == "alpha"
    assert config["immutable_files"] == ["extras/"]
    assert config["build_image"] == "example/alpha:1"


def test_task_level_build_image_overrides_project(fake_repo: Path):
    config = load_task_config("alpha/t2", fake_repo)
    assert config["build_image"] == "example/alpha-task:2"


def test_missing_build_image_falls_back_to_default(fake_repo: Path):
    config = load_task_config("beta/v1", fake_repo)
    assert config.get("build_image", DEFAULT_BUILD_IMAGE) == DEFAULT_BUILD_IMAGE
    assert config["pre_patches"] == ["a.diff", "b.diff"]


def test_unknown_task_raises(fake_repo: Path):
    with pytest.raises(ValueError, match="Unknown task"):
        load_task_config("alpha/nope", fake_repo)


def test_write_compose_file(tmp_path: Path):
    compose = write_compose_file(
        "alpha_t1",
        "example/alpha:1",
        tmp_path / "data",
        tmp_path / "scripts",
        tmp_path / "compose",
    )
    content = Path(compose).read_text()
    assert "image: example/alpha:1" in content
    assert f"{tmp_path}/data:/task_data:ro" in content
    assert f"{tmp_path}/scripts:/task_scripts:ro" in content


def test_stratified_sample_diverse_and_reproducible():
    from concordia_evals.cybergym_e2e.dataset import stratified_sample

    tasks = [f"proj{p}/t{i}" for p in range(5) for i in range(10)]
    a = stratified_sample(tasks, 12, seed=7)
    b = stratified_sample(tasks, 12, seed=7)
    assert a == b  # reproducible
    assert len(a) == 12
    projects = {t.split("/")[0] for t in a}
    assert len(projects) == 5  # all projects represented
    # more tasks than pools: falls back to depth
    c = stratified_sample(["a/1", "b/1", "b/2", "b/3"], 3, seed=1)
    assert len(c) == 3
    # n >= len(tasks) returns everything
    d = stratified_sample(["a/1", "b/1"], 10, seed=1)
    assert sorted(d) == ["a/1", "b/1"]
