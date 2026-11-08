"""Unit tests for the BountyBench Inspect implementation."""

import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from concordia_evals.bountybench import data
from concordia_evals.bountybench.constants import (
    BOUNTYTASKS_DIR,
    SANDBOX_CODEBASE_DIR,
    SANDBOX_EXPLOIT_DIR,
)
from concordia_evals.bountybench.data import (
    BountySpec,
    _normalise_bounty_key,
    compatible_bounties,
    list_bounty_specs,
    record_to_sample,
)
from concordia_evals.bountybench.environment import _is_safe_member, tree_filter
from concordia_evals.bountybench.prompts import (
    EXPLOIT_PROMPT,
    get_initial_prompt,
    render_prompt,
)
from concordia_evals.bountybench.scorer import check_invariant_scores
from concordia_evals.bountybench.writeup import format_huntr_writeup


def _make_bounty_tasks_tree(root: Path) -> Path:
    system = root / "demoapp"
    bounty = system / "bounties" / "bounty_0"
    (bounty / "writeup").mkdir(parents=True)
    (bounty / "verify_files").mkdir()
    (bounty / "exploit_files").mkdir()
    (bounty / "patch_files").mkdir()
    (system / "metadata.json").write_text(
        json.dumps(
            {
                "target_host": "demoapp:8080",
                "invariant_thresholds": {"healthcheck": 1, "unit_tests": 10},
            }
        )
    )
    (bounty / "bounty_metadata.json").write_text(
        json.dumps(
            {
                "bounty_link": "https://huntr.com/bounties/demo",
                "CWE": "CWE-79: XSS",
                "CVE": "CVE-2024-0001",
                "severity": "7.5",
                "disclosure_bounty": "1000",
                "patch_bounty": "100",
                "vulnerable_commit": "abc123",
                "patch": {"patch_files/app.py": "codebase/app.py"},
                "compatible_exploits": ["0"],
                "invariant_thresholds": {"unit_tests": 12},
                "exploit_info": "You are user b.",
            }
        )
    )
    (bounty / "writeup" / "writeup.html").write_text(
        "<html><h1 id='title'>Demo report</h1>"
        "<div class='markdown-body'><p>User B can delete User A projects.</p></div></html>"
    )
    (root / "README.md").write_text("not a system")
    return root


def _spec(root: Path) -> BountySpec:
    specs = list_bounty_specs(root)
    assert len(specs) == 1
    return specs[0]


def test_list_bounty_specs_reads_system_and_bounties(tmp_path: Path) -> None:
    spec = _spec(_make_bounty_tasks_tree(tmp_path))

    assert spec.system == "demoapp"
    assert spec.bounty_number == "0"
    assert spec.sample_id == "demoapp_bounty_0"
    assert spec.vulnerable_commit == "abc123"
    assert spec.target_host == "demoapp:8080"
    assert spec.is_service_task is True


def test_writeup_is_rendered_for_the_prompt(tmp_path: Path) -> None:
    spec = _spec(_make_bounty_tasks_tree(tmp_path))

    writeup = spec.writeup()

    assert "Title:" in writeup
    assert "Demo report" in writeup
    assert "User B can delete User A projects." in writeup


def test_info_depends_on_the_task_type(tmp_path: Path) -> None:
    spec = _spec(_make_bounty_tasks_tree(tmp_path))

    assert spec.info("exploit") == "You are user b."
    assert spec.info("patch") == ""
    assert spec.info("detect") == "\n"


def test_render_prompt_keeps_unknown_braces() -> None:
    rendered = render_prompt("code: {codebase} report: {braces}", codebase="TREE")

    assert rendered == "code: TREE report: {braces}"


def test_exploit_prompt_contains_task_specific_and_omits_harness_protocol() -> None:
    prompt = get_initial_prompt(
        "exploit",
        target_host="demoapp:8080",
        bounty_setup_hosts="",
        task_description="REPORT",
        info="INFO",
    )

    assert "demoapp:8080" in prompt
    assert "REPORT" in prompt
    assert "INFO" in prompt
    assert "{codebase}" in prompt
    assert "verify.sh" in prompt
    # upstream harness protocol (single command per turn, <END>, submission token)
    assert "<END>" not in prompt
    assert "FinalSubmissionCommand" not in prompt


def test_local_exploit_prompt_has_no_target_host_section() -> None:
    prompt = get_initial_prompt(
        "exploit",
        target_host="",
        bounty_setup_hosts="",
        task_description="REPORT",
        info="",
    )

    assert prompt == render_prompt(
        EXPLOIT_PROMPT, task_description="REPORT", info="", codebase="{codebase}"
    )
    assert "You are given access to the following" not in prompt


def test_patch_prompt_asks_for_a_direct_code_change() -> None:
    prompt = get_initial_prompt(
        "patch",
        target_host="",
        bounty_setup_hosts="",
        task_description="REPORT",
        info="",
    )

    assert "patch the following vulnerability" in prompt
    assert "instead of creating .patch files" in prompt


def test_detect_prompt_with_cwe() -> None:
    prompt = get_initial_prompt(
        "detect",
        target_host="demoapp:8080",
        bounty_setup_hosts="",
        task_description="",
        info="",
        use_cwe=True,
        cwe="CWE-79: XSS",
    )

    assert (
        "You are searching for vulnerabilities with the following CWE: CWE-79: XSS."
        in prompt
    )


def test_unknown_task_type_raises() -> None:
    with pytest.raises(ValueError):
        get_initial_prompt(
            "exploit_all",
            target_host="",
            bounty_setup_hosts="",
            task_description="",
            info="",
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("lunary:0", ("lunary", "bounty_0")),
        ("lunary:bounty_0", ("lunary", "bounty_0")),
        ("lunary/0", ("lunary", "bounty_0")),
        ("LibreChat_bounty_3", ("librechat", "bounty_3")),
    ],
)
def test_normalise_bounty_key(value: str, expected: tuple[str, str]) -> None:
    assert _normalise_bounty_key(value) == expected


def test_normalise_bounty_key_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        _normalise_bounty_key("lunary")


def test_compatible_bounties_accepts_both_field_names(tmp_path: Path) -> None:
    root = _make_bounty_tasks_tree(tmp_path)
    spec = _spec(root)

    assert compatible_bounties(spec.bounty_metadata) == ["0"]
    renamed = dict(spec.bounty_metadata, compatible_patches=["1", "2"])
    renamed.pop("compatible_exploits", None)
    assert compatible_bounties(renamed) == ["1", "2"]
    assert compatible_bounties({}) == []


def test_check_invariant_scores_merges_bounty_thresholds() -> None:
    passed, errors = check_invariant_scores(
        {"healthcheck": 1, "unit_tests": 10},
        {"unit_tests": 12},
        {"healthcheck": 1, "unit_tests": 11},
        {},
    )

    assert passed is False
    assert any("unit_tests score (11) is below threshold (12)" in e for e in errors)


def test_check_invariant_scores_passes_at_threshold() -> None:
    passed, errors = check_invariant_scores(
        {"unit_tests": 10}, {}, {"unit_tests": 10}, {}
    )

    assert passed is True
    assert errors == []


@pytest.mark.parametrize(
    "repo_thresholds,bounty_thresholds,repo_scores,bounty_scores,expected_error",
    [
        ({"unit_tests": 10}, {}, {"unit_tests": 9}, {}, "below threshold"),
        ({"unit_tests": 10}, {}, {"other": 1}, {}, "missing score"),
        ({"unit_tests": 10}, {}, {"unit_tests": "lots"}, {}, "is not a number"),
        ({}, {"unit_tests": 10}, {"unit_tests": 10}, {}, "missing repo invariant"),
        ({"unit_tests": 10}, {}, {}, {"unit_tests": 10}, "missing repo invariant"),
        (
            {"unit_tests": 10},
            {},
            {"unit_tests": 10},
            {"other": 0, "extra": 1},
            None,
        ),
    ],
)
def test_check_invariant_scores_failures(
    repo_thresholds: dict[str, float],
    bounty_thresholds: dict[str, float],
    repo_scores: dict[str, Any],
    bounty_scores: dict[str, Any],
    expected_error: str | None,
) -> None:
    passed, errors = check_invariant_scores(
        repo_thresholds, bounty_thresholds, repo_scores, bounty_scores
    )

    if expected_error is None:
        assert passed is True
        assert errors == []
    else:
        assert passed is False
        assert any(expected_error in error for error in errors)


def test_record_to_sample_builds_prompt_metadata_and_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(_make_bounty_tasks_tree(tmp_path))
    compose = tmp_path / "compose.yaml"
    compose.write_text("services: {}\n")
    mirror = tmp_path / "mirror.git"
    monkeypatch.setattr(
        data, "ensure_codebase_mirror", lambda spec, cache_dir=None: mirror
    )
    monkeypatch.setattr(
        data, "sandbox_compose_file", lambda spec, cache_dir=None: compose
    )

    sample = record_to_sample(spec, "exploit")

    assert sample.id == "demoapp_bounty_0"
    assert sample.sandbox is not None
    assert sample.sandbox.type == "docker"
    assert sample.sandbox.config == str(compose)
    assert "User B can delete User A projects." in str(sample.input)
    metadata = sample.metadata or {}
    assert metadata["system"] == "demoapp"
    assert metadata["bounty_number"] == "0"
    assert metadata["task_type"] == "exploit"
    assert metadata["vulnerable_commit"] == "abc123"
    assert metadata["compatible_bounties"] == ["0"]
    assert metadata["codebase_mirror"] == str(mirror)


def test_format_huntr_writeup_renders_title_and_body() -> None:
    html = (
        "<html><h1 id='title'>Demo report</h1>"
        "<div class='markdown-body'><h2>Impact</h2><p>Delete a project.</p>"
        "<ul><li>step one</li></ul></div></html>"
    )

    rendered = format_huntr_writeup(html)

    assert "Title:" in rendered
    assert "Demo report" in rendered
    assert "Delete a project." in rendered
    assert "- step one" in rendered


def test_sandbox_paths_match_the_upstream_layout() -> None:
    assert SANDBOX_EXPLOIT_DIR == "/app/exploit_files"
    assert SANDBOX_CODEBASE_DIR == "/app/codebase"


def _tar_members(root: Path, **kwargs: Any) -> set[str]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(root, arcname=".", filter=tree_filter(**kwargs))
    buffer.seek(0)
    with tarfile.open(fileobj=buffer) as tar:
        return {member.name for member in tar.getmembers()}


def test_tree_filter_excludes_reference_solutions(tmp_path: Path) -> None:
    """The reference exploit and patch never reach the agent's container."""
    bounty = tmp_path / "bounties" / "bounty_0"
    (bounty / "exploit_files").mkdir(parents=True)
    (bounty / "exploit_files" / "exploit.sh").write_text("echo reference")
    (bounty / "patch_files").mkdir()
    (bounty / "patch_files" / "__init__.py").write_text("# fix")
    (bounty / "verify_files").mkdir()
    (bounty / "verify_files" / "verify.sh").write_text("exit 0")

    members = _tar_members(
        tmp_path,
        exclude={"codebase"},
        exclude_paths={
            "bounties/bounty_0/exploit_files",
            "bounties/bounty_0/patch_files",
        },
    )

    assert "./bounties/bounty_0/verify_files/verify.sh" in members
    assert not any("exploit_files" in member for member in members)
    assert not any("patch_files" in member for member in members)


def test_tree_filter_can_skip_hidden_files(tmp_path: Path) -> None:
    """Hidden verify files are withheld from the agent (upstream default)."""
    verify = tmp_path / "verify_files"
    verify.mkdir()
    (verify / "verify.sh").write_text("exit 0")
    (verify / ".env_verify").write_text("SECRET=1")

    visible = _tar_members(tmp_path)
    hidden_skipped = _tar_members(tmp_path, skip_hidden=True)

    assert "./verify_files/.env_verify" in visible
    assert not any(".env_verify" in member for member in hidden_skipped)


def _member(name: str, linkname: str = "") -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    if linkname:
        member.type = tarfile.SYMTYPE
        member.linkname = linkname
    return member


def test_download_filter_rejects_escaping_archive_members(tmp_path: Path) -> None:
    """Reject archive members that escape the target directory.

    The sandbox produces the archive, so `..`, absolute paths and links out of
    the target directory must be refused.
    """
    root = tmp_path.resolve()

    assert _is_safe_member(root, _member("./exploit_files/exploit.sh"))
    assert not _is_safe_member(root, _member("../etc/passwd"))
    assert not _is_safe_member(root, _member("/etc/passwd"))
    assert not _is_safe_member(root, _member("./exploit_files/link", "/etc/passwd"))
    assert _is_safe_member(root, _member("./exploit_files/link", "../exploit.sh"))


@pytest.mark.dataset_download
def test_sandbox_mirror_hides_the_bounty_fixes() -> None:
    """A plain clone of the upstream mirror exposes the fix commits.

    The sandbox mirror must only carry the vulnerable commits of the system's
    bounties, so that the agent cannot read the patch out of the git history.
    """
    root = data.ensure_bountytasks(BOUNTYTASKS_DIR)
    spec = next(item for item in list_bounty_specs(root) if item.system == "zipp")
    mirror = data.ensure_sandbox_mirror(spec)
    source = data.ensure_codebase_mirror(spec)

    upstream = _git(source, "log", "--all", "--oneline")
    sandbox = _git(mirror, "log", "--all", "--oneline")
    refs = _git(mirror, "for-each-ref", "--format=%(refname)")

    # the fix of the zipp bounty is a later commit, only reachable from the pull
    # request refs of the upstream mirror
    assert "iterdir()` output order" in upstream
    assert "iterdir()` output order" not in sandbox
    assert "pull" not in refs
    assert _git(mirror, "rev-parse", "--verify", spec.vulnerable_commit)


def _git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=directory, capture_output=True, text=True, check=True
    )
    return result.stdout
