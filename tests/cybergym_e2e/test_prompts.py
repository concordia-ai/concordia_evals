"""Prompt generation tests: content fidelity with upstream run_agent.py."""

from concordia_evals.cybergym_e2e.prompts import get_prompt


def test_e2e_iterative_prompt_mentions_all_outputs():
    prompt = get_prompt("e2e", "curl", "iterative")
    assert "PROJECT DIRECTORY: /src/curl" in prompt
    assert "/output/poc.bin" in prompt
    assert "/output/fix.patch" in prompt
    assert "LLVMFuzzerTestOneInput" in prompt
    assert "/scripts/.venv/bin/python /scripts/validate.py" in prompt
    assert "Do NOT use network access" in prompt


def test_patch_only_iterative_prompt_mentions_inputs():
    prompt = get_prompt("patch-only", "curl", "iterative")
    assert "/src/crash.log" in prompt
    assert "/src/poc.bin" in prompt
    assert "/output/fix.patch" in prompt
    # patch-only prompt must not ask for a PoC
    assert "/output/poc.bin" not in prompt


def test_no_test_prompt_forbids_execution():
    e2e = get_prompt("e2e", "mruby", "no-test")
    patch = get_prompt("patch-only", "mruby", "no-test")
    assert "Do NOT compile or run any code" in e2e
    assert "Do NOT compile or run any tests" in patch
    assert "/src/mruby" in e2e


def test_feedback_is_appended():
    prompt = get_prompt("e2e", "curl", "iterative", feedback="PREVIOUS FAILED")
    assert prompt.endswith("Please fix the issues above and generate updated files.")
    assert "PREVIOUS FAILED" in prompt
