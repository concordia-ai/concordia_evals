"""Inspect AI entrypoint registry for all ``concordia_evals`` tasks and scorers."""

from __future__ import annotations

from concordia_evals.bixbench.bixbench import bixbench, bixbench_scorer
from concordia_evals.bountybench import bountybench
from concordia_evals.cot_control import cot_control, cot_control_qa_scorer
from concordia_evals.cot_immediacy import cot_immediacy, cot_immediacy_judge_scorer
from concordia_evals.cot_process import cot_process, openai_process_scorer
from concordia_evals.cve_bench import cve_bench
from concordia_evals.cybergym_e2e.cybergym_e2e import cybergym_e2e
from concordia_evals.cybergym_e2e.scorer import (
    cybergym_e2e_scorer,
    cybergym_e2e_scorer_both,
)
from concordia_evals.exploitbench import exploitbench
from concordia_evals.exploitgym import exploitgym
from concordia_evals.no_cot import (
    cot_arm_scorer,
    no_cot,
    no_cot_paired_scorer,
    no_cot_scorer,
)
from concordia_evals.web_exploit_bench import web_exploit_bench

__all__ = [
    "bixbench",
    "bixbench_scorer",
    "bountybench",
    "cot_arm_scorer",
    "cot_control",
    "cot_control_qa_scorer",
    "cot_immediacy",
    "cot_immediacy_judge_scorer",
    "cot_process",
    "cve_bench",
    "cybergym_e2e",
    "cybergym_e2e_scorer",
    "cybergym_e2e_scorer_both",
    "exploitbench",
    "exploitgym",
    "no_cot",
    "no_cot_paired_scorer",
    "no_cot_scorer",
    "openai_process_scorer",
    "web_exploit_bench",
]
