"""Inspect AI entrypoint registry for all ``concordia_evals`` tasks and scorers."""

from __future__ import annotations

from concordia_evals.bixbench.bixbench import bixbench, bixbench_scorer
from concordia_evals.cot_control import cot_control, cot_control_qa_scorer
from concordia_evals.cot_immediacy import cot_immediacy, cot_immediacy_judge_scorer
from concordia_evals.cot_process import cot_process, openai_process_scorer
from concordia_evals.no_cot import (
    cot_arm_scorer,
    no_cot,
    no_cot_paired_scorer,
    no_cot_scorer,
)

__all__ = [
    "bixbench",
    "bixbench_scorer",
    "cot_arm_scorer",
    "cot_control",
    "cot_control_qa_scorer",
    "cot_immediacy",
    "cot_immediacy_judge_scorer",
    "cot_process",
    "no_cot",
    "no_cot_paired_scorer",
    "no_cot_scorer",
    "openai_process_scorer",
]
