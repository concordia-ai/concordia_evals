"""CoT Immediacy (Legibility and Coverage) Inspect evaluation."""

from __future__ import annotations

from typing import Literal, cast

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from concordia_evals._cot_shared.science_qa import (
    FrontierScienceFormat,
    ScienceDatasetName,
    build_science_qa_dataset,
    filter_dataset_with_recorded_log,
)
from concordia_evals.cot_control._components import build_cot_control_qa_dataset
from concordia_evals.cot_immediacy._components import (
    coverage_pct,
    immediacy_generate_solver,
    legibility_pct,
    score_cot_immediacy,
)

ImmediacyDataset = Literal[
    "all",
    "chembench",
    "lab_bench",
    "frontierscience",
    "gpqa",
    "hle",
    "mmlu_pro",
]


@scorer(
    metrics=[
        legibility_pct(),
        coverage_pct(),
    ]
)
def cot_immediacy_judge_scorer(
    judge_model: str | Model | None = None,
    correct_only: bool = False,
) -> Scorer:
    """Autorater Judge from DeepMind arXiv:2510.23966 Appendix C."""

    async def score(state: TaskState, target: Target) -> Score:
        return await score_cot_immediacy(
            state,
            target,
            judge_model=judge_model,
            correct_only=correct_only,
        )

    return score


@task
def cot_immediacy(
    dataset: ImmediacyDataset = "all",
    target_reasoning_effort: str | None = "default",
    judge_model: str | None = None,
    correct_only: bool = False,
    source_log: str | None = None,
    filter_to_source_log: bool = True,
    frontierscience_format: FrontierScienceFormat = "all",
    frontierscience_subjects: str | list[str] | None = None,
    chembench_task: str | list[str] | None = None,
    lab_bench_subset: str | list[str] | None = None,
    limit_per_dataset: int | None = None,
    limit: int | None = None,
) -> Task:
    """DeepMind CoT Legibility and Coverage Judge benchmark (arXiv:2510.23966).

    Args:
        dataset: Dataset subset (``"all"``, ``"chembench"``, ``"lab_bench"``, ``"frontierscience"``, or legacy ``"gpqa"``, ``"hle"``, ``"mmlu_pro"``).
        target_reasoning_effort: Evaluated-model reasoning effort (or ``"default"``).
        judge_model: Fallback judge model (defaults to ``--model-role judge=...``, ``INSPECT_EVALS_JUDGE_MODEL``, ``openai/qwen3.8-27b`` when ``OPENAI_BASE_URL`` is set, or ``openai/gpt-4o``).
        correct_only: Only judge answers matching the reference answer.
        source_log: Optional ``.eval`` log file or directory to replay CoT traces without actor calls.
        filter_to_source_log: Keep only samples matched in ``source_log`` when ``source_log`` is provided.
        frontierscience_format: FrontierScience subset filter (``"all"``, ``"olympic"``, or ``"research"``).
        frontierscience_subjects: Optional FrontierScience subject filter (e.g. ``"biology,chemistry"`` or ``["biology", "chemistry"]``).
        chembench_task: Optional ChemBench subset filter (``None`` or ``"all"`` loads all 9 subsets).
        lab_bench_subset: Optional LAB-Bench subset filter (``None`` or ``"all"`` loads all 8 subsets).
        limit_per_dataset: Optional cap per dataset before merging.
        limit: Optional cap on total samples after filtering.

    Returns:
        Configured Inspect ``Task``.
    """
    if dataset in ("gpqa", "hle", "mmlu_pro"):
        ds = build_cot_control_qa_dataset(
            dataset=dataset,
            modes="baseline",
            sample_limit_per_mode=limit_per_dataset,
        )
    else:
        ds = build_science_qa_dataset(
            dataset=cast(ScienceDatasetName, dataset),
            frontierscience_format=frontierscience_format,
            frontierscience_subjects=frontierscience_subjects,
            chembench_task=chembench_task,
            lab_bench_subset=lab_bench_subset,
            limit_per_dataset=limit_per_dataset,
            prompt_mode="cot",
        )

    deferred_source_log: str | None = None
    if source_log:
        sci_ds: ScienceDatasetName = (
            cast(ScienceDatasetName, dataset)
            if dataset in ("all", "chembench", "lab_bench", "frontierscience")
            else "all"
        )
        deferred_source_log = filter_dataset_with_recorded_log(
            ds,
            log_path=source_log,
            param_name="source_log",
            metadata_key="recorded_source",
            filter_to_log=filter_to_source_log,
            dataset=sci_ds,
            frontierscience_subjects=frontierscience_subjects,
            chembench_task=chembench_task,
            lab_bench_subset=lab_bench_subset,
            accuracy_mode="first_epoch",
            include_problem_text=True,
        )

    if limit is not None:
        ds.samples = list(ds.samples)[:limit]

    return Task(
        config=GenerateConfig.model_validate(
            {
                "reasoning_effort": (
                    None if target_reasoning_effort == "default" else target_reasoning_effort
                )
            }
        ),
        dataset=ds,
        solver=[
            immediacy_generate_solver(
                deferred_source_log=deferred_source_log,
                filter_to_source_log=filter_to_source_log,
                dataset=dataset,
                frontierscience_subjects=frontierscience_subjects,
                chembench_task=chembench_task,
                lab_bench_subset=lab_bench_subset,
            )
        ],
        scorer=cot_immediacy_judge_scorer(
            judge_model=judge_model,
            correct_only=correct_only,
        ),
    )
