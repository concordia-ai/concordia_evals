"""Strict No-CoT Science QA and paired CoT uplift Inspect evaluation."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import Model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from concordia_evals._cot_shared.science_qa import (
    DEFAULT_GRADER_MODEL,
    FrontierScienceFormat,
    ScienceDatasetName,
    build_science_qa_dataset,
    filter_dataset_with_recorded_log,
)
from concordia_evals.no_cot._components import (
    NoCotMode,
    _score_no_cot_state,
    cot_accuracy,
    cot_mean_reasoning_tokens,
    cot_uplift,
    no_cot_accuracy,
    no_cot_compliance_rate,
    no_cot_solver,
)


@scorer(metrics=[no_cot_accuracy(), no_cot_compliance_rate()])
def no_cot_scorer(
    grader_model: str | Model | None = DEFAULT_GRADER_MODEL,
    strict_no_cot: bool = True,
    max_answer_chars: int | None = None,
) -> Scorer:
    """Default no-CoT scorer (accuracy + compliance rate)."""

    async def score(state: TaskState, target: Target) -> Score:
        return await _score_no_cot_state(
            state,
            target,
            mode="no_cot",
            has_cot_reference=False,
            grader_model=grader_model,
            strict_no_cot=strict_no_cot,
            max_answer_chars=max_answer_chars,
        )

    return score


@scorer(
    metrics=[
        cot_accuracy(),
        no_cot_accuracy(),
        cot_uplift(),
        no_cot_compliance_rate(),
    ]
)
def no_cot_paired_scorer(
    grader_model: str | Model | None = DEFAULT_GRADER_MODEL,
    strict_no_cot: bool = True,
    max_answer_chars: int | None = None,
) -> Scorer:
    """Paired CoT vs. no-CoT scorer reporting ``cot_accuracy``, ``no_cot_accuracy``, ``cot_uplift``, and ``no_cot_compliance_rate``."""

    async def score(state: TaskState, target: Target) -> Score:
        return await _score_no_cot_state(
            state,
            target,
            mode="paired",
            has_cot_reference=True,
            grader_model=grader_model,
            strict_no_cot=strict_no_cot,
            max_answer_chars=max_answer_chars,
        )

    return score


@scorer(metrics=[cot_accuracy(), cot_mean_reasoning_tokens()])
def cot_arm_scorer(
    grader_model: str | Model | None = DEFAULT_GRADER_MODEL,
) -> Scorer:
    """CoT-arm-only scorer reporting ``cot_accuracy`` and ``cot_mean_reasoning_tokens``."""

    async def score(state: TaskState, target: Target) -> Score:
        return await _score_no_cot_state(
            state,
            target,
            mode="cot",
            has_cot_reference=False,
            grader_model=grader_model,
            strict_no_cot=True,
            max_answer_chars=None,
        )

    return score


def make_no_cot_scorer(
    *,
    mode: NoCotMode = "no_cot",
    has_cot_reference: bool = False,
    grader_model: str | Model | None = DEFAULT_GRADER_MODEL,
    strict_no_cot: bool = True,
    max_answer_chars: int | None = None,
) -> Scorer:
    """Select the appropriate scorer for the configured ``no_cot`` mode."""
    if mode == "cot":
        return cot_arm_scorer(grader_model=grader_model)
    if mode == "paired" or has_cot_reference:
        return no_cot_paired_scorer(
            grader_model=grader_model,
            strict_no_cot=strict_no_cot,
            max_answer_chars=max_answer_chars,
        )
    return no_cot_scorer(
        grader_model=grader_model,
        strict_no_cot=strict_no_cot,
        max_answer_chars=max_answer_chars,
    )


@task
def no_cot(
    dataset: ScienceDatasetName = "all",
    mode: NoCotMode = "no_cot",
    frontierscience_format: FrontierScienceFormat = "all",
    frontierscience_subjects: str | list[str] | None = None,
    chembench_task: str | list[str] | None = None,
    lab_bench_subset: str | list[str] | None = None,
    limit_per_dataset: int | None = None,
    grader_model: str | None = DEFAULT_GRADER_MODEL,
    no_cot_model: str | None = None,
    cot_log: str | None = None,
    filter_to_cot_log: bool = True,
    strict_no_cot: bool = True,
    max_answer_chars: int | None = None,
    limit: int | None = None,
) -> Task:
    """Evaluate direct no-CoT answers and optional CoT uplift across Science QA subsets.

    Args:
        dataset: Science QA subset selector (``"all"``, ``"chembench"``, ``"lab_bench"``, or ``"frontierscience"``).
        mode: Execution mode (``"no_cot"``, ``"cot"``, or ``"paired"``).
        frontierscience_format: FrontierScience subset filter (``"all"``, ``"olympic"``, or ``"research"``).
        frontierscience_subjects: Optional FrontierScience subject filter (e.g. ``"biology,chemistry"`` or ``["biology", "chemistry"]``).
        chembench_task: Optional ChemBench subset filter (``None`` or ``"all"`` loads all 9 subsets).
        lab_bench_subset: Optional LAB-Bench subset filter (``None`` or ``"all"`` loads all 8 subsets).
        limit_per_dataset: Optional cap per dataset before merging.
        grader_model: Optional fallback rubric grader model for FrontierScience answers.
        no_cot_model: Optional explicit non-thinking model identifier for the no-CoT arm.
        cot_log: Optional ``.eval`` file or directory providing recorded CoT generations/scores.
        filter_to_cot_log: When ``cot_log`` is given, restrict samples to matched log entries.
        strict_no_cot: Score non-compliant no-CoT outputs as ``0.0`` even if answer matches.
        max_answer_chars: Optional override for maximum allowed no-CoT character length.
        limit: Optional cap on total samples after filtering.

    Returns:
        Configured Inspect ``Task``.
    """
    ds = build_science_qa_dataset(
        dataset=dataset,
        frontierscience_format=frontierscience_format,
        frontierscience_subjects=frontierscience_subjects,
        chembench_task=chembench_task,
        lab_bench_subset=lab_bench_subset,
        limit_per_dataset=limit_per_dataset,
        prompt_mode="cot" if mode == "cot" else "no_cot",
    )

    has_cot_ref = bool(cot_log)
    deferred_cot_log: str | None = None
    if cot_log:
        deferred_cot_log = filter_dataset_with_recorded_log(
            ds,
            log_path=cot_log,
            param_name="cot_log",
            metadata_key="recorded_cot",
            filter_to_log=filter_to_cot_log,
            dataset=dataset,
            frontierscience_subjects=frontierscience_subjects,
            chembench_task=chembench_task,
            lab_bench_subset=lab_bench_subset,
            accuracy_mode="mean",
            include_problem_text=False,
        )

    if limit is not None:
        ds.samples = list(ds.samples)[:limit]

    effective_mode: NoCotMode = "paired" if (has_cot_ref and mode == "no_cot") else mode
    cot_log_only = bool(cot_log and not filter_to_cot_log)
    return Task(
        dataset=ds,
        solver=[
            no_cot_solver(
                mode=effective_mode,
                no_cot_model=no_cot_model,
                cot_log_only=cot_log_only,
                deferred_cot_log=deferred_cot_log,
                filter_to_cot_log=filter_to_cot_log,
                dataset=dataset,
                frontierscience_subjects=frontierscience_subjects,
                chembench_task=chembench_task,
                lab_bench_subset=lab_bench_subset,
            )
        ],
        scorer=make_no_cot_scorer(
            mode=effective_mode,
            has_cot_reference=has_cot_ref,
            grader_model=grader_model,
            strict_no_cot=strict_no_cot,
            max_answer_chars=max_answer_chars,
        ),
    )
