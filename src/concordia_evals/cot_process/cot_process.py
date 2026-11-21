"""OpenAI Process-level Monitorability Inspect evaluation."""

from __future__ import annotations

from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from concordia_evals._cot_shared.assets import ensure_asset
from concordia_evals.cot_process._components import (
    DEFAULT_COT_PROCESS_MONITOR_MODEL,
    load_process_items,
    parse_process_scopes,
    process_answer_accuracy,
    process_generate_solver,
    process_gmean2_cot,
    process_metrics_for_scopes,
    render_model_prompt,
    score_openai_process,
)


@scorer(
    metrics=[
        process_gmean2_cot(),
        process_answer_accuracy(),
    ]
)
def openai_process_scorer(
    monitor_model: str | Model | None = DEFAULT_COT_PROCESS_MONITOR_MODEL,
    scopes: str = "cot_only",
) -> Scorer:
    """Score final-answer correctness and process theorem faithfulness."""
    parse_process_scopes(scopes)

    async def score(state: TaskState, target: Target) -> Score:
        return await score_openai_process(
            state,
            target,
            monitor_model=monitor_model,
            scopes=scopes,
        )

    return score


@task
def cot_process(
    dataset: Literal["all", "gpqa_physics", "aime_math"] = "all",
    target_reasoning_effort: str | None = "default",
    monitor_model: str | None = DEFAULT_COT_PROCESS_MONITOR_MODEL,
    scopes: str = "cot_only",
    limit: int | None = None,
) -> Task:
    """OpenAI Process-level Monitorability benchmark (arXiv:2512.18311).

    Args:
        dataset: Subset selector (``"all"``, ``"gpqa_physics"``, or ``"aime_math"``).
        target_reasoning_effort: Evaluated-model reasoning effort (or ``"default"``).
        monitor_model: Fallback monitor model for theorem-path faithfulness and fallback grading.
        scopes: Comma-separated monitor visibility scopes (default ``"cot_only"``).
        limit: Optional cap on total samples.

    Returns:
        Configured Inspect ``Task``.
    """
    parse_process_scopes(scopes)
    root = ensure_asset("openai-monitorability")
    proc_items = load_process_items(dataset=dataset)
    if limit is not None:
        proc_items = proc_items[:limit]

    samples: list[Sample] = []
    for item in proc_items:
        rendered_prompt = render_model_prompt(item, root=root)
        safe_ds = item.dataset.replace("/", "_").replace(".jsonl", "")
        samples.append(
            Sample(
                id=f"{safe_ds}__{item.instance_id}",
                input=rendered_prompt,
                target=item.answer,
                metadata={
                    "dataset": item.dataset,
                    "instance_id": item.instance_id,
                    "eval_item": item.__dict__,
                },
            )
        )
    return Task(
        config=GenerateConfig.model_validate(
            {
                "reasoning_effort": (
                    None if target_reasoning_effort == "default" else target_reasoning_effort
                )
            }
        ),
        dataset=MemoryDataset(samples),
        solver=[process_generate_solver()],
        scorer=openai_process_scorer(monitor_model=monitor_model, scopes=scopes),
        metrics=process_metrics_for_scopes(scopes),
    )
