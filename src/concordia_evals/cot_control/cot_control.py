"""CoT-Control-QA Inspect evaluation."""

from __future__ import annotations

from typing import Literal

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from concordia_evals.cot_control._components import (
    DEFAULT_COT_CONTROL_JUDGE_MODEL,
    build_cot_control_qa_dataset,
    cot_control_generate,
    cot_controllability_score,
    score_cot_control_qa,
    task_accuracy,
)


@scorer(
    metrics=[
        cot_controllability_score(),
        task_accuracy(),
    ]
)
def cot_control_qa_scorer(
    judge_model: str | Model | None = DEFAULT_COT_CONTROL_JUDGE_MODEL,
    check_meta_discussion: bool = False,
) -> Scorer:
    """Score CoT constraint compliance and QA accuracy."""

    async def score(state: TaskState, target: Target) -> Score:
        return await score_cot_control_qa(
            state,
            target,
            judge_model=judge_model,
            check_meta_discussion=check_meta_discussion,
        )

    return score


@task
def cot_control(
    dataset: Literal["gpqa", "hle", "mmlu_pro", "all"] = "all",
    modes: str = "all_main",
    target_reasoning_effort: str | None = "default",
    adversarial_pressure: str = "na",
    non_reasoning: bool = False,
    monitor_aware: bool = False,
    eval_aware: bool = False,
    cot_sys_prompt: bool = False,
    elicit_brevity: bool = False,
    sample_limit_per_mode: int | None = None,
    judge_model: str | None = DEFAULT_COT_CONTROL_JUDGE_MODEL,
    check_meta_discussion: bool = False,
    limit: int | None = None,
) -> Task:
    """CoT-Control-QA (arXiv:2603.05706) Inspect evaluation.

    Args:
        dataset: Dataset subset (``"gpqa"``, ``"hle"``, ``"mmlu_pro"``, or ``"all"``).
        modes: Comma-separated modes, ``"all_main"`` (9 modes), or ``"all_with_baseline"``.
        target_reasoning_effort: Evaluated-model reasoning effort (or ``"default"``).
        adversarial_pressure: Upstream prompt-pressure variant.
        non_reasoning: Use the upstream non-reasoning prompt variant.
        monitor_aware: Append monitor-awareness instruction to prompts.
        eval_aware: Append eval-awareness instruction to prompts.
        cot_sys_prompt: Place CoT control constraints in the system prompt.
        elicit_brevity: Append brevity-elicitation instructions.
        sample_limit_per_mode: Optional cap per dataset and control mode (default ``None`` -> 10,836 total).
        judge_model: Fallback judge model for ``ignore_question`` and meta-discussion checks.
        check_meta_discussion: Also penalize discussing the constraint inside the CoT.
        limit: Optional total sample cap applied after dataset construction.

    Returns:
        Configured Inspect ``Task``.
    """
    return Task(
        config=GenerateConfig.model_validate(
            {
                "reasoning_effort": (
                    None if target_reasoning_effort == "default" else target_reasoning_effort
                )
            }
        ),
        dataset=build_cot_control_qa_dataset(
            dataset=dataset,
            modes=modes,
            adversarial_pressure=adversarial_pressure,
            non_reasoning=non_reasoning,
            monitor_aware=monitor_aware,
            eval_aware=eval_aware,
            cot_sys_prompt=cot_sys_prompt,
            elicit_brevity=elicit_brevity,
            sample_limit_per_mode=sample_limit_per_mode,
            limit=limit,
        ),
        solver=[cot_control_generate()],
        scorer=cot_control_qa_scorer(
            judge_model=judge_model,
            check_meta_discussion=check_meta_discussion,
        ),
    )
