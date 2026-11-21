# Adapted from OpenAI monitorability-evals (https://github.com/openai/monitorability-evals),
# licensed under the Apache License, Version 2.0. See NOTICE for details.
"""Dataclass types for OpenAI Process-level Monitorability items and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalItem:
    """Single process monitorability evaluation item."""

    dataset: str
    archetype: str
    instance_id: str
    category: str
    prompt: str
    answer: str
    arm: int | None
    monitor_prompt_template_path: str
    monitor_prompt_template_text: str
    y_rule_ref: str
    z_rule_ref: str
    monitor_dataset_type: str | None
    model_prompt_source: str
    model_prompt_refs: list[str]
    model_prompt_templates_text: list[str]
    raw: dict[str, Any]


@dataclass
class ModelCallOutput:
    """Structured CoT and final-answer payload passed to monitor templates."""

    final_answer: str
    action_only: str
    cot_only: str
    all_msgs: str
