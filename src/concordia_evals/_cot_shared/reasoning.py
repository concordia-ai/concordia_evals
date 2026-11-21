"""Utilities for extracting unredacted reasoning blocks and resolving judge models."""

from __future__ import annotations

import os
import re
from typing import Any, Final
from urllib.parse import urlparse

from inspect_ai.model import (
    ChatMessageAssistant,
    ContentReasoning,
    GenerateConfig,
    Model,
    get_model,
)

DEFAULT_PORTABLE_JUDGE_MODEL: Final[str] = "openai/gpt-4o"
_WORD_TO_TOKEN_FACTOR: Final[float] = 1.3
_NON_TEMP_JUDGE_TAGS: Final[tuple[str, ...]] = (
    "/o1",
    "/o3",
    "/o4",
    "/gpt-5",
    "/gpt-6",
    "deepseek-r1",
    "deepseek-v4",
    "kimi-k3",
    "qwen",
)


def full_reasoning(message: ChatMessageAssistant) -> str:
    """Return concatenated plaintext reasoning blocks or raise if redacted/summary-only.

    Any block with ``block.redacted=True`` (even when accompanied by a summary, as used
    by summary-only providers such as Anthropic, OpenAI Responses, and OpenRouter) or
    any summary-only block (``not block.redacted`` with empty ``reasoning`` and non-empty
    ``summary``) raises ``ValueError``.

    Args:
        message: Assistant message emitted by Inspect AI.

    Returns:
        Joined plaintext reasoning blocks, or an empty string if none exist.

    Raises:
        ValueError: If any reasoning block is redacted or only contains a summary.
    """
    blocks = (
        [block for block in message.content if isinstance(block, ContentReasoning)]
        if isinstance(message.content, list)
        else []
    )
    parts: list[str] = []
    for block in blocks:
        if block.redacted or (not block.reasoning and block.summary):
            raise ValueError(
                "Full reasoning is unavailable: the provider returned only a summary "
                "or redacted reasoning. Use a provider that exposes the complete trace."
            )
        if block.reasoning:
            parts.append(block.reasoning)
    return "\n".join(parts).strip()


def _strip_and_extract_tag(text: str, tag: str) -> tuple[str, str]:
    """Extract closed and unclosed ``<tag>...</tag>`` contents and return ``(extracted, remainder)``."""
    if not text or not re.search(rf"<{re.escape(tag)}\b", text, flags=re.IGNORECASE):
        return "", text
    extracted: list[str] = []
    closed_re = re.compile(rf"(?is)<{re.escape(tag)}(?:\s[^>]*)?>(.*?)</{re.escape(tag)}>")

    def _collect(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if body:
            extracted.append(body)
        return ""

    remainder = closed_re.sub(_collect, text)
    unclosed_re = re.compile(rf"(?is)<{re.escape(tag)}(?:\s[^>]*)?>(.*)$")
    unclosed_match = unclosed_re.search(remainder)
    if unclosed_match:
        body = unclosed_match.group(1).strip()
        if body:
            extracted.append(body)
        remainder = remainder[: unclosed_match.start()]
    return "\n".join(extracted).strip(), remainder.strip()


def extract_cot_and_visible_text(
    message: Any,
    usage: Any = None,
    *,
    allow_output_reasoning_tag: bool = False,
) -> tuple[str, str, int]:
    """Extract ``(cot_trace, visible_text, cot_tokens)`` from a model output message."""
    if message is None:
        return "", "", 0

    cot = full_reasoning(message)
    text = (getattr(message, "text", None) or "").strip()

    if allow_output_reasoning_tag:
        out_cot, text = _strip_and_extract_tag(text, "output_reasoning")
        if not cot and out_cot:
            cot = out_cot

    think_cot, text = _strip_and_extract_tag(text, "think")
    if not cot and think_cot:
        cot = think_cot

    rtok = getattr(usage, "reasoning_tokens", None) if usage is not None else None
    if isinstance(rtok, int) and rtok > 0:
        cot_tokens = rtok
    else:
        cot_tokens = int(round(len(cot.split()) * _WORD_TO_TOKEN_FACTOR)) if cot else 0
    return cot, text, cot_tokens


def _is_custom_openai_gateway(base_url: str | None = None) -> bool:
    """Return True when ``OPENAI_BASE_URL`` points to a non-official OpenAI-compatible gateway."""
    raw = (base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL", "")).strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    host = (parsed.hostname or urlparse(f"https://{raw}").hostname or "").lower()
    if not host or host == "api.openai.com" or host.endswith(".openai.azure.com"):
        return False
    return True


_OPENAI_RESPONSES_PREFIXES: tuple[str, ...] = (
    "gpt-5",
    "gpt-6",
    "o1",
    "o3",
    "o4",
    "codex",
)


def gateway_openai_model_args(model_spec: str) -> dict[str, Any]:
    """Return ``responses_api=False`` for non-OpenAI aliases via custom ``OPENAI_BASE_URL``."""
    if not _is_custom_openai_gateway() or not model_spec.startswith("openai/"):
        return {}
    short = model_spec.removeprefix("openai/").lower()
    if short.startswith(_OPENAI_RESPONSES_PREFIXES) or "deep-research" in short:
        return {}
    return {"responses_api": False}


def judge_generate_config(
    model_or_name: str | Model | None,
) -> GenerateConfig:
    """Build judge ``GenerateConfig`` while avoiding ``temperature=0.0`` on reasoning models."""
    name_lower = str(model_or_name or "").lower()
    if name_lower.endswith("-nt") or any(tag in name_lower for tag in _NON_TEMP_JUDGE_TAGS):
        return GenerateConfig()
    return GenerateConfig(temperature=0.0)


def resolve_judge_model(
    explicit_model: str | Model | None = None,
    *,
    role: str = "judge",
    gateway_default: str | None = None,
    config: GenerateConfig | None = None,
) -> Model:
    """Resolve a judge/monitor model with portable fallbacks.

    Precedence:
    1. Explicit ``Model`` instance -> returned directly.
    2. ``--model-role <role>=...`` bound in Inspect's active role context.
    3. Explicit ``str`` model name (e.g. ``-T judge_model=...``) as default.
    4. ``INSPECT_EVALS_JUDGE_MODEL`` environment variable if non-empty.
    5. ``gateway_default`` when ``OPENAI_BASE_URL`` targets a custom gateway, else ``openai/gpt-4o``.
    """
    cfg = config or GenerateConfig()
    if isinstance(explicit_model, Model):
        return explicit_model

    if isinstance(explicit_model, str) and explicit_model.strip():
        fallback = explicit_model.strip()
        if fallback.startswith("openai-api/openai/") and not _is_custom_openai_gateway():
            fallback = "openai/" + fallback.removeprefix("openai-api/openai/")
    else:
        env_judge = os.environ.get("INSPECT_EVALS_JUDGE_MODEL", "").strip()
        if env_judge:
            fallback = env_judge
        elif gateway_default and _is_custom_openai_gateway():
            fallback = gateway_default
        else:
            fallback = DEFAULT_PORTABLE_JUDGE_MODEL
    return get_model(role=role, default=fallback, config=cfg, **gateway_openai_model_args(fallback))
