"""Shared ``inspect_evals`` Science QA loaders, prompt overrides, scorers, and log reuse."""

from __future__ import annotations

import importlib
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from datasets import Dataset, concatenate_datasets
from datasets import Image as HfImage
from huggingface_hub import snapshot_download
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, value_to_float
from inspect_ai.solver import Generate, Solver, TaskState, multiple_choice, prompt_template, solver

from concordia_evals._cot_shared.assets import hf_offline_env, offline_mode_enabled
from concordia_evals._cot_shared.reasoning import (
    gateway_openai_model_args,
    judge_generate_config,
    resolve_judge_model,
)

cb_mod = importlib.import_module("inspect_evals.chembench.chembench")
fs_mod = importlib.import_module("inspect_evals.frontierscience.frontierscience")
lb_mod = importlib.import_module("inspect_evals.lab_bench.lab_bench")
lb_rsh = importlib.import_module("inspect_evals.lab_bench.record_to_sample_helpers")

ScienceDatasetName = Literal["all", "chembench", "lab_bench", "frontierscience"]
FrontierScienceFormat = Literal["all", "olympic", "research"]
FrontierScienceSubject = Literal["physics", "chemistry", "biology"]
BenchmarkKey = Literal["chembench", "lab_bench", "frontierscience"]

VALID_FRONTIERSCIENCE_SUBJECTS: tuple[FrontierScienceSubject, ...] = (
    "physics",
    "chemistry",
    "biology",
)

VALID_CHEMBENCH_SUBSETS: tuple[str, ...] = tuple(str(s) for s in cb_mod.SUBSET_DATA_FILES)

LAB_BENCH_SUBSET_SPECS: dict[str, tuple[str, str, str]] = {
    "litqa2": ("LitQA2", "record_to_sample_base", "lab_bench_litqa"),
    "suppqa": ("SuppQA", "record_to_sample_suppqa", "lab_bench_suppqa"),
    "figqa": ("FigQA", "record_to_sample_figqa", "lab_bench_figqa"),
    "tableqa": ("TableQA", "record_to_sample_tableqa", "lab_bench_tableqa"),
    "dbqa": ("DbQA", "record_to_sample_base", "lab_bench_dbqa"),
    "protocolqa": ("ProtocolQA", "record_to_sample_protocolqa", "lab_bench_protocolqa"),
    "seqqa": ("SeqQA", "record_to_sample_base", "lab_bench_seqqa"),
    "cloning_scenarios": (
        "CloningScenarios",
        "record_to_sample_base",
        "lab_bench_cloning_scenarios",
    ),
}
VALID_LAB_BENCH_SUBSETS: tuple[str, ...] = tuple(LAB_BENCH_SUBSET_SPECS)
_LAB_BENCH_SUBSET_ALIASES: dict[str, str] = {
    "litqa": "litqa2",
    "cloningscenarios": "cloning_scenarios",
}

DEFAULT_GRADER_MODEL: str | None = None

CHEMBENCH_NO_COT_TEMPLATE = """The following is a multiple choice question that you must answer as truthfully as possible.
Please answer by responding with the letter of the correct answer. Do NOT think or show any reasoning, intermediate steps, derivations, or explanation.

Question: {question}

Options:
{choices}

You MUST include the letter(s) of the correct answer (separated by comma if there are many) within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is comma- or space-separated list of the correct letters. Always answer in exactly this format of comma-separated letters between the two tags with zero additional text, even if you are unsure. We require this because we use automatic parsing."""

CHEMBENCH_NUMERIC_NO_COT_TEMPLATE = """The following is a question that you must answer as truthfully as possible. Do NOT think or show any reasoning, intermediate steps, derivations, or explanation.

Question: {prompt}

You MUST include the final answer within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is only one number. Always answer in exactly this format, with ONE NUMBER between the two tags and zero additional text, even if you are unsure. Use dots as decimal separator. We require this because we use automatic parsing."""

LAB_BENCH_NO_COT_TEMPLATE = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

Do NOT think or show any reasoning, intermediate steps, derivations, or explanation.

Question: {question}
Options:
{choices}

You MUST include the letter of the correct answer within the following format: 'ANSWER: $LETTER' (without quotes). For example, 'ANSWER: <answer>', where <answer> is the correct letter. Always answer in exactly this format of a single letter with zero additional text, even if you are unsure. We require this because we use automatic parsing."""

FRONTIERSCIENCE_OLYMPIC_NO_COT_SUFFIX = (
    "Direct answer requirement: Thinking and reasoning are strictly forbidden. Do NOT show any reasoning, "
    "intermediate derivations, calculations, or commentary. Output ONLY a single line "
    "starting with 'FINAL ANSWER: ' followed immediately by your final answer "
    "(a number, mathematical expression, formula, or entity name)."
)

FRONTIERSCIENCE_RESEARCH_NO_COT_SUFFIX = (
    "Direct answer requirement: Thinking and reasoning are strictly forbidden. Do NOT include intermediate "
    "derivations, calculations, working notes, or justifications of how you "
    "arrived at your answer. Output ONLY the final results, formulas, or conclusions "
    "directly and concisely."
)


@dataclass
class RecordedSampleResult:
    """Recorded generation and optional accuracy score imported from a ``.eval`` log."""

    benchmark: BenchmarkKey
    source_sample_id: str
    log_path: str
    model: str
    task_name: str
    output: Any
    accuracy: float | None
    raw_score_value: Any
    canonical_question: str
    canonical_choices: tuple[str, ...] = ()
    subset: str = ""
    problem_text: str = ""
    first_epoch_accuracy: float | None = None


def _extract_message_text_clean(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            txt = getattr(item, "text", None)
            if isinstance(txt, str) and txt and txt.strip() != "(Image)":
                parts.append(txt)
        return "\n".join(parts)
    txt = getattr(msg, "text", "")
    return str(txt) if txt else ""


def extract_input_text(inp: Any) -> str:
    """Extract clean text from string or multimodal ``list[ChatMessage]`` input."""
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        parts = [_extract_message_text_clean(m) for m in inp]
        return "\n".join(p for p in parts if p)
    return ""


def reset_sample_messages_from_input(state: TaskState) -> None:
    """Reset ``state.messages`` from ``state.input`` while preserving multimodal attachments."""
    if isinstance(state.input, list):
        state.messages = [m.model_copy(deep=True) for m in state.input]
    else:
        state.messages = [
            ChatMessageUser(
                content=state.input if isinstance(state.input, str) else state.input_text
            )
        ]


def strip_cot_instructions(problem: str) -> str:
    """Strip trailing CoT or derivation directives from FrontierScience prompts."""
    text = problem.strip()
    stripped = re.sub(
        r"\s*Think step by step and solve the problem below\..*\Z",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    # Guard against residual CoT directives while keeping scientific phrases
    # such as "step-by-step protocol" intact.
    stripped = re.sub(
        r"\s*Think step by step\.?\s*",
        " ",
        stripped,
        flags=re.IGNORECASE,
    ).strip()
    lowered = stripped.lower()
    if "think step by step" in lowered or "include all intermediate derivations" in lowered:
        raise ValueError("Failed to strip residual CoT instruction from problem text")
    return stripped


def canonical_question_key(text: str) -> str:
    """Normalize raw problem text for cross-run ``.eval`` log matching."""
    stripped = strip_cot_instructions(text)
    stripped = re.sub(
        r"\s*Direct answer requirement:.*\Z",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"\s+", " ", stripped).strip().lower()


def canonical_choices_key(choices: Sequence[Any] | None) -> tuple[str, ...]:
    """Normalize MCQ options while ignoring position-shuffled ``Insufficient information`` choices."""
    if not choices:
        return ()
    filtered: list[str] = []
    for choice in choices:
        norm = re.sub(r"\s+", " ", str(choice)).strip().lower()
        if norm and "insufficient information" not in norm:
            filtered.append(norm)
    return tuple(sorted(filtered))


def _parse_subset_filter(
    raw_value: str | Sequence[str] | None,
    *,
    normalizer: Callable[[str], str],
    valid: tuple[str, ...],
    label: str,
    default_all: Sequence[str],
) -> list[str]:
    if raw_value is None:
        return list(default_all)
    if isinstance(raw_value, str):
        raw = normalizer(raw_value)
        if not raw or raw == "all":
            return list(default_all)
        parts = [normalizer(p) for p in raw_value.split(",") if p.strip()]
    else:
        parts = [normalizer(str(p)) for p in raw_value if str(p).strip()]
        if not parts or parts == ["all"]:
            return list(default_all)
    valid_set = set(valid)
    for part in parts:
        if part not in valid_set:
            raise ValueError(f"Unsupported {label} {part!r}; expected one of {valid} or 'all'")
    return list(dict.fromkeys(parts))


def _normalize_chembench_subset_token(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


def _parse_chembench_subsets(chembench_task: str | Sequence[str] | None) -> list[str]:
    return _parse_subset_filter(
        chembench_task,
        normalizer=_normalize_chembench_subset_token,
        valid=VALID_CHEMBENCH_SUBSETS,
        label="ChemBench subset",
        default_all=VALID_CHEMBENCH_SUBSETS,
    )


def _normalize_lab_bench_subset_token(raw: str) -> str:
    tok = raw.strip().lower().replace("-", "_")
    return _LAB_BENCH_SUBSET_ALIASES.get(tok, tok)


def _parse_lab_bench_subsets(lab_bench_subset: str | Sequence[str] | None) -> list[str]:
    return _parse_subset_filter(
        lab_bench_subset,
        normalizer=_normalize_lab_bench_subset_token,
        valid=VALID_LAB_BENCH_SUBSETS,
        label="LAB-Bench subset",
        default_all=VALID_LAB_BENCH_SUBSETS,
    )


def _parse_frontierscience_subjects(
    subjects: str | Sequence[str] | None,
) -> list[FrontierScienceSubject]:
    return cast(
        list[FrontierScienceSubject],
        _parse_subset_filter(
            subjects,
            normalizer=lambda s: s.strip().lower(),
            valid=VALID_FRONTIERSCIENCE_SUBJECTS,
            label="FrontierScience subject",
            default_all=(),
        ),
    )


def _read_cached_parquet_split(snap_dir: Path, subdir: str) -> list[dict[str, Any]]:
    parquets = sorted((snap_dir / subdir).glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No cached parquet shards under {snap_dir / subdir}")
    if len(parquets) == 1:
        ds = Dataset.from_parquet(str(parquets[0]))
    else:
        ds = concatenate_datasets([Dataset.from_parquet(str(p)) for p in parquets])
    for col, feat in list(ds.features.items()):
        if isinstance(feat, HfImage):
            ds = ds.cast_column(col, HfImage(decode=False))
        elif isinstance(getattr(feat, "feature", None), HfImage):
            ds = ds.cast_column(col, type(feat)(HfImage(decode=False)))
    return [dict(row) for row in ds]


def _load_chembench_samples(
    limit: int | None,
    chembench_task: str | Sequence[str] | None = None,
) -> list[Sample]:
    subsets = _parse_chembench_subsets(chembench_task)
    out: list[Sample] = []
    seen_ids: set[str] = set()

    with hf_offline_env():
        snap_dir: Path | None = None
        if offline_mode_enabled():
            snap_dir = Path(
                snapshot_download(
                    repo_id=cb_mod.DATASET_PATH,
                    repo_type="dataset",
                    revision=cb_mod.CHEMBENCH_DATASET_REVISION,
                    local_files_only=True,
                )
            )
        for sub_name in subsets:
            if snap_dir is not None:
                rows = _read_cached_parquet_split(snap_dir, sub_name)
                raw_samples = [
                    cast(Sample, cb_mod.record_to_sample(row, subset_name=sub_name)) for row in rows
                ]
            else:
                t_cb = cb_mod.chembench(task_name=sub_name, shuffle=False)
                raw_samples = list(t_cb.dataset)

            for sample in raw_samples:
                raw_id = str(sample.id)
                wrapped_id = (
                    f"chembench__{raw_id}"
                    if raw_id not in seen_ids
                    else f"chembench__{sub_name}__{raw_id}"
                )
                seen_ids.add(raw_id)
                q_text = extract_input_text(sample.input)
                meta = dict(sample.metadata or {})
                meta.update(
                    {
                        "benchmark": "chembench",
                        "subset": sub_name,
                        "subset_name": sub_name,
                        "source_sample_id": raw_id,
                        "canonical_question": canonical_question_key(q_text),
                        "canonical_choices": list(canonical_choices_key(sample.choices)),
                    }
                )
                out.append(
                    Sample(
                        id=wrapped_id,
                        input=sample.input,
                        choices=list(sample.choices) if sample.choices else None,
                        target=sample.target,
                        metadata=meta,
                    )
                )
                if limit is not None and len(out) >= limit:
                    return out
    return out


def _load_lab_bench_samples(
    limit: int | None,
    lab_bench_subset: str | Sequence[str] | None = None,
) -> list[Sample]:
    subsets = _parse_lab_bench_subsets(lab_bench_subset)
    out: list[Sample] = []
    seen_ids: set[str] = set()

    with hf_offline_env():
        snap_dir: Path | None = None
        if offline_mode_enabled():
            snap_dir = Path(
                snapshot_download(
                    repo_id=lb_mod.DATASET_PATH,
                    repo_type="dataset",
                    revision=lb_mod.LAB_BENCH_DATASET_REVISION,
                    local_files_only=True,
                )
            )
        for sub_key in subsets:
            enum_val, converter_name, task_fn_name = LAB_BENCH_SUBSET_SPECS[sub_key]
            if snap_dir is not None:
                converter_fn = cast(
                    Callable[[dict[str, Any]], Sample],
                    getattr(lb_rsh, converter_name),
                )
                rows = _read_cached_parquet_split(snap_dir, enum_val)
                sub_ds = MemoryDataset([converter_fn(row) for row in rows])
                sub_ds.shuffle_choices(seed=lb_mod.DEFAULT_SHUFFLE_SEED)
                raw_samples = list(sub_ds)
            else:
                t_lb = getattr(lb_mod, task_fn_name)(
                    shuffle=False,
                    shuffle_choices=lb_mod.DEFAULT_SHUFFLE_SEED,
                )
                raw_samples = list(t_lb.dataset)

            for sample in raw_samples:
                raw_id = str(sample.id)
                wrapped_id = (
                    f"lab_bench__{raw_id}"
                    if raw_id not in seen_ids
                    else f"lab_bench__{sub_key}__{raw_id}"
                )
                seen_ids.add(raw_id)
                q_text = extract_input_text(sample.input)
                meta = dict(sample.metadata or {})
                meta.update(
                    {
                        "benchmark": "lab_bench",
                        "subset": enum_val,
                        "subset_key": sub_key,
                        "source_sample_id": raw_id,
                        "canonical_question": canonical_question_key(q_text),
                        "canonical_choices": list(canonical_choices_key(sample.choices)),
                    }
                )
                out.append(
                    Sample(
                        id=wrapped_id,
                        input=sample.input,
                        choices=list(sample.choices) if sample.choices else None,
                        target=sample.target,
                        metadata=meta,
                    )
                )
                if limit is not None and len(out) >= limit:
                    return out
    return out


def _load_frontierscience_samples(
    frontierscience_format: FrontierScienceFormat,
    limit: int | None,
    frontierscience_subjects: str | Sequence[str] | None = None,
) -> list[Sample]:
    fs_mod._id_counter.clear()
    fmt_arg: Literal["olympic", "research"] | None = (
        None if frontierscience_format == "all" else frontierscience_format
    )
    subjects_arg = _parse_frontierscience_subjects(frontierscience_subjects)
    with hf_offline_env():
        t_fs = fs_mod.frontierscience(
            format=fmt_arg,
            subjects=cast(Any, subjects_arg),
            shuffle=False,
        )
    fs_mod._id_counter.clear()

    out: list[Sample] = []
    for sample in t_fs.dataset:
        raw_id = str(sample.id)
        base_id = raw_id.split("_")[0]
        raw_problem = sample.input if isinstance(sample.input, str) else ""
        stem = strip_cot_instructions(raw_problem)
        meta = dict(sample.metadata or {})
        fmt = str(meta.get("format", "olympic"))
        suffix = (
            FRONTIERSCIENCE_OLYMPIC_NO_COT_SUFFIX
            if fmt == "olympic"
            else FRONTIERSCIENCE_RESEARCH_NO_COT_SUFFIX
        )
        no_cot_input = f"{stem}\n\n{suffix}"
        meta.update(
            {
                "benchmark": "frontierscience",
                "subset": fmt,
                "source_sample_id": raw_id,
                "base_sample_id": base_id,
                "question_stem": stem,
                "cot_prompt": raw_problem,
                "no_cot_prompt": no_cot_input,
                "canonical_question": canonical_question_key(stem),
            }
        )
        out.append(
            Sample(
                id=f"frontierscience__{raw_id}",
                input=raw_problem,
                target=sample.target,
                metadata=meta,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def build_science_qa_dataset(
    dataset: ScienceDatasetName = "all",
    frontierscience_format: FrontierScienceFormat = "all",
    limit_per_dataset: int | None = None,
    prompt_mode: Literal["no_cot", "cot"] = "cot",
    frontierscience_subjects: str | Sequence[str] | None = None,
    chembench_task: str | Sequence[str] | None = None,
    lab_bench_subset: str | Sequence[str] | None = None,
) -> MemoryDataset:
    """Build wrapped ``inspect_evals`` samples for ChemBench, LAB-Bench, and FrontierScience.

    Args:
        dataset: Which subset(s) to include (``"all"``, ``"chembench"``, ``"lab_bench"``, or ``"frontierscience"``).
        frontierscience_format: FrontierScience split filter (``"all"``, ``"olympic"``, or ``"research"``).
        limit_per_dataset: Optional per-dataset cap applied before merging.
        prompt_mode: Use ``"cot"`` prompts or stripped ``"no_cot"`` prompts for FrontierScience.
        frontierscience_subjects: Optional FrontierScience subject filter (``"biology"``, ``"chemistry"``, ``"physics"``, comma-separated string, sequence, or ``"all"``).
        chembench_task: Optional ChemBench subset filter (``None`` / ``"all"`` loads all 9 subsets).
        lab_bench_subset: Optional LAB-Bench subset filter (``None`` / ``"all"`` loads all 8 subsets).

    Returns:
        MemoryDataset containing the wrapped samples.
    """
    samples: list[Sample] = []
    if dataset in ("all", "chembench"):
        samples.extend(_load_chembench_samples(limit_per_dataset, chembench_task=chembench_task))
    if dataset in ("all", "lab_bench"):
        samples.extend(
            _load_lab_bench_samples(limit_per_dataset, lab_bench_subset=lab_bench_subset)
        )
    if dataset in ("all", "frontierscience"):
        fs_samples = _load_frontierscience_samples(
            frontierscience_format,
            limit_per_dataset,
            frontierscience_subjects=frontierscience_subjects,
        )
        if prompt_mode == "no_cot":
            for sample in fs_samples:
                sample.input = str((sample.metadata or {})["no_cot_prompt"])
        samples.extend(fs_samples)
    return MemoryDataset(samples)


_CacheKey = tuple[
    str,
    str,
    frozenset[str] | None,
    frozenset[str] | None,
    frozenset[str] | None,
    tuple[tuple[str, int, int], ...],
]
_LOG_INDEX_CACHE: dict[_CacheKey, RecordedLogIndex] = {}


def _strip_model_suffixes(short_name: str) -> tuple[str, str]:
    no_nt = short_name.removesuffix("-nt")
    base = re.sub(r"-(?:st-)?(?:no)?guardrail$", "", no_nt).removesuffix("-st")
    return no_nt, base


def is_per_model_log_root(log_path: str | Path) -> bool:
    """Return True if ``log_path`` names a multi-model root directory or path template."""
    raw = str(log_path).strip()
    if "{model}" in raw or "{base_model}" in raw:
        return True
    p = Path(raw).expanduser()
    return p.name == "eval_log" or (p / "openai").is_dir()


def _dir_has_matching_science_log(probe: Path, dataset: ScienceDatasetName) -> bool:
    for candidate in sorted(probe.rglob("*.eval")):
        try:
            header = read_eval_log(candidate, header_only=True)
        except Exception:
            continue
        if header.status != "success":
            continue
        bm = infer_benchmark_from_log(header)
        if bm is not None and dataset in ("all", bm):
            return True
    return False


def resolve_recorded_log_path(
    log_path: str | Path,
    *,
    model_name: str | None = None,
    dataset: ScienceDatasetName = "all",
) -> Path | None:
    """Resolve a direct log path or a per-model directory under ``eval_log``.

    Returns ``None`` when ``log_path`` is a per-model root (or template) and no
    evaluated model name is known yet, signalling the caller to defer lookup to
    runtime solver execution.
    """
    raw = str(log_path).strip()
    if not is_per_model_log_root(raw):
        return Path(raw).expanduser()

    active_model = (model_name or os.environ.get("INSPECT_EVAL_MODEL") or "").strip()
    if not active_model:
        return None

    if "/" in active_model:
        provider, _, short_name = active_model.partition("/")
    else:
        provider, short_name = "", active_model
    no_nt, base = _strip_model_suffixes(short_name)
    candidates = list(dict.fromkeys([no_nt, base, short_name]))

    if "{model}" in raw or "{base_model}" in raw:
        return Path(
            raw.format(
                model=short_name,
                base_model=base,
                provider=provider or "openai",
            )
        ).expanduser()

    root_path = Path(raw).expanduser()
    candidate_roots = [root_path]
    if root_path.name == "eval_log":
        candidate_roots.extend([Path("..") / root_path.name, Path("../..") / root_path.name])

    ordered_probes: list[Path] = []
    for root in candidate_roots:
        for cand in candidates:
            if provider:
                ordered_probes.append(root / provider / cand)
            if provider != "openai":
                ordered_probes.append(root / "openai" / cand)
            ordered_probes.append(root / cand)

    existing_dirs: list[Path] = []
    non_empty_dirs: list[Path] = []
    seen_probes: set[Path] = set()
    for probe in ordered_probes:
        if probe in seen_probes or not probe.is_dir():
            continue
        seen_probes.add(probe)
        existing_dirs.append(probe)
        if next(probe.rglob("*.eval"), None) is not None:
            non_empty_dirs.append(probe)
            if _dir_has_matching_science_log(probe, dataset):
                return probe

    if non_empty_dirs:
        return non_empty_dirs[0]
    if existing_dirs:
        return existing_dirs[0]
    return candidate_roots[0] / (provider or "openai") / base


_LAB_BENCH_TASK_MARKERS: tuple[tuple[str, str], ...] = (
    ("cloning_scenarios", "cloning_scenarios"),
    ("cloningscenarios", "cloning_scenarios"),
    ("cloning", "cloning_scenarios"),
    ("litqa2", "litqa2"),
    ("litqa", "litqa2"),
    ("suppqa", "suppqa"),
    ("figqa", "figqa"),
    ("tableqa", "tableqa"),
    ("dbqa", "dbqa"),
    ("protocolqa", "protocolqa"),
    ("seqqa", "seqqa"),
)


def _infer_lab_bench_subset_key(task_name: str, meta: Mapping[str, Any]) -> str:
    raw_meta = str(
        meta.get("subset_key") or meta.get("subset") or meta.get("subtask") or ""
    ).strip()
    if raw_meta:
        return _normalize_lab_bench_subset_token(raw_meta)
    tn = (task_name or "").lower().replace("-", "_")
    for marker, sub_key in _LAB_BENCH_TASK_MARKERS:
        if marker in tn:
            return sub_key
    return ""


def _sample_subset_key(benchmark: BenchmarkKey, meta: Mapping[str, Any]) -> str:
    if benchmark == "chembench":
        raw_cb = meta.get("subset_name") if "subset_name" in meta else meta.get("subset")
        return _normalize_chembench_subset_token(str(raw_cb or ""))
    if benchmark == "lab_bench":
        return _infer_lab_bench_subset_key("", meta)
    if benchmark == "frontierscience":
        return str(meta.get("subject") or "").strip().lower()
    return ""


def infer_benchmark_from_log(log: EvalLog) -> BenchmarkKey | None:
    """Identify which upstream Science QA benchmark produced an Inspect ``EvalLog``."""
    task_name = (log.eval.task or "").lower()
    if "time_travel" in task_name or "time-travel" in task_name or "immediacy" in task_name:
        return None
    task_args = getattr(log.eval, "task_args_passed", None) or log.eval.task_args or {}
    if "no_cot" in task_name or "no-cot" in task_name:
        if task_args.get("mode") != "cot":
            return None
        ds_arg = str(task_args.get("dataset") or "").lower()
        if ds_arg in ("chembench", "lab_bench", "frontierscience"):
            return cast(BenchmarkKey, ds_arg)
        return None
    if "chembench" in task_name:
        return "chembench"
    norm_task = task_name.replace("-", "_")
    if "lab_bench" in norm_task or any(
        marker in norm_task for marker, _ in _LAB_BENCH_TASK_MARKERS
    ):
        return "lab_bench"
    if "frontierscience" in task_name:
        return "frontierscience"
    return None


def _validated_non_negative_float(raw: Any) -> float | None:
    try:
        fval = float(raw)
    except (ValueError, TypeError):
        return None
    if math.isnan(fval) or math.isinf(fval) or fval < 0.0:
        return None
    return fval


def _extract_score_float(scores: Mapping[str, Any] | None) -> tuple[float | None, Any]:
    if not scores:
        return None, None
    preferred_keys = (
        "chembench_scorer",
        "precision_choice",
        "frontierscience_scorer",
        "cot_arm_scorer",
        "no_cot_paired_scorer",
        "no_cot_scorer",
    )
    chosen: Any = None
    for key in preferred_keys:
        if key in scores:
            chosen = scores[key]
            break
    if chosen is None:
        chosen = next(iter(scores.values()))

    val = getattr(chosen, "value", chosen)
    if isinstance(val, dict):
        for field in (
            "cot_accuracy",
            "cot_raw_score",
            "raw_score",
            "accuracy",
            "correct",
            "no_cot_raw_score",
        ):
            if (
                field in val
                and isinstance(val[field], (int, float))
                and not isinstance(val[field], bool)
            ):
                parsed = _validated_non_negative_float(val[field])
                if parsed is not None:
                    return parsed, val
        return None, val
    if isinstance(val, str):
        upper = val.strip().upper()
        if upper == "C":
            return 1.0, val
        if upper in ("I", "N", "P"):
            return 0.0, val
    try:
        parsed = _validated_non_negative_float(value_to_float()(val))
        return parsed, val
    except (ValueError, TypeError):
        return None, val


def _extract_sample_question_text(sample: Any) -> str:
    raw_input = extract_input_text(sample.input)
    if not raw_input and getattr(sample, "messages", None):
        raw_input = extract_input_text([sample.messages[0]])
    return canonical_question_key(raw_input)


def _extract_sample_problem_text(sample: Any) -> str:
    if getattr(sample, "messages", None):
        msg_text = extract_input_text([sample.messages[0]]).strip()
        if msg_text:
            return msg_text
    return extract_input_text(sample.input).strip()


def _subsets_compatible(sample_subset: str, rec_subset: str) -> bool:
    return not sample_subset or not rec_subset or sample_subset == rec_subset


class RecordedLogIndex:
    """Lookup index over recorded ``.eval`` samples by ``(benchmark, subset, id)`` or question text."""

    def __init__(self, records: list[RecordedSampleResult]) -> None:
        self.records = records
        self._by_subset_id: dict[tuple[BenchmarkKey, str, str], RecordedSampleResult] = {}
        self._ambiguous_subset_ids: set[tuple[BenchmarkKey, str, str]] = set()
        self._by_id: dict[tuple[BenchmarkKey, str], RecordedSampleResult] = {}
        self._ambiguous_ids: set[tuple[BenchmarkKey, str]] = set()
        self._by_question: dict[tuple[BenchmarkKey, str], RecordedSampleResult] = {}
        self._by_question_choices: dict[
            tuple[BenchmarkKey, str, tuple[str, ...]], RecordedSampleResult
        ] = {}
        self._ambiguous_questions: set[tuple[BenchmarkKey, str]] = set()
        seen_q_ids: dict[tuple[BenchmarkKey, str], tuple[str, str]] = {}
        qc_accs: dict[tuple[BenchmarkKey, str, tuple[str, ...]], list[float]] = {}

        for rec in records:
            norm_id = rec.source_sample_id.removeprefix(f"{rec.benchmark}__")
            base_id = norm_id.split("_")[0] if rec.benchmark == "frontierscience" else norm_id
            for key_id in dict.fromkeys((norm_id, base_id)):
                if not key_id:
                    continue
                sub_id_key = (rec.benchmark, rec.subset, key_id)
                if sub_id_key not in self._ambiguous_subset_ids:
                    prev_sub = self._by_subset_id.get(sub_id_key)
                    if prev_sub is None:
                        self._by_subset_id[sub_id_key] = rec
                    elif (
                        prev_sub.source_sample_id,
                        prev_sub.canonical_question,
                    ) != (rec.source_sample_id, rec.canonical_question):
                        self._ambiguous_subset_ids.add(sub_id_key)
                        self._by_subset_id.pop(sub_id_key, None)

                id_key = (rec.benchmark, key_id)
                if id_key in self._ambiguous_ids:
                    continue
                existing = self._by_id.get(id_key)
                if existing is None:
                    self._by_id[id_key] = rec
                elif (
                    existing.subset,
                    existing.source_sample_id,
                    existing.canonical_question,
                ) != (
                    rec.subset,
                    rec.source_sample_id,
                    rec.canonical_question,
                ):
                    self._ambiguous_ids.add(id_key)
                    self._by_id.pop(id_key, None)

            if rec.canonical_question and rec.canonical_choices:
                qc_key = (rec.benchmark, rec.canonical_question, rec.canonical_choices)
                if rec.accuracy is not None:
                    qc_accs.setdefault(qc_key, []).append(rec.accuracy)
                if qc_key not in self._by_question_choices:
                    self._by_question_choices[qc_key] = rec

            if rec.canonical_question:
                q_key = (rec.benchmark, rec.canonical_question)
                rec_ident = (rec.subset, base_id if rec.benchmark == "frontierscience" else norm_id)
                prev_ident = seen_q_ids.get(q_key)
                if prev_ident is None:
                    seen_q_ids[q_key] = rec_ident
                    if q_key not in self._ambiguous_questions:
                        self._by_question[q_key] = rec
                elif prev_ident != rec_ident:
                    self._ambiguous_questions.add(q_key)
                    self._by_question.pop(q_key, None)
                elif q_key not in self._ambiguous_questions:
                    self._by_question[q_key] = rec

        for qc_key, accs in qc_accs.items():
            if len(accs) > 1:
                self._by_question_choices[qc_key] = replace(
                    self._by_question_choices[qc_key],
                    accuracy=sum(accs) / len(accs),
                )

    def find(self, sample: Sample) -> RecordedSampleResult | None:
        """Match a wrapped dataset ``Sample`` to a recorded ``.eval`` sample."""
        meta = sample.metadata or {}
        benchmark_raw = str(meta.get("benchmark", ""))
        if benchmark_raw not in ("chembench", "lab_bench", "frontierscience"):
            return None
        benchmark = cast(BenchmarkKey, benchmark_raw)
        sample_subset = _sample_subset_key(benchmark, meta)
        canon = str(meta.get("canonical_question", "")) or canonical_question_key(
            extract_input_text(sample.input)
        )
        raw_src = str(
            meta["source_sample_id"]
            if meta.get("source_sample_id") is not None
            else (sample.id or "")
        )
        src_id = raw_src.removeprefix(f"{benchmark}__")
        base_id = str(meta.get("base_sample_id", "")).removeprefix(f"{benchmark}__")

        def _compatible_rec(rec: RecordedSampleResult | None) -> RecordedSampleResult | None:
            if rec is None:
                return None
            if (
                benchmark == "frontierscience"
                and canon
                and rec.canonical_question
                and rec.canonical_question != canon
            ):
                return None
            return rec

        for candidate in (src_id, base_id):
            if not candidate:
                continue
            if sample_subset:
                exact_sub = _compatible_rec(
                    self._by_subset_id.get((benchmark, sample_subset, candidate))
                )
                if exact_sub is not None:
                    return exact_sub
                empty_sub = _compatible_rec(self._by_subset_id.get((benchmark, "", candidate)))
                if empty_sub is not None:
                    return empty_sub
            unambig = _compatible_rec(self._by_id.get((benchmark, candidate)))
            if unambig is not None and _subsets_compatible(sample_subset, unambig.subset):
                return unambig

        raw_cc = meta.get("canonical_choices")
        cc = (
            tuple(str(x) for x in raw_cc)
            if isinstance(raw_cc, (list, tuple)) and raw_cc
            else canonical_choices_key(sample.choices)
        )
        if canon and cc and (benchmark, canon, cc) in self._by_question_choices:
            qc_rec = self._by_question_choices[(benchmark, canon, cc)]
            if _subsets_compatible(sample_subset, qc_rec.subset):
                return qc_rec
        if canon and (benchmark, canon) in self._by_question:
            fallback = self._by_question[(benchmark, canon)]
            if not _subsets_compatible(sample_subset, fallback.subset):
                return None
            if cc and fallback.canonical_choices and cc != fallback.canonical_choices:
                return None
            return fallback
        return None


def _resolve_filter_frozenset(
    raw: str | Sequence[str] | None,
    parser: Callable[[str | Sequence[str] | None], Sequence[str]],
    all_values: Sequence[str],
) -> frozenset[str] | None:
    if raw is None:
        return None
    parsed = list(parser(raw))
    if not parsed or set(parsed) == set(all_values):
        return None
    return frozenset(parsed)


def _header_matches_filters(
    header: EvalLog,
    dataset: ScienceDatasetName,
    fs_filter: frozenset[str] | None,
    cb_filter: frozenset[str] | None,
    lb_filter: frozenset[str] | None,
) -> bool:
    if header.status != "success":
        return False
    bm = infer_benchmark_from_log(header)
    if bm is None or dataset not in ("all", bm):
        return False
    header_args = getattr(header.eval, "task_args_passed", None) or header.eval.task_args or {}
    if bm == "chembench" and cb_filter is not None:
        raw_cb_arg = header_args.get("task_name") or header_args.get("chembench_task")
        if raw_cb_arg:
            try:
                hdr_cb = set(_parse_chembench_subsets(raw_cb_arg))
            except ValueError:
                hdr_cb = set()
            if hdr_cb and hdr_cb.isdisjoint(cb_filter):
                return False
    elif bm == "lab_bench" and lb_filter is not None:
        raw_lb_arg = header_args.get("lab_bench_subset")
        if raw_lb_arg:
            try:
                hdr_lb = set(_parse_lab_bench_subsets(raw_lb_arg))
            except ValueError:
                hdr_lb = set()
            if hdr_lb and hdr_lb.isdisjoint(lb_filter):
                return False
        else:
            hdr_sub = _infer_lab_bench_subset_key(header.eval.task or "", {})
            if hdr_sub and hdr_sub not in lb_filter:
                return False
    elif bm == "frontierscience" and fs_filter is not None:
        raw_fs_arg = header_args.get("subjects") or header_args.get("frontierscience_subjects")
        if raw_fs_arg:
            try:
                hdr_fs = set(_parse_frontierscience_subjects(raw_fs_arg))
            except ValueError:
                hdr_fs = set()
            if hdr_fs and hdr_fs.isdisjoint(fs_filter):
                return False
    return True


def load_inspect_evals_log_records(
    log_path: str | Path,
    *,
    dataset: ScienceDatasetName = "all",
    frontierscience_subjects: str | Sequence[str] | None = None,
    chembench_task: str | Sequence[str] | None = None,
    lab_bench_subset: str | Sequence[str] | None = None,
) -> RecordedLogIndex:
    """Load recorded outputs and scores from a ``.eval`` file or directory."""
    root = Path(log_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Log path does not exist: {root}")

    fs_filter = _resolve_filter_frozenset(
        frontierscience_subjects,
        _parse_frontierscience_subjects,
        VALID_FRONTIERSCIENCE_SUBJECTS,
    )
    cb_filter = _resolve_filter_frozenset(
        chembench_task,
        _parse_chembench_subsets,
        VALID_CHEMBENCH_SUBSETS,
    )
    lb_filter = _resolve_filter_frozenset(
        lab_bench_subset,
        _parse_lab_bench_subsets,
        VALID_LAB_BENCH_SUBSETS,
    )

    if root.is_file():
        st = root.stat()
        sig_entries = [(str(root.resolve()), st.st_mtime_ns, st.st_size)]
        paths = [root]
    else:
        all_eval_files = sorted(root.rglob("*.eval"))
        if not all_eval_files:
            raise FileNotFoundError(f"No .eval files found under {root}")
        sortable: list[tuple[str, int, str, Path, int]] = []
        for candidate in all_eval_files:
            try:
                header = read_eval_log(candidate, header_only=True)
            except Exception:
                continue
            if not _header_matches_filters(header, dataset, fs_filter, cb_filter, lb_filter):
                continue
            st = candidate.stat()
            created = str(getattr(header.eval, "created", "") or "")
            sortable.append((created, st.st_mtime_ns, str(candidate), candidate, st.st_size))
        sortable.sort(key=lambda item: (item[0], item[1], item[2]))
        paths = [item[3] for item in sortable]
        sig_entries = [(item[2], item[1], item[4]) for item in sortable]

    cache_key: _CacheKey = (
        str(root.resolve()),
        dataset,
        fs_filter,
        cb_filter,
        lb_filter,
        tuple(sig_entries),
    )
    cached = _LOG_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    merged_records: dict[tuple[BenchmarkKey, str, str], RecordedSampleResult] = {}
    for path in paths:
        log = read_eval_log(
            path,
            resolve_attachments=False,
            exclude_fields={"events", "store", "attachments"},
        )
        inferred = infer_benchmark_from_log(log)
        log_args = getattr(log.eval, "task_args_passed", None) or log.eval.task_args or {}
        header_cb_sub = _normalize_chembench_subset_token(str(log_args.get("task_name") or ""))
        if header_cb_sub == "all":
            header_cb_sub = ""
        per_log_accs: dict[tuple[BenchmarkKey, str, str], list[float]] = {}
        per_log_first_acc: dict[tuple[BenchmarkKey, str, str], float | None] = {}
        per_log_first: dict[tuple[BenchmarkKey, str, str], RecordedSampleResult] = {}

        ordered_samples = sorted(
            list(log.samples or []),
            key=lambda s: int(getattr(s, "epoch", 1) or 1),
        )
        for sample in ordered_samples:
            meta = sample.metadata if isinstance(sample.metadata, dict) else {}
            bm_raw = meta.get("benchmark") or inferred
            if bm_raw not in ("chembench", "lab_bench", "frontierscience"):
                continue
            benchmark = cast(BenchmarkKey, bm_raw)
            sample_subset = ""
            if benchmark == "chembench":
                raw_cb_sub = (
                    meta.get("subset_name") if "subset_name" in meta else meta.get("subset")
                )
                sample_subset = _normalize_chembench_subset_token(
                    str(raw_cb_sub or header_cb_sub or "")
                )
                if cb_filter is not None and sample_subset not in cb_filter:
                    continue
            elif benchmark == "lab_bench":
                sample_subset = _infer_lab_bench_subset_key(log.eval.task or "", meta)
                if lb_filter is not None and sample_subset not in lb_filter:
                    continue
            elif benchmark == "frontierscience":
                sample_subset = str(meta.get("subject") or "").strip().lower()
                if fs_filter is not None and sample_subset and sample_subset not in fs_filter:
                    continue

            if not sample.output or not sample.output.choices:
                continue

            raw_id = str(
                meta["source_sample_id"]
                if meta.get("source_sample_id") is not None
                else (sample.id or "")
            )
            norm_id = raw_id.removeprefix(f"{benchmark}__")
            canon_q = _extract_sample_question_text(sample)
            rec_key_id = canon_q if (benchmark == "frontierscience" and canon_q) else norm_id
            rec_key = (benchmark, sample_subset, rec_key_id)
            acc, raw_val = _extract_score_float(sample.scores)
            if acc is not None:
                per_log_accs.setdefault(rec_key, []).append(acc)
            if rec_key not in per_log_first:
                per_log_first_acc[rec_key] = acc
                per_log_first[rec_key] = RecordedSampleResult(
                    benchmark=benchmark,
                    source_sample_id=raw_id,
                    log_path=str(path),
                    model=str(log.eval.model),
                    task_name=str(log.eval.task),
                    output=sample.output,
                    accuracy=acc,
                    raw_score_value=raw_val,
                    canonical_question=canon_q,
                    canonical_choices=canonical_choices_key(sample.choices),
                    subset=sample_subset,
                    problem_text=_extract_sample_problem_text(sample),
                    first_epoch_accuracy=acc,
                )

        for rec_key, first_rec in per_log_first.items():
            accs = per_log_accs.get(rec_key)
            if accs:
                first_rec.accuracy = sum(accs) / len(accs)
            if per_log_first_acc.get(rec_key) is not None:
                first_rec.first_epoch_accuracy = per_log_first_acc[rec_key]
            merged_records[rec_key] = first_rec

    index = RecordedLogIndex(list(merged_records.values()))
    _LOG_INDEX_CACHE[cache_key] = index
    return index


def serialize_recorded_result(
    rec: RecordedSampleResult,
    *,
    accuracy_mode: Literal["mean", "first_epoch"] = "mean",
    include_problem_text: bool = False,
) -> dict[str, Any]:
    """Serialize a ``RecordedSampleResult`` into lightweight sample metadata."""
    acc = (
        rec.first_epoch_accuracy
        if accuracy_mode == "first_epoch" and rec.first_epoch_accuracy is not None
        else rec.accuracy
    )
    payload: dict[str, Any] = {
        "output": rec.output.model_dump(mode="json"),
        "accuracy": acc,
        "log_path": rec.log_path,
        "model": rec.model,
        "task_name": rec.task_name,
    }
    if include_problem_text and rec.problem_text:
        payload["problem_text"] = rec.problem_text
    return payload


def filter_dataset_with_recorded_log(
    ds: MemoryDataset,
    *,
    log_path: str | Path,
    param_name: str,
    metadata_key: str,
    filter_to_log: bool,
    dataset: ScienceDatasetName,
    frontierscience_subjects: str | Sequence[str] | None = None,
    chembench_task: str | Sequence[str] | None = None,
    lab_bench_subset: str | Sequence[str] | None = None,
    accuracy_mode: Literal["mean", "first_epoch"] = "mean",
    include_problem_text: bool = False,
) -> str | None:
    """Eagerly match ``ds`` against ``log_path``, or return ``str(log_path)`` to defer."""
    resolved = resolve_recorded_log_path(log_path, dataset=dataset)
    if resolved is None:
        return str(log_path)
    index = load_inspect_evals_log_records(
        resolved,
        dataset=dataset,
        frontierscience_subjects=frontierscience_subjects,
        chembench_task=chembench_task,
        lab_bench_subset=lab_bench_subset,
    )
    matched: list[Sample] = []
    n_matched = 0
    for sample in ds:
        rec = index.find(sample)
        if rec is not None:
            meta = dict(sample.metadata or {})
            meta[metadata_key] = serialize_recorded_result(
                rec,
                accuracy_mode=accuracy_mode,
                include_problem_text=include_problem_text,
            )
            sample.metadata = meta
            matched.append(sample)
            n_matched += 1
        elif not filter_to_log:
            matched.append(sample)
    if n_matched == 0:
        raise ValueError(
            f"{param_name}={str(log_path)!r} matched 0 samples in the selected dataset"
        )
    ds.samples = matched
    return None


def attach_deferred_recorded_sample(
    state: TaskState,
    *,
    deferred_log: str,
    param_name: str,
    metadata_key: str,
    model_name: str,
    filter_to_log: bool,
    dataset: ScienceDatasetName,
    frontierscience_subjects: str | Sequence[str] | None = None,
    chembench_task: str | Sequence[str] | None = None,
    lab_bench_subset: str | Sequence[str] | None = None,
    accuracy_mode: Literal["mean", "first_epoch"] = "mean",
    include_problem_text: bool = False,
    index_cache: dict[str, RecordedLogIndex],
) -> bool:
    """Resolve and attach a recorded sample at runtime; return False if sample is skipped."""
    meta = state.metadata if isinstance(state.metadata, dict) else {}
    index = index_cache.get(model_name)
    if index is None:
        model_dir = resolve_recorded_log_path(
            deferred_log,
            model_name=model_name,
            dataset=dataset,
        )
        assert model_dir is not None
        index = load_inspect_evals_log_records(
            model_dir,
            dataset=dataset,
            frontierscience_subjects=frontierscience_subjects,
            chembench_task=chembench_task,
            lab_bench_subset=lab_bench_subset,
        )
        if not index.records:
            raise ValueError(
                f"{param_name}={deferred_log!r} ({model_dir}) matched 0 samples "
                f"for dataset={dataset!r}"
            )
        index_cache[model_name] = index

    rec = index.find(
        Sample(
            id=state.sample_id,
            input=state.input,
            choices=[c.value for c in state.choices] if state.choices else None,
            metadata=meta,
        )
    )
    if rec is not None:
        meta[metadata_key] = serialize_recorded_result(
            rec,
            accuracy_mode=accuracy_mode,
            include_problem_text=include_problem_text,
        )
        return True
    if filter_to_log:
        meta["skipped_unmatched_log"] = True
        state.completed = True
        return False
    return True


def build_arm_solver_inputs(
    sample_state: TaskState, *, prompt_mode: Literal["no_cot", "cot"]
) -> None:
    """Prepare ``TaskState.messages`` for ``chembench``, ``lab_bench``, or ``frontierscience``."""
    meta = sample_state.metadata or {}
    benchmark = str(meta.get("benchmark", ""))
    if benchmark == "frontierscience":
        key = "no_cot_prompt" if prompt_mode == "no_cot" else "cot_prompt"
        prompt_text = str(meta.get(key) or sample_state.input_text)
        sample_state.messages = [ChatMessageUser(content=prompt_text)]
        return

    reset_sample_messages_from_input(sample_state)


def reset_choice_flags(state: TaskState) -> None:
    """Clear ``correct`` flags on all choices in ``state``."""
    for choice in state.choices:
        choice.correct = None


def sync_mcq_choice_flags_from_output(state: TaskState) -> None:
    """Populate ``state.choices[i].correct`` from completion text when all flags are ``None``."""
    if not state.choices or state.output is None:
        return
    if any(choice.correct is not None for choice in state.choices):
        return
    text = (state.output.completion or "").strip()
    if not text:
        return
    ans_matches = re.findall(r"(?i)\bANSWER\s*:\s*\(?([A-Z])\)?", text)
    if ans_matches:
        letter = ans_matches[-1].upper()
    else:
        tail_match = re.search(r"\b([A-Z])\s*[.)]?\s*$", text)
        if tail_match is None:
            return
        letter = tail_match.group(1).upper()
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(state.choices):
        for i, choice in enumerate(state.choices):
            choice.correct = i == idx


@solver
def _chembench_no_cot_solver() -> Solver:
    mc_solve = multiple_choice(template=CHEMBENCH_NO_COT_TEMPLATE, cot=False)
    num_prompt_solve = prompt_template(template=CHEMBENCH_NUMERIC_NO_COT_TEMPLATE)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if len(state.choices) < 1:
            state = await num_prompt_solve(state, generate)
            return await generate(state)
        return await mc_solve(state, generate)

    return solve


def upstream_mc_solver(
    *,
    benchmark: str,
    prompt_mode: Literal["no_cot", "cot"],
) -> Solver:
    """Return the upstream ``multiple_choice`` (or freeform numeric) solver configured for CoT or No-CoT."""
    if benchmark == "chembench":
        if prompt_mode == "no_cot":
            return _chembench_no_cot_solver()
        return cast(Solver, cb_mod.multiple_choice_or_freeform(cot=True))
    if benchmark == "lab_bench":
        if prompt_mode == "no_cot":
            return multiple_choice(template=LAB_BENCH_NO_COT_TEMPLATE, cot=False)
        return multiple_choice(template=lb_mod.MULTIPLE_CHOICE_TEMPLATE, cot=True)
    raise ValueError(f"Not an MCQ benchmark: {benchmark!r}")


def upstream_scorer_for_benchmark(
    benchmark: BenchmarkKey,
    *,
    grader_model: str | Model | None = DEFAULT_GRADER_MODEL,
) -> Scorer:
    """Return the upstream ``inspect_evals`` scorer for the given benchmark."""
    if benchmark == "chembench":
        return cast(Scorer, cb_mod.chembench_scorer())
    if benchmark == "lab_bench":
        return cast(Scorer, lb_mod.precision_choice(no_answer=lb_mod.UNCERTAIN_ANSWER_CHOICE))
    if benchmark == "frontierscience":
        probe = resolve_judge_model(grader_model, role="judge")
        resolved = resolve_judge_model(
            grader_model,
            role="judge",
            config=judge_generate_config(probe),
        )
        return cast(Scorer, fs_mod.frontierscience_scorer(model=resolved))
    raise ValueError(f"Unsupported benchmark: {benchmark!r}")


async def grade_with_upstream_scorer(
    *,
    state: TaskState,
    target: Target,
    grader_model: str | Model | None = DEFAULT_GRADER_MODEL,
) -> tuple[float, Score]:
    """Grade ``state`` using the upstream ``inspect_evals`` scorer."""
    meta = state.metadata or {}
    bm_raw = str(meta.get("benchmark", ""))
    if bm_raw not in ("chembench", "lab_bench", "frontierscience"):
        raise ValueError(f"Unsupported benchmark in state.metadata: {bm_raw!r}")
    benchmark = cast(BenchmarkKey, bm_raw)
    scorer_fn = upstream_scorer_for_benchmark(benchmark, grader_model=grader_model)

    grade_state = state
    if benchmark == "frontierscience" and meta.get("question_stem"):
        grade_state = TaskState(
            model=state.model,
            sample_id=state.sample_id,
            epoch=state.epoch,
            input=str(meta["question_stem"]),
            messages=list(state.messages),
            target=target,
            choices=[c.value for c in state.choices] if state.choices else None,
            output=state.output,
            metadata=dict(meta),
        )
    elif (
        benchmark == "lab_bench"
        and grade_state.choices
        and all(c.correct is None for c in grade_state.choices)
    ):
        sync_mcq_choice_flags_from_output(grade_state)

    score_obj = await scorer_fn(grade_state, target)
    if score_obj is None:
        return 0.0, Score(value=0.0, answer="", explanation="Upstream scorer returned None")
    numeric, _ = _extract_score_float({"upstream": score_obj})
    if numeric is None:
        return math.nan, score_obj
    return numeric, score_obj


def resolve_no_cot_model_and_config(
    base_model: Model,
    *,
    no_cot_model: str | None = None,
) -> tuple[Model, GenerateConfig]:
    """Resolve model override and thinking-disabled ``GenerateConfig``."""
    target_model = (
        get_model(no_cot_model, **gateway_openai_model_args(no_cot_model))
        if no_cot_model
        else base_model
    )

    name_lower = str(target_model).lower()
    extra_body: dict[str, Any] = dict(target_model.config.extra_body or {})
    temp: float | None = 0.0
    reasoning_effort: Literal["none"] | None = "none"

    if name_lower.startswith("mockllm/"):
        extra_body = {}
    elif name_lower.endswith("-nt"):
        # LiteLLM proxy ``-nt`` aliases already inject per-vendor non-thinking params;
        # avoid sending conflicting reasoning/temperature fields.
        temp = None
        reasoning_effort = None
    elif any(tag in name_lower for tag in ("kimi-k3", "deepseek-v4", "deepseek-r1")):
        extra_body["thinking"] = {"type": "disabled"}
        temp = None
    elif "qwen" in name_lower:
        extra_body["enable_thinking"] = False
        extra_body["reasoning"] = {"enabled": False}
    elif name_lower.startswith("openrouter/"):
        extra_body["reasoning"] = {"enabled": False}
    elif any(tag in name_lower for tag in ("/o1", "/o3", "/o4", "/gpt-5")):
        temp = None

    cfg = GenerateConfig(
        temperature=temp,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body if extra_body else None,
    )
    return target_model, cfg


def make_generate_with_config(
    *,
    model: Model,
    config: GenerateConfig,
) -> Generate:
    """Create a ``Generate`` callable usable inside upstream ``multiple_choice`` solvers."""

    async def _generate(
        state: TaskState,
        tool_calls: Literal["loop", "single", "none"] = "loop",
        **kwargs: Any,
    ) -> TaskState:
        del tool_calls, kwargs
        messages = list(state.messages)
        out = await model.generate(messages, config=config)
        state.messages = [*messages, out.message]
        state.output = out
        return state

    return cast(Generate, _generate)
