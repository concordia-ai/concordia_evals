# Concordia Evals

This repository contains Inspect-native evaluations maintained by Concordia AI. Evaluations here are designed to be run with the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) framework.

## Relationship to `UKGovernmentBEIS/inspect_evals`

This repository is **not** part of the official [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) monorepo. Following the [Inspect Evals Register](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/EVAL_REGISTER.md) model, each evaluation is maintained independently here and registered with the official project via a small metadata file under `register/<eval_name>/eval.yaml` in `UKGovernmentBEIS/inspect_evals`. This keeps dependencies isolated while still making the evaluations discoverable through the official Inspect Evals documentation.

## Evaluations

| Evaluation | Description |
|------------|-------------|
| [BixBench](src/concordia_evals/bixbench) | Evaluates language model agents on bioinformatics data analysis tasks. Agents analyze biological datasets using Jupyter notebooks and answer open-ended research questions. |
| [CoT-Control](src/concordia_evals/cot_control) | Evaluates chain-of-thought controllability across 9 constraint modes on `gpqa`, `hle`, and `mmlu_pro` (`10,836` samples total by default, or `1,350` with `-T sample_limit_per_mode=50`). |
| [CoT Process Monitorability](src/concordia_evals/cot_process) | Evaluates theorem-path monitorability across GPQA physics (`52`) and AIME math (`87`) problems (`139` samples total). |
| [Strict No-CoT Science QA](src/concordia_evals/no_cot) | Evaluates direct-answer accuracy without chain-of-thought reasoning across all 9 ChemBench subsets (`2,785`), all 8 LAB-Bench subsets (`1,967`, including multimodal `FigQA`/`TableQA`), and FrontierScience (`160`) (`4,912` samples total by default; filterable via `chembench_task`, `lab_bench_subset`, and `frontierscience_subjects`), with optional paired CoT uplift. |
| [CoT Immediacy](src/concordia_evals/cot_immediacy) | Evaluates Chain-of-Thought legibility and coverage across the `4,912`-sample Science QA suite by default (filterable via `chembench_task`, `lab_bench_subset`, and `frontierscience_subjects`) using the DeepMind autorater rubric. |

## Installation

```bash
git clone https://github.com/concordia-ai/concordia_evals.git
cd concordia_evals
uv sync
```

## Usage

Each evaluation can be run directly from its task file or via the registered `concordia_evals/<task>` entrypoint. For example:

```bash
uv run inspect eval src/concordia_evals/bixbench/bixbench.py@bixbench --model openai/gpt-4o
uv run inspect eval src/concordia_evals/cot_control/cot_control.py@cot_control --model openai/gpt-4o
uv run inspect eval src/concordia_evals/cot_process/cot_process.py@cot_process --model openai/gpt-4o
uv run inspect eval src/concordia_evals/no_cot/no_cot.py@no_cot --model openai/gpt-4o
uv run inspect eval src/concordia_evals/cot_immediacy/cot_immediacy.py@cot_immediacy --model openai/gpt-4o
```

### Judge / Monitor Resolution

Judges (`cot_control`, `no_cot`, `cot_immediacy`) and monitors (`cot_process`) resolve in the following order:

1. Inspect role override: `--model-role judge=<model>` (or `--model-role monitor=<model>` for `cot_process`).
2. Explicit task parameter (`-T judge_model=...`, `-T monitor_model=...`, `-T grader_model=...`).
3. `INSPECT_EVALS_JUDGE_MODEL` environment variable (when set and non-empty).
4. `openai/qwen3.8-27b` in `cot_immediacy` when `OPENAI_BASE_URL` is non-empty, otherwise portable default `openai/gpt-4o`.

### Offline Mode and Asset Cache

- Set `CONCORDIA_EVALS_CACHE_DIR=/abs/path/to/cache` to store downloaded archives cleanly outside the package tree.
- Set `CONCORDIA_EVALS_OFFLINE=1` (or `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1`) to run strictly offline against pre-populated local assets and Hugging Face caches.

### Rescoring `.eval` Logs

- Logs produced from a task file (`src/concordia_evals/<task>/<task>.py@<task>`) auto-discover their scorer via `uv run inspect score path/to/log.eval`.
- Logs produced from a registered entrypoint (`concordia_evals/<task>`) pass `--scorer concordia_evals/<scorer>` (for example `--scorer concordia_evals/cot_immediacy_judge_scorer`).

See the individual evaluation directories under `src/concordia_evals/` for detailed usage, parameters, and environment setup.
