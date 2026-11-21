# Strict No-CoT Science QA (`no_cot`)

Evaluates direct-answer accuracy without chain-of-thought reasoning across `4,912` Science QA samples from `inspect_evals` by default:

- **ChemBench** (all 9 subsets, `2,785` samples including freeform numeric `mae` and MCQ; filterable via `chembench_task`) via `inspect_evals.chembench`
- **LAB-Bench** (all 8 subsets, `1,967` samples including multimodal `FigQA` and `TableQA`; filterable via `lab_bench_subset`) via `inspect_evals.lab_bench`
- **FrontierScience** (`olympic` `100` + `research` `60` = `160` samples; filterable via `frontierscience_subjects`) via `inspect_evals.frontierscience`

Upstream `"Think step by step..."` and derivation directives are replaced in-place with strict no-reasoning instructions inside the single user prompt (preserving upstream prompt structure, formatting requirements at the end, and domain phrases such as `"step-by-step protocol"`, with no extra system message), paired with non-thinking model/config overrides (`enable_thinking: false`, `thinking: disabled`, `reasoning.enabled: false`, `reasoning_effort="none"`, plus optional `-T no_cot_model=...` alias override), and audited against hidden reasoning blocks, `reasoning_tokens > 0`, visible CoT markers, and output-length caps.

## Running the evaluation

```bash
uv run inspect eval src/concordia_evals/no_cot/no_cot.py@no_cot \
  --model 'openai/<your-evaluated-model>' \
  --model-role 'judge=openai/<your-judge-model>' \
  --limit 10
```

Or via the registered entrypoint:

```bash
uv run inspect eval concordia_evals/no_cot \
  --model 'openai/<your-evaluated-model>' \
  --model-role 'judge=openai/<your-judge-model>' \
  --limit 10
```

To compute paired uplift (`Acc(CoT) - Acc(no_cot)`) while reusing recorded CoT `.eval` logs:

```bash
uv run inspect eval src/concordia_evals/no_cot/no_cot.py@no_cot \
  --model 'openai/<your-evaluated-model>' \
  -T cot_log=/path/to/cot_logs_dir
```

## Rescoring

- Task-file logs auto-discover the recorded scorer via `uv run inspect score path/to/log.eval`.
- Entrypoint logs specify one of `--scorer concordia_evals/no_cot_scorer`, `--scorer concordia_evals/no_cot_paired_scorer`, or `--scorer concordia_evals/cot_arm_scorer`.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `dataset` | `'all'` | `'all'`, `'chembench'`, `'lab_bench'`, or `'frontierscience'` |
| `mode` | `'no_cot'` | `'no_cot'`, `'cot'`, or `'paired'` |
| `frontierscience_format` | `'all'` | `'all'`, `'olympic'`, or `'research'` subset filter |
| `frontierscience_subjects` | `None` | Optional subject filter (comma-separated or list of `'biology'`, `'chemistry'`, `'physics'`, or `'all'`) |
| `chembench_task` | `None` | Optional ChemBench subset filter (`None` or `'all'` loads all 9 subsets) |
| `lab_bench_subset` | `None` | Optional LAB-Bench subset filter (`None` or `'all'` loads all 8 subsets) |
| `limit_per_dataset` | `None` | Optional cap per dataset before merging |
| `grader_model` | `None` | FrontierScience rubric grader (`--model-role judge=...`, `INSPECT_EVALS_JUDGE_MODEL`, else `'openai/gpt-4o'`) |
| `no_cot_model` | `None` | Explicit non-thinking model override for the no-CoT arm |
| `cot_log` | `None` | Recorded `.eval` file, directory, or per-model root (e.g. `eval_log`) for paired CoT reuse (raises if 0 samples match) |
| `filter_to_cot_log` | `True` | Restrict evaluation to samples matched in `cot_log` (`False` skips live CoT calls on unmatched samples) |
| `strict_no_cot` | `True` | Zero out accuracy when no-CoT compliance fails |
| `max_answer_chars` | `None` | Optional character ceiling override for compliance checks |
| `limit` | `None` | Optional cap on total samples |

## Scoring

- **No-CoT mode (`mode='no_cot'`)**: reports `no_cot_accuracy` and `no_cot_compliance_rate`.
- **Paired mode (`mode='paired'` or `cot_log=...`)**: reports `cot_accuracy`, `no_cot_accuracy`, `cot_uplift`, and `no_cot_compliance_rate`.
- **CoT mode (`mode='cot'`)**: reports `cot_accuracy` and `cot_mean_reasoning_tokens`.
- **FrontierScience grading**: `olympic` and `research` (`[0, 1]` rubric score, where `>= 0.5` is treated as passing in binary checks) use the resolved judge; unparseable judge responses omit `accuracy` / `raw_accuracy` from the score mapping so metric averages skip them cleanly while preserving strict RFC-8259 JSON values (non-compliant strict no-CoT outputs still score `0.0`).
