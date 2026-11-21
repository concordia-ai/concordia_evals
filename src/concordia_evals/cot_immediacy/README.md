# CoT Immediacy (`cot_immediacy`)

Measures Chain-of-Thought legibility and coverage using the 0-4 rubric autorater from Appendix C of [A Pragmatic Way to Measure Chain-of-Thought Monitorability](https://arxiv.org/abs/2510.23966). Questions come from the `4,912`-sample `inspect_evals` Science QA suite by default (`2,785` ChemBench across all 9 subsets, `1,967` LAB-Bench across all 8 subsets including multimodal `FigQA` and `TableQA`, and `160` FrontierScience `olympic` + `research`), with optional support for `gpqa`, `hle`, and `mmlu_pro`.

## Running the evaluation

```bash
uv run inspect eval src/concordia_evals/cot_immediacy/cot_immediacy.py@cot_immediacy \
  --model 'openai/<your-evaluated-model>' \
  --model-role 'judge=openai/<your-judge-model>' \
  --limit 10
```

Or via the registered entrypoint:

```bash
uv run inspect eval concordia_evals/cot_immediacy \
  --model 'openai/<your-evaluated-model>' \
  --model-role 'judge=openai/<your-judge-model>' \
  --limit 10
```

## Reusing recorded CoT traces

Replay existing `.eval` logs with zero actor calls via `-T source_log=...`:

```bash
uv run inspect eval src/concordia_evals/cot_immediacy/cot_immediacy.py@cot_immediacy \
  -T dataset=chembench \
  -T source_log=/path/to/chembench_or_logs_dir \
  --model mockllm/model
```

Or rescore an existing `.eval` log directly via `inspect score`:

```bash
# Logs recorded from src/concordia_evals/cot_immediacy/cot_immediacy.py@cot_immediacy:
uv run inspect score "<your-source-log>.eval" \
  --model-role 'judge=openai/<your-judge-model>' \
  --action append --output-file "<your-rescored-log>.eval"

# Arbitrary or entrypoint .eval logs:
uv run inspect score "<your-source-log>.eval" \
  --scorer concordia_evals/cot_immediacy_judge_scorer \
  --model-role 'judge=openai/<your-judge-model>' \
  --action append --output-file "<your-rescored-log>.eval"
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `dataset` | `'all'` | `'all'`, `'chembench'`, `'lab_bench'`, `'frontierscience'`, `'gpqa'`, `'hle'`, or `'mmlu_pro'` |
| `target_reasoning_effort` | `'default'` | Evaluated-model reasoning effort (`'default'` omits `reasoning_effort`) |
| `judge_model` | `None` | Fallback judge (`--model-role judge=...`, `INSPECT_EVALS_JUDGE_MODEL`, `'openai/qwen3.8-27b'` when `OPENAI_BASE_URL` is non-empty, else `'openai/gpt-4o'`) |
| `correct_only` | `False` | Skip judging samples below `0.5` answer score (covers MCQ/Olympic and FrontierScience `research >= 0.5`) |
| `source_log` | `None` | Path to `.eval` file, directory, or per-model root (e.g. `eval_log`) for replaying recorded CoT traces (raises if 0 samples match) |
| `filter_to_source_log` | `True` | Restrict evaluation to samples matched in `source_log` |
| `frontierscience_format` | `'all'` | `'all'`, `'olympic'`, or `'research'` subset filter |
| `frontierscience_subjects` | `None` | Optional subject filter (comma-separated or list of `'biology'`, `'chemistry'`, `'physics'`, or `'all'`) |
| `chembench_task` | `None` | Optional ChemBench subset filter (`None` or `'all'` loads all 9 subsets) |
| `lab_bench_subset` | `None` | Optional LAB-Bench subset filter (`None` or `'all'` loads all 8 subsets) |
| `limit_per_dataset` | `None` | Optional cap per dataset before merging |
| `limit` | `None` | Optional cap on total samples |

## Scoring

- `legibility_pct`: Mean legibility score scaled to 0-100%.
- `coverage_pct`: Mean coverage score scaled to 0-100%.
- Standard non-Qwen judges use `temperature=0.0`, `-nt` judge endpoints leave proxy thinking settings unmodified, and reasoning Qwen judges such as `openai/qwen3.8-27b` run with `reasoning_effort="high"` (stripping `<think>...</think>` blocks before JSON extraction). Invalid/non-integer responses raise a scoring error.
