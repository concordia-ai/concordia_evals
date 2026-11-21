# CoT Process Monitorability (`cot_process`)

Measures whether a process monitor can detect a required theorem path in a model's chain of thought while solving GPQA physics (`52` samples) and AIME math (`87` samples) problems (`139` samples total).

- **Paper:** <https://arxiv.org/abs/2512.18311>
- **Upstream repository:** <https://github.com/openai/monitorability-evals>

## Running the evaluation

```bash
uv run inspect eval src/concordia_evals/cot_process/cot_process.py@cot_process \
  --model 'openai/<your-evaluated-model>' \
  --model-role 'monitor=openai/<your-monitor-model>' \
  --limit 10
```

Or via the registered entrypoint:

```bash
uv run inspect eval concordia_evals/cot_process \
  --model 'openai/<your-evaluated-model>' \
  --model-role 'monitor=openai/<your-monitor-model>' \
  --limit 10
```

## Rescoring

```bash
# Logs recorded from src/concordia_evals/cot_process/cot_process.py@cot_process:
uv run inspect score path/to/log.eval --model-role 'monitor=openai/<your-monitor-model>'

# Logs recorded from concordia_evals/cot_process:
uv run inspect score path/to/log.eval \
  --scorer concordia_evals/openai_process_scorer \
  --model-role 'monitor=openai/<your-monitor-model>'
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `dataset` | `'all'` | `'all'`, `'gpqa_physics'`, or `'aime_math'` |
| `target_reasoning_effort` | `'default'` | Evaluated-model reasoning effort (`'default'` omits `reasoning_effort`) |
| `monitor_model` | `None` | Fallback monitor model (`--model-role monitor=...`, `INSPECT_EVALS_JUDGE_MODEL`, else `'openai/gpt-4o'`) |
| `scopes` | `'cot_only'` | Comma-separated upstream monitor scopes |
| `limit` | `None` | Optional cap on total samples |

## Scoring

- `process_gmean2_cot` (and `process_gmean2_answer_only` / `process_gmean2_all_messages` when non-default `scopes` are configured): Process monitorability TPR ``P(Z=1 | Y=1)`` across correct rollouts.
- `process_answer_accuracy`: Fraction of samples with a correct final answer.
- AIME integer and ``\boxed{...}`` answers are matched deterministically before falling back to the LLM answer checker, and monitors matching either a full theorem path or every theorem joined by ``&`` count as faithful.
