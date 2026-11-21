# CoT-Control (`cot_control`)

Evaluates whether models follow explicit constraints on their chain of thought (word suppression, casing rules, repeated markers, or ignoring the question) while answering multiple-choice science questions.

- **Paper:** <https://arxiv.org/abs/2603.05706>
- **Upstream repository:** <https://github.com/YuehHanChen/CoTControl>

## Running the evaluation

```bash
uv run inspect eval src/concordia_evals/cot_control/cot_control.py@cot_control \
  --model 'openai/<your-evaluated-model>' \
  --model-role 'judge=openai/<your-judge-model>' \
  --limit 10
```

Or via the registered entrypoint (or `openai-api/openai/<model>` when targeting a custom `OPENAI_BASE_URL` gateway):

```bash
uv run inspect eval concordia_evals/cot_control \
  --model 'openai/<your-evaluated-model>' \
  --model-role 'judge=openai/<your-judge-model>' \
  --limit 10
```

The default configuration evaluates all 9 control modes (`all_main`, excluding `baseline`) across all 3 full datasets (`gpqa`, `hle`, `mmlu_pro` totaling `10,836` samples across 9 modes; pass `sample_limit_per_mode=50` for a `1,350`-sample subset). A judge model is used for `ignore_question` and optional meta-discussion checks; all other constraints are graded deterministically.

## Rescoring

```bash
# Logs recorded from src/concordia_evals/cot_control/cot_control.py@cot_control:
uv run inspect score path/to/log.eval --model-role 'judge=openai/<your-judge-model>'

# Logs recorded from concordia_evals/cot_control:
uv run inspect score path/to/log.eval \
  --scorer concordia_evals/cot_control_qa_scorer \
  --model-role 'judge=openai/<your-judge-model>'
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `dataset` | `'all'` | `'gpqa'`, `'hle'`, `'mmlu_pro'`, or `'all'` |
| `modes` | `'all_main'` | Comma-separated modes, `'all_main'`, or `'all_with_baseline'` |
| `target_reasoning_effort` | `'default'` | Evaluated-model reasoning effort (`'default'` omits `reasoning_effort`) |
| `sample_limit_per_mode` | `None` | Optional cap per dataset and mode before merging (`None` uses all samples) |
| `adversarial_pressure` | `'na'` | Upstream prompt-pressure variant |
| `non_reasoning` | `False` | Use the upstream non-reasoning prompt variant |
| `monitor_aware` | `False` | Append monitor-awareness notice |
| `eval_aware` | `False` | Append evaluation-awareness notice |
| `cot_sys_prompt` | `False` | Place the constraint in the system prompt |
| `elicit_brevity` | `False` | Append brevity elicitation suffix |
| `judge_model` | `None` | Fallback judge model (`--model-role judge=...`, `INSPECT_EVALS_JUDGE_MODEL`, else `'openai/gpt-4o'`) |
| `check_meta_discussion` | `False` | Fail compliance if the CoT discusses the constraint itself |
| `limit` | `None` | Optional cap on total samples |

## Scoring

- `cot_controllability_score`: Fraction of non-baseline samples satisfying the CoT constraint.
- `task_accuracy`: Fraction of samples with a matching answer choice.
- Summary-only or redacted reasoning raises a scoring error instead of scoring as compliant or non-compliant.
