# Concordia Evals

This repository contains Inspect-native evaluations maintained by Concordia AI. Evaluations here are designed to be run with the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) framework.

## Relationship to `UKGovernmentBEIS/inspect_evals`

This repository is **not** part of the official [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) monorepo. Following the [Inspect Evals Register](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/EVAL_REGISTER.md) model, each evaluation is maintained independently here and registered with the official project via a small metadata file under `register/<eval_name>/eval.yaml` in `UKGovernmentBEIS/inspect_evals`. This keeps dependencies isolated while still making the evaluations discoverable through the official Inspect Evals documentation.

## Evaluations

| Evaluation | Description |
|------------|-------------|
| [BixBench](src/concordia_evals/bixbench) | Evaluates language model agents on bioinformatics data analysis tasks. Agents analyze biological datasets using Jupyter notebooks and answer open-ended research questions. |
| [Permission Awareness](src/concordia_evals/permission_awareness) | Measures whether coding agents condition risky shortcut-taking on the permission mode declared in the system prompt (human approval / hybrid / full-auto / control), across six validated scenarios with a fake approval layer. |
| [CyberGym-E2E](src/concordia_evals/cybergym_e2e) | Evaluates agents on the full vulnerability lifecycle — discovery, PoC generation, and patching — across 920 real-world vulnerabilities in 139 OSS projects, with four execution-based validation stages (S1-S4). Harness-agnostic via the shared [agents](src/concordia_evals/agents) layer (inspect_swe CLIs or native react). |

## Installation

```bash
git clone https://github.com/concordia-ai/concordia_evals.git
cd concordia_evals
uv sync
```

## Usage

Each evaluation can be run directly from its task file. For example:

```bash
uv run inspect eval src/concordia_evals/bixbench/bixbench.py@bixbench --model openai/gpt-4o
```

See the individual evaluation directories under `src/concordia_evals/` for detailed usage, parameters, and environment setup.
