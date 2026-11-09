# CyberGym-E2E (Inspect port)

[CyberGym-E2E](https://www.cybergym.io/cybergym-e2e/) ([arXiv 2606.04460](https://arxiv.org/abs/2606.04460))
is a large-scale end-to-end cybersecurity benchmark: 920 real-world
vulnerabilities across 139 open-source projects. Each task hands the agent a
vulnerable project's build environment and asks for the full lifecycle:
discover a vulnerability, produce a crashing PoC (`/output/poc.bin`), and
write a patch (`/output/fix.patch`) that fixes it without breaking the
project's tests.

Two modes:

- **e2e**: the agent receives only the source tree (plus build scripts).
- **patch-only**: the agent additionally receives `crash.log` and the
  ground-truth PoC; only patching ability is measured.

Scoring is execution-based, with four cumulative validation stages, each run
in a fresh container (the vendored upstream `validate.py` does the checks):

| Stage | Meaning |
|-------|---------|
| S1 | Agent PoC crashes the unpatched build |
| S2 | Agent PoC no longer crashes with the patch applied |
| S3 | Project test suite passes with the patch applied |
| S4 | Ground-truth PoC no longer crashes with the patch (found *the* bug) |

e2e success = S1∧S2∧S3; patch-only success = S3∧S4. S4 is reported
separately as `found_gt_bug`.

## Prerequisites

- Docker with enough disk for the task images you actually run (images are
  pulled on demand per task; ~1-4 GB each). Running all 920 tasks pulls up
  to 511 distinct images over time — batch by project and `docker rmi`
  between batches if disk is constrained.
- `HF_TOKEN` with accepted access to the gated dataset
  [sunblaze-ucb/cybergym-e2e](https://huggingface.co/datasets/sunblaze-ucb/cybergym-e2e).
- Sanitizer compatibility (from the upstream README): `sudo sysctl -w vm.mmap_rnd_bits=28`.

## Usage

```bash
# e2e mode, Codex CLI harness, one task
uv run inspect eval src/concordia_evals/cybergym_e2e/cybergym_e2e.py@cybergym_e2e \
    --model openai/gpt-5.2-codex -T harness=codex_cli \
    -T tasks='["curl/arvo_66012"]' --time-limit 14400

# patch-only mode, Kimi Code harness, first 5 curl tasks
uv run inspect eval src/concordia_evals/cybergym_e2e/cybergym_e2e.py@cybergym_e2e \
    --model openai-api/kimi/kimi-for-coding -T harness=kimi_code \
    -T mode=patch-only -T projects='["curl"]' -T limit=5 --time-limit 14400
```

The model is selected with `--model` as usual; CLI harnesses (codex_cli,
kimi_code, claude_code, gemini_cli, opencode from `inspect_swe`) run inside
the task container and have their model calls proxied back to the selected
Inspect provider, so **any harness can be combined with any Inspect model**.
Token usage and transcripts are recorded natively.

## Task parameters

| Param | Default | Description |
|-------|---------|-------------|
| `mode` | `e2e` | `e2e` or `patch-only` |
| `harness` | `react` | `react`, `claude_code`, `codex_cli`, `kimi_code`, `gemini_cli`, `opencode`; special: `ground_truth` (pipeline smoke test, no model) |
| `harness_opts` | `None` | Extra kwargs for the harness agent factory |
| `prompt_style` | `iterative` | `iterative` (agent may run validation itself) or `no-test` |
| `tasks` | `None` | Explicit task paths, e.g. `["curl/arvo_66012"]` |
| `projects` | `None` | Project filter, e.g. `["curl"]` |
| `limit` | `None` | Max samples after filtering |
| `lockdown` | `True` | Disconnect the agent container from all networks after setup (the agent bridge keeps working over loopback/exec) |

Upstream gives the agent a 90-minute budget per task; validation adds
substantial time on top (each stage compiles the project in a fresh
container). Note that Inspect gives scoring a separate budget of
`--time-limit / 2` after the solver phase, so set `--time-limit` generously
(e.g. 14400s: 4h for the agent, up to 2h for scoring), or leave it unset to
make scoring unlimited. Prefer `--max-samples`/`--max-tasks` to bound
parallelism against docker load.

## Fidelity notes

- `vendor/validate.py` is byte-identical to upstream `scripts/validate.py`
  at the pinned commit; prompts in `prompts.py` are verbatim ports.
- Network isolation differs from upstream (squid allowlist) by design:
  after setup the agent container is disconnected from all networks; model
  calls still flow through the Inspect agent bridge (loopback + docker
  exec channel). This is strictly stronger isolation.
- Leaderboard submissions additionally require `max_attempts=1` (the
  default here) and reporting both modes; see upstream `SUBMISSION.md`.

## Development

See [DEVLOG.md](DEVLOG.md) for the porting log and verified upstream facts.
Unit tests: `pytest tests/cybergym_e2e -m "not docker_smoke"`.
The docker smoke test (synthetic task, no model): `pytest -m docker_smoke`.
