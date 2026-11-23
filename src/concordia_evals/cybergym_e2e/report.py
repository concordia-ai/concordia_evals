"""Composite weighted security score report for the monitoring platform.

Reads .eval logs (or directories of them), groups runs by (model, harness),
and prints the e2e / patch-only success accuracy plus the composite
``weighted_security_score`` for each group.

Usage:
    uv run python -m concordia_evals.cybergym_e2e.report LOG [LOG ...]
"""

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, read_eval_log

from concordia_evals.cybergym_e2e.scorer import weighted_security_score

MODES = ("e2e", "patch-only")


@dataclass
class CompositeRow:
    model: str
    harness: str
    e2e_runs: int
    e2e_success: float | None
    patch_runs: int
    patch_only_success: float | None

    @property
    def composite(self) -> float | None:
        if self.e2e_success is None or self.patch_only_success is None:
            return None
        return weighted_security_score(self.e2e_success, self.patch_only_success)


def _success_accuracy(eval_log: EvalLog) -> float:
    results = eval_log.results
    for score in results.scores if results else []:
        if score.name == "success":
            metric = score.metrics.get("accuracy")
            if metric is not None:
                return float(metric.value)
    raise ValueError("no success/accuracy metric found in log results")


def composite_rows(logs: Iterable[EvalLog]) -> list[CompositeRow]:
    """Group logs by (model, harness) and average success accuracy per mode."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for eval_log in logs:
        task_args: dict[str, Any] = eval_log.eval.task_args or {}
        model = str(eval_log.eval.model)
        harness = str(task_args.get("harness", "?"))
        mode = str(task_args.get("mode", "e2e"))
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r} in log for model {model}")
        groups[(model, harness)][mode].append(_success_accuracy(eval_log))

    rows = []
    for (model, harness), by_mode in sorted(groups.items()):
        e2e_vals = by_mode.get("e2e", [])
        patch_vals = by_mode.get("patch-only", [])
        rows.append(
            CompositeRow(
                model=model,
                harness=harness,
                e2e_runs=len(e2e_vals),
                e2e_success=sum(e2e_vals) / len(e2e_vals) if e2e_vals else None,
                patch_runs=len(patch_vals),
                patch_only_success=sum(patch_vals) / len(patch_vals) if patch_vals else None,
            )
        )
    return rows


def _expand_paths(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.eval")))
        else:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help=".eval files or directories of them")
    args = parser.parse_args()

    logs = []
    for f in _expand_paths(args.logs):
        eval_log = read_eval_log(f)
        if eval_log.results is None:
            print(f"warning: skipping {f} (no results — run did not finish)", file=sys.stderr)
            continue
        logs.append(eval_log)
    rows = composite_rows(logs)

    header = f"{'model':<40} {'harness':<12} {'e2e':>8} {'patch-only':>12} {'composite':>10}  runs"
    print(header)
    print("-" * len(header))

    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "-"

    for row in rows:
        print(
            f"{row.model:<40} {row.harness:<12} {fmt(row.e2e_success):>8} "
            f"{fmt(row.patch_only_success):>12} {fmt(row.composite):>10}  "
            f"e2e:{row.e2e_runs} patch:{row.patch_runs}"
        )


if __name__ == "__main__":
    main()
