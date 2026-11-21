"""Concordia Inspect-native evaluations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from concordia_evals.bixbench import bixbench
    from concordia_evals.cot_control import cot_control
    from concordia_evals.cot_immediacy import cot_immediacy
    from concordia_evals.cot_process import cot_process
    from concordia_evals.no_cot import no_cot

__version__ = "0.1.0"
__all__ = [
    "bixbench",
    "cot_control",
    "cot_immediacy",
    "cot_process",
    "no_cot",
]

_TASK_MODULES = {
    "bixbench": "concordia_evals.bixbench.bixbench",
    "cot_control": "concordia_evals.cot_control.cot_control",
    "cot_immediacy": "concordia_evals.cot_immediacy.cot_immediacy",
    "cot_process": "concordia_evals.cot_process.cot_process",
    "no_cot": "concordia_evals.no_cot.no_cot",
}


def __getattr__(name: str) -> Any:
    if name in _TASK_MODULES:
        module = import_module(_TASK_MODULES[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
