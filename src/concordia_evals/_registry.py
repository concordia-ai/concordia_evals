# ruff: noqa: F401

# Importing the task modules registers them as `concordia_evals/<task>`,
# which lets inspect load them by name (e.g. `concordia_evals/bountybench`).
from concordia_evals.bountybench import bountybench
from concordia_evals.exploitbench import exploitbench
from concordia_evals.exploitgym import exploitgym
from concordia_evals.web_exploit_bench import web_exploit_bench
