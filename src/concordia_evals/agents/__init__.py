"""Shared agent (harness) layer for Concordia evals.

Benchmark-agnostic building blocks: a registry mapping harness names to
Inspect agents (wrapping inspect_swe and inspect's native react agent),
and solvers for network lockdown of agent sandboxes.
"""
