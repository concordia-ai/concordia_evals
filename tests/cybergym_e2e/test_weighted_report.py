"""Tests for the monitoring-platform composite score and its report grouping."""

from types import SimpleNamespace

import pytest
from inspect_ai.scorer import SampleScore, Score

from concordia_evals.cybergym_e2e.cybergym_e2e import sample_modes
from concordia_evals.cybergym_e2e.report import _success_accuracy, composite_rows
from concordia_evals.cybergym_e2e.scorer import (
    E2E_SCORE_WEIGHT,
    PATCH_ONLY_SCORE_WEIGHT,
    composite_weighted,
    weighted_security_score,
)


def _log(model: str, harness: str, mode: str, success: float):
    score = SimpleNamespace(name="success", metrics={"accuracy": SimpleNamespace(value=success)})
    results = SimpleNamespace(scores=[score])
    eval_info = SimpleNamespace(model=model, task_args={"harness": harness, "mode": mode})
    return SimpleNamespace(eval=eval_info, results=results)


class TestWeightedSecurityScore:
    def test_weights(self):
        assert E2E_SCORE_WEIGHT == 0.6
        assert PATCH_ONLY_SCORE_WEIGHT == 0.4

    def test_full_e2e_zero_patch(self):
        assert weighted_security_score(1.0, 0.0) == pytest.approx(0.6)

    def test_zero_e2e_full_patch(self):
        assert weighted_security_score(0.0, 1.0) == pytest.approx(0.4)

    def test_both_half(self):
        assert weighted_security_score(0.5, 0.5) == pytest.approx(0.5)

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            weighted_security_score(1.1, 0.5)
        with pytest.raises(ValueError):
            weighted_security_score(0.5, -0.01)


class TestSuccessAccuracy:
    def test_extracts_success_accuracy(self):
        assert _success_accuracy(_log("m", "h", "e2e", 0.75)) == 0.75

    def test_missing_metric_raises(self):
        log = _log("m", "h", "e2e", 1.0)
        log.results.scores = [SimpleNamespace(name="s3", metrics={})]
        with pytest.raises(ValueError):
            _success_accuracy(log)


class TestCompositeRows:
    def test_single_group_both_modes(self):
        rows = composite_rows(
            [
                _log("m1", "kimi_code", "e2e", 1.0),
                _log("m1", "kimi_code", "patch-only", 1.0),
            ]
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.model == "m1"
        assert row.harness == "kimi_code"
        assert row.e2e_success == 1.0
        assert row.patch_only_success == 1.0
        assert row.composite == pytest.approx(1.0)

    def test_missing_mode_has_no_composite(self):
        rows = composite_rows([_log("m1", "kimi_code", "e2e", 1.0)])
        assert rows[0].composite is None
        assert rows[0].patch_only_success is None
        assert rows[0].e2e_success == 1.0

    def test_duplicate_runs_are_averaged(self):
        rows = composite_rows(
            [
                _log("m1", "kimi_code", "e2e", 1.0),
                _log("m1", "kimi_code", "e2e", 0.0),
                _log("m1", "kimi_code", "patch-only", 1.0),
            ]
        )
        row = rows[0]
        assert row.e2e_runs == 2
        assert row.e2e_success == 0.5
        assert row.composite == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)

    def test_groups_are_split_by_model_and_harness(self):
        rows = composite_rows(
            [
                _log("m1", "kimi_code", "e2e", 1.0),
                _log("m1", "codex_cli", "e2e", 0.0),
                _log("m2", "kimi_code", "e2e", 1.0),
            ]
        )
        assert len(rows) == 3
        assert {(r.model, r.harness) for r in rows} == {
            ("m1", "kimi_code"),
            ("m1", "codex_cli"),
            ("m2", "kimi_code"),
        }

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            composite_rows([_log("m1", "h", "bogus", 1.0)])


class TestCompositeWeightedMetric:
    @staticmethod
    def _ss(mode: str, value: float) -> SampleScore:
        return SampleScore(score=Score(value=value), sample_metadata={"mode": mode})

    def test_combines_modes_with_weights(self):
        m = composite_weighted()
        scores = [
            self._ss("e2e", 1.0),
            self._ss("e2e", 0.0),
            self._ss("patch-only", 1.0),
        ]
        assert m(scores) == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)

    def test_missing_mode_raises(self):
        m = composite_weighted()
        with pytest.raises(ValueError):
            m([self._ss("e2e", 1.0)])


class TestSampleModes:
    def test_single_modes(self):
        assert sample_modes("e2e") == ["e2e"]
        assert sample_modes("patch-only") == ["patch-only"]

    def test_both_paired_modes(self):
        assert sample_modes("both") == ["e2e", "patch-only"]
