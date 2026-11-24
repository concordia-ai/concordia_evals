"""Tests for the monitoring-platform composite score and its report grouping."""

from types import SimpleNamespace

import pytest
from inspect_ai.scorer import SampleScore, Score

from concordia_evals.cybergym_e2e.cybergym_e2e import sample_modes
from concordia_evals.cybergym_e2e.report import _success_accuracy, composite_rows
from concordia_evals.cybergym_e2e.scorer import (
    E2E_SCORE_WEIGHT,
    PATCH_ONLY_SCORE_WEIGHT,
    capability_accuracy,
    composite_weighted,
    cybergym_e2e_scorer,
    cybergym_e2e_scorer_both,
    e2e_accuracy,
    patch_only_accuracy,
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
    def _ss(mode: str, value: float, task: str = "t1", infra: int = 0) -> SampleScore:
        return SampleScore(
            score=Score(value={"success": value, "infra_error": infra}),
            sample_metadata={"mode": mode, "task_path": task},
        )

    def test_combines_paired_tasks_with_weights(self):
        m = composite_weighted()
        scores = [
            self._ss("e2e", 1.0, "t1"),
            self._ss("e2e", 0.0, "t2"),
            self._ss("patch-only", 1.0, "t1"),
            self._ss("patch-only", 1.0, "t2"),
        ]
        assert m(scores) == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)

    def test_unpaired_task_is_excluded(self):
        m = composite_weighted()
        scores = [
            self._ss("e2e", 1.0, "t1"),
            self._ss("e2e", 0.0, "t2"),  # no patch-only partner
            self._ss("patch-only", 1.0, "t1"),
        ]
        assert m(scores) == pytest.approx(0.6 * 1.0 + 0.4 * 1.0)

    def test_infra_error_samples_are_excluded(self):
        m = composite_weighted()
        scores = [
            self._ss("e2e", 1.0, "t1"),
            self._ss("e2e", 0.0, "t2", infra=1),
            self._ss("patch-only", 1.0, "t1"),
            self._ss("patch-only", 0.0, "t2", infra=1),
        ]
        # only t1 pair remains
        assert m(scores) == pytest.approx(1.0)

    def test_no_pairs_raises(self):
        m = composite_weighted()
        with pytest.raises(ValueError):
            m([self._ss("e2e", 1.0, "t1")])


class TestCapabilityMetrics:
    def test_capability_accuracy_excludes_infra(self):
        m = capability_accuracy()
        scores = [
            TestCompositeWeightedMetric._ss("e2e", 1.0, "t1"),
            TestCompositeWeightedMetric._ss("e2e", 0.0, "t2", infra=1),
            TestCompositeWeightedMetric._ss("e2e", 0.0, "t3"),
        ]
        assert m(scores) == pytest.approx(0.5)

    def test_e2e_and_patch_only_breakdown(self):
        e2e_m = e2e_accuracy()
        patch_m = patch_only_accuracy()
        scores = [
            TestCompositeWeightedMetric._ss("e2e", 1.0, "t1"),
            TestCompositeWeightedMetric._ss("e2e", 0.0, "t2"),
            TestCompositeWeightedMetric._ss("patch-only", 1.0, "t1"),
        ]
        assert e2e_m(scores) == pytest.approx(0.5)
        assert patch_m(scores) == pytest.approx(1.0)


class TestScorerRegistration:
    def test_both_variants_registered_at_import(self):
        from inspect_ai._util.registry import registry_info
        from inspect_ai.scorer._scorer import SCORER_METRICS

        s_default = cybergym_e2e_scorer()
        s_both = cybergym_e2e_scorer_both()
        assert s_default is not s_both  # distinct registry-tagged wrappers

        m_default = registry_info(s_default).metadata[SCORER_METRICS]
        m_both = registry_info(s_both).metadata[SCORER_METRICS]
        assert len(m_default["success"]) == 2  # capability_accuracy + stderr
        assert len(m_both["success"]) == 5  # + e2e/patch-only breakdown + composite

        for key in ("s1", "s4", "found_gt_bug"):
            assert key in m_default and key in m_both

    def test_bare_names_resolvable_by_inspect_score(self):
        """Logs record unqualified names; fresh processes must resolve them.

        The @scorer decorator registers package-scoped names
        ("concordia_evals/..."); `inspect score` looks up the bare name
        recorded in the log, so module import must alias the bare names.
        """
        from inspect_ai._eval.loader import scorer_from_registry

        assert callable(scorer_from_registry("cybergym_e2e_scorer"))
        assert callable(scorer_from_registry("cybergym_e2e_scorer_both"))


class TestSampleModes:
    def test_single_modes(self):
        assert sample_modes("e2e") == ["e2e"]
        assert sample_modes("patch-only") == ["patch-only"]

    def test_both_paired_modes(self):
        assert sample_modes("both") == ["e2e", "patch-only"]
