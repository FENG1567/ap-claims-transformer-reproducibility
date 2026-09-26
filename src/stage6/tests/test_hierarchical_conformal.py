import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

from fit_hierarchical_conformal import (
    finite_sample_quantile,
    mondrian_risk_sets,
    risk_sets,
    validate_operating_point,
)


def test_finite_sample_quantile_uses_ceil_n_plus_one_rank():
    scores = np.arange(1, 11) / 10
    # ceil((10+1)*0.8)=9 -> ninth order statistic.
    assert finite_sample_quantile(scores, 0.2) == 0.9


def test_risk_set_never_empty_and_only_expands_by_argmax():
    p = np.array([[0.6, 0.3, 0.1], [0.34, 0.33, 0.33]])
    sets = risk_sets(p, q=0.2)
    assert sets.sum(axis=1).tolist() == [1, 1]
    assert sets[0, 0] and sets[1, 0]


def test_mondrian_q_changes_the_corresponding_group_output():
    p = np.tile(np.array([[0.6, 0.2, 0.1, 0.07, 0.03]]), (4, 1))
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    groups = np.array(["low", "low", "high", "high"])

    sets, entries = mondrian_risk_sets(
        p, scores, groups, alpha=0.20, global_q=0.50, min_rows=2
    )

    assert entries["low"]["q"] == 0.2
    assert entries["high"]["q"] == 0.9
    assert entries["low"]["status"] == "CALIBRATED"
    assert entries["high"]["status"] == "CALIBRATED"
    assert not np.array_equal(sets[0], sets[2])
    assert sets[0].tolist() == [True, False, False, False, False]
    assert sets[2].tolist() == [True, True, True, False, False]


def test_mondrian_small_group_uses_global_q_and_is_audited():
    p = np.tile(np.array([[0.6, 0.2, 0.1, 0.07, 0.03]]), (5, 1))
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
    groups = np.array(["small", "small", "large", "large", "large"])

    sets, entries = mondrian_risk_sets(
        p, scores, groups, alpha=0.20, global_q=0.50, min_rows=3
    )

    assert entries["small"] == {
        "n": 2,
        "q": 0.5,
        "status": "GLOBAL_FALLBACK_SMALL_CALIBRATION_GROUP",
    }
    assert np.array_equal(sets[:2], risk_sets(p[:2], q=0.5))
    assert entries["large"]["status"] == "CALIBRATED"


def test_operating_point_is_hash_bound_and_2021a_only(tmp_path):
    threshold = tmp_path / "transformer_operating_thresholds_2021A.json"
    threshold.write_text(json.dumps({
        "any_readmission": {
            "probability_column": "p_any_readmission_calibrated",
            "threshold": 0.2,
        }
    }), encoding="utf-8")
    digest = hashlib.sha256(threshold.read_bytes()).hexdigest()
    manifest = {
        "status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "artifacts": {threshold.name: {"bytes": threshold.stat().st_size, "sha256": digest}},
    }
    path = tmp_path / "transformer_operating_point_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_operating_point(tmp_path)["threshold"] == 0.2
    manifest["2021B_accessed"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not sealed"):
        validate_operating_point(tmp_path)
