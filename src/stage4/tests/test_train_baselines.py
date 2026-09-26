import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_baselines import (
    apply_calibrator,
    assess_prediction_validity,
    capacity_constrained_flags,
    engineer_static,
    fit_token_map,
    token_matrix,
)


def test_token_map_is_development_frequency_filtered_and_special_free():
    s = pd.Series([[0, 1, 2, 3, 4, 5], [4, 6], [4, 5]])
    mapping = fit_token_map(s, min_frequency=2)
    assert set(mapping) == {4, 5}
    x = token_matrix(s, mapping)
    assert x.shape == (3, 2)
    assert x.nnz == 5


def test_prday_after_los_is_invalid_not_real_timing():
    frame = pd.DataFrame(
        {
            "year": [2020], "AGE": [50], "LOS": [2], "I10_NDX": [2], "I10_NPR": [4],
            "cost_2021_usd": [100.0], "TOTAL_DISC": [1000], "N_DISC_U": [1], "N_HOSP_U": [1],
            "S_DISC_U": [1], "S_HOSP_U": [1], "CCR_NRD": [0.4], "WAGEINDEX": [1.0],
            "dx_tokens": [[4, 2, 0]], "pr_tokens": [[4, 5, 6, 3]], "prday": [[0, 3, -99, -99]],
            "prior_count_ytd": [0], "prior_count_30d": [0], "prior_count_90d": [0],
            "prior_count_180d": [0], "prior_ed_count_180d": [0],
            "prior_nonelective_count_180d": [0], "prior_ap_count_180d": [0],
            "prior_biliary_count_180d": [0], "prior_sepsis_or_organ_count_180d": [0],
            "prior_los_sum_180d": [0], "prior_max_severity_180d": [0],
            "prior_max_mortality_risk_180d": [0], "days_since_prior_discharge": [-1],
            "history_30d_fully_observable": [True], "history_90d_fully_observable": [True],
            "history_180d_fully_observable": [False],
            **{name: [1] for name in ["AWEEKEND", "DISPUNIFORM", "DMONTH", "DRG", "DRGVER",
                "ELECTIVE", "FEMALE", "HCUP_ED", "MDC", "PAY1", "PL_NCHS", "RESIDENT",
                "ZIPINC_QRTL", "APRDRG", "APRDRG_Risk_Mortality", "APRDRG_Severity",
                "HOSP_BEDSIZE", "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH"]},
        }
    )
    x = engineer_static(frame)
    assert x.loc[0, "prday_day0_count"] == 1
    assert x.loc[0, "prday_invalid_after_discharge_count"] == 1
    assert x.loc[0, "prday_missing_count"] == 1


def test_none_calibrator_is_identity_with_safe_clipping():
    p = np.array([0.0, 0.2, 1.0])
    out = apply_calibrator("none", None, p)
    assert np.allclose(out, [1e-7, 0.2, 1 - 1e-7])


def test_invalid_or_inverse_elastic_candidate_cannot_be_selected():
    # The former unstable SGD run could emit finite but backwards rankings with
    # extremely large coefficients.  Both a non-converged state and inverse
    # ranking must keep such a candidate out of model selection.
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    inverse_risk = np.array([0.9, 0.8, 0.2, 0.1])
    report = assess_prediction_validity(
        y, inverse_risk, converged=False, coefficients=np.array([[148.0, -172.0]])
    )
    assert report["finite_predictions"]
    assert not report["ranking_direction_plausible"]
    assert not report["valid_for_selection"]
    assert "optimizer_not_converged" in report["rejection_reasons"]
    assert "inverse_or_undefined_2021A_ranking" in report["rejection_reasons"]


def test_capacity_rule_is_exact_and_deterministic_when_every_score_ties():
    risk = np.full(10, 0.12)
    hashes = np.array([9, 3, 8, 2, 6, 1, 10, 5, 7, 4], dtype=np.uint64)
    flagged, report = capacity_constrained_flags(risk, hashes, capacity_fraction=0.20)
    assert flagged.sum() == 2
    assert set(hashes[flagged]) == {1, 2}
    assert report["capacity_rows_requested"] == 2
    assert report["capacity_fraction_realized"] == 0.20
    assert report["boundary_tie_rows"] == 10
    assert report["boundary_tie_rows_selected"] == 2
