import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_ap_history import _clean_tokens, derive_patient_history


def test_clean_tokens_accepts_arrow_style_numpy_arrays():
    assert _clean_tokens(np.array([0, 3, 4, 5], dtype=np.int32)) == {4, 5}
    assert _clean_tokens(np.array([], dtype=np.int32)) == set()


def test_history_is_strictly_prior_nonoverlapping_and_windowed():
    admissions = pd.DataFrame(
        {
            "encounter_hash": [1, 2, 3, 4],
            "patient_hash": [9, 9, 9, 9],
            "NRD_DaysToEvent": [100, 130, 150, 160],
            "LOS": [5, 40, 2, 1],
            "DMONTH": [4, 5, 6, 6],
            "HCUP_ED": [1, 0, 1, 1],
            "ELECTIVE": [0, 1, 0, 0],
            "APRDRG_Severity": [2, 4, 1, 3],
            "APRDRG_Risk_Mortality": [1, 4, 1, 2],
            "principal_ap": [True, False, False, True],
            "principal_biliary": [False, True, False, False],
            "principal_sepsis_or_organ": [False, False, True, False],
            "planned_status": [0, 1, 0, 0],
            "dx_tokens": [[4, 0], [5], [6], [7]],
            "pr_tokens": [[10, 3], [11], [12], [13]],
        }
    )
    indexes = pd.DataFrame(
        {
            "year": [2020], "encounter_hash": [4], "patient_hash": [9],
            "NRD_DaysToEvent": [160], "DMONTH": [6], "analysis_partition": ["development"],
        }
    )
    rec = derive_patient_history(admissions, indexes)[0]
    # Encounter 2 overlaps the index (130 + 40 > 160), encounter 3 is 8 days prior.
    assert rec["prior_count_ytd"] == 2
    assert rec["prior_count_30d"] == 1
    assert rec["days_since_prior_discharge"] == 8
    assert rec["prior_dx_tokens_180d"] == [4, 6]
    assert rec["prior_pr_tokens_180d"] == [10, 12]
    assert rec["history_30d_fully_observable"] is True
    assert rec["history_90d_fully_observable"] is True
    assert rec["history_180d_fully_observable"] is False


def test_patient_with_no_prior_history_gets_explicit_zeroes():
    admissions = pd.DataFrame(
        {
            "encounter_hash": [10], "patient_hash": [99], "NRD_DaysToEvent": [200],
            "LOS": [3], "DMONTH": [7], "HCUP_ED": [1], "ELECTIVE": [0],
            "APRDRG_Severity": [2], "APRDRG_Risk_Mortality": [1],
            "principal_ap": [True], "principal_biliary": [False],
            "principal_sepsis_or_organ": [False], "planned_status": [0],
            "dx_tokens": [[4]], "pr_tokens": [[10]],
        }
    )
    indexes = pd.DataFrame(
        {
            "year": [2020], "encounter_hash": [10], "patient_hash": [99],
            "NRD_DaysToEvent": [200], "DMONTH": [7], "analysis_partition": ["development"],
        }
    )
    rec = derive_patient_history(admissions, indexes)[0]
    assert rec["prior_count_ytd"] == 0
    assert rec["prior_count_30d"] == 0
    assert rec["prior_dx_tokens_180d"] == []
    assert rec["days_since_prior_discharge"] == -1
