from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "repair_locked_2022_episode_order_v8.py"
SPEC = importlib.util.spec_from_file_location("repair_v8", MODULE_PATH)
assert SPEC and SPEC.loader
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def row(encounter: str, patient: str, day: str, *, ap: bool, planned: int, cause: int,
        month: str = "5") -> dict:
    return {
        "year": 2022, "encounter_hash": encounter, "patient_hash": patient,
        "hospital_hash": "h", "NRD_STRATUM": "1", "DISCWT": "1", "AGE": "50",
        "FEMALE": "1", "PAY1": "1", "ZIPINC_QRTL": "1", "NRD_DaysToEvent": day,
        "DMONTH": month, "LOS": "1", "DISPUNIFORM": "1", "principal_ap": ap,
        "planned_status": planned, "primary_cause_code": cause,
        "in_hospital_death_label": 0, "high_cost_label": 0, "prolonged_los_label": 0,
        "dx_tokens": [4], "pr_tokens": [3], "prday": [-999],
        "APRDRG_Severity": "1", "APRDRG_Risk_Mortality": "1", "HCUP_ED": "0",
        "ELECTIVE": "0", "principal_biliary": False,
        "principal_sepsis_or_organ": False,
    }


def frozen_spec() -> dict:
    sample = row("e", "p", "1", ap=True, planned=0, cause=1)
    return {"episode_output_columns": [
        *[name for name in sample if name not in {"principal_ap", "planned_status", "primary_cause_code",
                                                   "DISPUNIFORM", "HCUP_ED", "ELECTIVE",
                                                   "principal_biliary", "principal_sepsis_or_organ"}],
        "analysis_partition", "primary_analysis_eligible", "any_unplanned_readmission_30d",
        "readmission_leaf", "ap_specific_readmission_30d", "biliary_readmission_30d",
        "sepsis_or_organ_readmission_30d",
    ]}


def test_numeric_order_restores_negative_and_preserves_positive() -> None:
    admissions = pd.DataFrame([
        row("p1-index", "p1", "1", ap=True, planned=0, cause=1),
        row("p1-event", "p1", "5", ap=False, planned=0, cause=2),
        # Lexical ordering places "100" before the truly prior "90".  The old
        # implementation therefore saw a negative immediate gap and discarded
        # this otherwise eligible no-readmission index admission.
        row("p2-prior", "p2", "90", ap=False, planned=0, cause=4),
        row("p2-index", "p2", "100", ap=True, planned=0, cause=1),
    ])
    episodes, qc = repair.build_episodes_numeric(admissions, frozen_spec())
    result = episodes.set_index("encounter_hash")
    assert set(result.index) == {"p1-index", "p2-index"}
    assert int(result.loc["p1-index", "any_unplanned_readmission_30d"]) == 1
    assert int(result.loc["p1-index", "readmission_leaf"]) == 2
    assert int(result.loc["p2-index", "any_unplanned_readmission_30d"]) == 0
    assert int(result.loc["p2-index", "readmission_leaf"]) == 0
    assert qc["eligible_events"] == 1 and qc["eligible_non_events"] == 1


def test_unknown_first_gate_and_history_use_numeric_time() -> None:
    admissions = pd.DataFrame([
        row("prior", "p", "90", ap=False, planned=0, cause=4),
        row("index", "p", "100", ap=True, planned=0, cause=1),
        row("unknown", "p", "105", ap=False, planned=-1, cause=4),
        row("event", "p", "110", ap=False, planned=0, cause=2),
        row("negative", "q", "200", ap=True, planned=0, cause=1, month="7"),
    ])
    episodes, qc = repair.build_episodes_numeric(admissions, frozen_spec())
    assert set(episodes["encounter_hash"]) == {"negative"}
    assert qc["excluded_unknown_priority"] == 1
    history = repair.build_history_numeric(admissions, episodes)
    assert len(history) == 1 and int(history.loc[0, "prior_count_ytd"]) == 0
    assert bool(history.loc[0, "history_180d_fully_observable"])
