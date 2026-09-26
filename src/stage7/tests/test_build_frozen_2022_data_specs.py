"""No-data tests for frozen 2022 ETL/derivative specification construction."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_frozen_2022_data_specs.py"
SPEC = importlib.util.spec_from_file_location("build_frozen_2022_data_specs", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(builder)


def write(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8"); return path


def fake_inputs(tmp_path: Path) -> dict[str, Path]:
    dx = [f"I10_DX{i}" for i in range(1, 41)]; pr = [f"I10_PR{i}" for i in range(1, 26)]; prday = [f"PRDAY{i}" for i in range(1, 26)]
    required = ["KEY_NRD", "NRD_VisitLink", "HOSP_NRD", "NRD_DaysToEvent", "AGE", "LOS", "DMONTH", "DIED", "DISPUNIFORM", "DISCWT", "TOTCHG", "NRD_STRATUM", "AWEEKEND", "ELECTIVE", "FEMALE", "HCUP_ED", "I10_NDX", "I10_NPR", "PAY1", "PL_NCHS", "RESIDENT", "ZIPINC_QRTL", *dx, *pr, *prday]
    core = list(dict.fromkeys(required + [f"CORE_FILL_{i}" for i in range(127)]))[:127]
    hospital = ["HOSP_NRD", "HOSP_BEDSIZE", "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH", "N_DISC_U", "N_HOSP_U", "S_DISC_U", "S_HOSP_U", "TOTAL_DISC", "H_FILL1", "H_FILL2"]
    severity = ["KEY_NRD", "HOSP_NRD", "APRDRG", "APRDRG_Risk_Mortality", "APRDRG_Severity"]
    layouts = [{"year": 2022, "group": group, "variables": [{"name": x} for x in values]} for group, values in (("CORE", core), ("HOSPITAL", hospital), ("SEVERITY", severity))]
    paths = {"nrd_specs": write(tmp_path / "nrd_specs.json", layouts),
             "ontology": tmp_path / "ontology.csv", "pra": tmp_path / "pra.parquet", "mapping": tmp_path / "mapping.parquet",
             "dx": write(tmp_path / "dx.json", {"frozen_after_year": 2020, "processed_years": [2018, 2019, 2020], "token_to_id": {"K850": 4}}),
             "pr": write(tmp_path / "pr.json", {"frozen_after_year": 2020, "processed_years": [2018, 2019, 2020], "token_to_id": {"0FC98ZZ": 4}}),
             "aux": write(tmp_path / "aux.json", {"year_2022_accessed": False, "high_cost": {"threshold_2021_usd": 100.0}, "prolonged_los": {"days": 7}}),
             "cpi": write(tmp_path / "cpi.json", {"nrd_2022_accessed": False, "cpi_to_2021_factor": .9})}
    pd.DataFrame({"label_node": ["readmission.biliary_event", "readmission.sepsis_or_acute_organ_dysfunction"], "code": ["K83", "A41"]}).to_csv(paths["ontology"], index=False)
    pd.DataFrame({"table": ["PR.1", "PR.2", "PR.3", "PR.4"], "code": ["0A", "K85", "0B", "Z00"]}).to_parquet(paths["pra"], index=False)
    pd.DataFrame({"code_system": ["CM", "PCS"], "code": ["K850", "0FC98ZZ"]}).to_parquet(paths["mapping"], index=False)
    return paths


def test_builder_uses_only_metadata_and_writes_hash_bound_contracts(tmp_path: Path) -> None:
    x = fake_inputs(tmp_path); out = tmp_path / "specs"
    result = builder.build_specs(out, nrd_specs=x["nrd_specs"], ontology=x["ontology"], pra=x["pra"], pra_mapping=x["mapping"], dx_vocab=x["dx"], pr_vocab=x["pr"], auxiliary=x["aux"], cpi=x["cpi"])
    assert result["status"] == "PASS_FROZEN_2022_DATA_SPECS" and result["nrd_2022_accessed"] is False
    etl = json.loads((out / "frozen_2022_etl_spec.json").read_text(encoding="utf-8"))
    derivative = json.loads((out / "frozen_2022_derivative_spec.json").read_text(encoding="utf-8"))
    assert etl["source_all_columns"]["core"].__len__() == 127
    assert etl["memory_contract"].startswith("Core CSV streamed")
    assert etl["archive_members"]["ccr"] == "cc2022NRD.csv"
    assert etl["source_formats"]["ccr"] == "csv" and etl["csv_has_header"]["ccr"] is True
    assert etl["csv_quotechar"] == {"core": '"', "severity": '"', "hospital": '"', "ccr": "'"}
    assert etl["source_columns"]["ccr"] == ["HOSP_NRD", "YEAR", "CCR_NRD", "WAGEINDEX"]
    assert set(etl["source_metadata_provenance"]) == {"nrd_file_specs", "stage2_ontology", "pra_code_sets", "pra_mapping", "diagnosis_vocabulary", "procedure_vocabulary", "auxiliary_thresholds", "cpi_constants"}
    stage5_static = {"AGE", "LOS", "I10_NDX", "I10_NPR", "CCR_NRD", "WAGEINDEX", "APRDRG_Severity", "APRDRG_Risk_Mortality", "AWEEKEND", "DMONTH", "ELECTIVE", "FEMALE", "HCUP_ED", "PAY1", "PL_NCHS", "RESIDENT", "ZIPINC_QRTL", "HOSP_BEDSIZE", "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH", "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d", "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d", "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d", "prior_max_severity_180d", "prior_max_mortality_risk_180d", "days_since_prior_discharge", "history_30d_fully_observable", "history_90d_fully_observable", "history_180d_fully_observable"}
    derivative_available = set(derivative["episode_columns"]) | set(derivative["history_columns"]) | {"AGE", "FEMALE", "PAY1", "ZIPINC_QRTL"}
    assert stage5_static <= derivative_available and {"LOS", "DMONTH"} <= set(derivative["static_columns"])
    assert not (set(derivative["static_columns"]) & set(derivative["outcomes"].values()))
    assert derivative["high_cost_missing_allowed"] is True
    for name in ("frozen_2022_etl_spec.json", "frozen_2022_derivative_spec.json"):
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert (out / f"{name}.sha256").read_text().split()[0] == actual
    with pytest.raises(RuntimeError, match="immutable"):
        builder.build_specs(out, nrd_specs=x["nrd_specs"], ontology=x["ontology"], pra=x["pra"], pra_mapping=x["mapping"], dx_vocab=x["dx"], pr_vocab=x["pr"], auxiliary=x["aux"], cpi=x["cpi"])
