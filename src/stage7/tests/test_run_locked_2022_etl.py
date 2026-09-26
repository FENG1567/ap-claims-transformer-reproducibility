"""Synthetic tests for the lock-first standalone 2022 local ETL controller."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import csv
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_locked_2022_etl.py"
SPEC = importlib.util.spec_from_file_location("run_locked_2022_etl", MODULE_PATH)
assert SPEC and SPEC.loader
etl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(etl)


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(path: Path) -> dict[str, object]: return {"bytes": path.stat().st_size, "sha256": digest(path)}
def write_json(path: Path, value: dict) -> None: path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def dependencies(tmp_path: Path) -> dict[str, Path]:
    values = {
        "stage2_ontology": {"name": "locked"}, "pra_code_sets": {"name": "locked"}, "nrd_file_specs": {"name": "locked"}, "pra_mapping": {"name": "locked"},
        "diagnosis_vocabulary": {"frozen_after_year": 2020, "processed_years": [2018, 2019, 2020], "token_to_id": {"K850": 4, "K83": 5, "A41": 6}},
        "procedure_vocabulary": {"frozen_after_year": 2020, "processed_years": [2018, 2019, 2020], "token_to_id": {"0FC98ZZ": 4}},
        "auxiliary_thresholds": {"name": "locked"}, "cpi_constants": {"name": "locked"},
    }
    result = {}
    for role, value in values.items():
        path = tmp_path / f"{role}.json"; write_json(path, value); result[role] = path
    return result


def frozen_spec(path: Path, deps: dict[str, Path]) -> dict:
    core = ["KEY_NRD", "NRD_VisitLink", "HOSP_NRD", "NRD_DaysToEvent", "AGE", "LOS", "DMONTH", "DIED", "DISPUNIFORM", "DISCWT", "TOTCHG", "NRD_STRATUM", "CCR_NRD", "WAGEINDEX", "FEMALE", "PAY1", "ZIPINC_QRTL", "I10_DX1", "I10_DX2", "I10_PR1", "PRDAY1", "HCUP_ED", "ELECTIVE"]
    value = {
        "status": "FROZEN_2022_LOCAL_ETL_SPEC", "sealed_test_year": 2022, "test_partition": "test", "data_dependent_adaptation": False, "code_set_calendar_year": 2022, "pra_version": "PRA_v4.0", "ontology_version": "locked-v1",
        "frozen_dependencies": {role: str(deps[role].resolve()) for role in etl.DEPENDENCY_ROLES},
        "source_metadata_provenance": {role: {"path": str(item.resolve()), "bytes": item.stat().st_size, "sha256": digest(item)} for role, item in deps.items()},
        "source_columns": {"core": core, "severity": ["KEY_NRD", "HOSP_NRD", "APRDRG", "APRDRG_Risk_Mortality", "APRDRG_Severity"],
                           "hospital": ["HOSP_NRD", "HOSP_BEDSIZE", "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH", "N_DISC_U", "N_HOSP_U", "S_DISC_U", "S_HOSP_U", "TOTAL_DISC"],
                           "ccr": ["HOSP_NRD", "YEAR", "CCR_NRD", "WAGEINDEX"]},
        "diagnosis_columns": ["I10_DX1", "I10_DX2"], "procedure_columns": ["I10_PR1"], "prday_columns": ["PRDAY1"],
        "core_passthrough_columns": ["AGE", "LOS", "DMONTH", "DIED", "DISPUNIFORM", "DISCWT", "TOTCHG", "NRD_STRATUM", "FEMALE", "PAY1", "ZIPINC_QRTL", "NRD_DaysToEvent", "HCUP_ED", "ELECTIVE"],
        "episode_output_columns": ["year", "encounter_hash", "patient_hash", "hospital_hash", "NRD_STRATUM", "DISCWT", "AGE", "FEMALE", "PAY1", "ZIPINC_QRTL", "NRD_DaysToEvent", "DMONTH", "LOS", "analysis_partition", "primary_analysis_eligible", "any_unplanned_readmission_30d", "readmission_leaf", "ap_specific_readmission_30d", "biliary_readmission_30d", "sepsis_or_organ_readmission_30d", "high_cost_label", "prolonged_los_label", "in_hospital_death_label", "dx_tokens", "pr_tokens", "prday", "APRDRG_Severity", "APRDRG_Risk_Mortality"],
        "code_sets": {"known_cm": ["K850", "K83", "A41"], "known_pcs": ["0FC98ZZ"], "ap_prefixes": ["K85"], "biliary": ["K83"], "sepsis_or_organ": ["A41"], "PR.1": [], "PR.2": [], "PR.3": [], "PR.4": []},
        "cpi_to_2021_factor": 1.0, "auxiliary_thresholds": {"high_cost_threshold_2021_usd": 100, "prolonged_los_days": 7},
        "archive_members": {role: f"{role}.parquet" for role in etl.SOURCE_ROLES},
    }
    write_json(path, value); return value


def unlock(tmp_path: Path, spec: Path, deps: dict[str, Path]) -> Path:
    path = tmp_path / "unlock.json"
    entries = {str(MODULE_PATH.resolve()): identity(MODULE_PATH), str(spec.resolve()): identity(spec)}
    entries.update({str(item.resolve()): identity(item) for item in deps.values()})
    write_json(path, {"status": "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION", "sealed_test_year": 2022,
                      "2022_access_before_lock": False, "frozen_code": entries})
    path.with_suffix(".json.sha256").write_text(f"{digest(path)}  {path.name}\n", encoding="ascii")
    return path


def sources(tmp_path: Path, spec: dict) -> dict[str, Path]:
    core = pd.DataFrame([
        {"KEY_NRD": 1, "NRD_VisitLink": "a", "HOSP_NRD": 10, "NRD_DaysToEvent": 1, "AGE": 50, "LOS": 2, "DMONTH": 1, "DIED": 0, "DISPUNIFORM": 1, "DISCWT": 1.0, "TOTCHG": 50, "NRD_STRATUM": 1, "CCR_NRD": 1.0, "WAGEINDEX": 1., "FEMALE": 1, "PAY1": 1, "ZIPINC_QRTL": 1, "I10_DX1": "K85.0", "I10_DX2": "", "I10_PR1": "", "PRDAY1": None, "HCUP_ED": 1, "ELECTIVE": 0},
        {"KEY_NRD": 2, "NRD_VisitLink": "a", "HOSP_NRD": 10, "NRD_DaysToEvent": 5, "AGE": 50, "LOS": 2, "DMONTH": 1, "DIED": 0, "DISPUNIFORM": 1, "DISCWT": 1.0, "TOTCHG": 200, "NRD_STRATUM": 1, "CCR_NRD": 1.0, "WAGEINDEX": 1., "FEMALE": 1, "PAY1": 1, "ZIPINC_QRTL": 1, "I10_DX1": "K83", "I10_DX2": "", "I10_PR1": "", "PRDAY1": None, "HCUP_ED": 0, "ELECTIVE": 0},
    ])
    severity = pd.DataFrame([{"KEY_NRD": 1, "HOSP_NRD": 10, "APRDRG": 1, "APRDRG_Risk_Mortality": 1, "APRDRG_Severity": 1}, {"KEY_NRD": 2, "HOSP_NRD": 10, "APRDRG": 1, "APRDRG_Risk_Mortality": 1, "APRDRG_Severity": 1}])
    hospital = pd.DataFrame([{"HOSP_NRD": 10, "HOSP_BEDSIZE": 1, "H_CONTRL": 1, "HOSP_URCAT4": 1, "HOSP_UR_TEACH": 1, "N_DISC_U": 1, "N_HOSP_U": 1, "S_DISC_U": 1, "S_HOSP_U": 1, "TOTAL_DISC": 2}])
    ccr = pd.DataFrame([{"HOSP_NRD": 10, "YEAR": 2022, "CCR_NRD": 1.0, "WAGEINDEX": 1.0}])
    result = {}
    for role, frame in {"core": core, "severity": severity, "hospital": hospital, "ccr": ccr}.items():
        path = tmp_path / f"{role}.parquet"; pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path); result[role] = path
    return result


def test_preflight_is_lock_first_and_tamper_fails(tmp_path: Path) -> None:
    deps = dependencies(tmp_path); spec = tmp_path / "spec.json"; frozen_spec(spec, deps); lock = unlock(tmp_path, spec, deps)
    assert etl.validate_preflight(lock, spec)["sealed_test_year"] == 2022
    bad = json.loads(lock.read_text()); bad["frozen_code"][str(next(iter(deps.values())).resolve())]["sha256"] = "0" * 64; write_json(lock, bad)
    lock.with_suffix(".json.sha256").write_text(f"{digest(lock)}  {lock.name}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="Frozen dependency"):
        etl.validate_etl_spec(spec, etl.validate_preflight(lock, spec))


def test_metadata_and_pra_mapping_provenance_tampering_fails(tmp_path: Path) -> None:
    deps = dependencies(tmp_path); spec = tmp_path / "spec.json"; frozen_spec(spec, deps); lock = unlock(tmp_path, spec, deps)
    write_json(deps["nrd_file_specs"], {"name": "tampered"})
    with pytest.raises(RuntimeError, match="build-input provenance mismatch: nrd_file_specs"):
        etl.validate_etl_spec(spec, etl.validate_preflight(lock, spec))
    (tmp_path / "mapping").mkdir(exist_ok=True); deps = dependencies(tmp_path / "mapping")
    spec = tmp_path / "mapping" / "spec.json"; frozen_spec(spec, deps); lock = unlock(tmp_path / "mapping", spec, deps)
    write_json(deps["pra_mapping"], {"name": "tampered"})
    with pytest.raises(RuntimeError, match="build-input provenance mismatch: pra_mapping"):
        etl.validate_etl_spec(spec, etl.validate_preflight(lock, spec))


def test_one_shot_etl_produces_derivative_compatible_test_contract(tmp_path: Path) -> None:
    deps = dependencies(tmp_path); spec_path = tmp_path / "spec.json"; spec = frozen_spec(spec_path, deps); lock = unlock(tmp_path, spec_path, deps); raw = sources(tmp_path, spec)
    output = tmp_path / "out"; result = etl.run_locked_2022_etl(raw, lock, spec_path, output, threads=1)
    assert result["status"] == "PASS_LOCKED_2022_LOCAL_ETL" and result["test_partition"] == "test"
    episodes = pd.read_parquet(output / "ap_episodes.parquet"); history = pd.read_parquet(output / "ap_history_2022.parquet")
    assert set(episodes.analysis_partition) == {"test"} == set(history.analysis_partition)
    assert "hospital_hash" in episodes and not (set(episodes.columns) & etl.FORBIDDEN_IDENTIFIERS)
    assert set(episodes.encounter_hash) == set(history.encounter_hash)
    assert episodes.loc[0, "dx_tokens"][0] == 4  # frozen known token; vocabulary not extended
    with pytest.raises(RuntimeError, match="immutable"):
        etl.run_locked_2022_etl(raw, lock, spec_path, output, threads=1)


def test_schema_drift_partial_and_oov_are_rejected_or_explicit(tmp_path: Path) -> None:
    deps = dependencies(tmp_path); spec_path = tmp_path / "spec.json"; spec = frozen_spec(spec_path, deps); lock = unlock(tmp_path, spec_path, deps); raw = sources(tmp_path, spec)
    core = pq.read_table(raw["core"]).to_pandas().drop(columns=["I10_DX2"]); pq.write_table(pa.Table.from_pandas(core, preserve_index=False), raw["core"])
    with pytest.raises(RuntimeError, match="schema drift"):
        etl.run_locked_2022_etl(raw, lock, spec_path, tmp_path / "drift", threads=1)
    raw = sources(tmp_path, spec); core = pq.read_table(raw["core"]).to_pandas(); core.loc[0, "I10_DX2"] = "NEW2022"; pq.write_table(pa.Table.from_pandas(core, preserve_index=False), raw["core"])
    result = etl.run_locked_2022_etl(raw, lock, spec_path, tmp_path / "oov", threads=1)
    admissions = pd.read_parquet(Path(result["output"]) / "admissions_model.parquet")
    assert admissions.loc[0, "dx_tokens"][1] == 2
    partial = tmp_path / "partial.partial-old"; partial.mkdir()
    with pytest.raises(RuntimeError, match="partial output"):
        etl.run_locked_2022_etl(raw, lock, spec_path, tmp_path / "partial", threads=1)


def test_ccr_year_must_equal_sealed_year(tmp_path: Path) -> None:
    deps = dependencies(tmp_path); spec_path = tmp_path / "spec.json"; spec = frozen_spec(spec_path, deps); lock = unlock(tmp_path, spec_path, deps); raw = sources(tmp_path, spec)
    ccr = pq.read_table(raw["ccr"]).to_pandas(); ccr.loc[0, "YEAR"] = 2021
    pq.write_table(pa.Table.from_pandas(ccr, preserve_index=False), raw["ccr"])
    with pytest.raises(RuntimeError, match="non-2022 YEAR"):
        etl.run_locked_2022_etl(raw, lock, spec_path, tmp_path / "bad-ccr-year", threads=1)


def test_missing_cost_is_masked_and_core_csv_transform_is_chunk_bounded(tmp_path: Path, monkeypatch) -> None:
    deps = dependencies(tmp_path); spec_path = tmp_path / "spec.json"; spec = frozen_spec(spec_path, deps); lock = unlock(tmp_path, spec_path, deps); raw = sources(tmp_path, spec)
    core = pq.read_table(raw["core"]).to_pandas(); core.loc[0, "TOTCHG"] = None; pq.write_table(pa.Table.from_pandas(core, preserve_index=False), raw["core"])
    result = etl.run_locked_2022_etl(raw, lock, spec_path, tmp_path / "masked", threads=1)
    admissions = pd.read_parquet(Path(result["output"]) / "admissions_model.parquet")
    assert pd.isna(admissions.loc[0, "high_cost_label"]) and not bool(admissions.loc[0, "high_cost_observed"])
    # Use a headerless synthetic Core CSV and require chunksize; a full-table
    # pd.read_csv call is made to fail so this guards against regression.
    csv_spec = dict(spec); csv_spec["source_formats"] = {"core": "csv", "severity": "parquet", "hospital": "parquet", "ccr": "parquet"}; csv_spec["source_all_columns"] = {role: list(spec["source_columns"][role]) for role in etl.SOURCE_ROLES}; csv_spec["csv_has_header"] = {role: False for role in etl.SOURCE_ROLES}; csv_spec["csv_quotechar"] = {role: '"' for role in etl.SOURCE_ROLES}
    csv_path = tmp_path / "core.csv"; core.loc[:, csv_spec["source_all_columns"]["core"]].to_csv(csv_path, index=False, header=False)
    calls = []; original = pd.read_csv
    def guarded(*args, **kwargs):
        calls.append(kwargs.get("chunksize")); assert kwargs.get("chunksize") == 1; return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guarded)
    spool = tmp_path / "core_spool.parquet"; report = etl.stream_core_csv_to_parquet(csv_path, csv_spec, {k: v for k, v in deps.items()}, spool, chunksize=1)
    assert report["rows"] == 2 and calls == [1] and pq.ParquetFile(spool).metadata.num_rows == 2


def test_full_csv_core_route_streams_then_uses_disk_join(tmp_path: Path, monkeypatch) -> None:
    """The production API must take the bounded path, not merely expose a helper."""
    deps = dependencies(tmp_path)
    spec_path = tmp_path / "csv-spec.json"
    spec = frozen_spec(spec_path, deps)
    spec["source_formats"] = {"core": "csv", "severity": "parquet", "hospital": "parquet", "ccr": "csv"}
    spec["source_all_columns"] = {role: list(spec["source_columns"][role]) for role in etl.SOURCE_ROLES}
    spec["csv_has_header"] = {role: False for role in etl.SOURCE_ROLES}; spec["csv_has_header"]["ccr"] = True
    spec["csv_quotechar"] = {role: '"' for role in etl.SOURCE_ROLES}; spec["csv_quotechar"]["ccr"] = "'"
    write_json(spec_path, spec)
    lock = unlock(tmp_path, spec_path, deps)
    raw = sources(tmp_path, spec)
    core = pq.read_table(raw["core"]).to_pandas()
    core["TOTCHG"] = core["TOTCHG"].astype(object)
    core.loc[0, "TOTCHG"] = ""  # CSV path must preserve an unobserved cost label.
    csv_core = tmp_path / "core.csv"
    core.loc[:, spec["source_all_columns"]["core"]].to_csv(csv_core, index=False, header=False)
    raw["core"] = csv_core
    ccr_csv = tmp_path / "cc2022NRD.csv"
    pq.read_table(raw["ccr"]).to_pandas().to_csv(ccr_csv, index=False, quotechar="'", quoting=csv.QUOTE_ALL)
    raw["ccr"] = ccr_csv
    original_read_csv = pd.read_csv
    calls: list[int | None] = []

    def guarded_read_csv(*args, **kwargs):
        if Path(args[0]).resolve() == csv_core.resolve():
            calls.append(kwargs.get("chunksize"))
            assert kwargs.get("chunksize") == 131072
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    output = tmp_path / "csv-out"
    result = etl.run_locked_2022_etl(raw, lock, spec_path, output, threads=1)
    assert result["status"] == "PASS_LOCKED_2022_LOCAL_ETL" and calls == [131072]
    assert (output / "admissions_model.parquet").is_file()
    admissions = pd.read_parquet(output / "admissions_model.parquet")
    assert len(admissions) == 2 and not (set(admissions.columns) & etl.FORBIDDEN_IDENTIFIERS)
    assert pd.isna(admissions.loc[0, "high_cost_label"]) and not bool(admissions.loc[0, "high_cost_observed"])


def test_missing_ccr_masks_cost_only_and_preserves_readmission_in_parquet_and_csv(tmp_path: Path) -> None:
    """CCR coverage is a cost-label condition, never a main-cohort condition."""
    def make_missing_ccr_raw(base: Path, spec: dict) -> dict[str, Path]:
        raw = sources(base, spec)
        core = pq.read_table(raw["core"]).to_pandas()
        core["HOSP_NRD"] = 11  # CCR has only hospital 10, while Hospital has 11 below.
        pq.write_table(pa.Table.from_pandas(core, preserve_index=False), raw["core"])
        hospital = pq.read_table(raw["hospital"]).to_pandas()
        hospital.loc[0, "HOSP_NRD"] = 11
        pq.write_table(pa.Table.from_pandas(hospital, preserve_index=False), raw["hospital"])
        return raw

    # Internal parquet route: its former INNER JOIN must no longer remove rows.
    (tmp_path / "parquet").mkdir(); deps = dependencies(tmp_path / "parquet")
    spec_path = tmp_path / "parquet" / "spec.json"
    spec = frozen_spec(spec_path, deps); lock = unlock(spec_path.parent, spec_path, deps)
    raw = make_missing_ccr_raw(spec_path.parent, spec)
    parquet_output = tmp_path / "parquet-out"
    etl.run_locked_2022_etl(raw, lock, spec_path, parquet_output, threads=1)
    parquet_admissions = pd.read_parquet(parquet_output / "admissions_model.parquet")
    parquet_episodes = pd.read_parquet(parquet_output / "ap_episodes.parquet")
    parquet_aux = pd.read_parquet(parquet_output / "aux_compact.parquet")
    assert len(parquet_admissions) == 2 and int(parquet_episodes.loc[0, "biliary_readmission_30d"]) == 1
    assert parquet_admissions["cost_2021_usd"].isna().all() and parquet_admissions["high_cost_label"].isna().all()
    assert not parquet_admissions["high_cost_observed"].any() and len(parquet_aux) == 1 and parquet_aux["CCR_NRD"].isna().all()

    # Formal CSV production route: preserve the identical outcome and mask.
    csv_root = tmp_path / "csv"; csv_root.mkdir()
    csv_deps = dependencies(csv_root)
    csv_spec_path = csv_root / "spec.json"; csv_spec = frozen_spec(csv_spec_path, csv_deps)
    csv_spec["source_formats"] = {"core": "csv", "severity": "parquet", "hospital": "parquet", "ccr": "csv"}
    csv_spec["source_all_columns"] = {role: list(csv_spec["source_columns"][role]) for role in etl.SOURCE_ROLES}
    csv_spec["csv_has_header"] = {role: False for role in etl.SOURCE_ROLES}; csv_spec["csv_has_header"]["ccr"] = True
    csv_spec["csv_quotechar"] = {role: '"' for role in etl.SOURCE_ROLES}; csv_spec["csv_quotechar"]["ccr"] = "'"
    write_json(csv_spec_path, csv_spec); csv_lock = unlock(csv_root, csv_spec_path, csv_deps)
    csv_raw = make_missing_ccr_raw(csv_root, csv_spec)
    csv_core = pq.read_table(csv_raw["core"]).to_pandas(); csv_path = csv_root / "core.csv"
    csv_core.loc[:, csv_spec["source_all_columns"]["core"]].to_csv(csv_path, index=False, header=False)
    csv_raw["core"] = csv_path
    ccr_csv = csv_root / "cc2022NRD.csv"
    pq.read_table(csv_raw["ccr"]).to_pandas().to_csv(ccr_csv, index=False, quotechar="'", quoting=csv.QUOTE_ALL)
    csv_raw["ccr"] = ccr_csv
    csv_output = tmp_path / "csv-out"
    etl.run_locked_2022_etl(csv_raw, csv_lock, csv_spec_path, csv_output, threads=1)
    csv_admissions = pd.read_parquet(csv_output / "admissions_model.parquet")
    csv_episodes = pd.read_parquet(csv_output / "ap_episodes.parquet")
    csv_aux = pd.read_parquet(csv_output / "aux_compact.parquet")
    assert len(csv_admissions) == 2 and int(csv_episodes.loc[0, "biliary_readmission_30d"]) == 1
    assert csv_admissions["cost_2021_usd"].isna().all() and csv_admissions["high_cost_label"].isna().all()
    assert not csv_admissions["high_cost_observed"].any() and len(csv_aux) == 1 and csv_aux["CCR_NRD"].isna().all()
