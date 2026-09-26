#!/usr/bin/env python3
"""Create immutable, no-2022-data ETL and derivative specifications.

Only public schema metadata and pre-test frozen artifacts are read.  This is a
lock-preparation program: it deliberately accepts no NRD archive, member, data
directory, password, or output derived from 2022 patients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).parents[2]
HISTORY = [
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d", "prior_ed_count_180d",
    "prior_nonelective_count_180d", "prior_ap_count_180d", "prior_biliary_count_180d",
    "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d", "prior_max_severity_180d",
    "prior_max_mortality_risk_180d", "days_since_prior_discharge", "history_30d_fully_observable",
    "history_90d_fully_observable", "history_180d_fully_observable", "prior_dx_tokens_180d", "prior_pr_tokens_180d",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file(): raise RuntimeError(f"Missing frozen metadata/artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def stable(value: Any) -> str: return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def _schema(path: Path, group: str) -> list[str]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    hit = [entry for entry in entries if entry.get("year") == 2022 and entry.get("group") == group]
    if len(hit) != 1: raise RuntimeError(f"Expected one 2022 {group} layout")
    return [str(v["name"]) for v in hit[0]["variables"]]


def _codes(ontology: Path, pra: Path, mapping: Path) -> dict[str, list[str]]:
    labels = pd.read_csv(ontology, dtype=str)
    required_nodes = {"readmission.biliary_event": "biliary", "readmission.sepsis_or_acute_organ_dysfunction": "sepsis_or_organ"}
    out: dict[str, list[str]] = {}
    for node, name in required_nodes.items():
        value = labels.loc[labels["label_node"].eq(node), "code"].dropna().astype(str).str.replace(".", "", regex=False).str.upper().unique().tolist()
        if not value: raise RuntimeError(f"Frozen Stage2 ontology lacks {node}")
        out[name] = sorted(value)
    pra_frame = pd.read_parquet(pra)
    for table in ("PR.1", "PR.2", "PR.3", "PR.4"):
        value = pra_frame.loc[pra_frame["table"].eq(table), "code"].dropna().astype(str).str.replace(".", "", regex=False).str.upper().unique().tolist()
        if not value: raise RuntimeError(f"Frozen 2022 PRA code set lacks {table}")
        out[table] = sorted(value)
    maps = pd.read_parquet(mapping)
    out["known_cm"] = sorted(maps.loc[maps["code_system"].eq("CM"), "code"].dropna().astype(str).str.replace(".", "", regex=False).str.upper().unique())
    out["known_pcs"] = sorted(maps.loc[maps["code_system"].eq("PCS"), "code"].dropna().astype(str).str.replace(".", "", regex=False).str.upper().unique())
    if not out["known_cm"] or not out["known_pcs"]: raise RuntimeError("Frozen 2022 mapping lacks CM/PCS codes")
    out["ap_prefixes"] = ["K85"]
    return out


def build_specs(output_dir: Path, *, nrd_specs: Path, ontology: Path, pra: Path, pra_mapping: Path,
                dx_vocab: Path, pr_vocab: Path, auxiliary: Path, cpi: Path,
                archive_members: dict[str, str] | None = None) -> dict[str, Any]:
    """Build immutable specs from frozen metadata; never open an NRD patient file."""
    output_dir = output_dir.resolve()
    if output_dir.exists() or list(output_dir.parent.glob(output_dir.name + ".partial*")):
        raise RuntimeError("Frozen 2022 data spec output already exists and is immutable")
    inputs = {"nrd_file_specs": nrd_specs, "stage2_ontology": ontology, "pra_code_sets": pra,
              "pra_mapping": pra_mapping, "diagnosis_vocabulary": dx_vocab, "procedure_vocabulary": pr_vocab,
              "auxiliary_thresholds": auxiliary, "cpi_constants": cpi}
    for path in inputs.values(): identity(path)
    core, hospital, severity = _schema(nrd_specs, "CORE"), _schema(nrd_specs, "HOSPITAL"), _schema(nrd_specs, "SEVERITY")
    if (len(core), len(hospital), len(severity)) != (127, 12, 5):
        raise RuntimeError("Frozen 2022 NRD layout must be Core/Hospital/Severity = 127/12/5 fields")
    dx = [f"I10_DX{i}" for i in range(1, 41)]; pr = [f"I10_PR{i}" for i in range(1, 26)]; prday = [f"PRDAY{i}" for i in range(1, 26)]
    required = {"KEY_NRD", "NRD_VisitLink", "HOSP_NRD", "NRD_DaysToEvent", "AGE", "LOS", "DMONTH", "DIED", "DISPUNIFORM", "DISCWT", "TOTCHG", "NRD_STRATUM", *dx, *pr, *prday}
    if not required.issubset(core): raise RuntimeError("Official 2022 Core layout lacks frozen ETL fields")
    aux_value, cpi_value = json.loads(auxiliary.read_text(encoding="utf-8")), json.loads(cpi.read_text(encoding="utf-8"))
    if aux_value.get("year_2022_accessed") is not False or cpi_value.get("nrd_2022_accessed") is not False: raise RuntimeError("A frozen input reports prohibited 2022 access")
    high = float(aux_value["high_cost"]["threshold_2021_usd"]); los_cut = float(aux_value["prolonged_los"].get("days", 7))
    members = archive_members or {"core": "NRD_2022_Core.csv", "severity": "NRD_2022_Severity.csv", "hospital": "NRD_2022_Hospital.csv", "ccr": "cc2022NRD.csv"}
    core_projection = ["KEY_NRD", "NRD_VisitLink", "HOSP_NRD", "NRD_DaysToEvent", "AGE", "LOS", "DMONTH", "DIED", "DISPUNIFORM", "DISCWT", "TOTCHG", "NRD_STRATUM", "AWEEKEND", "ELECTIVE", "FEMALE", "HCUP_ED", "I10_NDX", "I10_NPR", "PAY1", "PL_NCHS", "RESIDENT", "ZIPINC_QRTL", *dx, *pr, *prday]
    if not set(core_projection).issubset(core): raise RuntimeError("Official 2022 Core layout lacks one or more frozen projection fields")
    source_cols = {"core": core_projection, "severity": severity, "hospital": hospital, "ccr": ["HOSP_NRD", "YEAR", "CCR_NRD", "WAGEINDEX"]}
    static = ["LOS", "I10_NDX", "I10_NPR", "CCR_NRD", "WAGEINDEX", "APRDRG_Severity", "APRDRG_Risk_Mortality", "AWEEKEND", "DMONTH", "ELECTIVE", "FEMALE", "HCUP_ED", "PAY1", "PL_NCHS", "RESIDENT", "ZIPINC_QRTL", "HOSP_BEDSIZE", "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH"]
    episode = list(dict.fromkeys(["year", "encounter_hash", "patient_hash", "hospital_hash", "NRD_STRATUM", "DISCWT", "AGE", "FEMALE", "PAY1", "ZIPINC_QRTL", "NRD_DaysToEvent", "DMONTH", "LOS", "analysis_partition", "primary_analysis_eligible", "any_unplanned_readmission_30d", "readmission_leaf", "ap_specific_readmission_30d", "biliary_readmission_30d", "sepsis_or_organ_readmission_30d", "high_cost_label", "prolonged_los_label", "in_hospital_death_label", "dx_tokens", "pr_tokens", "prday", *static]))
    etl = {"status": "FROZEN_2022_LOCAL_ETL_SPEC", "sealed_test_year": 2022, "test_partition": "test", "data_dependent_adaptation": False,
           "code_set_calendar_year": 2022, "pra_version": "CMS_PRA_v4.0_calendar_2022_reporting_2024", "ontology_version": "stage2_locked_raw_icd",
           "frozen_dependencies": {key: str(value.resolve()) for key, value in inputs.items() if key != "nrd_file_specs" and key != "pra_mapping"},
           # Bind every metadata input that defined a test-year schema/code
           # contract, including inputs not needed by the runtime API.
           "source_metadata_provenance": {key: identity(value) for key, value in inputs.items()},
           "source_columns": source_cols, "source_all_columns": {"core": core, "severity": severity, "hospital": hospital, "ccr": source_cols["ccr"]},
           "source_formats": {"core": "csv", "severity": "csv", "hospital": "csv", "ccr": "csv"}, "csv_has_header": {"core": False, "severity": False, "hospital": False, "ccr": True},
           "csv_quotechar": {"core": '"', "severity": '"', "hospital": '"', "ccr": "'"}, "archive_members": members, "diagnosis_columns": dx, "procedure_columns": pr, "prday_columns": prday,
           "core_passthrough_columns": [column for column in core_projection if column not in {"KEY_NRD", "NRD_VisitLink", "HOSP_NRD", *dx, *pr, *prday}],
           "episode_output_columns": episode, "code_sets": _codes(ontology, pra, pra_mapping), "cpi_to_2021_factor": float(cpi_value["cpi_to_2021_factor"]),
           "auxiliary_thresholds": {"high_cost_threshold_2021_usd": high, "prolonged_los_days": los_cut}, "core_csv_chunksize": 131072,
           "memory_contract": "Core CSV streamed in 131072-row frozen projections; no full-year object DataFrame."}
    derivative = {"status": "FROZEN_2022_DERIVATIVE_SPEC", "sealed_test_year": 2022, "test_partition": "test", "data_dependent_adaptation": False,
                  "year_column": "year", "partition_column": "analysis_partition", "eligibility_column": "primary_analysis_eligible", "eligibility_value": True, "leaf_column": "readmission_leaf",
                  # LOS and DMONTH are frozen Stage5 inputs and remain
                  # available at the index-discharge prediction anchor.  The
                  # other three fields already occur in identity columns.
                  "episode_columns": [column for column in episode if column not in {"NRD_DaysToEvent"}],
                  "history_columns": ["encounter_hash", "patient_hash", "analysis_partition", *HISTORY], "token_sequence_columns": ["dx_tokens", "pr_tokens", "prday"], "static_columns": [x for x in static if x not in {"FEMALE", "PAY1", "ZIPINC_QRTL"}],
                  "outcomes": {"any_readmission": "any_unplanned_readmission_30d", "ap_specific_readmission": "ap_specific_readmission_30d", "biliary_event": "biliary_readmission_30d", "sepsis_or_organ_complication": "sepsis_or_organ_readmission_30d", "high_cost": "high_cost_label", "prolonged_los": "prolonged_los_label", "in_hospital_death": "in_hospital_death_label"},
                  "high_cost_missing_allowed": True}
    output_dir.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))
    try:
        for name, value in (("frozen_2022_etl_spec.json", etl), ("frozen_2022_derivative_spec.json", derivative)):
            p = temporary / name; p.write_text(stable(value), encoding="utf-8"); (temporary / f"{name}.sha256").write_text(f"{sha256(p)}  {name}\n", encoding="ascii")
        os.replace(temporary, output_dir)
    except Exception:
        import shutil; shutil.rmtree(temporary, ignore_errors=True); raise
    return {"status": "PASS_FROZEN_2022_DATA_SPECS", "nrd_2022_accessed": False, "output": str(output_dir), "artifacts": {p.name: identity(p) for p in output_dir.iterdir() if p.suffix == ".json"}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--nrd-file-specs", type=Path, required=True); p.add_argument("--ontology", type=Path, required=True); p.add_argument("--pra", type=Path, required=True); p.add_argument("--pra-mapping", type=Path, required=True); p.add_argument("--dx-vocab", type=Path, required=True); p.add_argument("--pr-vocab", type=Path, required=True); p.add_argument("--auxiliary", type=Path, required=True); p.add_argument("--cpi", type=Path, required=True)
    a = p.parse_args(); print(stable(build_specs(a.output_dir, nrd_specs=a.nrd_file_specs, ontology=a.ontology, pra=a.pra, pra_mapping=a.pra_mapping, dx_vocab=a.dx_vocab, pr_vocab=a.pr_vocab, auxiliary=a.auxiliary, cpi=a.cpi)))

if __name__ == "__main__": main()
