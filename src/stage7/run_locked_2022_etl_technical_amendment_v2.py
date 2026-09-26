#!/usr/bin/env python3
"""One-shot, lock-first local NRD-2022 ETL for the sealed temporal test.

The controller intentionally owns the 2022-only implementations rather than
calling the development-era ``local_pipeline`` CLIs, which correctly reject
2022.  A hash-locked JSON specification supplies the exact raw-source schema,
2022 annual PRA/ontology code sets, 2018--2020 frozen vocabularies, auxiliary
thresholds, CPI factor and output projection.  Consequently no 2022 values can
extend a vocabulary, choose a threshold, fit a normaliser/imputer, or change a
code set.

``--validate-only`` checks only the unlock lock, sidecar and frozen identities;
it does not inspect an archive, source directory, source schema or output.
Production input is explicit archive role/path pairs.  Archive passwords are
never accepted as arguments: 7-Zip is launched without ``-p`` and inherits the
interactive terminal when an encrypted archive asks for a password.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import duckdb


SCHEMA_VERSION = "stage7_locked_2022_local_etl_technical_amendment_v2"
SPECIAL_TOKEN_IDS = {"[PAD]": 0, "[MASK]": 1, "[OOV]": 2, "[MISSING]": 3}
SOURCE_ROLES = ("core", "severity", "hospital", "ccr")
DEPENDENCY_ROLES = (
    "stage2_ontology", "pra_code_sets", "diagnosis_vocabulary", "procedure_vocabulary",
    "auxiliary_thresholds", "cpi_constants",
)
FORBIDDEN_IDENTIFIERS = {"KEY_NRD", "NRD_VISITLINK", "NRD_VisitLink", "HOSP_NRD"}
HISTORY_COLUMNS = (
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d",
    "prior_max_severity_180d", "prior_max_mortality_risk_180d", "days_since_prior_discharge",
    "history_30d_fully_observable", "history_90d_fully_observable",
    "history_180d_fully_observable", "prior_dx_tokens_180d", "prior_pr_tokens_180d",
)
# The only pandas materialisation in the CSV-Core route is restricted to the
# AP index admissions and the other same-year admissions belonging to those
# patients.  This deliberately fixed guard is a memory safety contract, not a
# value learned from the sealed test year.
MAX_AP_RELATED_ROWS = 2_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable frozen JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _locked_identity(lock: dict[str, Any], path: Path) -> bool:
    path = path.resolve()
    entry = lock.get("frozen_code", {}).get(str(path))
    return bool(isinstance(entry, dict) and path.is_file() and entry.get("sha256") == sha256(path)
                and int(entry.get("bytes", -1)) == path.stat().st_size)


def validate_preflight(unlock_lock: Path, frozen_etl_spec: Path | None = None) -> dict[str, Any]:
    """Check the pre-2022 lock without opening any 2022 source."""
    unlock_lock = unlock_lock.resolve()
    if not unlock_lock.is_file():
        raise RuntimeError("Missing pre-2022 unlock lock")
    sidecar = unlock_lock.with_suffix(unlock_lock.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != f"{sha256(unlock_lock)}  {unlock_lock.name}":
        raise RuntimeError("Pre-2022 unlock lock SHA256 sidecar mismatch")
    lock = read_json(unlock_lock)
    if (lock.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
            or lock.get("sealed_test_year") != 2022 or lock.get("2022_access_before_lock") is not False):
        raise RuntimeError("Pre-2022 unlock lock is not authorized for the one 2022 evaluation")
    if not _locked_identity(lock, Path(__file__).resolve()):
        raise RuntimeError("2022 ETL controller path/bytes/SHA256 are not frozen in unlock lock")
    if frozen_etl_spec is not None and not _locked_identity(lock, frozen_etl_spec.resolve()):
        raise RuntimeError("Frozen 2022 ETL specification is not hash-locked in unlock lock")
    return lock


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise RuntimeError(f"Frozen ETL spec requires non-empty string list: {field}")
    if len(value) != len(set(value)):
        raise RuntimeError(f"Frozen ETL spec has duplicate names: {field}")
    return tuple(value)


def validate_etl_spec(spec_path: Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Validate every frozen non-2022 dependency before any source is opened."""
    spec_path = spec_path.resolve()
    if not _locked_identity(lock, spec_path):
        raise RuntimeError("Frozen 2022 ETL specification is not hash-locked in unlock lock")
    spec = read_json(spec_path)
    if (spec.get("status") != "FROZEN_2022_LOCAL_ETL_SPEC" or spec.get("sealed_test_year") != 2022
            or spec.get("test_partition") != "test" or spec.get("data_dependent_adaptation") is not False):
        raise RuntimeError("Invalid frozen 2022 ETL seal or adaptation contract")
    if spec.get("code_set_calendar_year") != 2022 or not isinstance(spec.get("pra_version"), str) or not isinstance(spec.get("ontology_version"), str):
        raise RuntimeError("Frozen ETL spec must identify the predeclared 2022 PRA/ontology code-set version")
    dependencies = spec.get("frozen_dependencies")
    if not isinstance(dependencies, dict) or set(dependencies) != set(DEPENDENCY_ROLES):
        raise RuntimeError("Frozen ETL spec must name all ontology/PRA/vocabulary/threshold/CPI dependencies")
    for role, raw_path in dependencies.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(f"Invalid frozen dependency path: {role}")
        if not _locked_identity(lock, Path(raw_path)):
            raise RuntimeError(f"Frozen dependency is not path/bytes/SHA256 locked: {role}")
    provenance = spec.get("source_metadata_provenance", spec.get("build_inputs"))
    expected_provenance = {"nrd_file_specs", "stage2_ontology", "pra_code_sets", "pra_mapping", "diagnosis_vocabulary", "procedure_vocabulary", "auxiliary_thresholds", "cpi_constants"}
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise RuntimeError("Frozen ETL spec lacks complete build-input provenance")
    for role, entry in provenance.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("bytes"), int) or not isinstance(entry.get("sha256"), str):
            raise RuntimeError(f"Frozen ETL build-input provenance is malformed: {role}")
        source = Path(entry["path"])
        if not source.is_file() or source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
            raise RuntimeError(f"Frozen ETL build-input provenance mismatch: {role}")
    source_columns = spec.get("source_columns")
    if not isinstance(source_columns, dict) or set(source_columns) != set(SOURCE_ROLES):
        raise RuntimeError("Frozen ETL spec must name exactly core/severity/hospital/ccr source projections")
    for role, columns in source_columns.items():
        _string_list(columns, f"source_columns.{role}")
    source_formats = spec.get("source_formats", {role: "parquet" for role in SOURCE_ROLES})
    if not isinstance(source_formats, dict) or set(source_formats) != set(SOURCE_ROLES) or any(
        value not in ("parquet", "csv") for value in source_formats.values()
    ):
        raise RuntimeError("Frozen ETL source formats must be parquet or CSV for every source role")
    if any(value == "csv" for value in source_formats.values()):
        source_all_columns = spec.get("source_all_columns")
        if not isinstance(source_all_columns, dict) or set(source_all_columns) != set(SOURCE_ROLES):
            raise RuntimeError("Frozen CSV source layout must name all source columns for every role")
        for role in SOURCE_ROLES:
            all_columns = _string_list(source_all_columns[role], f"source_all_columns.{role}")
            if not set(source_columns[role]).issubset(all_columns):
                raise RuntimeError(f"Frozen CSV layout omits required {role} fields")
        quotechars = spec.get("csv_quotechar")
        if not isinstance(quotechars, dict) or set(quotechars) != set(SOURCE_ROLES) or any(not isinstance(value, str) or len(value) != 1 for value in quotechars.values()):
            raise RuntimeError("Frozen CSV layout must provide one quote character per source role")
    for field in ("diagnosis_columns", "procedure_columns", "core_passthrough_columns", "episode_output_columns"):
        _string_list(spec.get(field), field)
    derivative_contract = {
        "year", "encounter_hash", "patient_hash", "hospital_hash", "NRD_STRATUM", "DISCWT", "AGE", "FEMALE", "PAY1", "ZIPINC_QRTL",
        "analysis_partition", "primary_analysis_eligible", "any_unplanned_readmission_30d", "readmission_leaf",
        "ap_specific_readmission_30d", "biliary_readmission_30d", "sepsis_or_organ_readmission_30d",
        "high_cost_label", "prolonged_los_label", "in_hospital_death_label", "dx_tokens", "pr_tokens", "prday",
    }
    if not derivative_contract.issubset(spec["episode_output_columns"]):
        raise RuntimeError("Frozen episode projection is not compatible with the locked 2022 derivative contract")
    if not set(spec["diagnosis_columns"]).issubset(spec["source_columns"]["core"]):
        raise RuntimeError("Frozen diagnosis source projection is incomplete")
    if not set(spec["procedure_columns"]).issubset(spec["source_columns"]["core"]):
        raise RuntimeError("Frozen procedure source projection is incomplete")
    required_core = {"KEY_NRD", "NRD_VisitLink", "HOSP_NRD", "NRD_DaysToEvent", "AGE", "LOS", "DMONTH", "DIED", "DISPUNIFORM", "DISCWT", "TOTCHG", "NRD_STRATUM"}
    if not required_core.issubset(spec["source_columns"]["core"]):
        raise RuntimeError("Frozen core projection lacks necessary linkage/eligibility fields")
    if not {"KEY_NRD", "HOSP_NRD", "APRDRG", "APRDRG_Risk_Mortality", "APRDRG_Severity"}.issubset(spec["source_columns"]["severity"]):
        raise RuntimeError("Frozen severity projection lacks required fields")
    if "HOSP_NRD" not in spec["source_columns"]["hospital"] or not {"HOSP_NRD", "YEAR", "CCR_NRD", "WAGEINDEX"}.issubset(spec["source_columns"]["ccr"]):
        raise RuntimeError("Frozen hospital/CCR projections lack HOSP_NRD")
    code_sets = spec.get("code_sets")
    needed_sets = {"known_cm", "known_pcs", "ap_prefixes", "biliary", "sepsis_or_organ", "PR.1", "PR.2", "PR.3", "PR.4"}
    if not isinstance(code_sets, dict) or set(code_sets) != needed_sets:
        raise RuntimeError("Frozen ETL spec lacks exact 2022 ontology/PRA code sets")
    if not all(isinstance(values, list) and all(isinstance(x, str) for x in values) for values in code_sets.values()):
        raise RuntimeError("Frozen code sets must be lists of strings")
    if not isinstance(spec.get("cpi_to_2021_factor"), (int, float)) or not math.isfinite(float(spec["cpi_to_2021_factor"])):
        raise RuntimeError("CPI conversion must be a frozen finite constant")
    thresholds = spec.get("auxiliary_thresholds")
    if not isinstance(thresholds, dict) or not isinstance(thresholds.get("high_cost_threshold_2021_usd"), (int, float)):
        raise RuntimeError("Frozen ETL spec lacks high-cost threshold")
    if not isinstance(thresholds.get("prolonged_los_days"), (int, float)):
        raise RuntimeError("Frozen ETL spec lacks prolonged-LOS threshold")
    return spec


def normalize_code(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip().upper().replace(".", "")


def hash_identifier(year: int, values: pd.Series) -> pd.Series:
    """Stable year-scoped opaque ID; raw source identifiers never leave memory."""
    def one(value: object) -> str:
        raw = "" if value is None or pd.isna(value) else str(value)
        return hashlib.sha256(f"{year}:{raw}".encode("utf-8")).hexdigest()[:32]
    return values.map(one)


def _load_vocab(path: Path) -> dict[str, int]:
    value = read_json(path)
    processed_years = value.get("processed_years")
    if (value.get("frozen_after_year") != 2020 or not isinstance(processed_years, list)
            or processed_years[:3] != [2018, 2019, 2020]
            or any(not isinstance(year, int) or year < 2018 for year in processed_years)
            or processed_years != sorted(set(processed_years))):
        raise RuntimeError(f"Vocabulary is not exclusively frozen from 2018-2020: {path}")
    mapping = value.get("token_to_id", value)
    if not isinstance(mapping, dict):
        raise RuntimeError(f"Frozen vocabulary has no token_to_id map: {path}")
    declared_specials = {name: mapping[name] for name in SPECIAL_TOKEN_IDS if name in mapping}
    if declared_specials and declared_specials != SPECIAL_TOKEN_IDS:
        raise RuntimeError("Frozen vocabulary special-token mapping is incomplete or changed")
    observed_pairs = [(normalize_code(code), int(token)) for code, token in mapping.items()
                      if code not in SPECIAL_TOKEN_IDS]
    out = dict(observed_pairs)
    if "" in out or len(out) != len(observed_pairs) or any(token <= 3 for token in out.values()):
        raise RuntimeError("Frozen vocabulary assigns observed code to special token")
    return out


def _tokens(frame: pd.DataFrame, columns: tuple[str, ...], vocab: dict[str, int]) -> list[list[int]]:
    result: list[list[int]] = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        encoded: list[int] = []
        for value in row:
            code = normalize_code(value)
            encoded.append(3 if not code else int(vocab.get(code, 2)))  # missing / frozen OOV
        result.append(encoded)
    return result


def _code_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, list(columns)].map(normalize_code).to_numpy(dtype=object)


def _exact_code_membership(matrix: np.ndarray, values: set[str]) -> np.ndarray:
    """Exact O(N) membership against a frozen set, preserving input shape."""
    array = np.asarray(matrix, dtype=object)
    flat = array.reshape(-1)
    mask = np.fromiter((code in values for code in flat), dtype=bool, count=flat.size)
    return mask.reshape(array.shape)


def _flags_from_raw_codes(core: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Use raw ICD strings and frozen exact sets; never infer labels from OOV IDs."""
    dx = _code_matrix(core, tuple(spec["diagnosis_columns"]))
    pr = _code_matrix(core, tuple(spec["procedure_columns"]))
    sets = {name: set(map(normalize_code, values)) for name, values in spec["code_sets"].items()}
    principal = dx[:, 0]
    code_mask = _exact_code_membership
    principal_ap = np.fromiter((any(code.startswith(prefix) for prefix in sets["ap_prefixes"]) for code in principal), dtype=bool, count=len(core))
    any_ap = np.asarray([np.fromiter((any(code.startswith(prefix) for prefix in sets["ap_prefixes"]) for code in row), dtype=bool).any() for row in dx])
    principal_biliary = code_mask(principal, sets["biliary"])
    principal_sepsis = code_mask(principal, sets["sepsis_or_organ"])
    any_biliary = code_mask(dx, sets["biliary"]).any(axis=1)
    any_sepsis = code_mask(dx, sets["sepsis_or_organ"]).any(axis=1)
    nonempty_pr = pr != ""
    unknown_pr = (nonempty_pr & ~code_mask(pr, sets["known_pcs"])).sum(axis=1).astype(np.int16)
    unknown_principal = (principal == "") | ~code_mask(principal, sets["known_cm"])
    planned_unknown = unknown_principal | (unknown_pr > 0)
    pr1, pr3 = code_mask(pr, sets["PR.1"]).any(axis=1), code_mask(pr, sets["PR.3"]).any(axis=1)
    pr2, pr4 = code_mask(principal, sets["PR.2"]), code_mask(principal, sets["PR.4"])
    primary = np.select([principal_ap, principal_biliary, principal_sepsis], [1, 2, 3], default=4).astype(np.int8)
    return pd.DataFrame({
        "principal_ap": principal_ap, "any_ap": any_ap, "principal_biliary": principal_biliary,
        "any_biliary": any_biliary, "principal_sepsis_or_organ": principal_sepsis,
        "any_sepsis_or_organ": any_sepsis, "primary_cause_code": primary,
        "planned_status": np.where(planned_unknown, -1, (pr1 | pr2 | (pr3 & ~pr4)).astype(np.int8)).astype(np.int8),
        "algorithm_unknown": planned_unknown, "unknown_principal_diagnosis": unknown_principal,
        "unknown_procedure_count": unknown_pr, "pra_pr1": pr1, "pra_pr2": pr2, "pra_pr3": pr3, "pra_pr4": pr4,
    })


def _check_source_schema(path: Path, columns: tuple[str, ...], role: str) -> None:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(columns) - available
    if missing:
        raise RuntimeError(f"2022 {role} schema drift: missing frozen fields {sorted(missing)}")


def _read_source(path: Path, role: str, spec: dict[str, Any]) -> pd.DataFrame:
    columns = tuple(spec["source_columns"][role])
    fmt = spec.get("source_formats", {}).get(role, "parquet")
    if fmt == "parquet":
        _check_source_schema(path, columns, role)
        return pq.read_table(path, columns=list(columns)).to_pandas()
    if role == "core":
        # A 16.5m-row Core must enter through stream_core_csv_to_parquet.
        # Keeping this guard here prevents a future caller from accidentally
        # reinstating an apparently harmless, but unbounded, object DataFrame.
        raise RuntimeError("2022 Core CSV must use bounded stream_core_csv_to_parquet")
    # Hospital/CCR/Severity are narrow projections; the complete
    # physical column order is supplied by a locked HCUP file-spec projection;
    # pandas is told both that order and the exact minimal usecols list.
    all_columns = tuple(spec["source_all_columns"][role])
    header = 0 if bool(spec.get("csv_has_header", {}).get(role, False)) else None
    names = None if header == 0 else list(all_columns)
    try:
        frame = pd.read_csv(path, header=header, names=names, usecols=list(columns), dtype=object,
                            low_memory=False, quotechar=spec["csv_quotechar"][role])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"2022 {role} CSV source/schema drift") from exc
    if set(columns) - set(frame.columns):
        raise RuntimeError(f"2022 {role} schema drift after frozen CSV parse")
    return frame.loc[:, list(columns)]


def stream_core_csv_to_parquet(path: Path, spec: dict[str, Any], deps: dict[str, Path], output: Path,
                               *, chunksize: int = 131072) -> dict[str, int]:
    """Project and transform Core one bounded CSV chunk at a time.

    The private spool is inside the caller's atomic temporary directory and
    may contain linkage keys solely until fixed joins complete.  It is never
    published.  Peak Core memory is O(chunksize × frozen projection), not O(N).
    """
    if chunksize < 1:
        raise RuntimeError("Core CSV chunksize must be positive")
    all_columns = tuple(spec["source_all_columns"]["core"])
    columns = tuple(spec["source_columns"]["core"])
    header = 0 if bool(spec.get("csv_has_header", {}).get("core", False)) else None
    names = None if header == 0 else list(all_columns)
    dx_vocab, pr_vocab = _load_vocab(deps["diagnosis_vocabulary"]), _load_vocab(deps["procedure_vocabulary"])
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        reader = pd.read_csv(path, header=header, names=names, usecols=list(columns), dtype=object,
                             low_memory=False, quotechar=spec["csv_quotechar"]["core"], chunksize=chunksize)
        for core in reader:
            if set(columns) - set(core.columns):
                raise RuntimeError("2022 Core CSV schema drift after frozen streaming parse")
            if core.empty:
                continue
            flags = _flags_from_raw_codes(core, spec)
            chunk = core.loc[:, list(spec["core_passthrough_columns"])].copy()
            # Private joins need only opaque derived keys: no raw identifier is
            # persisted, even in the temporary Core spool.
            chunk.insert(0, "year", 2022)
            chunk.insert(1, "encounter_hash", hash_identifier(2022, core["KEY_NRD"]))
            chunk.insert(2, "patient_hash", hash_identifier(2022, core["NRD_VisitLink"]))
            chunk.insert(3, "hospital_hash", hash_identifier(2022, core["HOSP_NRD"]))
            # Stable schemas across chunks: CSV inference must not turn an
            # all-missing first chunk into Arrow null and later chunks into a
            # different physical type.  Downstream numeric use is explicit.
            for column in spec["core_passthrough_columns"]:
                chunk[column] = chunk[column].astype("string").fillna("")
            chunk["dx_tokens"] = _tokens(core, tuple(spec["diagnosis_columns"]), dx_vocab)
            chunk["pr_tokens"] = _tokens(core, tuple(spec["procedure_columns"]), pr_vocab)
            prday_columns = tuple(spec.get("prday_columns", ()))
            chunk["prday"] = (core.loc[:, list(prday_columns)].apply(pd.to_numeric, errors="coerce").fillna(-999).astype("int16").values.tolist()
                              if prday_columns else [[None] * len(spec["procedure_columns"]) for _ in range(len(core))])
            chunk = pd.concat([chunk.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
            if writer is None:
                writer = pq.ParquetWriter(output, pa.Table.from_pandas(chunk, preserve_index=False).schema,
                                          compression="zstd", compression_level=6)
            writer.write_table(pa.Table.from_pandas(chunk, preserve_index=False), row_group_size=131072)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    if not rows:
        raise RuntimeError("2022 Core CSV streaming projection is empty")
    return {"rows": rows, "chunksize": chunksize}


def read_frozen_sources(paths: dict[str, Path], spec: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Read exactly the frozen projections from already extracted source parquet files."""
    if set(paths) != set(SOURCE_ROLES):
        raise RuntimeError("Exactly core/severity/hospital/ccr explicit source paths are required")
    tables: dict[str, pd.DataFrame] = {}
    for role in SOURCE_ROLES:
        path = paths[role].resolve()
        if not path.is_file():
            raise RuntimeError(f"Missing explicit 2022 {role} source")
        tables[role] = _read_source(path, role, spec)
        if tables[role].empty:
            raise RuntimeError(f"2022 {role} source is empty")
    return tables


def _write_table(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd", compression_level=6,
                   row_group_size=131072, use_dictionary=True, write_statistics=True)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def _sql_identifier(value: str) -> str:
    """Quote a schema identifier that was already validated in the frozen spec."""
    return '"' + value.replace('"', '""') + '"'


def _sql_string(value: str | Path) -> str:
    """Quote a local path for a DuckDB statement (never shell-interpolated)."""
    return "'" + str(value).replace("'", "''") + "'"


def _copy_duckdb(con: duckdb.DuckDBPyConnection, query: str, output: Path) -> None:
    """Materialise an immutable parquet artifact without materialising a DataFrame."""
    con.execute("COPY (" + query + ") TO " + _sql_string(output) + " (FORMAT PARQUET, COMPRESSION ZSTD)")


def _hash_noncore_sources_to_parquet(paths: dict[str, Path], spec: dict[str, Any], scratch: Path) -> dict[str, Path]:
    """Build narrow hashed source projections for disk joins.

    Severity has only five frozen fields and hospital/CCR are hospital-level
    tables.  Unlike Core, their source projections are bounded enough for a
    narrow in-memory conversion; raw linkage values are discarded before any
    private spool is written.
    """
    hashed: dict[str, Path] = {}
    for role in ("severity", "hospital", "ccr"):
        frame = _read_source(paths[role], role, spec)
        if frame.empty:
            raise RuntimeError(f"2022 {role} source is empty")
        if role == "severity":
            if frame["KEY_NRD"].isna().any() or frame["KEY_NRD"].duplicated().any():
                raise RuntimeError("2022 Severity encounter linkage key is missing or duplicated")
            compact = frame.drop(columns=["KEY_NRD", "HOSP_NRD"], errors="ignore").copy()
            compact.insert(0, "year", 2022)
            compact.insert(1, "encounter_hash", hash_identifier(2022, frame["KEY_NRD"]))
        else:
            if frame["HOSP_NRD"].isna().any() or frame["HOSP_NRD"].duplicated().any():
                raise RuntimeError(f"2022 {role} hospital linkage key is missing or duplicated")
            if role == "ccr":
                ccr_year = pd.to_numeric(frame["YEAR"], errors="coerce")
                if ccr_year.isna().any() or not ccr_year.eq(2022).all():
                    raise RuntimeError("2022 CCR source/schema has a non-2022 YEAR value")
            compact = frame.drop(columns=["HOSP_NRD", "YEAR"] if role == "ccr" else ["HOSP_NRD"]).copy()
            compact.insert(0, "hospital_hash", hash_identifier(2022, frame["HOSP_NRD"]))
        _reject_raw_identifiers(compact, f"{role}_hashed_spool")
        output = scratch / f"{role}_hashed.parquet"
        _write_table(compact, output)
        hashed[role] = output
    return hashed


def _build_csv_core_artifacts(paths: dict[str, Path], spec: dict[str, Any], deps: dict[str, Path],
                              temporary: Path, threads: int) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Build the CSV-Core route with disk joins and a bounded AP-only DataFrame.

    The complete Core never returns to pandas after streaming.  DuckDB joins
    and writes all annual artifacts directly to Parquet; only admissions for
    AP index patients are fetched for the prespecified episode/history logic.
    """
    scratch = temporary / "_private_spool"
    scratch.mkdir(parents=True, exist_ok=False)
    core_spool = scratch / "core_projected.parquet"
    try:
        stream_core_csv_to_parquet(paths["core"], spec, deps, core_spool)
        noncore = _hash_noncore_sources_to_parquet(paths, spec, scratch)
        con = duckdb.connect(str(scratch / "joins.duckdb"))
        try:
            con.execute(f"PRAGMA threads={threads}")
            # Keep DuckDB spilling to the private spool rather than silently
            # consuming host memory for the annual sort/join.
            con.execute("SET memory_limit='16GB'")
            con.execute("SET temp_directory=" + _sql_string(scratch / "duckdb_tmp"))
            con.execute("CREATE VIEW core AS SELECT * FROM read_parquet(" + _sql_string(core_spool) + ")")
            con.execute("CREATE VIEW severity AS SELECT * FROM read_parquet(" + _sql_string(noncore["severity"]) + ")")
            con.execute("CREATE VIEW hospital AS SELECT * FROM read_parquet(" + _sql_string(noncore["hospital"]) + ")")
            con.execute("CREATE VIEW ccr AS SELECT * FROM read_parquet(" + _sql_string(noncore["ccr"]) + ")")
            core_count = int(con.execute("SELECT count(*) FROM core").fetchone()[0])
            if core_count < 1 or int(con.execute("SELECT count(DISTINCT encounter_hash) FROM core").fetchone()[0]) != core_count:
                raise RuntimeError("2022 Core streaming spool has missing or duplicate encounter keys")

            passthrough = [_sql_identifier(column) for column in spec["core_passthrough_columns"]]
            severity_fields = [_sql_identifier(column) for column in spec["source_columns"]["severity"] if column not in {"KEY_NRD", "HOSP_NRD"}]
            hospital_fields = [_sql_identifier(column) for column in spec["source_columns"]["hospital"] if column != "HOSP_NRD"]
            ccr_fields = [_sql_identifier(column) for column in spec["source_columns"]["ccr"] if column not in {"HOSP_NRD", "YEAR"}]
            select_core = ["c.year", "c.encounter_hash", "c.patient_hash", "c.hospital_hash"] + [f"c.{column}" for column in passthrough]
            select_core += ["c.dx_tokens", "c.pr_tokens", "c.prday", "c.principal_ap", "c.any_ap", "c.principal_biliary", "c.any_biliary", "c.principal_sepsis_or_organ", "c.any_sepsis_or_organ", "c.primary_cause_code", "c.planned_status", "c.algorithm_unknown", "c.unknown_principal_diagnosis", "c.unknown_procedure_count", "c.pra_pr1", "c.pra_pr2", "c.pra_pr3", "c.pra_pr4"]
            select_aux = [f"s.{column}" for column in severity_fields] + [f"h.{column}" for column in hospital_fields] + [f"r.{column}" for column in ccr_fields]
            # CCR is a cost-conversion lookup, not an eligibility source.  A
            # missing hospital CCR must therefore mask only cost-derived
            # labels; it must never remove an AP index or readmission.
            base_query = "SELECT " + ", ".join(select_core + select_aux) + " FROM core c INNER JOIN severity s USING (encounter_hash) INNER JOIN hospital h USING (hospital_hash) LEFT JOIN ccr r USING (hospital_hash)"
            joined_count = int(con.execute("SELECT count(*) FROM (" + base_query + ")").fetchone()[0])
            if joined_count != core_count:
                raise RuntimeError("2022 disk join lost or expanded Core encounters")
            # Cost validity is intentionally separate from the label: invalid
            # charge/CCR combinations remain nullable labels, never negatives.
            threshold = float(spec["auxiliary_thresholds"]["high_cost_threshold_2021_usd"])
            cpi = float(spec["cpi_to_2021_factor"])
            prolonged = float(spec["auxiliary_thresholds"]["prolonged_los_days"])
            admissions_query = "SELECT base.*, " + (
                "CASE WHEN try_cast(TOTCHG AS DOUBLE) > 0 AND try_cast(CCR_NRD AS DOUBLE) > 0 "
                f"THEN try_cast(TOTCHG AS DOUBLE) * try_cast(CCR_NRD AS DOUBLE) * {cpi!r} ELSE NULL END AS cost_2021_usd, "
                "CASE WHEN try_cast(TOTCHG AS DOUBLE) > 0 AND try_cast(CCR_NRD AS DOUBLE) > 0 "
                f"THEN CAST((try_cast(TOTCHG AS DOUBLE) * try_cast(CCR_NRD AS DOUBLE) * {cpi!r}) > {threshold!r} AS TINYINT) ELSE NULL END AS high_cost_label, "
                "CAST(try_cast(TOTCHG AS DOUBLE) > 0 AND try_cast(CCR_NRD AS DOUBLE) > 0 AS BOOLEAN) AS high_cost_observed, "
                f"CAST(COALESCE(try_cast(LOS AS DOUBLE) > {prolonged!r}, FALSE) AS TINYINT) AS prolonged_los_label, "
                "CAST(COALESCE(try_cast(DIED AS INTEGER) = 1, FALSE) AS TINYINT) AS in_hospital_death_label "
                "FROM (" + base_query + ") base"
            )
            _copy_duckdb(con, admissions_query, temporary / "admissions_model.parquet")
            _copy_duckdb(con, "SELECT year, encounter_hash, patient_hash, " + ", ".join(f"{column}" for column in passthrough) + ", dx_tokens, pr_tokens, prday FROM core", temporary / "core_compact.parquet")
            _copy_duckdb(con, "SELECT * FROM severity", temporary / "severity_hashed.parquet")
            _copy_duckdb(con, "SELECT * FROM hospital", temporary / "hospital_compact.parquet")
            _copy_duckdb(con, "SELECT * FROM ccr", temporary / "ccr_compact.parquet")
            _copy_duckdb(con, "SELECT h.* EXCLUDE (hospital_hash), h.hospital_hash, r.* EXCLUDE (hospital_hash) FROM hospital h LEFT JOIN ccr r USING (hospital_hash)", temporary / "aux_compact.parquet")
            _copy_duckdb(con, "SELECT year, encounter_hash, principal_ap, any_ap, principal_biliary, any_biliary, principal_sepsis_or_organ, any_sepsis_or_organ, primary_cause_code, planned_status, algorithm_unknown, unknown_principal_diagnosis, unknown_procedure_count, pra_pr1, pra_pr2, pra_pr3, pra_pr4 FROM core", temporary / "exact_label_flags.parquet")
            con.execute("CREATE VIEW admissions AS SELECT * FROM read_parquet(" + _sql_string(temporary / "admissions_model.parquet") + ")")
            related_query = "SELECT a.* FROM admissions a INNER JOIN (SELECT DISTINCT patient_hash FROM admissions WHERE principal_ap) p USING (patient_hash)"
            related_count = int(con.execute("SELECT count(*) FROM (" + related_query + ")").fetchone()[0])
            if related_count > MAX_AP_RELATED_ROWS:
                raise RuntimeError(f"2022 AP-related in-memory extraction exceeds fixed safety bound ({MAX_AP_RELATED_ROWS})")
            admissions_for_episodes = con.execute(related_query).fetchdf()
        finally:
            con.close()
        artifacts = {name: identity(temporary / f"{name}.parquet") for name in (
            "core_compact", "severity_hashed", "hospital_compact", "ccr_compact", "aux_compact", "exact_label_flags", "admissions_model"
        )}
        return artifacts, admissions_for_episodes
    finally:
        # All spool contents are private, non-published intermediates.  This
        # also removes the DuckDB database and any temporary sort runs.
        shutil.rmtree(scratch, ignore_errors=True)


def _reject_raw_identifiers(frame: pd.DataFrame, context: str) -> None:
    survived = FORBIDDEN_IDENTIFIERS & set(frame.columns)
    if survived:
        raise RuntimeError(f"Raw NRD identifier survived in {context}: {sorted(survived)}")


def _build_admissions(sources: dict[str, pd.DataFrame], spec: dict[str, Any], deps: dict[str, Path]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    core, severity, hospital, ccr = (sources[role].copy() for role in SOURCE_ROLES)
    if core["KEY_NRD"].isna().any() or core["NRD_VisitLink"].isna().any() or core["HOSP_NRD"].isna().any():
        raise RuntimeError("Core source has missing raw linkage keys")
    if core["KEY_NRD"].duplicated().any() or severity["KEY_NRD"].duplicated().any():
        raise RuntimeError("Core/severity source has duplicate frozen encounter key")
    if hospital["HOSP_NRD"].duplicated().any() or ccr["HOSP_NRD"].duplicated().any():
        raise RuntimeError("Hospital/CCR source has duplicate HOSP_NRD")
    ccr_year = pd.to_numeric(ccr["YEAR"], errors="coerce")
    if ccr_year.isna().any() or not ccr_year.eq(2022).all():
        raise RuntimeError("2022 CCR source/schema has a non-2022 YEAR value")
    dx_vocab, pr_vocab = _load_vocab(deps["diagnosis_vocabulary"]), _load_vocab(deps["procedure_vocabulary"])
    flags = _flags_from_raw_codes(core, spec)
    core_compact = core.loc[:, list(spec["core_passthrough_columns"])].copy()
    core_compact.insert(0, "year", 2022)
    core_compact.insert(1, "encounter_hash", hash_identifier(2022, core["KEY_NRD"]))
    core_compact.insert(2, "patient_hash", hash_identifier(2022, core["NRD_VisitLink"]))
    core_compact["dx_tokens"] = _tokens(core, tuple(spec["diagnosis_columns"]), dx_vocab)
    core_compact["pr_tokens"] = _tokens(core, tuple(spec["procedure_columns"]), pr_vocab)
    prday_columns = tuple(spec.get("prday_columns", ()))
    if prday_columns:
        if not set(prday_columns).issubset(core.columns):
            raise RuntimeError("Frozen PRDAY projection is absent from 2022 core")
        core_compact["prday"] = core.loc[:, list(prday_columns)].apply(pd.to_numeric, errors="coerce").values.tolist()
    else:
        core_compact["prday"] = [[None] * len(spec["procedure_columns"]) for _ in range(len(core))]
    severity_hashed = severity.loc[:, ["APRDRG", "APRDRG_Risk_Mortality", "APRDRG_Severity"]].copy()
    severity_hashed.insert(0, "year", 2022); severity_hashed.insert(1, "encounter_hash", hash_identifier(2022, severity["KEY_NRD"]))
    hospital_compact = hospital.drop(columns=["HOSP_NRD"]).copy(); hospital_compact.insert(0, "hospital_hash", hash_identifier(2022, hospital["HOSP_NRD"]))
    ccr_compact = ccr.drop(columns=["HOSP_NRD", "YEAR"]).copy(); ccr_compact.insert(0, "hospital_hash", hash_identifier(2022, ccr["HOSP_NRD"]))
    for frame, name in ((core_compact, "core_compact"), (severity_hashed, "severity_hashed"), (hospital_compact, "hospital_compact"), (ccr_compact, "ccr_compact")):
        _reject_raw_identifiers(frame, name)
    raw = core.copy()
    raw["encounter_hash"] = hash_identifier(2022, raw["KEY_NRD"])
    raw["patient_hash"] = hash_identifier(2022, raw["NRD_VisitLink"])
    raw["hospital_hash"] = hash_identifier(2022, raw["HOSP_NRD"])
    raw["year"] = 2022
    raw["dx_tokens"] = core_compact["dx_tokens"]
    raw["pr_tokens"] = core_compact["pr_tokens"]
    raw["prday"] = core_compact["prday"]
    raw = pd.concat([raw.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
    raw = raw.merge(severity.drop(columns=["HOSP_NRD"]), on="KEY_NRD", how="inner", validate="one_to_one")
    raw = raw.merge(hospital, on="HOSP_NRD", how="inner", validate="many_to_one")
    # CCR absence is valid cost-label missingness, never a reason to alter the
    # main admission/readmission cohort.
    raw = raw.merge(ccr.drop(columns=["YEAR"]), on="HOSP_NRD", how="left", validate="many_to_one", suffixes=("", "__ccr"))
    if len(raw) != len(core):
        raise RuntimeError("2022 admissions join lost or expanded core encounters")
    ccr_column = "CCR_NRD__ccr" if "CCR_NRD__ccr" in raw.columns else "CCR_NRD"
    cost = pd.to_numeric(raw["TOTCHG"], errors="coerce") * pd.to_numeric(raw[ccr_column], errors="coerce")
    raw["nominal_cost"] = np.where((cost > 0) & np.isfinite(cost), cost, np.nan)
    raw["cost_2021_usd"] = raw["nominal_cost"] * float(spec["cpi_to_2021_factor"])
    thresholds = spec["auxiliary_thresholds"]
    valid_cost = np.isfinite(pd.to_numeric(raw["TOTCHG"], errors="coerce")) & (pd.to_numeric(raw["TOTCHG"], errors="coerce") > 0) & np.isfinite(pd.to_numeric(raw[ccr_column], errors="coerce")) & (pd.to_numeric(raw[ccr_column], errors="coerce") > 0)
    raw["high_cost_label"] = pd.Series(np.where(valid_cost, (raw["cost_2021_usd"] > float(thresholds["high_cost_threshold_2021_usd"])).astype(np.int8), pd.NA), dtype="Int8")
    raw["high_cost_observed"] = valid_cost.astype(bool)
    raw["prolonged_los_label"] = (pd.to_numeric(raw["LOS"], errors="coerce") > float(thresholds["prolonged_los_days"])).fillna(False).astype(np.int8)
    raw["in_hospital_death_label"] = pd.to_numeric(raw["DIED"], errors="coerce").eq(1).astype(np.int8)
    exact = pd.concat([raw[["year", "encounter_hash"]], flags.reset_index(drop=True)], axis=1)
    admissions = raw.drop(columns=[column for column in FORBIDDEN_IDENTIFIERS if column in raw.columns])
    _reject_raw_identifiers(admissions, "admissions_model")
    aux_compact = hospital_compact.merge(ccr_compact, on="hospital_hash", how="left", validate="one_to_one")
    return {"core_compact": core_compact, "severity_hashed": severity_hashed, "hospital_compact": hospital_compact,
            "ccr_compact": ccr_compact, "aux_compact": aux_compact, "exact_label_flags": exact}, admissions


def _build_episodes(admissions: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Freeze the prespecified annual linkage/eligibility rules; never inspect results to choose them."""
    records: list[dict[str, Any]] = []
    for _, group in admissions.sort_values(["patient_hash", "NRD_DaysToEvent", "encounter_hash"], kind="mergesort").groupby("patient_hash", sort=False):
        group = group.reset_index(drop=True)
        for pos, index in group.iterrows():
            day, los = pd.to_numeric(pd.Series([index["NRD_DaysToEvent"]]), errors="coerce").iloc[0], pd.to_numeric(pd.Series([index["LOS"]]), errors="coerce").iloc[0]
            if not np.isfinite(day) or not np.isfinite(los):
                continue
            later = group.iloc[pos + 1:].copy()
            gap = pd.to_numeric(later["NRD_DaysToEvent"], errors="coerce") - float(day) - max(0.0, float(los))
            immediate = gap.iloc[0] if len(gap) else np.nan
            base = bool(index["principal_ap"]) and float(index["AGE"]) >= 18 and int(index["in_hospital_death_label"]) == 0 and int(index["DISPUNIFORM"]) not in (2, 20) and int(index["DMONTH"]) <= 11 and (not np.isfinite(immediate) or immediate >= 1)
            candidates = later.loc[gap.between(1, 30, inclusive="both")]
            unknown = candidates.loc[candidates["planned_status"].eq(-1)]
            unplanned = candidates.loc[candidates["planned_status"].eq(0)]
            first_unknown = float(unknown["NRD_DaysToEvent"].min()) if len(unknown) else math.inf
            first_unplanned = float(unplanned["NRD_DaysToEvent"].min()) if len(unplanned) else math.inf
            uncertain = first_unknown <= first_unplanned
            if not base or uncertain:
                continue
            readmit = None if not len(unplanned) else unplanned.sort_values(["NRD_DaysToEvent", "encounter_hash"], kind="mergesort").iloc[0]
            leaf = 0 if readmit is None else int(readmit["primary_cause_code"])
            row = index.to_dict()
            row.update({"analysis_partition": "test", "primary_analysis_eligible": True,
                        "any_unplanned_readmission_30d": int(leaf > 0), "readmission_leaf": leaf,
                        "ap_specific_readmission_30d": int(leaf == 1), "biliary_readmission_30d": int(leaf == 2),
                        "sepsis_or_organ_readmission_30d": int(leaf == 3)})
            records.append(row)
    episodes = pd.DataFrame(records)
    if episodes.empty:
        raise RuntimeError("Frozen 2022 AP eligibility produced no episode rows")
    needed = set(spec["episode_output_columns"])
    if missing := needed - set(episodes.columns):
        raise RuntimeError(f"Frozen 2022 episode output projection cannot be satisfied: {sorted(missing)}")
    episodes = episodes.loc[:, list(spec["episode_output_columns"])].copy()
    if not episodes["analysis_partition"].eq("test").all() or episodes["encounter_hash"].duplicated().any():
        raise RuntimeError("2022 episode test partition/key invariant failed")
    _reject_raw_identifiers(episodes, "ap_episodes")
    return episodes


def _history(admissions: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    by_patient = {patient: group.sort_values(["NRD_DaysToEvent", "encounter_hash"], kind="mergesort")
                  for patient, group in admissions.groupby("patient_hash", sort=False)}
    for index in episodes.itertuples(index=False):
        group = by_patient[index.patient_hash]
        start = float(index.NRD_DaysToEvent)
        prior = group[(pd.to_numeric(group["NRD_DaysToEvent"], errors="coerce") < start)
                      & (pd.to_numeric(group["NRD_DaysToEvent"], errors="coerce") + pd.to_numeric(group["LOS"], errors="coerce").fillna(0).clip(lower=0) <= start)].copy()
        if len(prior): prior["gap"] = start - (pd.to_numeric(prior["NRD_DaysToEvent"], errors="coerce") + pd.to_numeric(prior["LOS"], errors="coerce").fillna(0).clip(lower=0))
        else: prior["gap"] = pd.Series(dtype=float)
        record: dict[str, Any] = {"year": 2022, "encounter_hash": index.encounter_hash, "patient_hash": index.patient_hash,
                                  "analysis_partition": "test", "prior_count_ytd": int(len(prior)),
                                  "days_since_prior_discharge": int(prior["gap"].min()) if len(prior) else -1,
                                  "history_30d_fully_observable": int(index.DMONTH) >= 2,
                                  "history_90d_fully_observable": int(index.DMONTH) >= 4,
                                  "history_180d_fully_observable": int(index.DMONTH) >= 7}
        for window in (30, 90, 180):
            current = prior[prior["gap"].between(0, window, inclusive="both")]
            record[f"prior_count_{window}d"] = int(len(current))
            if window == 180:
                record.update({"prior_ed_count_180d": int((pd.to_numeric(current.get("HCUP_ED", 0), errors="coerce").fillna(0) > 0).sum()),
                               "prior_nonelective_count_180d": int((pd.to_numeric(current.get("ELECTIVE", 0), errors="coerce").fillna(0) != 1).sum()),
                               "prior_ap_count_180d": int(current["principal_ap"].sum()), "prior_biliary_count_180d": int(current["principal_biliary"].sum()),
                               "prior_sepsis_or_organ_count_180d": int(current["principal_sepsis_or_organ"].sum()),
                               "prior_los_sum_180d": int(pd.to_numeric(current["LOS"], errors="coerce").fillna(0).clip(lower=0).sum()),
                               "prior_max_severity_180d": int(pd.to_numeric(current["APRDRG_Severity"], errors="coerce").max()) if len(current) else 0,
                               "prior_max_mortality_risk_180d": int(pd.to_numeric(current["APRDRG_Risk_Mortality"], errors="coerce").max()) if len(current) else 0,
                               "prior_dx_tokens_180d": sorted({int(token) for values in current["dx_tokens"] for token in values if int(token) > 3}),
                               "prior_pr_tokens_180d": sorted({int(token) for values in current["pr_tokens"] for token in values if int(token) > 3})})
        records.append(record)
    frame = pd.DataFrame(records)
    if len(frame) != len(episodes) or frame["encounter_hash"].duplicated().any():
        raise RuntimeError("2022 same-year history cardinality failure")
    return frame.loc[:, ["year", "encounter_hash", "patient_hash", "analysis_partition", *HISTORY_COLUMNS]]


def _prepare_output(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    partials = list(output_dir.parent.glob(output_dir.name + ".partial*")) if output_dir.parent.exists() else []
    if output_dir.exists() or partials:
        raise RuntimeError("2022 ETL output or partial output already exists and is immutable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))


def run_locked_2022_etl(source_paths: dict[str, Path], unlock_lock: Path, frozen_etl_spec: Path,
                        output_dir: Path, *, threads: int = 8) -> dict[str, Any]:
    """Production core: source files are opened only after every lock/dependency gate."""
    if not 1 <= threads <= 8:
        raise RuntimeError("--threads must be between 1 and 8")
    lock = validate_preflight(unlock_lock, frozen_etl_spec)
    spec = validate_etl_spec(frozen_etl_spec, lock)
    temporary = _prepare_output(output_dir)
    try:
        pa.set_cpu_count(threads)
        deps = {role: Path(path).resolve() for role, path in spec["frozen_dependencies"].items()}
        core_format = spec.get("source_formats", {}).get("core", "parquet")
        if core_format == "csv":
            if not source_paths["core"].is_file():
                raise RuntimeError("Missing explicit 2022 core source")
            manifest_artifacts, admissions = _build_csv_core_artifacts(source_paths, spec, deps, temporary, threads)
        else:
            sources = read_frozen_sources(source_paths, spec)
            artifacts, admissions = _build_admissions(sources, spec, deps)
            artifacts["admissions_model"] = admissions
            manifest_artifacts = {}
            for name, frame in artifacts.items():
                manifest_artifacts[name] = _write_table(frame, temporary / f"{name}.parquet")
        episodes = _build_episodes(admissions, spec)
        history = _history(admissions, episodes)
        if core_format == "csv":
            # The full annual admissions table was already written by DuckDB;
            # only these AP-derived artifacts originate from the bounded frame.
            manifest_artifacts = {
                name: {"file": Path(item["path"]).name, "bytes": item["bytes"], "sha256": item["sha256"]}
                for name, item in manifest_artifacts.items()
            }
        manifest_artifacts["ap_episodes"] = _write_table(episodes, temporary / "ap_episodes.parquet")
        manifest_artifacts["ap_history_2022"] = _write_table(history, temporary / "ap_history_2022.parquet")
        manifest = {"status": "PASS_LOCKED_2022_LOCAL_ETL", "schema_version": SCHEMA_VERSION,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "2022_accessed": True, "one_shot": True,
                    "test_partition": "test", "data_dependent_adaptation": False,
                    "frozen_vocabulary_only": True, "new_2022_codes_mapped_to_oov_only": True,
                    "raw_identifiers_removed": True, "no_outcome_small_cells_logged": True,
                    "high_cost_missingness_preserved_as_mask": True,
                    "artifacts": manifest_artifacts,
                    "input_identity": {role: identity(path) for role, path in source_paths.items()},
                    "unlock_lock": identity(unlock_lock), "frozen_etl_spec": identity(frozen_etl_spec),
                    }
        manifest_path = temporary / "manifest.json"; manifest_path.write_text(stable_json(manifest), encoding="utf-8")
        manifest_digest = sha256(manifest_path)
        (temporary / "manifest.json.sha256").write_text(f"{manifest_digest}  manifest.json\n", encoding="ascii")
        os.replace(temporary, output_dir.resolve())
        return {**manifest, "manifest_sha256": manifest_digest, "output": str(output_dir.resolve())}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def extract_archives(archive_paths: dict[str, Path], spec: dict[str, Any], temporary_root: Path,
                     seven_zip: Path) -> dict[str, Path]:
    """Extract fixed members only; credentials are always supplied interactively by 7-Zip."""
    members = spec.get("archive_members")
    if not isinstance(members, dict) or set(members) != set(SOURCE_ROLES):
        raise RuntimeError("Frozen ETL spec must name one exact archive member per source role")
    if set(archive_paths) != set(SOURCE_ROLES):
        raise RuntimeError("Exactly one explicit archive path per source role is required")
    if not seven_zip.is_file():
        raise RuntimeError("Frozen 7-Zip executable is unavailable")
    paths: dict[str, Path] = {}
    for role in SOURCE_ROLES:
        archive, member = archive_paths[role].resolve(), members[role]
        if not archive.is_file() or not isinstance(member, str) or not member or ".." in Path(member).parts:
            raise RuntimeError(f"Invalid frozen archive/member for {role}")
        destination = temporary_root / "raw" / role; destination.mkdir(parents=True, exist_ok=True)
        # No '-pPASSWORD', stdin payload, log file, or command-line credential.
        completed = subprocess.run([str(seven_zip), "x", "-y", f"-o{destination}", str(archive), member], check=False)
        if completed.returncode:
            raise RuntimeError(f"Secure interactive extraction failed for 2022 {role}")
        extracted = destination / member
        if not extracted.is_file():
            raise RuntimeError(f"Frozen archive member was not extracted: {role}")
        fmt = spec.get("source_formats", {}).get(role, "parquet")
        expected_suffix = ".parquet" if fmt == "parquet" else ".csv"
        if extracted.suffix.lower() != expected_suffix:
            raise RuntimeError(f"Frozen {role} archive member suffix does not match its locked source format")
        paths[role] = extracted
    return paths


def _parse_pairs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, sep, path = value.partition("=")
        if sep != "=" or role not in SOURCE_ROLES or not path or role in result:
            raise SystemExit("Each source/archive must be one unique role=path for core,severity,hospital,ccr")
        result[role] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlock-lock", type=Path, required=True)
    parser.add_argument("--frozen-etl-spec", type=Path)
    parser.add_argument("--archive", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seven-zip", type=Path, default=Path(r"C:\Program Files\7-Zip\7z.exe"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        if args.archive or args.output_dir is not None:
            raise SystemExit("--validate-only accepts no archive/source/output arguments")
        result = validate_preflight(args.unlock_lock, args.frozen_etl_spec)
        print(stable_json({"status": "PASS_PRE_2022_LOCAL_ETL_PREFLIGHT", "sealed_test_year": result["sealed_test_year"]}))
        return
    if args.frozen_etl_spec is None or args.output_dir is None:
        raise SystemExit("Production run requires --frozen-etl-spec and --output-dir")
    # CLI deliberately permits only archive paths; direct parquet sources are an internal, synthetic-test API.
    archives = _parse_pairs(args.archive)
    lock = validate_preflight(args.unlock_lock, args.frozen_etl_spec)
    spec = validate_etl_spec(args.frozen_etl_spec, lock)
    outer = _prepare_output(args.output_dir)
    try:
        source_paths = extract_archives(archives, spec, outer, args.seven_zip)
        # run into a sibling temporary output; the archive extraction root is not published.
        result = run_locked_2022_etl(source_paths, args.unlock_lock, args.frozen_etl_spec, outer / "published", threads=args.threads)
        os.replace(outer / "published", args.output_dir.resolve())
        shutil.rmtree(outer, ignore_errors=True)
        print(stable_json(result))
    except Exception:
        shutil.rmtree(outer, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
