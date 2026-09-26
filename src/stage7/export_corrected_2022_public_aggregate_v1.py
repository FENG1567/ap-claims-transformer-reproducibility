#!/usr/bin/env python3
"""Export the corrected 2022 results as an aggregate-only public package.

The corrected evaluation and drift directories contain restricted artifacts as
well as the aggregate tables needed for a manuscript.  This module has a
small, explicit allow-list: it authenticates only the aggregate parquet files,
converts them to CSV, and never copies prediction tables or bootstrap
replicates.  A row containing a prespecified count field at or below ten is
suppressed before export.  Suppression is deliberately conservative: all
count and outcome-derived rate, interval, coverage, and performance fields
are made null while only non-identifying keys and status metadata remain.

The destination is immutable.  Existing destinations and stale partial
destinations fail closed, and publication is an atomic directory rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "stage7_public_aggregate_export_v1"
EVALUATION_STATUS = "PASS_LOCKED_2022_PRIMARY_EVALUATION"
DRIFT_STATUS = "PASS_LOCKED_2022_DRIFT_ANALYSIS"
EXPECTED_N_ROWS = 119_114
EXPECTED_BOOTSTRAPS = 1_000

# These are the only source artifacts this exporter is allowed to open.
EVALUATION_ARTIFACTS: dict[str, str] = {
    "metrics.parquet": "metrics.csv",
    "comparisons_patient_bootstrap.parquet": "comparisons_patient_bootstrap.csv",
    "comparisons_hospital_bootstrap.parquet": "comparisons_hospital_bootstrap.csv",
    "decision_curve.parquet": "decision_curve.csv",
    "conformal.parquet": "conformal.csv",
    "conformal_subgroups.parquet": "conformal_subgroups_public.csv",
}
DRIFT_ARTIFACTS: dict[str, str] = {
    "covariate_drift.parquet": "covariate_drift.csv",
    "label_drift.parquet": "label_drift_public.csv",
    "calibration_drift.parquet": "calibration_drift.csv",
    "coverage_drift.parquet": "coverage_drift_public.csv",
}

# The first five are the count contract.  The remaining names are count-like
# values that must also be removed when a row is suppressed, but do not by
# themselves trigger suppression (for example, n_replicates is a protocol
# property rather than a patient count).
TRIGGER_COUNT_FIELDS = ("n_rows", "event_count", "non_events", "missing_n", "valid_n")
OTHER_COUNT_FIELDS = {
    "events",
    "non_event_count",
    "missing_count",
    "weighted_event_count",
    "valid_weight",
    "weight_sum",
}
ALL_COUNT_FIELDS = set(TRIGGER_COUNT_FIELDS) | OTHER_COUNT_FIELDS

SAFE_NUMERIC_FIELDS = {
    "n_replicates",
    "replicates",
    "seed",
    "threshold_probability",
    "nominal_coverage",
    "nominal",
}

FORBIDDEN_COLUMN_TOKENS = (
    "patient_hash",
    "hospital_hash",
    "patient_id",
    "subject_id",
    "encounter_id",
    "record_id",
    "row_id",
    "sample_id",
    "prediction",
    "probability",
    "logit",
    "replicate",
)

DEPENDENT_FIELD_TOKENS = (
    "rate",
    "prevalence",
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "calibration",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "net_benefit",
    "coverage",
    "set_size",
    "singleton",
    "empty_set",
    "full_set",
    "abstention",
    "retained_",
    "abstained_",
    "error_rate",
    "ci",
    "change",
    "proportion",
    "population_stability",
    "standardized_mean_difference",
    "weighted_mean",
    "weighted_sd",
    "observed_",
)

KEY_FIELD_ORDER = (
    "domain",
    "outcome",
    "model",
    "comparison",
    "weighting",
    "cluster_unit",
    "conformal_set",
    "subgroup_dimension",
    "subgroup_value",
    "covariate",
    "frozen_level",
    "source_column",
    "kind",
    "metric",
    "threshold_probability",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable manifest: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Manifest must be a JSON object: {path.name}")
    return value


def _manifest_path(directory: Path, explicit: Path | None) -> Path:
    path = (explicit if explicit is not None else directory / "manifest.json").resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing input manifest: {path.name}")
    return path


def _safe_relative_name(name: str) -> str:
    candidate = Path(name)
    if candidate.is_absolute() or candidate.name != name or name in ("", ".", "..") or ".." in candidate.parts:
        raise RuntimeError(f"Manifest artifact name is not a simple relative filename: {name!r}")
    return name


def _artifact_from_manifest(directory: Path, manifest: dict[str, Any], source_name: str) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Input manifest lacks an artifacts object")
    if source_name not in artifacts or not isinstance(artifacts[source_name], dict):
        raise RuntimeError(f"Input manifest lacks required aggregate artifact: {source_name}")
    entry = artifacts[source_name]
    filename = _safe_relative_name(source_name)
    path = (directory / filename).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise RuntimeError(f"Missing aggregate artifact: {filename}")
    expected_bytes = entry.get("bytes")
    expected_hash = entry.get("sha256")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise RuntimeError(f"Invalid byte count for aggregate artifact: {filename}")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise RuntimeError(f"Invalid SHA256 for aggregate artifact: {filename}")
    actual_bytes = path.stat().st_size
    actual_hash = sha256(path)
    if actual_bytes != expected_bytes or actual_hash.lower() != expected_hash.lower():
        raise RuntimeError(f"Aggregate artifact hash/size mismatch: {filename}")
    return path


def _validate_input_manifest(directory: Path, explicit: Path | None, expected_status: str) -> tuple[dict[str, Any], Path]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Input directory is missing: {directory.name}")
    path = _manifest_path(directory, explicit)
    manifest = _load_json(path)
    if manifest.get("status") != expected_status:
        raise RuntimeError(f"Unexpected input manifest status for {directory.name}")
    return manifest, path


def _validate_evaluation_overview(manifest: dict[str, Any]) -> None:
    if manifest.get("n_rows") != EXPECTED_N_ROWS:
        raise RuntimeError("Evaluation manifest has unexpected n_rows")
    models = manifest.get("models")
    outcomes = manifest.get("co_primary_outcomes")
    if not isinstance(models, list) or len(models) != 13 or len(set(models)) != 13:
        raise RuntimeError("Evaluation manifest must contain exactly 13 frozen models")
    if not isinstance(outcomes, list) or len(outcomes) != 2 or len(set(outcomes)) != 2:
        raise RuntimeError("Evaluation manifest must contain exactly two co-primary outcomes")
    patient = manifest.get("patient_bootstrap")
    if not isinstance(patient, dict) or patient.get("replicates") != EXPECTED_BOOTSTRAPS:
        raise RuntimeError("Evaluation manifest lacks the required 1000 patient bootstrap")
    if manifest.get("hospital_bootstrap_included") is not True:
        raise RuntimeError("Evaluation manifest lacks the required hospital bootstrap")


def _validate_drift_overview(manifest: dict[str, Any]) -> None:
    if manifest.get("n_rows") != EXPECTED_N_ROWS:
        raise RuntimeError("Drift manifest has unexpected n_rows")
    bootstrap = manifest.get("bootstrap")
    if not isinstance(bootstrap, dict) or bootstrap.get("n_replicates") != EXPECTED_BOOTSTRAPS:
        raise RuntimeError("Drift manifest lacks the required 1000 patient bootstrap")
    if bootstrap.get("cluster_column", "patient_hash") != "patient_hash":
        raise RuntimeError("Drift bootstrap is not patient-clustered")


def _read_table(directory: Path, manifest: dict[str, Any], source_name: str) -> pd.DataFrame:
    path = _artifact_from_manifest(directory, manifest, source_name)
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover - engine-specific error text
        raise RuntimeError(f"Unable to read aggregate artifact: {source_name}") from exc
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"Aggregate artifact is not tabular: {source_name}")
    _reject_restricted_columns(frame, source_name)
    return frame


def _reject_restricted_columns(frame: pd.DataFrame, source_name: str) -> None:
    for column in frame.columns:
        normalized = str(column).strip().lower()
        if normalized in SAFE_NUMERIC_FIELDS:
            continue
        if any(token in normalized for token in FORBIDDEN_COLUMN_TOKENS):
            raise RuntimeError(f"Restricted row-level/replicate column in {source_name}: {column}")


def _validate_evaluation_tables(tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["metrics.parquet"]
    required = {"model", "outcome", "weighting"}
    if not required.issubset(metrics.columns):
        raise RuntimeError("Metrics table lacks model/outcome/weighting keys")
    if len(metrics) != 52 or metrics["model"].nunique() != 13 or metrics["outcome"].nunique() != 2 or metrics["weighting"].nunique() != 2:
        raise RuntimeError("Metrics table must contain exactly 52 rows (13 models x 2 outcomes x 2 weightings)")
    for source_name, expected_cluster in (("comparisons_patient_bootstrap.parquet", "patient_hash"), ("comparisons_hospital_bootstrap.parquet", "hospital_hash")):
        table = tables[source_name]
        needed = {"outcome", "weighting", "cluster_unit", "n_replicates", "n_defined"}
        if not needed.issubset(table.columns) or len(table) != 4:
            raise RuntimeError(f"{source_name} must contain four primary comparison rows")
        if table["cluster_unit"].astype(str).nunique() != 1 or table["cluster_unit"].astype(str).iloc[0] != expected_cluster:
            raise RuntimeError(f"{source_name} has the wrong cluster unit")
        if not (pd.to_numeric(table["n_replicates"], errors="coerce") == EXPECTED_BOOTSTRAPS).all():
            raise RuntimeError(f"{source_name} is not a 1000-replicate comparison")
        if not (pd.to_numeric(table["n_defined"], errors="coerce") == EXPECTED_BOOTSTRAPS).all():
            raise RuntimeError(f"{source_name} contains undefined bootstrap replicates")


def _validate_drift_tables(tables: dict[str, pd.DataFrame]) -> None:
    required = {"covariate_drift.parquet", "label_drift.parquet", "calibration_drift.parquet", "coverage_drift.parquet"}
    if set(tables) != required:
        raise RuntimeError("Drift table allow-list is incomplete")
    for name, table in tables.items():
        if table.empty:
            raise RuntimeError(f"Drift aggregate table is empty: {name}")


def _as_finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_triggered(value: Any) -> bool:
    number = _as_finite_number(value)
    return number is not None and number <= 10


def _row_key(row: pd.Series, index: int) -> str:
    parts: list[str] = []
    for name in KEY_FIELD_ORDER:
        if name not in row.index:
            continue
        value = row[name]
        if pd.isna(value):
            text = "<MISSING>"
        else:
            text = str(value)
        parts.append(f"{name}={text}")
    return "|".join(parts) if parts else f"aggregate_row={index + 1}"


def _is_reference_or_protocol_field(column: str) -> bool:
    normalized = column.lower()
    if normalized in SAFE_NUMERIC_FIELDS:
        return True
    if normalized.startswith("reference_") or normalized.endswith("_reference"):
        return True
    if normalized in {"scope", "mondrian_dimension", "abstention_rule", "analysis", "ci95_method", "coverage_ci95_method"}:
        return True
    return False


def _is_dependent_field(column: str) -> bool:
    normalized = column.lower()
    if normalized in ALL_COUNT_FIELDS or normalized in TRIGGER_COUNT_FIELDS:
        return True
    if normalized in {"event_tier", "status", "calibration_status", "discrimination_status", "coverage_ci95_status"}:
        return True
    if _is_reference_or_protocol_field(normalized):
        return False
    return any(token in normalized for token in DEPENDENT_FIELD_TOKENS)


def _suppress_row(row: pd.Series, index: int) -> tuple[pd.Series, dict[str, Any] | None]:
    trigger_fields = [name for name in TRIGGER_COUNT_FIELDS if name in row.index and _is_triggered(row[name])]
    if not trigger_fields:
        return row, None
    result = row.copy()
    for column in row.index:
        name = str(column)
        normalized = name.lower()
        if normalized in ALL_COUNT_FIELDS or normalized in TRIGGER_COUNT_FIELDS:
            result[column] = None
        elif normalized in {"event_tier", "status", "calibration_status", "discrimination_status", "coverage_ci95_status"}:
            result[column] = "SUPPRESSED_SMALL_CELL"
        elif _is_dependent_field(name):
            result[column] = None
        elif pd.api.types.is_number(row[column]) and not _is_reference_or_protocol_field(name):
            # Unknown numeric estimates are treated as outcome-derived rather
            # than risk accidentally publishing a count-derived statistic.
            result[column] = None
    audit = {
        "row_key": _row_key(row, index),
        "trigger_fields": ";".join(trigger_fields),
        "suppressed": True,
    }
    return result, audit


def suppress_small_cells(frame: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Suppress rows whose explicit count contract contains a value <= 10.

    The audit intentionally contains only a non-identifying key and the names
    of the triggering fields.  It never stores the original count values.
    """
    public_rows: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(frame.iterrows()):
        public_row, audit = _suppress_row(row, index)
        public_rows.append(public_row)
        if audit is not None:
            audit_rows.append({"source_file": source_file, **audit})
    if not public_rows:
        public = frame.copy()
    else:
        public = pd.DataFrame(public_rows, columns=frame.columns)
    return public, audit_rows


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", na_rep="")


def _prepare_output(output: Path, input_dirs: Iterable[Path]) -> Path:
    output = output.resolve()
    if output.exists() or list(output.parent.glob(output.name + ".partial*")):
        raise RuntimeError("Public aggregate output or partial output already exists and is immutable")
    input_resolved = [directory.resolve() for directory in input_dirs]
    if any(output == directory or directory in output.parents for directory in input_resolved):
        raise RuntimeError("Public output cannot be inside a restricted input directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))


def _evidence_entry(role: str, path: Path) -> dict[str, Any]:
    return {"role": role, "filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(stable_json(value) + "\n", encoding="utf-8")


def export_public_aggregate(
    evaluation_dir: Path,
    drift_dir: Path,
    output_dir: Path,
    evaluation_manifest: Path | None = None,
    drift_manifest: Path | None = None,
    evidence_manifests: Iterable[Path] = (),
) -> dict[str, Any]:
    """Authenticate and publish the fixed aggregate-only allow-list."""
    evaluation_dir = evaluation_dir.resolve()
    drift_dir = drift_dir.resolve()
    eval_info, eval_manifest_path = _validate_input_manifest(evaluation_dir, evaluation_manifest, EVALUATION_STATUS)
    drift_info, drift_manifest_path = _validate_input_manifest(drift_dir, drift_manifest, DRIFT_STATUS)
    _validate_evaluation_overview(eval_info)
    _validate_drift_overview(drift_info)

    evaluation_tables = {name: _read_table(evaluation_dir, eval_info, name) for name in EVALUATION_ARTIFACTS}
    drift_tables = {name: _read_table(drift_dir, drift_info, name) for name in DRIFT_ARTIFACTS}
    _validate_evaluation_tables(evaluation_tables)
    _validate_drift_tables(drift_tables)

    temporary = _prepare_output(output_dir, (evaluation_dir, drift_dir))
    try:
        audits: list[dict[str, Any]] = []
        for source_name, destination_name in {**EVALUATION_ARTIFACTS, **DRIFT_ARTIFACTS}.items():
            source_frame = evaluation_tables.get(source_name, drift_tables.get(source_name))
            if source_frame is None:  # pragma: no cover - allow-list is static
                raise RuntimeError(f"Missing allow-listed source table: {source_name}")
            public_frame, rows = suppress_small_cells(source_frame, source_name)
            audits.extend(rows)
            _write_csv(public_frame, temporary / destination_name)

        audit_frame = pd.DataFrame(audits, columns=["source_file", "row_key", "trigger_fields", "suppressed"])
        _write_csv(audit_frame, temporary / "public_suppression_audit.csv")

        summary = {
            "status": "PASS_PUBLIC_AGGREGATE_EXPORT",
            "schema_version": SCHEMA_VERSION,
            "public_aggregate_only": True,
            "hcup_small_cell_suppression_le_10": True,
            "no_patient_level": True,
            "evaluation_n_rows": EXPECTED_N_ROWS,
            "drift_n_rows": EXPECTED_N_ROWS,
            "metrics_rows": int(len(evaluation_tables["metrics.parquet"])),
            "patient_comparison_rows": int(len(evaluation_tables["comparisons_patient_bootstrap.parquet"])),
            "hospital_comparison_rows": int(len(evaluation_tables["comparisons_hospital_bootstrap.parquet"])),
            "suppressed_rows": int(len(audits)),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(temporary / "results_summary.json", summary)

        artifacts: dict[str, dict[str, Any]] = {}
        for path in sorted(temporary.iterdir()):
            if path.name == "public_evidence_manifest.json":
                continue
            artifacts[path.name] = {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        evidence = [_evidence_entry("evaluation_manifest", eval_manifest_path), _evidence_entry("drift_manifest", drift_manifest_path)]
        for path in evidence_manifests:
            evidence_path = Path(path).resolve()
            if not evidence_path.is_file():
                raise RuntimeError(f"Missing optional evidence manifest: {Path(path).name}")
            evidence.append(_evidence_entry("optional_evidence_manifest", evidence_path))
        public_manifest = {
            "status": "PASS_PUBLIC_AGGREGATE_EXPORT",
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "public_aggregate_only": True,
            "hcup_small_cell_suppression_le_10": True,
            "no_patient_level": True,
            "no_replicate_level": True,
            "source_manifests": evidence,
            "artifacts": artifacts,
            "suppression_audit": "public_suppression_audit.csv",
            "restricted_artifacts_not_copied": True,
        }
        manifest_path = temporary / "public_evidence_manifest.json"
        _write_json(manifest_path, public_manifest)
        digest = sha256(manifest_path)
        (temporary / "public_evidence_manifest.json.sha256").write_text(f"{digest}  public_evidence_manifest.json\n", encoding="ascii")

        final = output_dir.resolve()
        competing_partials = [path for path in final.parent.glob(final.name + ".partial*") if path.resolve() != temporary.resolve()]
        if final.exists() or competing_partials:
            raise RuntimeError("Public aggregate output appeared before atomic publish")
        os.replace(temporary, final)
        return {**summary, "output": str(final), "manifest_sha256": digest}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--drift-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--drift-manifest", type=Path)
    parser.add_argument("--evidence-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = export_public_aggregate(
        args.evaluation_dir,
        args.drift_dir,
        args.output_dir,
        args.evaluation_manifest,
        args.drift_manifest,
        args.evidence_manifest,
    )
    print(stable_json(result))


if __name__ == "__main__":
    main()
