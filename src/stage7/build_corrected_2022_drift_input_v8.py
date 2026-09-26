#!/usr/bin/env python3
"""Package the exact frozen drift columns beside corrected 2022 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STATUS = "PASS_TECHNICAL_AMENDMENT_V8_CORRECTED_FROZEN_DRIFT_INPUT_PACKAGING"
MAPPING = {
    "APRDRG_Risk_Mortality": "APRDRG_Risk_Mortality",
    "APRDRG_Severity": "APRDRG_Severity",
    "HOSP_BEDSIZE": "HOSP_BEDSIZE",
    "HOSP_URCAT4": "HOSP_URCAT4",
    "HOSP_UR_TEACH": "HOSP_UR_TEACH",
    "H_CONTRL": "H_CONTRL",
    "PL_NCHS": "PL_NCHS",
    "outcome_biliary_readmission": "biliary_readmission_30d",
    "outcome_sepsis_or_organ_readmission": "sepsis_or_organ_readmission_30d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def identity(path: Path, *, relative_name: bool = False) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    return {"file": path.name if relative_name else str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def assert_identity(path: Path, declared: dict[str, Any] | None, label: str) -> None:
    actual = identity(path)
    if (not isinstance(declared, dict) or declared.get("sha256") != actual["sha256"]
            or int(declared.get("bytes", -1)) != actual["bytes"]):
        raise RuntimeError(f"{label} bytes/SHA256 identity mismatch")


def assert_sidecar(path: Path, label: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError(f"{label} SHA256 sidecar mismatch")


def artifact_from_manifest(directory: Path, manifest: dict[str, Any], key: str, filename: str) -> Path:
    path = directory / filename
    assert_identity(path, manifest.get(key), f"{directory.name}/{filename}")
    return path


def build(prediction_dir: Path, derivative_dir: Path, unlock_lock: Path,
          evaluation_spec: Path, drift_analyzer: Path, output_dir: Path) -> dict[str, Any]:
    prediction_dir, derivative_dir = prediction_dir.resolve(), derivative_dir.resolve()
    unlock_lock, evaluation_spec, drift_analyzer = (unlock_lock.resolve(), evaluation_spec.resolve(), drift_analyzer.resolve())
    output_dir = output_dir.resolve()
    for path, label in ((unlock_lock, "unlock lock"), (evaluation_spec, "evaluation spec")):
        assert_sidecar(path, label)
    lock = read_json(unlock_lock)
    if (lock.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
            or lock.get("technical_amendment", {}).get("status") != "PASS_TECHNICAL_AMENDMENT_V8_NUMERIC_EPISODE_ORDER_LOCKED"):
        raise RuntimeError("Drift input is not bound to the corrected v8 lock")
    for path, label in ((evaluation_spec, "evaluation spec"), (drift_analyzer, "drift analyzer")):
        assert_identity(path, lock.get("frozen_code", {}).get(str(path)), label)
    prediction_manifest_path, derivative_manifest_path = prediction_dir / "manifest.json", derivative_dir / "manifest.json"
    for path, label in ((prediction_manifest_path, "prediction manifest"), (derivative_manifest_path, "derivative manifest")):
        assert_sidecar(path, label)
    prediction_manifest, derivative_manifest = read_json(prediction_manifest_path), read_json(derivative_manifest_path)
    if prediction_manifest.get("status") != "PASS_LOCKED_2022_STANDARDIZED_PREDICTIONS":
        raise RuntimeError("Prediction manifest is not the locked 2022 standardized output")
    if derivative_manifest.get("status") != "PASS_LOCKED_2022_MINIMUM_DERIVATIVE":
        raise RuntimeError("Derivative manifest is not the locked 2022 derivative")
    predictions = artifact_from_manifest(prediction_dir, prediction_manifest, "artifact", "predictions_2022.parquet")
    derivative = artifact_from_manifest(derivative_dir, derivative_manifest, "derivative", "locked_2022_derivative.parquet")
    for manifest, label in ((prediction_manifest, "prediction"), (derivative_manifest, "derivative")):
        assert_identity(unlock_lock, manifest.get("unlock_lock"), f"{label} unlock binding")
    if output_dir.exists() or list(output_dir.parent.glob(output_dir.name + ".partial*")):
        raise RuntimeError("Corrected drift-input output or partial already exists and is immutable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))
    try:
        prediction_frame = pq.read_table(predictions).to_pandas()
        derivative_columns = ["encounter_hash", *MAPPING.values()]
        derivative_frame = pq.read_table(derivative, columns=derivative_columns).to_pandas()
        if prediction_frame["encounter_hash"].duplicated().any() or derivative_frame["encounter_hash"].duplicated().any():
            raise RuntimeError("Drift-input join keys are not one-to-one")
        if set(MAPPING) & set(prediction_frame.columns):
            raise RuntimeError("Prediction table already contains a frozen drift addition")
        derivative_frame = derivative_frame.rename(columns={source: target for target, source in MAPPING.items()})
        combined = prediction_frame.merge(derivative_frame, on="encounter_hash", how="inner", validate="one_to_one")
        if len(combined) != len(prediction_frame) or len(combined) != len(derivative_frame):
            raise RuntimeError("Corrected drift-input join lost or expanded encounters")
        if not combined["analysis_year"].eq(2022).all() or not combined["analysis_partition"].eq("test").all():
            raise RuntimeError("Corrected drift input is not exclusively the 2022 test partition")
        artifact_path = temporary / "predictions_2022_with_frozen_drift_columns.parquet"
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), artifact_path,
                       compression="zstd", compression_level=6, row_group_size=131072,
                       use_dictionary=True, write_statistics=True)
        manifest = {
            "status": STATUS, "schema_version": "stage7_corrected_drift_input_v8",
            "created_utc": datetime.now(timezone.utc).isoformat(), "row_count": int(len(combined)),
            "artifact": identity(artifact_path, relative_name=True), "fields_added": list(MAPPING),
            "frozen_source_mapping": MAPPING, "one_to_one_encounter_join": True,
            "selection_basis": "EXACT_PRE_2022_FROZEN_DRIFT_REFERENCE_COLUMNS_ONLY",
            "no_2022_value_based_selection_or_adaptation": True,
            "no_model_probability_threshold_or_conformal_change": True,
            "inputs": {"predictions": identity(predictions), "derivative": identity(derivative),
                       "unlock_lock": identity(unlock_lock), "evaluation_spec": identity(evaluation_spec),
                       "drift_analyzer": identity(drift_analyzer), "packaging_runtime": identity(Path(__file__).resolve())},
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(stable_json(manifest), encoding="utf-8")
        digest = sha256(manifest_path)
        (temporary / "manifest.json.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
        temporary.replace(output_dir)
        return {**manifest, "manifest_sha256": digest, "output": str(output_dir)}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--derivative-dir", type=Path, required=True)
    parser.add_argument("--unlock-lock", type=Path, required=True)
    parser.add_argument("--evaluation-spec", type=Path, required=True)
    parser.add_argument("--drift-analyzer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build(args.prediction_dir, args.derivative_dir, args.unlock_lock,
                            args.evaluation_spec, args.drift_analyzer, args.output_dir)), end="")


if __name__ == "__main__":
    main()
