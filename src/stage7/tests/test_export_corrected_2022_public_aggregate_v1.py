from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "export_corrected_2022_public_aggregate_v1.py"
SPEC = importlib.util.spec_from_file_location("public_export", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
public_export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(public_export)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(directory: Path, name: str, frame: pd.DataFrame) -> None:
    path = directory / name
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def _metrics() -> pd.DataFrame:
    rows = []
    for model in [f"model_{index:02d}" for index in range(13)]:
        for outcome in ("any_readmission", "ap_readmission"):
            for weighting in ("unweighted", "DISCWT_weighted"):
                rows.append({
                    "model": model,
                    "outcome": outcome,
                    "weighting": weighting,
                    "n_rows": 119114,
                    "event_count": 100,
                    "auroc": 0.75,
                    "ci95_lower": 0.70,
                    "ci95_upper": 0.80,
                    "status": "OK",
                })
    return pd.DataFrame(rows)


def _comparison(cluster_unit: str) -> pd.DataFrame:
    rows = []
    for outcome in ("any_readmission", "ap_readmission"):
        for weighting in ("unweighted", "DISCWT_weighted"):
            rows.append({
                "comparison": "transformer_minus_lightgbm",
                "outcome": outcome,
                "weighting": weighting,
                "cluster_unit": cluster_unit,
                "n_replicates": 1000,
                "n_defined": 1000,
                "auprc_difference_observed": 0.01,
                "ci95_percentile_lower": -0.02,
                "ci95_percentile_upper": 0.04,
                "status": "OK",
            })
    return pd.DataFrame(rows)


def _source_tables() -> dict[str, pd.DataFrame]:
    return {
        "metrics.parquet": _metrics(),
        "comparisons_patient_bootstrap.parquet": _comparison("patient_hash"),
        "comparisons_hospital_bootstrap.parquet": _comparison("hospital_hash"),
        "decision_curve.parquet": pd.DataFrame([{"model": "model_00", "outcome": "any_readmission", "weighting": "unweighted", "threshold_probability": 0.20, "net_benefit": 0.10}]),
        "conformal.parquet": pd.DataFrame([
            {"conformal_set": "global_80", "n_rows": 5, "event_count": 6, "coverage": 0.4, "coverage_ci95_lower": 0.1, "coverage_ci95_upper": 0.8, "status": "OK"},
            {"conformal_set": "global_90", "n_rows": 100, "event_count": 20, "coverage": 0.9, "coverage_ci95_lower": 0.8, "coverage_ci95_upper": 0.95, "status": "OK"},
        ]),
        "conformal_subgroups.parquet": pd.DataFrame([
            {"conformal_set": "global_80", "subgroup_dimension": "sex", "subgroup_value": "F", "n_rows": 20, "event_count": 0, "coverage": 0.5, "status": "OK"},
        ]),
        "covariate_drift.parquet": pd.DataFrame([
            {"covariate": "age", "frozen_level": "[0,20)", "observed_weighted_mean": 50.0, "status": "OK"},
        ]),
        "label_drift.parquet": pd.DataFrame([
            {"outcome": "high_cost", "weighting": "unweighted", "n_rows": 50, "event_count": 20, "non_events": 5, "observed_rate": 0.4, "ci95_lower": 0.3, "ci95_upper": 0.5, "status": "OK"},
        ]),
        "calibration_drift.parquet": pd.DataFrame([
            {"model": "model_00", "outcome": "any_readmission", "valid_n": 10, "calibration_intercept": 0.1, "calibration_slope": 0.9, "status": "OK"},
        ]),
        "coverage_drift.parquet": pd.DataFrame([
            {"conformal_set": "global_80", "subgroup_dimension": "sex", "subgroup_value": "M", "n_rows": 20, "event_count": 20, "missing_n": 10, "coverage_observed": 0.8, "coverage_ci95_lower": 0.6, "coverage_ci95_upper": 0.9, "status": "OK"},
        ]),
    }


def _make_input_dirs(tmp_path: Path) -> tuple[Path, Path]:
    evaluation_dir = tmp_path / "evaluation"
    drift_dir = tmp_path / "drift"
    evaluation_dir.mkdir()
    drift_dir.mkdir()
    eval_names = set(public_export.EVALUATION_ARTIFACTS)
    drift_names = set(public_export.DRIFT_ARTIFACTS)
    for name, frame in _source_tables().items():
        _write_parquet(evaluation_dir if name in eval_names else drift_dir, name, frame)
    # These restricted inputs are intentionally present but are never opened
    # or copied by the public allow-list.
    _write_parquet(evaluation_dir, "patient_bootstrap_replicates.parquet", pd.DataFrame({"patient_hash": ["p1"], "replicate": [1]}))
    _write_parquet(drift_dir, "label_rate_bootstrap.parquet", pd.DataFrame({"patient_hash": ["p1"], "replicate": [1]}))
    eval_artifacts = {}
    for name in sorted(eval_names):
        path = evaluation_dir / name
        eval_artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    drift_artifacts = {}
    for name in sorted(drift_names):
        path = drift_dir / name
        drift_artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    (evaluation_dir / "manifest.json").write_text(json.dumps({
        "status": public_export.EVALUATION_STATUS,
        "n_rows": public_export.EXPECTED_N_ROWS,
        "models": [f"model_{index:02d}" for index in range(13)],
        "co_primary_outcomes": ["any_readmission", "ap_readmission"],
        "patient_bootstrap": {"replicates": 1000, "seed": 1},
        "hospital_bootstrap_included": True,
        "artifacts": eval_artifacts,
    }), encoding="utf-8")
    (drift_dir / "manifest.json").write_text(json.dumps({
        "status": public_export.DRIFT_STATUS,
        "n_rows": public_export.EXPECTED_N_ROWS,
        "bootstrap": {"n_replicates": 1000, "cluster_column": "patient_hash", "seed": 2},
        "artifacts": drift_artifacts,
    }), encoding="utf-8")
    return evaluation_dir, drift_dir


def test_suppression_contract_covers_all_required_small_cells() -> None:
    source = pd.DataFrame([
        {"outcome": "events_small", "n_rows": 5, "event_count": 6, "non_events": 20, "observed_rate": 0.2, "ci95_lower": 0.1, "status": "OK"},
        {"outcome": "event_none", "n_rows": 20, "event_count": 0, "non_events": 20, "coverage": 0.4, "status": "OK"},
        {"outcome": "non_events_small", "n_rows": 20, "event_count": 15, "non_events": 5, "auroc": 0.8, "status": "OK"},
        {"outcome": "missing_small", "n_rows": 20, "event_count": 15, "missing_n": 10, "calibration_slope": 0.9, "status": "OK"},
        {"outcome": "valid_small", "n_rows": 20, "event_count": 15, "valid_n": 10, "brier": 0.2, "status": "OK"},
        {"outcome": "large", "n_rows": 20, "event_count": 15, "non_events": 15, "missing_n": 15, "valid_n": 20, "auroc": 0.8, "status": "OK"},
    ])
    public, audit = public_export.suppress_small_cells(source, "fixture.parquet")
    assert len(audit) == 5
    assert {field for row in audit for field in row["trigger_fields"].split(";")} == {"n_rows", "event_count", "non_events", "missing_n", "valid_n"}
    assert public.iloc[0]["n_rows"] is None or pd.isna(public.iloc[0]["n_rows"])
    assert public.iloc[0]["event_count"] is None or pd.isna(public.iloc[0]["event_count"])
    assert public.iloc[0]["observed_rate"] is None or pd.isna(public.iloc[0]["observed_rate"])
    assert public.iloc[0]["ci95_lower"] is None or pd.isna(public.iloc[0]["ci95_lower"])
    assert public.iloc[0]["status"] == "SUPPRESSED_SMALL_CELL"
    assert public.iloc[5]["n_rows"] == 20
    assert public.iloc[5]["event_count"] == 15
    assert public.iloc[5]["auroc"] == 0.8
    audit_text = json.dumps(audit)
    assert '"n_rows":' not in audit_text and '"event_count":' not in audit_text and '"non_events":' not in audit_text


def test_export_is_aggregate_only_suppressed_and_immutable(tmp_path: Path) -> None:
    evaluation_dir, drift_dir = _make_input_dirs(tmp_path)
    output_dir = tmp_path / "public"
    result = public_export.export_public_aggregate(evaluation_dir, drift_dir, output_dir)
    assert result["status"] == "PASS_PUBLIC_AGGREGATE_EXPORT"
    assert output_dir.is_dir()
    assert not (output_dir / "patient_bootstrap_replicates.parquet").exists()
    assert not (output_dir / "label_rate_bootstrap.parquet").exists()
    assert not any("replicate" in path.name or "prediction" in path.name for path in output_dir.iterdir())
    assert (output_dir / "public_evidence_manifest.json").is_file()
    assert (output_dir / "results_summary.json").is_file()

    audit = pd.read_csv(output_dir / "public_suppression_audit.csv")
    assert len(audit) >= 5
    assert set(audit.columns) == {"source_file", "row_key", "trigger_fields", "suppressed"}
    audit_text = (output_dir / "public_suppression_audit.csv").read_text(encoding="utf-8")
    assert "n_rows" in audit_text and "event_count" in audit_text
    assert ",6," not in audit_text and ",5," not in audit_text and ",0," not in audit_text and ",10," not in audit_text
    assert "patient_hash" not in (output_dir / "public_evidence_manifest.json").read_text(encoding="utf-8")

    public_metrics = pd.read_csv(output_dir / "metrics.csv")
    assert len(public_metrics) == 52
    public_conformal = pd.read_csv(output_dir / "conformal.csv")
    small = public_conformal.loc[public_conformal["conformal_set"] == "global_80"].iloc[0]
    assert pd.isna(small["n_rows"]) and pd.isna(small["event_count"]) and pd.isna(small["coverage"])
    large = public_conformal.loc[public_conformal["conformal_set"] == "global_90"].iloc[0]
    assert large["n_rows"] == 100 and large["event_count"] == 20 and large["coverage"] == 0.9

    with pytest.raises(RuntimeError, match="immutable"):
        public_export.export_public_aggregate(evaluation_dir, drift_dir, output_dir)


def test_manifest_tampering_and_partial_destination_fail_closed(tmp_path: Path) -> None:
    evaluation_dir, drift_dir = _make_input_dirs(tmp_path)
    output_dir = tmp_path / "public"
    (tmp_path / "public.partial-stale").mkdir()
    with pytest.raises(RuntimeError, match="immutable"):
        public_export.export_public_aggregate(evaluation_dir, drift_dir, output_dir)
    (tmp_path / "public.partial-stale").rmdir()
    manifest = json.loads((evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["metrics.parquet"]["sha256"] = "0" * 64
    (evaluation_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        public_export.export_public_aggregate(evaluation_dir, drift_dir, output_dir)
    assert not output_dir.exists()
