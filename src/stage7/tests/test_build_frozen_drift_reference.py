"""Synthetic, no-NRD regression tests for the frozen drift-reference builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_frozen_drift_reference.py"
SPEC = importlib.util.spec_from_file_location("frozen_drift_builder", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def ident(path: Path) -> dict: return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}
def write_json(path: Path, value: dict) -> None: path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _base(year: int, partition: str) -> pd.DataFrame:
    n = 10; leaf = np.array([0, 1, 2, 3, 4] * 2)
    return pd.DataFrame({"year": year, "analysis_partition": partition, "encounter_hash": [f"{partition}-{x}" for x in range(n)], "patient_hash": [f"p{x // 2}" for x in range(n)], "DISCWT": np.linspace(1., 2., n), "any_unplanned_readmission_30d": (leaf > 0).astype(int), "ap_specific_readmission_30d": (leaf == 1).astype(int), "readmission_leaf": leaf, "AGE": [30, 50, 66, 78, 43] * 2, "FEMALE": [0, 1] * 5, "PAY1": [1, 2] * 5, "ZIPINC_QRTL": [1, 2] * 5, "HOSP_BEDSIZE": [1, 2] * 5, "cost_2021_usd": [100, 2000, 300, 1000, 4000] * 2, "LOS": [2, 8, 4, 10, 3] * 2, "DIED": [0, 0, 0, 1, 0] * 2})


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = _base(2021, "2021A")
    details = {}
    for model in mod.TRANSFORMER_ROSTER:
        root = tmp_path / model; root.mkdir(); raw_path = root / "raw.parquet"; raw.to_parquet(raw_path, index=False)
        cal = raw[["year", "analysis_partition", "encounter_hash", "patient_hash"]].copy()
        cal["p_any_readmission_calibrated"] = np.where(raw.any_unplanned_readmission_30d.eq(1), .8, .1)
        cal["p_ap_specific_readmission_calibrated"] = np.where(raw.readmission_leaf.eq(1), .8, .1)
        cal_path = root / "predictions_2021A_calibrated.parquet"; cal.to_parquet(cal_path, index=False)
        manifest_path = root / "transformer_operating_point_manifest.json"; write_json(manifest_path, {"source": ident(raw_path)})
        details[model] = {"manifest": {"source": ident(raw_path)}, "files": {"predictions_2021A_calibrated.parquet": cal_path}, "manifest_path": manifest_path}
    base_dir = tmp_path / "baseline"; base_dir.mkdir(); baseline = raw.copy()
    for model in mod.BASELINE_ROSTER:
        baseline[f"p_cal__any_readmission__{model}"] = np.where(raw.any_unplanned_readmission_30d.eq(1), .7, .2)
        baseline[f"p_cal__ap_specific_readmission__{model}"] = np.where(raw.ap_specific_readmission_30d.eq(1), .7, .1)
    baseline_path = base_dir / "predictions_2021A.parquet"; baseline.to_parquet(baseline_path, index=False)
    baseline_lock = base_dir / "baseline_lock.json"; write_json(baseline_lock, {})
    baseline_detail = {"directory": base_dir, "lock": {"artifacts": {baseline_path.name: {"bytes": baseline_path.stat().st_size, "sha256": digest(baseline_path)}}}}
    # Authorised 2021B predictor and its 15 materialized conformal set columns.
    b = _base(2021, "2021B"); pred_path = tmp_path / "pred2021b.parquet"; b.to_parquet(pred_path, index=False)
    sets = b[["encounter_hash"]].copy()
    for source in mod.MONDRIAN.values(): sets[f"group_{source}"] = b[{"sex": "FEMALE", "age_group": "AGE", "payer": "PAY1", "zip_income_quartile": "ZIPINC_QRTL"}[source]].astype(str)
    outputs = {"global": {}, "mondrian": {}}
    for label, _ in mod.NOMINAL:
        sets[f"gmask{label}"] = np.array([1, 2, 4, 8, 16] * 2, dtype=np.uint8); sets[f"gsize{label}"] = 1
        outputs["global"][label] = {"leaf_set_mask": f"gmask{label}", "leaf_set_size": f"gsize{label}"}
        for source in mod.MONDRIAN.values():
            sets[f"mmask{source}{label}"] = sets[f"gmask{label}"]; sets[f"msize{source}{label}"] = 1
            outputs["mondrian"].setdefault(source, {"levels": {}})["levels"][label] = {"leaf_set_mask": f"mmask{source}{label}", "leaf_set_size": f"msize{source}{label}"}
    sets["abstain_90"] = False; sets_path = tmp_path / "sets.parquet"; sets.to_parquet(sets_path, index=False)
    pred_manifest = tmp_path / "prediction_manifest.json"; write_json(pred_manifest, {"status": "PASS_PREDICTIONS_2021B_ONLY", "partition": "2021B", "year_2022_accessed": False, "model_or_threshold_selection_on_2021B": False, "artifact": {"bytes": pred_path.stat().st_size, "sha256": digest(pred_path)}})
    conformal_manifest = tmp_path / "conformal_calibrator.json"; write_json(conformal_manifest, {})
    stage6_registry = tmp_path / "stage6.json"; write_json(stage6_registry, {"prediction": {"manifest": str(pred_manifest), "manifest_sha256": digest(pred_manifest), "artifact": str(pred_path), "artifact_sha256": digest(pred_path)}})
    stage6_detail = {"artifact_path": sets_path, "manifest_path": conformal_manifest, "calibrator": {"risk_set_outputs": outputs}}
    # These validated-source functions are owned and independently tested by the evaluation-contract module; synthetic unit tests inject their completed results.
    monkeypatch.setattr(mod.builder, "validate_all_operating_points", lambda _: details)
    monkeypatch.setattr(mod.builder, "validate_baseline_lock", lambda _: baseline_detail)
    monkeypatch.setattr(mod.builder, "validate_stage6_conformal", lambda *_: stage6_detail)
    all_points = tmp_path / "all.json"; baseline_lock_input = tmp_path / "lock.json"; all_points.write_text("{}", encoding="utf-8"); baseline_lock_input.write_text("{}", encoding="utf-8")
    dev = pd.concat([_base(2018, "development"), _base(2019, "development"), _base(2020, "development")], ignore_index=True)
    dev["encounter_hash"] = [f"dev-{year}-{index}" for index, year in enumerate(dev["year"])]
    dev_path = tmp_path / "development.parquet"; dev.to_parquet(dev_path, index=False)
    labels_path = tmp_path / "labels_2021a.parquet"; raw.to_parquet(labels_path, index=False)
    aux = tmp_path / "aux.json"; write_json(aux, {"high_cost": {"threshold_2021_usd": 1500}})
    return {"all": all_points, "baseline": baseline_lock_input, "stage6": stage6_registry, "dev": dev_path, "labels": labels_path, "aux": aux, "sets": sets_path}


def test_builds_consumer_valid_13_model_reference_and_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch); output = tmp_path / "reference.json"
    result = mod.build_frozen_drift_reference(paths["all"], paths["baseline"], paths["stage6"], paths["dev"], paths["labels"], paths["aux"], output)
    reference = json.loads(output.read_text(encoding="utf-8"))
    assert result["reference_sha256"] == digest(output)
    assert output.with_suffix(".json.sha256").read_text(encoding="ascii").split()[0] == digest(output)
    assert set(reference["calibration"]) == set(mod.TRANSFORMER_ROSTER + mod.BASELINE_ROSTER)
    assert len(reference["label_outcomes"]) == 7
    assert len({(x["conformal_set"], x["subgroup_dimension"], x["subgroup_value"]) for x in reference["coverage"]}) == len(reference["coverage"])
    mod.drift.validate_drift_reference({"models": {name: {} for name in reference["calibration"]}, "conformal_sets": [{"name": f"global_{label}"} for label, _ in mod.NOMINAL], "drift_reference": reference})
    with pytest.raises(RuntimeError, match="immutable"):
        mod.build_frozen_drift_reference(paths["all"], paths["baseline"], paths["stage6"], paths["dev"], paths["labels"], paths["aux"], output)


def test_future_partition_and_conformal_hash_drift_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    bad = pd.read_parquet(paths["dev"]); bad.loc[:, "year"] = 2022; bad.to_parquet(paths["dev"], index=False)
    with pytest.raises(RuntimeError, match="authorized development"):
        mod.build_frozen_drift_reference(paths["all"], paths["baseline"], paths["stage6"], paths["dev"], paths["labels"], paths["aux"], tmp_path / "bad.json")
    paths = fixture(tmp_path / "hash", monkeypatch)
    registry = json.loads(paths["stage6"].read_text(encoding="utf-8")); registry["prediction"]["artifact_sha256"] = "0" * 64; write_json(paths["stage6"], registry)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        mod.build_frozen_drift_reference(paths["all"], paths["baseline"], paths["stage6"], paths["dev"], paths["labels"], paths["aux"], tmp_path / "hash" / "bad.json")


def test_label_source_must_match_all_frozen_2021a_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    labels = pd.read_parquet(paths["labels"]); labels.loc[0, "patient_hash"] = "different-patient"; labels.to_parquet(paths["labels"], index=False)
    with pytest.raises(RuntimeError, match="one-to-one aligned"):
        mod.build_frozen_drift_reference(paths["all"], paths["baseline"], paths["stage6"], paths["dev"], paths["labels"], paths["aux"], tmp_path / "mismatch.json")


def test_hash_bound_legacy_baseline_without_partition_is_accepted_only_for_2021(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline" / "predictions_2021A.parquet"
    frame = pd.read_parquet(baseline).drop(columns="analysis_partition")
    frame.to_parquet(baseline, index=False)
    # Re-seal the synthetic baseline lock exactly as the real historical lock
    # binds its prediction artifact.
    monkeypatch.setattr(
        mod.builder,
        "validate_baseline_lock",
        lambda _: {
            "directory": baseline.parent,
            "lock": {"artifacts": {baseline.name: ident(baseline)}},
        },
    )
    output = tmp_path / "legacy.json"
    mod.build_frozen_drift_reference(paths["all"], paths["baseline"], paths["stage6"], paths["dev"], paths["labels"], paths["aux"], output)
    assert output.is_file()

    paths = fixture(tmp_path / "future", monkeypatch)
    baseline = tmp_path / "future" / "baseline" / "predictions_2021A.parquet"
    frame = pd.read_parquet(baseline).drop(columns="analysis_partition")
    frame.loc[:, "year"] = 2022
    frame.to_parquet(baseline, index=False)
    monkeypatch.setattr(
        mod.builder,
        "validate_baseline_lock",
        lambda _: {
            "directory": baseline.parent,
            "lock": {"artifacts": {baseline.name: ident(baseline)}},
        },
    )
    with pytest.raises(RuntimeError, match="authorized 2021A"):
        mod.build_frozen_drift_reference(paths["all"], paths["baseline"], paths["stage6"], paths["dev"], paths["labels"], paths["aux"], tmp_path / "future" / "bad.json")


def test_missing_cost_is_excluded_not_negative_in_frozen_high_cost_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    labels = pd.read_parquet(paths["labels"])
    labels.loc[0, "cost_2021_usd"] = np.nan
    labels.loc[1, "cost_2021_usd"] = 0.
    labels.to_parquet(paths["labels"], index=False)
    assert np.isnan(mod._labels(labels, 1500.)["high_cost"][:2]).all()
    reference = mod.build_label_reference(labels, 1500.)
    high = reference["high_cost"]
    # One missing cost and one zero cost are both masked.  Of the remaining
    # eight strictly-positive costs, three are high.  Turning either into a
    # negative label would produce a different denominator and rate.
    assert high["unweighted_rate"] == pytest.approx(3 / 8)
    assert high["validity"]["policy"] == "EXCLUDE_NULL_HIGH_COST_LABELS_FROM_ALL_RATES_AND_BOOTSTRAPS"
    assert high["validity"]["reference_valid_n"] is None
    assert high["validity"]["reference_missing_n"] is None
