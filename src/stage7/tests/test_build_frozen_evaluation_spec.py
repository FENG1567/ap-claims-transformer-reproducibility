"""Synthetic regression tests for the fail-closed pre-2022 spec builder.

Fixtures intentionally contain only JSON and tiny opaque files.  They never
contain an NRD/MIMIC row, a parquet reader, or a server location.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_frozen_evaluation_spec.py"
SPEC = importlib.util.spec_from_file_location("build_frozen_evaluation_spec", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _operating_point(root: Path, run: str) -> tuple[dict, Path]:
    directory = root / f"operating_{run}"
    directory.mkdir()
    source_prediction = directory / "predictions_2021A.parquet"
    source_prediction.write_bytes(f"synthetic-{run}".encode("ascii"))
    checkpoint = directory / "best_checkpoint.pt"
    checkpoint.write_bytes(f"checkpoint-{run}".encode("ascii"))
    finetune = directory / "finetune_manifest.json"
    write_json(finetune, {"status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022", "run": run})
    names = (
        "predictions_2021A_calibrated.parquet", "transformer_binary_calibrators.joblib",
        "transformer_operating_thresholds_2021A.json", "transformer_calibration_selection_2021A.json",
    )
    (directory / names[0]).write_bytes(b"opaque-calibrated-predictions")
    (directory / names[1]).write_bytes(b"opaque-calibrators")
    write_json(directory / names[2], {
        "any_readmission": {"probability_column": "p_any_readmission_calibrated", "threshold": 0.21,
                             "rule": "fixed 20% capacity on 2021A; threshold transported unchanged"},
        "ap_specific_readmission": {"probability_column": "p_ap_specific_readmission_calibrated", "threshold": 0.17,
                                     "rule": "fixed 20% capacity on 2021A; threshold transported unchanged"},
    })
    write_json(directory / names[3], {"synthetic": "selection provenance"})
    manifest = {
        "status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False,
        "capacity_fraction": 0.20, "endpoints": list(builder.OUTCOMES),
        "source": {"sha256": digest(source_prediction), "bytes": source_prediction.stat().st_size},
        "artifacts": {name: {"bytes": (directory / name).stat().st_size, "sha256": digest(directory / name)} for name in names},
    }
    manifest_path = directory / "transformer_operating_point_manifest.json"
    write_json(manifest_path, manifest)
    registry_entry = {
        "operating_point_directory": str(directory.resolve()), "operating_point_manifest": artifact(manifest_path),
        "source": {"prediction": {"bytes": source_prediction.stat().st_size, "sha256": digest(source_prediction)},
                   "checkpoint": artifact(checkpoint), "finetune_manifest": artifact(finetune)},
    }
    return registry_entry, manifest_path


def _all_operating_points(root: Path) -> tuple[Path, Path]:
    runs: dict[str, dict] = {}
    joint_manifest: Path | None = None
    for run in builder.TRANSFORMER_ROSTER:
        entry, manifest = _operating_point(root, run)
        runs[run] = entry
        if run == "joint_main":
            joint_manifest = manifest
    assert joint_manifest is not None
    registry = {
        "status": "PASS_ALL_TRANSFORMER_CALIBRATIONS_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False,
        "exact_roster": list(builder.TRANSFORMER_ROSTER), "runs": runs,
    }
    path = root / "all_operating_points.json"
    write_json(path, registry)
    return path, joint_manifest


def _baseline(root: Path) -> Path:
    directory = root / "baselines"
    directory.mkdir()
    threshold = {
        outcome: {model: {"threshold": 0.30 + index / 100,
                          "rule": "fixed 20% capacity: synthetic 2021A calibration"}
                  for index, model in enumerate(builder.BASELINE_ROSTER)}
        for outcome in builder.OUTCOMES
    }
    calibration = {outcome: {model: {"selected_method": "none"} for model in builder.BASELINE_ROSTER}
                   for outcome in builder.OUTCOMES}
    selection = {outcome: {model: {"selected_config": {"synthetic": True}} for model in builder.BASELINE_ROSTER}
                 for outcome in builder.OUTCOMES}
    write_json(directory / "operating_thresholds_2021A.json", threshold)
    write_json(directory / "calibration_selection_2021A.json", calibration)
    write_json(directory / "model_selection_2021A.json", selection)
    model_directory = directory / "models"
    model_directory.mkdir()
    for outcome in builder.OUTCOMES:
        for model in builder.BASELINE_ROSTER:
            (model_directory / f"{outcome}__{model}.joblib").write_bytes(f"{outcome}-{model}".encode("ascii"))
    registered_paths = [directory / name for name in ("operating_thresholds_2021A.json", "calibration_selection_2021A.json", "model_selection_2021A.json")]
    registered_paths.extend(model_directory / f"{outcome}__{model}.joblib" for outcome in builder.OUTCOMES for model in builder.BASELINE_ROSTER)
    registered = {str(item.relative_to(directory)): {"bytes": item.stat().st_size, "sha256": digest(item)} for item in registered_paths}
    lock = {
        "status": "PASS_BASELINES_FROZEN_PRE_2021B_PRE_2022", "2021B_outcomes_accessed": False,
        "year_2022_accessed": False, "artifacts": registered,
    }
    path = directory / "baseline_lock.json"
    write_json(path, lock)
    return path


def _stage6(root: Path, joint_manifest: Path) -> Path:
    directory = root / "stage6"
    directory.mkdir()
    sets = directory / "conformal_sets_2021B.parquet"
    sets.write_bytes(b"opaque-conformal-sets")
    outputs = {"global": {}, "mondrian": {}}
    calibration = {"global": {}, "mondrian": {}}
    for label, _ in builder.NOMINAL_COVERAGES:
        outputs["global"][label] = {"leaf_set_mask": f"leaf_set_mask_{label}", "leaf_set_size": f"leaf_set_size_{label}"}
        calibration["global"][label] = {"q": 0.1, "n": 100}
    for dimension in builder.MONDRIAN_SOURCE_DIMENSIONS.values():
        outputs["mondrian"][dimension] = {"levels": {}}
        calibration["mondrian"][dimension] = {}
        for label, _ in builder.NOMINAL_COVERAGES:
            outputs["mondrian"][dimension]["levels"][label] = {
                "leaf_set_mask": f"leaf_set_mask_{dimension}_{label}",
                "leaf_set_size": f"leaf_set_size_{dimension}_{label}", "q_source": "synthetic q source",
            }
            calibration["mondrian"][dimension][label] = {"group": {"q": 0.1, "n": 100, "status": "CALIBRATED"}}
    manifest = {
        "status": "PASS_LOCKED_PRE_2022", "calibration_partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False,
        "nominal_coverages": [0.90, 0.80, 0.95], "ancestor_closure": True,
        "any_readmission_probability_column": "p_any_readmission_calibrated",
        "any_readmission_threshold_from_2021A": 0.21,
        "operating_point_source": {"manifest_sha256": digest(joint_manifest)},
        "calibration": calibration, "risk_set_outputs": outputs,
        "artifact": {"bytes": sets.stat().st_size, "sha256": digest(sets)},
    }
    manifest_path = directory / "conformal_calibrator.json"
    write_json(manifest_path, manifest)
    registry = {
        "status": "PASS_2021B_CONFORMAL_LOCKED_PRE_2022", "partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False,
        "nominal_coverages": [0.90, 0.80, 0.95],
        "conformal": {"manifest": str(manifest_path.resolve()), "manifest_sha256": digest(manifest_path),
                      "artifact": str(sets.resolve()), "artifact_sha256": digest(sets)},
    }
    path = directory / "stage6_registry.json"
    write_json(path, registry)
    return path


def _frozen_drift_reference(root: Path) -> Path:
    coverage = []
    for label, _ in builder.NOMINAL_COVERAGES:
        names = [f"global_{label}"] + [f"mondrian_{dimension}_{label}" for dimension in builder.SUBGROUPS]
        for name in names:
            coverage.append({"conformal_set": name, "subgroup_dimension": "ALL", "subgroup_value": "ALL",
                             "coverage": 0.90, "mean_set_size": 1.1, "abstention_rate": 0.1})
            for dimension in builder.SUBGROUPS:
                coverage.append({"conformal_set": name, "subgroup_dimension": dimension, "subgroup_value": "synthetic",
                                 "coverage": 0.90, "mean_set_size": 1.1, "abstention_rate": 0.1})
    models = builder.TRANSFORMER_ROSTER + builder.BASELINE_ROSTER
    reference = {
        "status": "FROZEN_2022_DRIFT_REFERENCE", "year_2022_accessed": False,
        "source_partitions": {"development_years": [2018, 2019, 2020], "selection_partition": "2021A",
                              "conformal_partition": "2021B only", "year_2022_accessed": False,
                              "model_or_threshold_selection_on_2021B": False},
        "covariates": {
            "age": {"column": "AGE", "kind": "numeric_binned", "bins": [18, 45, 120], "reference_missing_rate": 0.,
                    "reference_weighted_distribution": {"[18,45)": .5, "[45,120]": .5, "<NONFINITE_OR_OUT_OF_RANGE>": 0.},
                    "reference_weighted_mean": 58., "reference_weighted_sd": 15.},
            "sex": {"column": "FEMALE", "kind": "categorical", "categories": [0, 1], "reference_missing_rate": 0.,
                    "reference_weighted_distribution": {"0": .5, "1": .5, "<UNSEEN_OR_OTHER>": 0.}},
        },
        "label_outcomes": {name: {"column": column, "unweighted_rate": 0.1, "DISCWT_weighted_rate": 0.1} for name, column in (
            ("any_readmission", "outcome_any_readmission"), ("ap_specific_readmission", "outcome_ap_specific_readmission"),
            ("biliary_readmission", "outcome_biliary_readmission"), ("sepsis_or_organ_readmission", "outcome_sepsis_or_organ_readmission"),
            ("high_cost", "outcome_high_cost"), ("prolonged_los", "outcome_prolonged_los"), ("in_hospital_death", "outcome_in_hospital_death"),
        )},
        "calibration": {model: {outcome: {"calibration_intercept": 0., "calibration_slope": 1., "brier": .1, "log_loss": .2}
                                for outcome in builder.OUTCOMES} for model in models},
        "coverage": coverage,
        "bootstrap": {"n_replicates": 1000, "seed": 20220914, "max_threads": 8, "cluster_column": "patient_hash"},
    }
    path = root / "frozen_drift_reference.json"
    write_json(path, reference)
    path.with_suffix(".json.sha256").write_text(f"{digest(path)}  {path.name}\n", encoding="ascii")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    all_points, joint = _all_operating_points(tmp_path)
    baseline = _baseline(tmp_path)
    stage6 = _stage6(tmp_path, joint)
    drift_reference = _frozen_drift_reference(tmp_path)
    evaluator = tmp_path / "evaluate_locked_2022.py"
    evaluator.write_text("# fixed synthetic evaluator\n", encoding="utf-8")
    return all_points, baseline, stage6, evaluator, drift_reference


def test_builds_exact_13_model_contract_and_immutable_sidecar(tmp_path: Path) -> None:
    all_points, baseline, stage6, evaluator, drift_reference = _fixture(tmp_path)
    output = tmp_path / "frozen_evaluation_spec.json"
    result = builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, drift_reference, output)
    spec = builder.validate_frozen_evaluation_spec(output, evaluator, drift_reference)
    assert result["spec_sha256"] == digest(output)
    assert spec["transformer_roster"] == list(builder.TRANSFORMER_ROSTER)
    assert spec["baseline_roster"] == list(builder.BASELINE_ROSTER)
    assert set(spec["models"]) == set(builder.TRANSFORMER_ROSTER + builder.BASELINE_ROSTER)
    assert len(spec["models"]) == 13
    for name in builder.TRANSFORMER_ROSTER:
        assert set(spec["models"][name]["inference_artifacts"]) == {"checkpoint", "finetune_manifest"}
    for name in builder.BASELINE_ROSTER:
        assert set(spec["models"][name]["inference_artifacts"]) == set(builder.OUTCOMES)
    assert len(spec["conformal_sets"]) == 15
    assert spec["evaluation_protocol"]["paired_bootstrap"]["cluster_units"] == ["patient_hash", "hospital_hash"]
    assert spec["evaluation_protocol"]["paired_bootstrap"]["replicates_each"] == 1000
    assert spec["evaluation_protocol"]["decision_curve"]["threshold_probability_grid"] == [round(i / 100, 2) for i in range(1, 51)]
    assert set(spec["evaluation_protocol"]["drift"]) == {"covariate", "label", "calibration", "coverage"}
    assert output.with_suffix(".json.sha256").read_text(encoding="ascii").split()[0] == digest(output)
    assert builder._load_drift_validator().validate_drift_reference(spec)["status"] == "FROZEN_2022_DRIFT_REFERENCE"
    with pytest.raises(RuntimeError, match="already exists"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, drift_reference, output)


def test_tampered_operating_artifact_and_missing_rosters_fail_closed(tmp_path: Path) -> None:
    all_points, baseline, stage6, evaluator, drift_reference = _fixture(tmp_path)
    registry = json.loads(all_points.read_text(encoding="utf-8"))
    joint_dir = Path(registry["runs"]["joint_main"]["operating_point_directory"])
    (joint_dir / "transformer_operating_thresholds_2021A.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, drift_reference, tmp_path / "bad1.json")

    all_points, baseline, stage6, evaluator, drift_reference = _fixture(tmp_path / "missing")
    threshold = baseline.parent / "operating_thresholds_2021A.json"
    value = json.loads(threshold.read_text(encoding="utf-8"))
    del value["any_readmission"]["lightgbm"]
    write_json(threshold, value)
    lock = json.loads(baseline.read_text(encoding="utf-8"))
    lock["artifacts"][threshold.name] = {"bytes": threshold.stat().st_size, "sha256": digest(threshold)}
    write_json(baseline, lock)
    with pytest.raises(RuntimeError, match="Baseline model roster is incomplete"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, drift_reference, tmp_path / "bad2.json")

    all_points, baseline, stage6, evaluator, drift_reference = _fixture(tmp_path / "outcome")
    registry = json.loads(all_points.read_text(encoding="utf-8"))
    target = Path(registry["runs"]["scratch"]["operating_point_directory"]) / "transformer_operating_thresholds_2021A.json"
    values = json.loads(target.read_text(encoding="utf-8"))
    del values["ap_specific_readmission"]
    write_json(target, values)
    manifest_path = target.parent / "transformer_operating_point_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][target.name] = {"bytes": target.stat().st_size, "sha256": digest(target)}
    write_json(manifest_path, manifest)
    registry["runs"]["scratch"]["operating_point_manifest"] = artifact(manifest_path)
    write_json(all_points, registry)
    with pytest.raises(RuntimeError, match="threshold outcome roster"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, drift_reference, tmp_path / "bad3.json")


def test_future_access_flags_and_evaluator_hash_changes_fail_closed(tmp_path: Path) -> None:
    all_points, baseline, stage6, evaluator, drift_reference = _fixture(tmp_path)
    registry = json.loads(all_points.read_text(encoding="utf-8"))
    registry["year_2022_accessed"] = True
    write_json(all_points, registry)
    with pytest.raises(RuntimeError, match="exact pre-2021B/pre-2022 seal"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, drift_reference, tmp_path / "future.json")

    all_points, baseline, stage6, evaluator, drift_reference = _fixture(tmp_path / "code")
    output = tmp_path / "code" / "spec.json"
    builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, drift_reference, output)
    evaluator.write_text("# changed evaluator\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evaluation code SHA256 mismatch"):
        builder.validate_frozen_evaluation_spec(output, evaluator, drift_reference)


def test_spec_sidecar_tampering_is_detected(tmp_path: Path) -> None:
    all_points, baseline, stage6, evaluator, drift_reference = _fixture(tmp_path)
    output = tmp_path / "spec.json"
    builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, drift_reference, output)
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sidecar mismatch"):
        builder.validate_frozen_evaluation_spec(output, evaluator, drift_reference)


def _reseal_reference(path: Path, value: dict) -> None:
    write_json(path, value)
    path.with_suffix(".json.sha256").write_text(f"{digest(path)}  {path.name}\n", encoding="ascii")


def test_drift_reference_is_required_hash_bound_and_consumer_complete_before_publication(tmp_path: Path) -> None:
    all_points, baseline, stage6, evaluator, reference = _fixture(tmp_path / "no-sidecar")
    reference.with_suffix(".json.sha256").unlink()
    output = tmp_path / "no-sidecar" / "missing.json"
    with pytest.raises(RuntimeError, match="reference or SHA256 sidecar is missing"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, reference, output)
    assert not output.exists()

    all_points, baseline, stage6, evaluator, reference = _fixture(tmp_path / "tampered")
    reference.write_text(reference.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "tampered" / "tampered.json"
    with pytest.raises(RuntimeError, match="reference SHA256 sidecar mismatch"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, reference, output)
    assert not output.exists()

    all_points, baseline, stage6, evaluator, reference = _fixture(tmp_path / "seven")
    value = json.loads(reference.read_text(encoding="utf-8"))
    value["label_outcomes"].pop("in_hospital_death")
    _reseal_reference(reference, value)
    with pytest.raises(RuntimeError, match="exactly seven label outcomes"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, reference, tmp_path / "seven" / "bad.json")

    all_points, baseline, stage6, evaluator, reference = _fixture(tmp_path / "calibration")
    value = json.loads(reference.read_text(encoding="utf-8"))
    value["calibration"].pop("lightgbm")
    _reseal_reference(reference, value)
    with pytest.raises(RuntimeError, match="exactly every frozen model"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, reference, tmp_path / "calibration" / "bad.json")

    all_points, baseline, stage6, evaluator, reference = _fixture(tmp_path / "coverage")
    value = json.loads(reference.read_text(encoding="utf-8"))
    value["coverage"] = [item for item in value["coverage"] if not (item["conformal_set"] == "global_80" and item["subgroup_dimension"] == "sex")]
    _reseal_reference(reference, value)
    with pytest.raises(RuntimeError, match="prespecified subgroup coverage"):
        builder.build_frozen_evaluation_spec(all_points, baseline, stage6, evaluator, reference, tmp_path / "coverage" / "bad.json")
