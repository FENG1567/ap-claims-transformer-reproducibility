"""Synthetic no-NRD tests for the locked all-model 2022 prediction producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "predict_all_locked_2022.py"
MODULE_SPEC = importlib.util.spec_from_file_location("predict_all_locked_2022", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
producer = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(producer)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, object]:
    return {"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def sidecar(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest(path)}  {path.name}\n", encoding="ascii")


def marker(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def conformal_entries() -> list[dict]:
    entries = []
    for label, nominal in (("80", 0.8), ("90", 0.9), ("95", 0.95)):
        entries.append({"name": f"global_{label}", "scope": "global", "nominal": nominal,
                        "mask_column": f"mask_global_{label}", "size_column": f"size_global_{label}",
                        **({"abstain_column": "abstain_90"} if label == "90" else {})})
        for display in ("sex", "age", "PAY1", "ZIPINC_QRTL"):
            entries.append({"name": f"mondrian_{display}_{label}", "scope": "mondrian", "nominal": nominal,
                            "mondrian_dimension": display, "mask_column": f"mask_{display}_{label}",
                            "size_column": f"size_{display}_{label}"})
    return entries


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    artifacts = tmp_path / "artifacts"; artifacts.mkdir(parents=True)
    baseline_dir = artifacts / "baselines"; model_dir = baseline_dir / "models"; model_dir.mkdir(parents=True)
    preprocessor, token_maps = marker(model_dir / "static_preprocessor.joblib", "pre"), marker(model_dir / "development_token_maps.joblib", "maps")
    baseline_artifacts = {str(preprocessor.relative_to(baseline_dir)): identity(preprocessor), str(token_maps.relative_to(baseline_dir)): identity(token_maps)}
    baseline_paths: dict[str, dict[str, Path]] = {}
    for name in producer.BASELINES:
        baseline_paths[name] = {}
        for outcome in producer.OUTCOMES:
            path = marker(model_dir / f"{outcome}__{name}.joblib", f"{name}-{outcome}")
            baseline_paths[name][outcome] = path
            baseline_artifacts[str(path.relative_to(baseline_dir))] = identity(path)
    baseline_reports = {}
    for report_name in ("operating_thresholds_2021A.json", "calibration_selection_2021A.json", "model_selection_2021A.json"):
        report = marker(baseline_dir / report_name, report_name)
        baseline_artifacts[str(report.relative_to(baseline_dir))] = identity(report)
        baseline_reports[report_name] = report
    baseline_lock = baseline_dir / "baseline_lock.json"
    write_json(baseline_lock, {"status": "PASS_BASELINES_FROZEN_PRE_2021B_PRE_2022", "2021B_outcomes_accessed": False,
                               "year_2022_accessed": False, "artifacts": baseline_artifacts})
    transformer_models = {}
    for name in producer.TRANSFORMERS:
        directory = artifacts / name; directory.mkdir()
        checkpoint = marker(directory / "best_checkpoint.pt", name)
        finetune = directory / "finetune_manifest.json"
        write_json(finetune, {"status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022", "prediction_partition": "2021A",
                              "2021B_accessed": False, "year_2022_accessed": False, "checkpoint": identity(checkpoint),
                              "training_contract": {"max_tokens": 8}})
        op = {}
        for label in ("manifest", "thresholds", "calibrator", "calibration_selection"):
            path = marker(directory / f"{label}.bin", f"{name}-{label}")
            op[label] = identity(path)
        transformer_models[name] = {"family": "claims_transformer",
                                    "probabilities": {outcome: f"p_cal__{outcome}__{name}" for outcome in producer.OUTCOMES},
                                    "thresholds": {outcome: .5 for outcome in producer.OUTCOMES},
                                    "inference_artifacts": {"checkpoint": identity(checkpoint), "finetune_manifest": identity(finetune)},
                                    "2021A_operating_point_source": op}
    baseline_models = {}
    for name in producer.BASELINES:
        baseline_models[name] = {"family": "baseline",
                                 "probabilities": {outcome: f"p_cal__{outcome}__{name}" for outcome in producer.OUTCOMES},
                                 "thresholds": {outcome: .5 for outcome in producer.OUTCOMES},
                                 "inference_artifacts": {outcome: identity(baseline_paths[name][outcome]) for outcome in producer.OUTCOMES},
                                 "2021A_operating_point_source": {"baseline_lock": identity(baseline_lock),
                                                                    "thresholds": identity(baseline_reports["operating_thresholds_2021A.json"]),
                                                                    "calibration_selection": identity(baseline_reports["calibration_selection_2021A.json"]),
                                                                    "model_selection": identity(baseline_reports["model_selection_2021A.json"])}}
    conformal_artifact = marker(artifacts / "conformal_sets_2021B.parquet", "not-opened")
    calibrator = artifacts / "conformal_calibrator.json"
    levels = {"0": {"q": .6}, "1": {"q": .6}, "18-44": {"q": .6}, "45-64": {"q": .6}, "65-74": {"q": .6}, "75+": {"q": .6}, "2": {"q": .6}}
    write_json(calibrator, {"status": "PASS_LOCKED_PRE_2022", "calibration_partition": "2021B only",
                            "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False,
                            "operating_point_source": {"manifest_sha256": transformer_models["joint_main"]["2021A_operating_point_source"]["manifest"]["sha256"]},
                            "calibration": {"global": {x: {"q": .6} for x in ("80", "90", "95")},
                                            "mondrian": {dim: {x: levels for x in ("80", "90", "95")} for dim in producer.MONDRIAN.values()}}})
    stage6 = marker(artifacts / "stage6_registry.json", "registry")
    spec_path = tmp_path / "evaluation_spec.json"
    spec = {"status": "FROZEN_2022_EVALUATION_SPEC", "sealed_test_year": 2022,
            "transformer_roster": list(producer.TRANSFORMERS), "baseline_roster": list(producer.BASELINES),
            "co_primary_outcomes": {"any_readmission": "outcome_any_readmission", "ap_specific_readmission": "outcome_ap_specific_readmission"},
            "models": {**transformer_models, **baseline_models}, "conformal_sets": conformal_entries(),
            "conformal_2021B_source": {"stage6_registry": identity(stage6), "calibrator": identity(calibrator), "sets_artifact": identity(conformal_artifact)}}
    write_json(spec_path, spec); sidecar(spec_path)
    derivative_dir = tmp_path / "derivative"; derivative_dir.mkdir()
    derivative = derivative_dir / "locked_2022_derivative.parquet"
    rows = []
    for index, leaf in enumerate((0, 1, 2, 3, 4)):
        rows.append({"encounter_hash": f"e{index}", "patient_hash": f"p{index // 2}", "hospital_hash": f"h{index % 2}",
                     "NRD_STRATUM": f"s{index % 2}", "DISCWT": 1.0, "analysis_year": 2022, "analysis_partition": "test",
                     "AGE": 50 + index, "FEMALE": index % 2, "PAY1": 2, "ZIPINC_QRTL": 2, "readmission_leaf": leaf,
                     "dx_tokens": [4, 5], "pr_tokens": [8], "prday": [0], "prior_dx_tokens_180d": [9], "prior_pr_tokens_180d": [10],
                     "any_unplanned_readmission_30d": int(leaf > 0), "ap_specific_readmission_30d": int(leaf == 1),
                     "biliary_readmission_30d": int(leaf == 2), "sepsis_or_organ_readmission_30d": int(leaf == 3),
                     # Upstream cost validity is intentionally nullable and must
                     # not remove this otherwise eligible primary-test episode.
                     "high_cost_label": None if index == 4 else index % 2,
                     "prolonged_los_label": 0, "in_hospital_death_label": 0})
    pq.write_table(pa.Table.from_pylist(rows), derivative)
    derivative_manifest = derivative_dir / "manifest.json"
    # The actual unlock identity is filled after the lock is written below.
    write_json(derivative_manifest, {"status": "PASS_LOCKED_2022_MINIMUM_DERIVATIVE", "2022_accessed": True, "one_shot": True,
                                     "data_dependent_adaptation": False, "derivative": identity(derivative), "unlock_lock": {"sha256": "PENDING"}})
    lock_path = tmp_path / "unlock.json"
    lock = {"status": "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION", "sealed_test_year": 2022,
            "2022_access_before_lock": False, "frozen_code": {str(MODULE_PATH.resolve()): identity(MODULE_PATH), str(spec_path.resolve()): identity(spec_path)}}
    write_json(lock_path, lock); sidecar(lock_path)
    manifest = json.loads(derivative_manifest.read_text(encoding="utf-8")); manifest["unlock_lock"] = identity(lock_path)
    write_json(derivative_manifest, manifest); sidecar(derivative_manifest)
    return lock_path, spec_path, derivative_dir, tmp_path / "prediction_output"


def synthetic_predictor(name: str, frame: pd.DataFrame):
    n = len(frame)
    if name in producer.TRANSFORMERS:
        leaves = np.tile(np.array([[.55, .25, .10, .06, .04]]), (n, 1))
        return leaves[:, 1:].sum(axis=1), leaves
    return {"any_readmission": np.full(n, .4), "ap_specific_readmission": np.full(n, .2)}


def _future_oov_transformer_fixture(tmp_path: Path):
    """Build a real tiny frozen Torch checkpoint; no NRD data are involved."""
    torch = pytest.importorskip("torch", reason="real Transformer runtime is not installed in this local test environment")
    (ClaimsTransformer, ClaimsTransformerConfig, _ExampleDataset, StaticPreprocessor, _collate,
     _load_hierarchy, make_examples, _predict) = producer._import_transformer_components()
    root = tmp_path / "tiny_root"; hierarchy = tmp_path / "tiny_hierarchy"
    (root / "data" / "nrd").mkdir(parents=True); hierarchy.mkdir()
    vocabulary = {"token_to_id": {str(index): index for index in range(5)}}
    write_json(root / "data" / "nrd" / "diagnosis_vocabulary_state.json", vocabulary)
    write_json(root / "data" / "nrd" / "procedure_vocabulary_state.json", vocabulary)
    np.savez(hierarchy / "hierarchy_arrays.npz", dx_category=np.zeros(5, dtype=np.int64),
             dx_domain=np.zeros(5, dtype=np.int64), pr_category=np.zeros(5, dtype=np.int64),
             pr_domain=np.zeros(5, dtype=np.int64))
    write_json(hierarchy / "hierarchy_vocabularies.json", {
        "diagnosis_category": ["PAD"], "diagnosis_domain": ["PAD"],
        "procedure_category": ["PAD"], "procedure_domain": ["PAD"],
    })
    bundle = _load_hierarchy(root, hierarchy)
    static = StaticPreprocessor([], [], {}, {}, {})
    config = ClaimsTransformerConfig(unified_vocab_size=10, category_vocab_size=1, domain_vocab_size=1,
                                     static_dim=0, admission_time_static_dim=0, d_model=8, nhead=2,
                                     num_layers=1, dim_feedforward=16, dropout=0.0, max_encounters=2,
                                     year_vocab_size=5)
    model = ClaimsTransformer(config, bundle["token_to_category"], bundle["token_to_domain"])
    checkpoint = tmp_path / "tiny_checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "model_config": config.to_dict(),
                "static_preprocessor": static.to_dict(), "training_contract": {"max_tokens": 8}}, checkpoint)
    frame = pd.DataFrame([{
        "year": 2022, "encounter_hash": "future", "patient_hash": "patient", "analysis_partition": "test",
        "primary_analysis_eligible": True, "any_unplanned_readmission_30d": 0, "readmission_leaf": 0,
        "dx_tokens": [4], "pr_tokens": [4], "prday": [0], "LOS": 2,
        "prior_count_ytd": 0, "prior_count_30d": 0, "prior_count_90d": 0, "prior_count_180d": 0,
        "prior_ed_count_180d": 0, "prior_nonelective_count_180d": 0, "prior_ap_count_180d": 0,
        "prior_biliary_count_180d": 0, "prior_sepsis_or_organ_count_180d": 0, "prior_los_sum_180d": 0,
        "prior_max_severity_180d": 0, "prior_max_mortality_risk_180d": 0, "days_since_prior_discharge": 999,
        "history_30d_fully_observable": 1, "history_90d_fully_observable": 1,
        "history_180d_fully_observable": 1, "prior_dx_tokens_180d": [], "prior_pr_tokens_180d": [],
        "AGE": 55, "FEMALE": 1, "PAY1": 2, "ZIPINC_QRTL": 2, "DISCWT": 1.0,
    }])
    return root, hierarchy, checkpoint, frame, static, bundle, make_examples


def test_preflight_authenticates_all_artifacts_before_opening_derivative_rows(tmp_path: Path) -> None:
    lock, spec, derivative, _ = make_fixture(tmp_path)
    gate = producer.validate_preflight(lock, spec, derivative)
    assert set(gate["transformers"]) == set(producer.TRANSFORMERS)
    assert set(gate["baselines"]) == set(producer.BASELINES)
    # Tampering with a per-ablation 2021A calibrator must fail before parquet is opened.
    calibrator = Path(gate["spec"]["models"]["no_prday"]["2021A_operating_point_source"]["calibrator"]["file"])
    calibrator.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="2021A calibrator path/bytes/SHA256 mismatch"):
        producer.validate_preflight(lock, spec, derivative)


def test_atomic_all_model_synthetic_output_has_exact_schema_and_is_one_shot(tmp_path: Path) -> None:
    lock, spec, derivative, output = make_fixture(tmp_path)
    result = producer.produce_locked_2022(derivative, lock, spec, output, predictors=synthetic_predictor, threads=1)
    assert result["status"] == "PASS_LOCKED_2022_STANDARDIZED_PREDICTIONS"
    assert (output / "manifest.json.sha256").is_file()
    table = pd.read_parquet(output / "predictions_2022.parquet")
    assert table["encounter_hash"].is_unique
    assert table["hospital_hash"].tolist() == ["h0", "h1", "h0", "h1", "h0"]
    assert len([c for c in table if c.startswith("p_cal__")]) == 26
    assert len([c for c in table if c.startswith("mask_")]) == 15
    assert len([c for c in table if c.startswith("size_")]) == 15
    assert "abstain_90" in table and "binary_hierarchy_conflict_90" in table
    assert table["analysis_year"].eq(2022).all() and table["analysis_partition"].eq("test").all()
    assert len(table) == 5 and pd.isna(table.loc[table["encounter_hash"].eq("e4"), "outcome_high_cost"]).all()
    assert result["high_cost_label_completeness"] == {
        "valid_n": None, "valid_n_suppressed": True,
        "missing_n": None, "missing_n_suppressed": True,
        "small_cell_threshold": "n<=10", "missing_is_not_reclassified_as_low_cost": True,
    }
    with pytest.raises(RuntimeError, match="immutable"):
        producer.produce_locked_2022(derivative, lock, spec, output, predictors=synthetic_predictor, threads=1)


def test_derivative_lock_binding_and_missing_static_production_interface_fail_closed(tmp_path: Path) -> None:
    lock, spec, derivative, output = make_fixture(tmp_path)
    manifest_path = derivative / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["unlock_lock"]["sha256"] = "0" * 64
    write_json(manifest_path, manifest); sidecar(manifest_path)
    with pytest.raises(RuntimeError, match="not bound to the supplied"):
        producer.validate_preflight(lock, spec, derivative)
    # A real (non-stub) invocation cannot silently fall back to a synthetic model.
    lock, spec, derivative, output = make_fixture(tmp_path / "fresh")
    with pytest.raises(RuntimeError, match="requires --root and --hierarchy-dir"):
        producer.produce_locked_2022(derivative, lock, spec, output, threads=1)


def test_real_transformer_2022_uses_only_explicit_future_oov_year_bucket(tmp_path: Path) -> None:
    root, hierarchy, checkpoint, frame, static, bundle, make_examples = _future_oov_transformer_fixture(tmp_path)
    values = static.transform(frame)
    # The ordinary training/selection default remains sealed to 2018--2021.
    with pytest.raises(RuntimeError, match="Unauthorized year 2022"):
        make_examples(frame, bundle["dx_size"], 8, values)
    allowed = make_examples(frame, bundle["dx_size"], 8, values, allow_future_oov_year=True)
    assert allowed[0]["year_index"] == [4, 4]
    source = {"checkpoint": checkpoint, "manifest": {"training_contract": {"max_tokens": 8}}}
    any_probability, leaves = producer._transformer_predict(frame, source, root, hierarchy, 1, 0, "cpu")
    assert any_probability.shape == (1,) and leaves.shape == (1, 5)
    assert np.isfinite(any_probability).all() and np.isfinite(leaves).all()
