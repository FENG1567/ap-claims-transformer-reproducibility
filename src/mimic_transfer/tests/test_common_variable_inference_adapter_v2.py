import hashlib
import importlib.util
import json
import pathlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[3]
STAGE5 = ROOT / "work" / "stage5"
sys.path.insert(0, str(STAGE5))
from claims_transformer import ClaimsTransformer, ClaimsTransformerConfig
from finetune_ap_transformer_v2 import (
    COMMON_VARIABLE_CATEGORICAL_COLUMNS, COMMON_VARIABLE_NUMERIC_COLUMNS,
    StaticPreprocessor,
)

MODULE = Path(__file__).parents[1] / "common_variable_inference_adapter_v2.py"
spec = importlib.util.spec_from_file_location("common_adapter_v2", MODULE)
adapter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


def identity(path: Path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class _ForeignPosixPathPayload:
    """Forces a Linux ``pathlib.PosixPath`` global into a torch checkpoint."""

    def __reduce__(self):
        return pathlib.PosixPath, ("/sealed/linux-produced-checkpoint",)


def fixture_lock(tmp_path: Path, monkeypatch, *, foreign_path=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    static_source = {name: [1.0] for name in COMMON_VARIABLE_NUMERIC_COLUMNS}
    static_source.update({name: [1] for name in COMMON_VARIABLE_CATEGORICAL_COLUMNS})
    static_source.update({"AGE": [50.0], "LOS": [3.0], "I10_NDX": [2.0], "I10_NPR": [1.0]})
    static = StaticPreprocessor.fit(
        __import__("pandas").DataFrame(static_source),
        numeric_columns=COMMON_VARIABLE_NUMERIC_COLUMNS,
        categorical_columns=COMMON_VARIABLE_CATEGORICAL_COLUMNS,
    )
    hierarchy = tmp_path / "hierarchy_arrays.npz"
    np.savez(hierarchy, dx_category=np.array([0, 0, 0, 0, 1]), pr_category=np.array([0, 0, 0, 0, 1]),
             dx_domain=np.array([0, 0, 0, 0, 1]), pr_domain=np.array([0, 0, 0, 0, 1]))
    hv = tmp_path / "hierarchy_vocabularies.json"
    write_json(hv, {"diagnosis_category": {"pad": 0, "x": 1}, "procedure_category": {"pad": 0, "p": 1},
                    "diagnosis_domain": {"pad": 0, "x": 1}, "procedure_domain": {"pad": 0, "p": 1}})
    dx, pr = tmp_path / "dx.json", tmp_path / "pr.json"
    write_json(dx, {"token_to_id": {str(i): i for i in range(5)}}); write_json(pr, {"token_to_id": {str(i): i for i in range(5)}})
    config = ClaimsTransformerConfig(unified_vocab_size=10, category_vocab_size=3, domain_vocab_size=3,
        static_dim=static.dimension, admission_time_static_dim=0, d_model=8, nhead=2, num_layers=1,
        dim_feedforward=16, dropout=0.0, use_year_version=False)
    category = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 2]); domain = category.clone()
    model = ClaimsTransformer(config, category, domain)
    contract = {"max_tokens": 8}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint_payload = {"model_config": config.to_dict(), "model_state": model.state_dict(), "static_preprocessor": static.to_dict(), "training_contract": contract}
    if foreign_path:
        checkpoint_payload["foreign_platform_bookkeeping_path"] = _ForeignPosixPathPayload()
    torch.save(checkpoint_payload, checkpoint)
    prediction = tmp_path / "prediction.bin"; prediction.write_bytes(b"2021A only")
    finetune = tmp_path / "finetune.json"
    write_json(finetune, {"status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022", "prediction_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False, "ablations": {"common_variable_only": True}, "checkpoint": {"sha256": identity(checkpoint)["sha256"]}, "prediction": {"sha256": identity(prediction)["sha256"]}, "model_config": config.to_dict(), "static_preprocessor": static.to_dict(), "static_feature_set": {"numeric": list(static.numeric_columns), "categorical": list(static.categorical_columns), "dimension": static.dimension}, "training_contract": contract})
    cal = tmp_path / "cal.joblib"; joblib.dump({"version": 1, "endpoints": {"any_readmission": {"method": "none", "model": None}, "ap_specific_readmission": {"method": "none", "model": None}}}, cal)
    thresholds = tmp_path / "thresholds.json"
    write_json(thresholds, {x: {"probability_column": f"p_{x}_calibrated", "rule": "fixed 20% capacity on 2021A; threshold transported unchanged", "threshold": .2} for x in ("any_readmission", "ap_specific_readmission")})
    operating = tmp_path / "operating.json"
    write_json(operating, {"status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A", "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False, "source": {"manifest_sha256": identity(finetune)["sha256"], "sha256": identity(prediction)["sha256"]}, "artifacts": {"transformer_binary_calibrators.joblib": {"bytes": cal.stat().st_size, "sha256": identity(cal)["sha256"]}, "transformer_operating_thresholds_2021A.json": {"bytes": thresholds.stat().st_size, "sha256": identity(thresholds)["sha256"]}}})
    stage6 = tmp_path / "stage6_2021B" / "conformal_2021B"
    stage6.mkdir(parents=True)
    cm, cs = stage6 / "conformal_calibrator.json", stage6 / "conformal_sets_2021B.parquet"
    write_json(cm, {}); cs.write_bytes(b"sets")
    def foreign_identity(path: Path):
        result = identity(path)
        result["path"] = "/foreign/immutable-copy/" + str(path.relative_to(tmp_path)).replace("\\\\", "/")
        return result
    group = {str(c): {"0": {"q": .4}, "1": {"q": .5}, "18-44": {"q": .6}, "45-64": {"q": .6}, "65-74": {"q": .6}, "75+": {"q": .6}} for c in (80, 90, 95)}
    binding = tmp_path / "binding.json"
    write_json(binding, {"status": "PASS_MIMIC_CONFORMAL_PROJECTION_PRE_DATA", "calibration_partition": "2021B only", "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False, "allowed_mondrian_dimensions": ["sex", "age_group"], "nominal_coverages": [.8, .9, .95], "conformal_manifest": foreign_identity(cm), "conformal_sets": foreign_identity(cs), "global": {str(c): {"q": .4} for c in (80, 90, 95)}, "mondrian": {"sex": {str(c): {"0": {"q": .4}, "1": {"q": .5}} for c in (80, 90, 95)}, "age_group": {str(c): {x: {"q": .6} for x in ("18-44", "45-64", "65-74", "75+")} for c in (80, 90, 95)}}})
    files = {"checkpoint": checkpoint, "finetune_manifest": finetune, "operating_manifest": operating, "calibrators": cal, "thresholds": thresholds, "hierarchy_arrays": hierarchy, "hierarchy_vocabularies": hv, "dx_vocabulary": dx, "pr_vocabulary": pr, "conformal_binding": binding}
    # The production adapter pins these two registered identities.  Small,
    # production-isomorphic state_dict fixtures replace only the constants in
    # this module, never the identity validation code path.
    monkeypatch.setattr(adapter, "EXPECTED_COMMON_VARIABLE_CHECKPOINT_SHA256", identity(checkpoint)["sha256"])
    monkeypatch.setattr(adapter, "EXPECTED_COMMON_VARIABLE_FINETUNE_MANIFEST_SHA256", identity(finetune)["sha256"])
    lock = {"inference_adapter": {"model_id": "common_variable_only", "mimic_fit": False, "mimic_recalibration": False, "mimic_threshold_optimization": False, "vocabulary_growth": False, **{key: identity(value) for key, value in files.items()}}}
    feature = {**{name: 1.0 for name in COMMON_VARIABLE_NUMERIC_COLUMNS}, **{name: 1 for name in COMMON_VARIABLE_CATEGORICAL_COLUMNS}, "AGE": 50, "FEMALE": 1, "LOS": 3, "I10_NDX": 2, "I10_NPR": 1, "dx_tokens": [4], "pr_tokens": [4], "prday": [1], "prior_dx_tokens_180d": [3], "prior_pr_tokens_180d": [], "oov_audit": {"diagnosis": 1, "procedure": 0, "total": 1}}
    return lock, feature, files


def test_loads_real_model_class_state_dict_and_returns_locked_outputs(tmp_path, monkeypatch):
    lock, feature, _ = fixture_lock(tmp_path, monkeypatch)
    adapter.clear_model_cache(); out = adapter.predict_common_variable_episode(feature, lock)
    assert out["model_id"] == "common_variable_only" and out["calendar_year_used"] is False
    assert set(out["decisions"]) == {"any_readmission", "ap_specific_readmission"}
    for coverage in ("80", "90", "95"):
        assert set(out["conformal"][coverage]) >= {"set", "abstain", "global", "sex", "age_group"}
        assert out["conformal"][coverage]["sex"]["group"] == "1"
        assert out["conformal"][coverage]["age_group"]["group"] == "45-64"


def test_foreign_path_checkpoint_load_is_isolated_and_model_still_runs(tmp_path, monkeypatch):
    lock, feature, files = fixture_lock(tmp_path, monkeypatch, foreign_path=True)
    adapter.clear_model_cache()
    original_posix_path = pathlib.PosixPath
    payload = adapter._load_checkpoint(files["checkpoint"])
    assert isinstance(payload["foreign_platform_bookkeeping_path"], pathlib.Path)
    assert pathlib.PosixPath is original_posix_path
    out = adapter.predict(feature, lock)
    assert out["calendar_year_used"] is False


def test_identity_tamper_and_year_gate_fail_closed(tmp_path, monkeypatch):
    lock, feature, files = fixture_lock(tmp_path, monkeypatch)
    files["thresholds"].write_text("{}", encoding="utf-8")
    adapter.clear_model_cache()
    with pytest.raises(RuntimeError, match="identity mismatch"):
        adapter.predict(feature, lock)
    lock, feature, _ = fixture_lock(tmp_path / "second", monkeypatch)
    (tmp_path / "second").mkdir(exist_ok=True)
    feature["year"] = 2022
    adapter.clear_model_cache()
    with pytest.raises(RuntimeError, match="Calendar year"):
        adapter.predict(feature, lock)


def test_common_variable_year_config_static_and_oov_contracts(tmp_path, monkeypatch):
    lock, feature, _ = fixture_lock(tmp_path, monkeypatch)
    feature["oov_audit"] = {"diagnosis": 1, "procedure": 0, "total": 2}
    with pytest.raises(RuntimeError, match="OOV audit"):
        adapter.predict(feature, lock)
    feature["oov_audit"] = {"diagnosis": 0, "procedure": 0, "total": 0}
    out = adapter.predict(feature, lock)
    assert 0 <= out["p_any_readmission_calibrated"] <= 1


def test_rejects_checkpoint_that_enables_year_embedding(tmp_path, monkeypatch):
    lock, feature, files = fixture_lock(tmp_path, monkeypatch)
    payload = torch.load(files["checkpoint"], map_location="cpu", weights_only=False)
    payload["model_config"]["use_year_version"] = True
    torch.save(payload, files["checkpoint"])
    finetune = json.loads(files["finetune_manifest"].read_text(encoding="utf-8"))
    finetune["checkpoint"]["sha256"] = identity(files["checkpoint"])["sha256"]
    finetune["model_config"] = payload["model_config"]
    write_json(files["finetune_manifest"], finetune)
    operating = json.loads(files["operating_manifest"].read_text(encoding="utf-8"))
    operating["source"]["manifest_sha256"] = identity(files["finetune_manifest"])["sha256"]
    write_json(files["operating_manifest"], operating)
    lock["inference_adapter"]["checkpoint"] = identity(files["checkpoint"])
    lock["inference_adapter"]["finetune_manifest"] = identity(files["finetune_manifest"])
    lock["inference_adapter"]["operating_manifest"] = identity(files["operating_manifest"])
    monkeypatch.setattr(adapter, "EXPECTED_COMMON_VARIABLE_CHECKPOINT_SHA256", identity(files["checkpoint"])["sha256"])
    monkeypatch.setattr(adapter, "EXPECTED_COMMON_VARIABLE_FINETUNE_MANIFEST_SHA256", identity(files["finetune_manifest"])["sha256"])
    adapter.clear_model_cache()
    with pytest.raises(RuntimeError, match="calendar-year embedding"):
        adapter.predict(feature, lock)


def test_relocated_foreign_provenance_uses_local_canonical_stage6_files(tmp_path, monkeypatch):
    lock, feature, files = fixture_lock(tmp_path, monkeypatch)
    binding = json.loads(files["conformal_binding"].read_text(encoding="utf-8"))
    assert binding["conformal_manifest"]["path"].startswith("/foreign/")
    adapter.clear_model_cache()
    output = adapter.predict_common_variable_episode(feature, lock)
    assert set(output["conformal"]) == {"80", "90", "95"}


def test_relocated_conformal_tamper_and_wrong_suffix_fail_closed(tmp_path, monkeypatch):
    lock, _feature, files = fixture_lock(tmp_path, monkeypatch)
    binding_path = files["conformal_binding"]
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    manifest = binding_path.parent / "stage6_2021B" / "conformal_2021B" / "conformal_calibrator.json"
    manifest.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="MIMIC conformal nested identity: conformal_manifest"):
        adapter._validate_conformal(binding_path)
    wrong = dict(binding["conformal_sets"])
    wrong["path"] = "/foreign/not-the-locked-stage6.parquet"
    with pytest.raises(RuntimeError, match="MIMIC conformal provenance suffix: conformal_sets"):
        adapter._canonical_conformal_file(
            binding_path, wrong, adapter.CONFORMAL_CANONICAL_SUFFIXES["conformal_sets"], "conformal_sets"
        )


def test_relocated_conformal_symlink_is_rejected(tmp_path, monkeypatch):
    _lock, _feature, files = fixture_lock(tmp_path, monkeypatch)
    binding_path = files["conformal_binding"]
    target = binding_path.parent / "stage6_2021B" / "conformal_2021B" / "conformal_sets_2021B.parquet"
    alternate = target.with_name("alternate.parquet")
    target.replace(alternate)
    try:
        target.symlink_to(alternate)
    except OSError:
        pytest.skip("host disallows test symlink creation")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="MIMIC conformal symlink: conformal_sets"):
        adapter._canonical_conformal_file(
            binding_path, binding["conformal_sets"], adapter.CONFORMAL_CANONICAL_SUFFIXES["conformal_sets"], "conformal_sets"
        )


def test_relocated_conformal_symlink_guard_is_exercised_without_host_symlink_privilege(tmp_path, monkeypatch):
    _lock, _feature, files = fixture_lock(tmp_path, monkeypatch)
    binding_path = files["conformal_binding"]
    target = binding_path.parent / "stage6_2021B" / "conformal_2021B" / "conformal_sets_2021B.parquet"
    original_is_symlink = Path.is_symlink

    def synthetic_symlink(path: Path) -> bool:
        return path == target or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_symlink)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="MIMIC conformal symlink: conformal_sets"):
        adapter._canonical_conformal_file(
            binding_path, binding["conformal_sets"], adapter.CONFORMAL_CANONICAL_SUFFIXES["conformal_sets"], "conformal_sets"
        )
