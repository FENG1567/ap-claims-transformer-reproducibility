"""Subprocess E2E tests for the independent v7 restricted runtime."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "run_locked_mimic_transfer_v7.py"
spec = importlib.util.spec_from_file_location("mimic_v7_test", MODULE)
v = importlib.util.module_from_spec(spec); sys.modules[spec.name] = v; spec.loader.exec_module(v)

def h(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(p: Path) -> dict: return {"path": str(p), "bytes": p.stat().st_size, "sha256": h(p)}
def write_json(p: Path, x: object) -> Path: p.write_text(json.dumps(x, sort_keys=True), encoding="utf-8"); return p
def gz(rows, fields):
    b = io.StringIO(); w = csv.DictWriter(b, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return gzip.compress(b.getvalue().encode("utf-8"))
def cli(*args, test_only=True, check=True):
    env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    if test_only: env["AP_CLAIMS_TEST_ONLY"] = "1"
    else: env.pop("AP_CLAIMS_TEST_ONLY", None)
    return subprocess.run([sys.executable, str(MODULE), *map(str, args)], text=True, capture_output=True, check=check, env=env)

def fixture(tmp_path: Path, invalid_admissions=()):
    admissions, patients, dx, pr = [], [], [], []
    for i in range(1, 12):
        patients.append({"subject_id": str(i), "anchor_age": "50", "anchor_year": "2020", "gender": "F" if i % 2 else "M", "dod": ""})
        admissions.extend([
            {"subject_id": str(i), "hadm_id": str(i*10), "admittime": "2020-01-01 00:00:00", "dischtime": "2020-01-02 00:00:00", "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME", "admission_type": "EMERGENCY", "hospital_expire_flag": "0", "deathtime": ""},
            {"subject_id": str(i), "hadm_id": str(i*10+1), "admittime": "2020-01-10 00:00:00", "dischtime": "2020-01-11 00:00:00", "admission_location": "EMERGENCY ROOM", "discharge_location": "HOME", "admission_type": "EMERGENCY", "hospital_expire_flag": "0", "deathtime": ""},
        ])
        dx.extend([{"subject_id": str(i), "hadm_id": str(i*10), "seq_num": "1", "icd_code": "K85.9", "icd_version": "10"}, {"subject_id": str(i), "hadm_id": str(i*10+1), "seq_num": "1", "icd_code": "K80.1", "icd_version": "10"}])
        pr.append({"subject_id": str(i), "hadm_id": str(i*10), "seq_num": "1", "icd_code": "0FJD0ZZ", "icd_version": "10", "chartdate": ""})
    admissions.extend(invalid_admissions)
    bodies = {"mimic-iv-3.1/hosp/admissions.csv.gz": gz(admissions, list(admissions[0])), "mimic-iv-3.1/hosp/patients.csv.gz": gz(patients, list(patients[0])), "mimic-iv-3.1/hosp/diagnoses_icd.csv.gz": gz(dx, list(dx[0])), "mimic-iv-3.1/hosp/procedures_icd.csv.gz": gz(pr, list(pr[0]))}
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, body in bodies.items(): z.writestr(name, body)
    with zipfile.ZipFile(archive) as z:
        members = {name: {"bytes": z.getinfo(name).file_size, "compressed_bytes": z.getinfo(name).compress_size, "crc32": f"{z.getinfo(name).CRC:08x}", "sha256": hashlib.sha256(z.read(name)).hexdigest()} for name in bodies}
    method = write_json(tmp_path / "method.json", {"status": v.METHOD_STATUS, "source": {"archive_bytes": archive.stat().st_size, "archive_sha256": h(archive), "required_members": list(bodies), "required_member_identities": members}, "planned_readmission_under_shifted_dates": {"annual_rule_sets": list(v.YEARS)}, "prediction_anchor_and_history": {"calendar_year_embedding": "DISABLED"}, "model_and_calibration": {"required_pending_bindings": ["real"]}})
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"\x80binary torch checkpoint - never json parsed")
    finetune = write_json(tmp_path / "finetune.json", {"status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022", "prediction_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False, "ablations": {"common_variable_only": True}, "model_config": {"use_year_version": False}, "checkpoint": identity(checkpoint), "static_preprocessor": {"numeric_columns": ["AGE", "LOS"], "categorical_columns": ["FEMALE"]}})
    calibrators = tmp_path / "calibrators.joblib"; calibrators.write_bytes(b"not opened by fake adapter")
    thresholds = write_json(tmp_path / "thresholds.json", {"any": 0.2})
    operating = write_json(tmp_path / "operating.json", {"status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A", "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False, "source": {"manifest_sha256": h(finetune)}, "artifacts": {"transformer_binary_calibrators.joblib": identity(calibrators), "transformer_operating_thresholds_2021A.json": identity(thresholds)}})
    dxv, prv = write_json(tmp_path / "dx.json", {"token_to_id": {"K859": 5, "K801": 6}}), write_json(tmp_path / "pr.json", {"token_to_id": {"0FJD0ZZ": 6}})
    arrays = tmp_path / "hierarchy_arrays.npz"; arrays.write_bytes(b"immutable synthetic hierarchy")
    hv = write_json(tmp_path / "hierarchy_vocabularies.json", {"diagnosis_category": ["x"], "procedure_category": ["x"], "diagnosis_domain": ["x"], "procedure_domain": ["x"]})
    stage6 = tmp_path / "stage6_2021B" / "conformal_2021B"; stage6.mkdir(parents=True)
    cm, cs = stage6 / "conformal_calibrator.json", stage6 / "conformal_sets_2021B.parquet"; cm.write_text("{}", encoding="utf-8"); cs.write_bytes(b"sets")
    def remote_identity(path):
        item = identity(path); item["path"] = "/foreign/immutable-copy/" + str(path.relative_to(tmp_path)).replace("\\\\", "/"); return item
    conformal_manifest, conformal_sets = remote_identity(cm), remote_identity(cs)
    binding = write_json(tmp_path / "mimic_binding.json", {"status": v.TRANSFER_STATUS, "calibration_partition": "2021B only", "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False, "allowed_mondrian_dimensions": ["sex", "age_group"], "nominal_coverages": [.8, .9, .95], "global": {str(q): {"q": .8} for q in (80, 90, 95)}, "mondrian": {"sex": {str(q): {"0": {"q": .8}, "1": {"q": .8}} for q in (80, 90, 95)}, "age_group": {str(q): {"18-44": {"q": .8}, "45-64": {"q": .8}, "65-74": {"q": .8}, "75+": {"q": .8}} for q in (80, 90, 95)}}, "conformal_manifest": conformal_manifest, "conformal_sets": conformal_sets})
    model_lock = write_json(tmp_path / "model_lock.json", {"status": v.MODEL_LOCK_STATUS, "selection_partition": "2021A", "selected_id": v.MODEL_ID, "selection_mode": "pre_specified_common_variable_only_no_metric_selection", "2021B_accessed": False, "year_2022_accessed": False, "selected_model": {"configuration": {"common_variable_only": True}, "checkpoint": identity(checkpoint), "finetune_manifest": identity(finetune)}, "operating_point": {"manifest": identity(operating), "binary_calibrators": identity(calibrators), "thresholds": identity(thresholds)}})
    registry = write_json(tmp_path / "registry.json", {"status": v.REGISTRY_STATUS, "model_id": v.MODEL_ID, "selection_partition": "2021A", "calibration_partition": "2021B only", "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False, "model_lock": identity(model_lock), "mimic_transfer_conformal_binding": identity(binding), "stage6": {"conformal_manifest": conformal_manifest, "conformal_sets": conformal_sets}})
    labels = write_json(tmp_path / "ontology.csv", {})
    labels.write_text("label_node,code\nreadmission.ap_specific,K859\nreadmission.biliary_event,K801\nreadmission.sepsis_or_acute_organ_dysfunction,A419\n", encoding="utf-8")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("def predict_common_variable_episode(feature, execution_lock):\n assert 'inference_adapter' in execution_lock\n assert all(k not in feature for k in ('year','calendar_year','year_index'))\n c={}\n for q in ('80','90','95'):\n  c[q]={'set':['none','biliary'],'abstain':False,'sex':{'group':str(int(feature['FEMALE'])),'set':['none','biliary'],'abstain':False},'age_group':{'group':'45-64','set':['none','biliary'],'abstain':False}}\n return {'p_any_readmission_calibrated':.4,'conformal':c}\n", encoding="utf-8")
    root = Path(__file__).parents[3]
    old = root / "outputs" / "stage2_lock" / "planned_readmission" / "planned_readmission_lock.json"; new = root / "outputs" / "stage7_pre2022_lock" / "planned_readmission_2022" / "planned_readmission_2022_lock.json"
    artifacts = {"checkpoint": checkpoint, "finetune_manifest": finetune, "operating_manifest": operating, "calibrators": calibrators, "thresholds": thresholds, "hierarchy_arrays": arrays, "hierarchy_vocabularies": hv, "dx_vocabulary": dxv, "pr_vocabulary": prv, "conformal_binding": binding}
    bindings = {name: identity(path) for name, path in artifacts.items()}; bindings.update({"common_variable_model_lock": identity(model_lock), "conformal_registry": identity(registry), "adapter": identity(adapter), "stage2_pra_lock": identity(old), "pra_2022_lock": identity(new), "ontology_labels": identity(labels)})
    adapter_lock = {"model_id": v.MODEL_ID, "module": "adapter", "function": "predict_common_variable_episode", "mimic_fit": False, "mimic_recalibration": False, "mimic_threshold_optimization": False, "vocabulary_growth": False, **{name: bindings[name] for name in v.ARTIFACTS}}
    specpath = write_json(tmp_path / "bindings.json", {"contract": {"projection": v.MODEL_ID, "calibration_partition": "2021A", "conformal_partition": "2021B", "mimic_fit": False, "mimic_recalibration": False, "mimic_threshold_optimization": False, "vocabulary_growth": False}, "bindings": bindings, "inference_adapter": adapter_lock, "evaluation": {"endpoints": {"any_readmission": {"probability_field": "p_any_readmission_calibrated", "threshold": .2}}, "coverages": [.8, .9, .95], "subgroups": ["sex", "age_group"]}})
    return method, archive, specpath, finetune

def test_subprocess_four_modes_binary_checkpoint_and_formal_run(tmp_path):
    method, archive, bindings, _ = fixture(tmp_path); lock, out = tmp_path / "execution.json", tmp_path / "out"
    assert "NO_ARCHIVE_ACCESS" in cli("--method-lock", method, "--validate-only").stdout
    frozen = cli("--method-lock", method, "--freeze-execution-lock", "--bindings", bindings, "--execution-lock-out", lock); assert v.EXECUTION_STATUS in frozen.stdout
    assert json.loads(lock.read_text())["execution_lock_version"] == 7
    assert "NO_ARCHIVE_ACCESS" in cli("--method-lock", method, "--validate-only", "--execution-lock", lock).stdout
    final = cli("--method-lock", method, "--execution-lock", lock, "--archive", archive, "--output-dir", out, "--test-only-bootstrap-replicates", 3); assert "PASS_RESTRICTED_LOCKED_MIMIC_TRANSFER_V7" in final.stdout
    manifest = json.loads((out / "restricted" / "manifest.json").read_text()); assert manifest["row_count"] == 11 and manifest["evaluation"]["subject_cluster_bootstrap"]["replicates"] == 3
    assert {"features", "prediction", "label"}.issubset(manifest["restricted_row_field_schema"]) and manifest["adapter_code_identity"]["sha256"] == manifest["binding_identities"]["adapter"]["sha256"]
    assert manifest["admission_data_quality_audit"] == {"input_rows": 22, "valid_rows": 22, "excluded_total": 0, "reason_counts": {"missing_ids": 0, "invalid_or_missing_timestamps": 0, "discharge_before_admit": 0}}
    assert "subject_id" not in (out / "public" / "aggregate.json").read_text()

def test_invalid_admissions_are_excluded_before_collapse_with_aggregate_audit_and_manifest_binding(tmp_path, monkeypatch):
    invalid = [
        {"subject_id": "", "hadm_id": "x", "admittime": "2020-01-01", "dischtime": "2020-01-02"},
        {"subject_id": "99", "hadm_id": "990", "admittime": "", "dischtime": "2020-01-02"},
        {"subject_id": "98", "hadm_id": "980", "admittime": "2020-01-03", "dischtime": "2020-01-02"},
    ]
    valid = [{"subject_id": "1", "hadm_id": "1", "admittime": "2020-01-01", "dischtime": "2020-01-02"}, {"subject_id": "1", "hadm_id": "2", "admittime": "2020-01-05", "dischtime": "2020-01-06"}]
    filtered, audit = v.valid_admissions([*valid, *invalid])
    assert audit == {"input_rows": 5, "valid_rows": 2, "excluded_total": 3, "reason_counts": {"missing_ids": 1, "invalid_or_missing_timestamps": 1, "discharge_before_admit": 1}}
    assert v.collapse(filtered) == v.collapse(valid)
    method, archive, bindings, _ = fixture(tmp_path, invalid); lock, out = tmp_path / "execution.json", tmp_path / "out"
    v.freeze(method, bindings, lock)
    monkeypatch.setenv("AP_CLAIMS_TEST_ONLY", "1")
    result = v.run(method, lock, archive, out, test_only_bootstrap_replicates=3)
    manifest = json.loads((out / "restricted" / "manifest.json").read_text())
    expected = {"input_rows": 25, "valid_rows": 22, "excluded_total": 3, "reason_counts": {"missing_ids": 1, "invalid_or_missing_timestamps": 1, "discharge_before_admit": 1}}
    assert result["rows"] == manifest["row_count"] == 11
    assert result["admission_data_quality_audit"] == manifest["admission_data_quality_audit"] == expected
    assert "subject_id" not in json.dumps(manifest["admission_data_quality_audit"])

def test_tamper_prearchive_fails_before_archive_open(tmp_path):
    method, _, bindings, finetune = fixture(tmp_path); lock = tmp_path / "execution.json"; v.freeze(method, bindings, lock)
    finetune.write_text("{}", encoding="utf-8")
    sentinel = tmp_path / "archive_sentinel"; sentinel.write_text("must never open", encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding identity: finetune_manifest"): v.run(method, lock, sentinel, tmp_path / "out", test_only_bootstrap_replicates=3)

def test_relocated_conformal_nested_records_are_local_and_fail_closed(tmp_path):
    """Foreign provenance is accepted only for the exact local Stage 6 copy."""
    method, _, bindings, _ = fixture(tmp_path)
    # This invokes the full prearchive semantic chain during freeze.  The
    # fixture's binding and registry preserve a non-existent foreign root.
    lock = tmp_path / "execution.json"; v.freeze(method, bindings, lock)
    assert v.validate_only(method, lock)["archive_opened"] is False
    spec = json.loads(bindings.read_text(encoding="utf-8")); binding_path = Path(spec["bindings"]["conformal_binding"]["path"])
    conformal = json.loads(binding_path.read_text(encoding="utf-8")); record = conformal["conformal_manifest"]
    target = binding_path.parent / "stage6_2021B" / "conformal_2021B" / "conformal_calibrator.json"
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="MIMIC conformal nested identity: conformal_manifest"):
        v._canonical_conformal_file(binding_path, record, "stage6_2021B/conformal_2021B/conformal_calibrator.json", "conformal_manifest")
    wrong_suffix = dict(record); wrong_suffix["path"] = "/foreign/not-the-locked-stage6.json"
    with pytest.raises(RuntimeError, match="MIMIC conformal provenance suffix: conformal_manifest"):
        v._canonical_conformal_file(binding_path, wrong_suffix, "stage6_2021B/conformal_2021B/conformal_calibrator.json", "conformal_manifest")

def test_relocated_conformal_symlink_is_rejected(tmp_path):
    _, _, bindings, _ = fixture(tmp_path)
    spec = json.loads(bindings.read_text(encoding="utf-8")); binding_path = Path(spec["bindings"]["conformal_binding"]["path"])
    conformal = json.loads(binding_path.read_text(encoding="utf-8")); record = conformal["conformal_sets"]
    target = binding_path.parent / "stage6_2021B" / "conformal_2021B" / "conformal_sets_2021B.parquet"; alternate = target.with_name("alternate.parquet")
    target.replace(alternate)
    try:
        target.symlink_to(alternate)
    except OSError:
        pytest.skip("host disallows test symlink creation")
    with pytest.raises(RuntimeError, match="MIMIC conformal symlink: conformal_sets"):
        v._canonical_conformal_file(binding_path, record, "stage6_2021B/conformal_2021B/conformal_sets_2021B.parquet", "conformal_sets")

def test_model_lock_chain_and_leaf_mapping_and_subgroups(tmp_path):
    method, _, bindings, _ = fixture(tmp_path); lock = tmp_path / "bad_lock.json"; v.freeze(method, bindings, lock)
    bad = json.loads(lock.read_text()); bad["inference_adapter"]["checkpoint"]["sha256"] = "0" * 64; lock.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="inference adapter/top-level mismatch"): v.prearchive(method, lock)
    assert v.REPORT_TO_MODEL["AP_specific"] == "ap" and v.MODEL_TO_REPORT["sepsis_or_organ"] == "sepsis_or_acute_organ_dysfunction"
    rows = [{"subject_id": "1", "sex": "1", "age_group": "45-64", "label": {"leaf": "AP_specific", "any_readmission": 0}, "prediction": {"p": .1, "conformal": {q: {"set": ["ap"], "abstain": False, "sex": {"group": "1", "set": ["ap"], "abstain": False}, "age_group": {"group": "45-64", "set": ["ap"], "abstain": True}} for q in ("80", "90", "95")}}}]
    metrics = v.evaluate(rows, {"endpoints": {"x": {"probability_field": "p", "threshold": .2}}, "coverages": [.8, .9, .95], "subgroups": ["sex", "age_group"]}, 1)
    assert metrics["conformal"]["80"]["sex"]["abstention_rate"] == 0 and metrics["conformal"]["80"]["age_group"]["abstention_rate"] == 1
    endpoint_rows = [{"label": {"x": 0}, "prediction": {"p": .1}}, {"label": {"x": 1}, "prediction": {"p": .9}}]
    scored = v.metrics(endpoint_rows, {"probability_field": "p", "threshold": .2}, "x")
    assert scored["prevalence"] == .5 and scored["auprc"] == 1 and "log_loss" in scored and "calibration_gate" in scored
    assert any(key.startswith("algorithm_unknown_extreme_bounds") for key in metrics["subject_cluster_bootstrap"]["estimands"])
    assert any(key.startswith("conformal.80.sex") for key in metrics["subject_cluster_bootstrap"]["estimands"])

def test_transfer_collapse_prday_oov_and_missing_sex_exclusion():
    ad = [{"subject_id": "1", "hadm_id": "1", "admittime": "2020-01-01", "dischtime": "2020-01-02", "discharge_location": "ACUTE HOSPITAL", "admission_location": "ER", "hospital_expire_flag": "0"}, {"subject_id": "1", "hadm_id": "2", "admittime": "2020-01-03", "dischtime": "2020-01-04", "admission_location": "ACUTE HOSPITAL", "hospital_expire_flag": "0"}]
    dx = [{"hadm_id": "2", "seq_num": "1", "icd_version": "10", "icd_code": "K85.9"}]
    assert not v.eligible(ad, [{"subject_id": "1", "anchor_age": "50", "anchor_year": "2020", "gender": "", "dod": ""}], dx)[0]
    ix, eps, people = v.eligible(ad, [{"subject_id": "1", "anchor_age": "50", "anchor_year": "2020", "gender": "F", "dod": ""}], dx); assert ix[0]["hadm_ids"] == ("1", "2")
    f = v.features(ix[0], eps, people, dx, [{"hadm_id": "2", "seq_num": "1", "icd_code": "NEW", "icd_version": "10", "chartdate": ""}], {}, {}, lambda e: "other")
    assert f["prday"] == [None] and f["oov_audit"]["total"] >= 1 and not set(f) & {"year", "calendar_year", "year_index"}

def test_indexed_bootstrap_preserves_metrics_and_1000_replicate_contract():
    rows = []
    for i in range(40):
        any_label = i % 3 == 0
        rows.append({"subject_id": str(i // 2), "sex": str(i % 2), "age_group": "45-64", "label": {"any_readmission": int(any_label), "label_status": "qualified", "leaf": "biliary" if any_label else "none"}, "prediction": {"p_any_readmission_calibrated": 0.2 + 0.01 * (i % 10), "conformal": {q: {"set": ["biliary"], "abstain": False, "sex": {"group": str(i % 2), "set": ["biliary"], "abstain": False}, "age_group": {"group": "45-64", "set": ["biliary"], "abstain": False}} for q in ("80", "90", "95")}}})
    spec = {"endpoints": {"any_readmission": {"probability_field": "p_any_readmission_calibrated", "threshold": .2}}, "coverages": [.8, .9, .95], "subgroups": ["sex", "age_group"]}
    base = v.metrics(rows, spec["endpoints"]["any_readmission"], "any_readmission")
    arrays = v._metrics_arrays([r["label"]["any_readmission"] for r in rows], [r["prediction"]["p_any_readmission_calibrated"] for r in rows], spec["endpoints"]["any_readmission"])
    assert arrays == base
    indexed = v._conformal_metrics_indexed(rows, list(range(len(rows))), "80", "sex")
    assert indexed == v._conformal_metrics(rows, "80", "sex")
    started = time.perf_counter(); result = v.evaluate(rows, spec, 1000); elapsed = time.perf_counter() - started
    assert result["subject_cluster_bootstrap"]["replicates"] == 1000
    assert result["subject_cluster_bootstrap"]["unit"] == "subject_id"
    assert elapsed < 20.0, f"indexed 1000-replicate synthetic benchmark exceeded bound: {elapsed:.3f}s"

def test_lower_bootstrap_rejected_without_test_environment_and_for_real_identity(tmp_path, monkeypatch):
    method, archive, bindings, _ = fixture(tmp_path); lock, out = tmp_path / "execution.json", tmp_path / "out"
    cli("--method-lock", method, "--freeze-execution-lock", "--bindings", bindings, "--execution-lock-out", lock)
    denied = cli("--method-lock", method, "--execution-lock", lock, "--archive", archive, "--output-dir", out, "--test-only-bootstrap-replicates", 3, test_only=False, check=False)
    assert denied.returncode and "AP_CLAIMS_TEST_ONLY" in denied.stderr
    monkeypatch.setenv("AP_CLAIMS_TEST_ONLY", "1")
    with pytest.raises(RuntimeError, match="registered real MIMIC archive"):
        v._bootstrap_replicates({"source": {"archive_sha256": v.REAL_MIMIC_ARCHIVE_SHA}}, 3)
