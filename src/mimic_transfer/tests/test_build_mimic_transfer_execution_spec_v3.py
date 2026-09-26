"""Contract tests for the pre-data v7 execution-binding spec builder v3."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / "build_mimic_transfer_execution_spec_v3.py"
spec = importlib.util.spec_from_file_location("mimic_binding_builder_test", MODULE)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)


def ident(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    method = write_json(tmp_path / "method.json", {
        "status": builder.METHOD_STATUS,
        "source": {
            # This is deliberately a non-existent path.  The builder must
            # validate method metadata but must never open this archive.
            "archive": r"C:\private\mimic\mimic-iv_3.1.0.zip",
            "archive_bytes": 123,
            "archive_sha256": "a" * 64,
            "required_members": ["a", "b", "c", "d"],
            "required_member_identities": {x: {"bytes": 1, "sha256": "b" * 64} for x in ("a", "b", "c", "d")},
        },
        "planned_readmission_under_shifted_dates": {"annual_rule_sets": list(builder.YEARS)},
        "prediction_anchor_and_history": {"calendar_year_embedding": "DISABLED"},
    })

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    checkpoint = artifact_dir / "best_checkpoint.pt"
    checkpoint.write_bytes(b"binary checkpoint fixture")
    finetune = write_json(artifact_dir / "finetune_manifest.json", {
        "status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022",
        "prediction_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "ablations": {builder.MODEL_ID: True},
        "checkpoint": ident(checkpoint),
        "model_config": {"use_year_version": False},
        "static_preprocessor": {"numeric_columns": ["AGE"], "categorical_columns": ["FEMALE"]},
    })
    calibrators = artifact_dir / "transformer_binary_calibrators.joblib"
    calibrators.write_bytes(b"frozen calibrator bytes")
    thresholds = write_json(artifact_dir / "transformer_operating_thresholds_2021A.json", {
        "any_readmission": {
            "probability_column": "p_any_readmission_calibrated",
            "threshold": 0.321,
            "rule": "fixed 20% capacity on 2021A; threshold transported unchanged",
        },
        "ap_specific_readmission": {
            "probability_column": "p_ap_specific_readmission_calibrated",
            "threshold": 0.123,
            "rule": "fixed 20% capacity on 2021A; threshold transported unchanged",
        },
    })
    operating = write_json(artifact_dir / "transformer_operating_point_manifest.json", {
        "status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "source": {"manifest_sha256": ident(finetune)["sha256"]},
        "artifacts": {
            "transformer_binary_calibrators.joblib": ident(calibrators),
            "transformer_operating_thresholds_2021A.json": ident(thresholds),
        },
    })
    # These binary/JSON fixtures are only identity-bound here.  v7 is the
    # component that later loads and semantically validates them.
    (artifact_dir / "hierarchy_arrays.npz").write_bytes(b"hierarchy arrays")
    write_json(artifact_dir / "hierarchy_vocabularies.json", {"fixture": True})
    write_json(artifact_dir / "diagnosis_vocabulary_state.json", {"token_to_id": {"K859": 4}})
    write_json(artifact_dir / "procedure_vocabulary_state.json", {"token_to_id": {"0F": 4}})

    runtime = MODULE.parent / "run_locked_mimic_transfer_v7.py"
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "def predict_common_variable_episode(feature, execution_lock):\n"
        "    return {'p_any_readmission_calibrated': 0.1, 'conformal': {}}\n",
        encoding="utf-8",
    )

    conformal_root = tmp_path / "conformal"
    conformal_dir = conformal_root / "stage6_2021B" / "conformal_2021B"
    conformal_dir.mkdir(parents=True)
    conformal_manifest = conformal_dir / "conformal_calibrator.json"
    conformal_manifest.write_text("{}", encoding="utf-8")
    conformal_sets = conformal_dir / "conformal_sets_2021B.parquet"
    conformal_sets.write_bytes(b"conformal sets")
    binding = {
        "status": "PASS_MIMIC_CONFORMAL_PROJECTION_PRE_DATA",
        "calibration_partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False,
        "year_2022_accessed": False,
        "allowed_mondrian_dimensions": list(builder.SUBGROUPS),
        "nominal_coverages": list(builder.COVERAGES),
        "global": {str(c): {"q": 0.2} for c in (80, 90, 95)},
        "mondrian": {
            "sex": {str(c): {"0": {"q": 0.2}, "1": {"q": 0.2}} for c in (80, 90, 95)},
            "age_group": {str(c): {"18-44": {"q": 0.2}, "45-64": {"q": 0.2}} for c in (80, 90, 95)},
        },
        # The absolute provenance path belongs to the remote producer.  The
        # builder must validate only its exact canonical suffix and local copy.
        "conformal_manifest": {**ident(conformal_manifest), "path": "/foreign/producer/stage6_2021B/conformal_2021B/conformal_calibrator.json"},
        "conformal_sets": {**ident(conformal_sets), "path": "/foreign/producer/stage6_2021B/conformal_2021B/conformal_sets_2021B.parquet"},
    }
    write_json(conformal_root / "mimic_conformal_binding.json", binding)
    (conformal_root / "common_variable_model_lock_2021A.json").write_text("{}", encoding="utf-8")
    (conformal_root / "common_variable_2021b_conformal_registry.json").write_text("{}", encoding="utf-8")

    repo = MODULE.parents[2]
    paths = {
        "method": method,
        "runtime": runtime,
        "adapter": adapter,
        "artifact_dir": artifact_dir,
        "conformal_root": conformal_root,
        "stage2": repo / "outputs/stage2_lock/planned_readmission/planned_readmission_lock.json",
        "pra2022": repo / "outputs/stage7_pre2022_lock/planned_readmission_2022/planned_readmission_2022_lock.json",
        "labels": repo / "outputs/stage2_lock/ontology/icd_ccsr_labels.csv",
    }
    # The production builder must reject any non-registered artifact.  A
    # miniature fixture replaces only that registration table so this test
    # can exercise the same identity gate without copying the 457 MB model.
    monkeypatch.setattr(builder, "EXPECTED_METHOD_LOCK_SHA256", ident(method)["sha256"])
    monkeypatch.setattr(builder, "EXPECTED_RUNTIME_SHA256", ident(runtime)["sha256"])
    monkeypatch.setattr(builder, "EXPECTED_ADAPTER_SHA256", ident(adapter)["sha256"])
    monkeypatch.setattr(builder, "EXPECTED_ONTOLOGY_LABELS_SHA256", ident(paths["labels"])["sha256"])
    monkeypatch.setattr(builder, "EXPECTED_ARTIFACT_SHA256", {
        role: ident(paths["artifact_dir"] / filename)["sha256"]
        for role, filename in builder.ARTIFACT_FILES.items()
    })
    return paths


def build_args(paths: dict[str, Path], output: Path) -> list[object]:
    return [paths["method"], paths["runtime"], paths["adapter"], paths["artifact_dir"],
            paths["conformal_root"], paths["stage2"], paths["pra2022"], paths["labels"], output]


def test_success_binds_exact_v7_roster_and_reads_threshold_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    output = tmp_path / "binding_spec.json"
    result = builder.build_spec(*build_args(paths, output))
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert set(result) == {"method_lock", "runtime", "contract", "bindings", "inference_adapter", "evaluation"}
    assert set(persisted["bindings"]) == set(builder.ROLES)
    assert set(persisted["inference_adapter"]) >= {"model_id", "module", "function", *builder.ARTIFACT_ROLES, "conformal_binding"}
    assert persisted["bindings"]["checkpoint"]["bytes"] > 0
    assert persisted["evaluation"]["endpoints"]["any_readmission"]["probability_field"] == "p_any_readmission_calibrated"
    assert persisted["evaluation"]["endpoints"]["ap_specific_readmission"]["probability_field"] == "p_ap_specific_readmission_calibrated"
    assert persisted["evaluation"]["endpoints"]["any_readmission"]["threshold"] == 0.321
    assert persisted["evaluation"]["endpoints"]["ap_specific_readmission"]["threshold"] == 0.123
    assert persisted["evaluation"]["coverages"] == [0.8, 0.9, 0.95]
    assert persisted["evaluation"]["subgroups"] == ["sex", "age_group"]
    assert r"C:\private\mimic\mimic-iv_3.1.0.zip" not in output.read_text(encoding="utf-8")


def test_artifact_hash_tamper_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    finetune = paths["artifact_dir"] / "finetune_manifest.json"
    finetune.write_text(finetune.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="registered production identity mismatch: finetune_manifest"):
        builder.build_spec(*build_args(paths, tmp_path / "binding_spec.json"))


def test_conformal_roster_tamper_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    binding = paths["conformal_root"] / "mimic_conformal_binding.json"
    payload = json.loads(binding.read_text(encoding="utf-8"))
    del payload["mondrian"]["age_group"]
    binding.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="MIMIC conformal subgroup roster"):
        builder.build_spec(*build_args(paths, tmp_path / "binding_spec.json"))


def test_relocated_conformal_local_tamper_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    local_sets = paths["conformal_root"] / "stage6_2021B" / "conformal_2021B" / "conformal_sets_2021B.parquet"
    local_sets.write_bytes(local_sets.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="MIMIC conformal nested identity: conformal_sets"):
        builder.build_spec(*build_args(paths, tmp_path / "binding_spec.json"))


def test_relocated_conformal_wrong_provenance_suffix_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    binding = paths["conformal_root"] / "mimic_conformal_binding.json"
    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["conformal_manifest"]["path"] = "/foreign/producer/not-the-locked-file.json"
    binding.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="MIMIC conformal provenance suffix: conformal_manifest"):
        builder.build_spec(*build_args(paths, tmp_path / "binding_spec.json"))


def test_relocated_conformal_symlink_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    local_manifest = paths["conformal_root"] / "stage6_2021B" / "conformal_2021B" / "conformal_calibrator.json"
    outside = tmp_path / "outside_conformal_calibrator.json"
    outside.write_bytes(local_manifest.read_bytes())
    local_manifest.unlink()
    try:
        local_manifest.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(RuntimeError, match="symlink input is not permitted|symlink escape"):
        builder.build_spec(*build_args(paths, tmp_path / "binding_spec.json"))


def test_output_is_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    output = tmp_path / "binding_spec.json"
    builder.build_spec(*build_args(paths, output))
    with pytest.raises(RuntimeError, match="immutable output exists"):
        builder.build_spec(*build_args(paths, output))


def test_symlink_escape_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    link = paths["artifact_dir"] / "best_checkpoint.pt"
    link.unlink()
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(RuntimeError, match="symlink input is not permitted|symlink escape"):
        builder.build_spec(*build_args(paths, tmp_path / "binding_spec.json"))


def test_cli_has_no_archive_parameter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    command = [sys.executable, str(MODULE), "--method-lock", str(paths["method"]), "--runtime", str(paths["runtime"]),
               "--adapter", str(paths["adapter"]), "--artifact-dir", str(paths["artifact_dir"]),
               "--conformal-output-root", str(paths["conformal_root"]), "--stage2-lock", str(paths["stage2"]),
               "--pra-2022-lock", str(paths["pra2022"]), "--ontology-labels", str(paths["labels"]),
               "--output-spec", str(tmp_path / "binding_spec.json"), "--archive", str(tmp_path / "must-not-open.zip")]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr
    assert not (tmp_path / "must-not-open.zip").exists()


 
def test_registered_v7_runtime_and_v2_adapter_pins() -> None:
    runtime = MODULE.parent / "run_locked_mimic_transfer_v7.py"
    adapter = MODULE.parent / "common_variable_inference_adapter_v2.py"
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == builder.EXPECTED_RUNTIME_SHA256 == "e08d98f3143a96df70ba5a701b0e4240712ae73af342e7b812ea0081acb45c5c"
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == builder.EXPECTED_ADAPTER_SHA256 == "ce8d97545f5f214f394ddaff086e1ca0ae90db67f6b498e7e49afd9e75868333"


def test_adapter_v2_identity_tamper_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    registered_adapter = MODULE.parent / "common_variable_inference_adapter_v2.py"
    tampered_adapter = tmp_path / "common_variable_inference_adapter_v2.py"
    tampered_adapter.write_bytes(registered_adapter.read_bytes() + b"\n# tampered")
    paths["adapter"] = tampered_adapter
    monkeypatch.setattr(builder, "EXPECTED_ADAPTER_SHA256", "ce8d97545f5f214f394ddaff086e1ca0ae90db67f6b498e7e49afd9e75868333")
    with pytest.raises(RuntimeError, match="registered production identity mismatch: inference adapter"):
        builder.build_spec(*build_args(paths, tmp_path / "binding_spec.json"))
