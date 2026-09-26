import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import freeze_pre2022_lock as freezer
from freeze_pre2022_lock import (
    ARTIFACT_ROLES, CODE_ROLES, ETL_DEPENDENCY_ROLES, GATE_ROLES, TRANSFORMER_ROSTER,
    create_lock, identity, sha256, validate_gate,
)


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def sidecar(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def valid_gates() -> dict:
    return {
        "analysis_lock": {"status": "LOCKED_PRE_2022", "sealed_test_year": 2022, "conformal_only_partition": "2021B"},
        "baseline_lock": {"status": "PASS_BASELINES_FROZEN_PRE_2021B_PRE_2022", "2021B_outcomes_accessed": False, "year_2022_accessed": False},
        "model_selection_lock": {"status": "LOCKED_ON_2021A_PRE_2021B_PRE_2022", "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False},
        "ablation_registry": {"status": "PASS_PRE_2021B_PRE_2022", "2021B_accessed": False, "year_2022_accessed": False, "operating_point": {}},
        "operating_point_manifest": {"status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A", "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False},
        "stage6_registry": {"status": "PASS_2021B_CONFORMAL_LOCKED_PRE_2022", "partition": "2021B only", "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False},
    }


def valid_bundle(tmp_path: Path):
    inputs = {role: write_json(tmp_path / "gates" / f"{role}.json", value) for role, value in valid_gates().items()}
    runs = {}
    for run in TRANSFORMER_ROSTER:
        directory = tmp_path / "operating" / run; directory.mkdir(parents=True)
        manifest = write_json(directory / "transformer_operating_point_manifest.json", valid_gates()["operating_point_manifest"])
        runs[run] = {"operating_point_directory": str(directory.resolve()), "operating_point_manifest": identity(manifest)}
    registry = write_json(tmp_path / "all_operating_points.json", {"status": "PASS_ALL_TRANSFORMER_CALIBRATIONS_AND_THRESHOLDS_LOCKED_ON_2021A", "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False, "exact_roster": list(TRANSFORMER_ROSTER), "runs": runs})
    drift = write_json(tmp_path / "drift.json", {"status": "FROZEN_2022_DRIFT_REFERENCE", "year_2022_accessed": False, "model_or_threshold_selection_on_2021B": False, "source_partitions": {"selection_partition": "2021A", "conformal_partition": "2021B only"}}); sidecar(drift)
    evaluation = write_json(tmp_path / "evaluation.json", {"status": "FROZEN_2022_EVALUATION_SPEC", "sealed_test_year": 2022, "2021B_model_or_threshold_selection": False, "year_2022_accessed_before_unlock": False, "transformer_roster": list(TRANSFORMER_ROSTER)}); sidecar(evaluation)
    dependencies = {}
    for role in ETL_DEPENDENCY_ROLES:
        target = tmp_path / "dependencies" / f"{role}.bin"; target.parent.mkdir(exist_ok=True); target.write_text(role, encoding="utf-8"); dependencies[role] = str(target.resolve())
    etl = write_json(tmp_path / "etl.json", {"status": "FROZEN_2022_LOCAL_ETL_SPEC", "sealed_test_year": 2022, "test_partition": "test", "data_dependent_adaptation": False, "frozen_dependencies": dependencies}); sidecar(etl)
    derivative = write_json(tmp_path / "derivative.json", {"status": "FROZEN_2022_DERIVATIVE_SPEC", "sealed_test_year": 2022, "test_partition": "test", "data_dependent_adaptation": False}); sidecar(derivative)
    pra_dir = tmp_path / "pra"; pra_dir.mkdir(); mapping = pra_dir / "mapping.xlsx"; mapping.write_text("map", encoding="utf-8")
    ccs = pra_dir / "annual_ccs_mapping_2022.parquet"; codes = pra_dir / "annual_algorithm_code_sets_2022.parquet"; ccs.write_text("ccs", encoding="utf-8"); codes.write_text("codes", encoding="utf-8")
    pra = write_json(pra_dir / "planned_readmission_2022_lock.json", {"status": "PASS_2022_PRA_LOCKED_PRE_TEST_ACCESS", "calendar_year": 2022, "nrd_2022_accessed": False, "sources": {"yale_modified_ccs_2024_mapping": {**identity(mapping), "path": str(mapping.resolve())}}, "artifacts": {"ccs_mapping": identity(ccs), "algorithm_code_sets": identity(codes)}}); sidecar(pra)
    artifacts = {"all_transformer_registry": registry, "drift_reference": drift, "evaluation_spec": evaluation, "etl_spec": etl, "derivative_spec": derivative, "pra_lock": pra}
    code = {}
    for role in CODE_ROLES:
        target = tmp_path / "code" / f"{role}.py"; target.parent.mkdir(exist_ok=True); target.write_text(f"# {role}\n", encoding="utf-8"); code[role] = target
    assert set(inputs) == set(GATE_ROLES) and set(artifacts) == set(ARTIFACT_ROLES)
    return inputs, artifacts, code


def test_every_gate_rejects_a_2022_access_flag():
    values = valid_gates()
    for name, value in values.items(): validate_gate(name, value)
    values["stage6_registry"]["year_2022_accessed"] = True
    with pytest.raises(RuntimeError, match="stage6_registry"):
        validate_gate("stage6_registry", values["stage6_registry"])


def test_lock_binds_fixed_roles_all_models_and_is_exclusive(tmp_path):
    inputs, artifacts, code = valid_bundle(tmp_path); output = tmp_path / "unlock.json"
    result = create_lock(inputs, artifacts, code, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
    assert set(saved["frozen_code_roles"]) == set(CODE_ROLES)
    assert str(artifacts["etl_spec"].resolve()) in saved["frozen_code"]
    assert output.with_suffix(".json.sha256").read_text(encoding="ascii").split()[0] == sha256(output)
    with pytest.raises(RuntimeError, match="already exists"):
        create_lock(inputs, artifacts, code, output)


def test_sidecar_tampering_incomplete_roster_and_duplicate_code_fail_closed(tmp_path):
    inputs, artifacts, code = valid_bundle(tmp_path)
    artifacts["drift_reference"].with_suffix(".json.sha256").write_text("bad\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="sidecar mismatch"):
        create_lock(inputs, artifacts, code, tmp_path / "bad-sidecar.json")
    sidecar(artifacts["drift_reference"])
    registry = json.loads(artifacts["all_transformer_registry"].read_text(encoding="utf-8")); registry["exact_roster"] = list(TRANSFORMER_ROSTER[:-1]); write_json(artifacts["all_transformer_registry"], registry)
    with pytest.raises(RuntimeError, match="2021A-only"):
        create_lock(inputs, artifacts, code, tmp_path / "bad-roster.json")
    registry["exact_roster"] = list(TRANSFORMER_ROSTER); write_json(artifacts["all_transformer_registry"], registry)
    code["prediction"] = code["etl"]
    with pytest.raises(RuntimeError, match="unique paths"):
        create_lock(inputs, artifacts, code, tmp_path / "duplicate-code.json")


def test_sidecar_publish_failure_leaves_no_orphan_lock_and_preserves_existing_files(tmp_path, monkeypatch):
    inputs, artifacts, code = valid_bundle(tmp_path); output = tmp_path / "transaction.json"
    real_link = freezer.os.link

    def fail_only_sidecar(source, target, *args, **kwargs):
        if str(target).endswith(".sha256"):
            raise FileExistsError("simulated concurrent sidecar")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(freezer.os, "link", fail_only_sidecar)
    with pytest.raises(FileExistsError, match="simulated"):
        create_lock(inputs, artifacts, code, output)
    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()
    assert not list(tmp_path.glob(".transaction.json.*.tmp"))
    assert not list(tmp_path.glob(".transaction.json.sha256.*.tmp"))
