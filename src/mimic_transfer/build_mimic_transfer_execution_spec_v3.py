#!/usr/bin/env python3
"""Build a v7 MIMIC-transfer binding specification from frozen artifacts.

This controller is deliberately a *pre-data* operation.  It only hashes and
checks the already-produced model, conformal, ontology, and planned-
readmission lock artifacts.  It has no ``--archive`` option and never opens a
MIMIC table or a patient-level prediction artifact.  The resulting JSON is a
 binding *specification*; ``run_locked_mimic_transfer_v7.py`` remains the only
program that can turn it into an immutable execution lock.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


MODEL_ID = "common_variable_only"
METHOD_STATUS = "LOCKED_METHOD_PRE_FORMAL_MIMIC_ANALYSIS_PENDING_MODEL_BINDINGS"
YEARS = (2018, 2019, 2020, 2021, 2022)
OLD_PRA_SHA256 = "e1e682fea14da07e5b9ac531be13859781b7f8ad277bfdc1d6e73509aca64fa1"
NEW_PRA_SHA256 = "8ad5391afb31c72aa54b41875cac0d458b05182fe02f3140ae464683b8855010"
EXPECTED_METHOD_LOCK_SHA256 = "69114be50eccfa37c2a560e4afd1e6a64d902a7f510ba04463400b83ed063a3c"
EXPECTED_RUNTIME_SHA256 = "e08d98f3143a96df70ba5a701b0e4240712ae73af342e7b812ea0081acb45c5c"
EXPECTED_ADAPTER_SHA256 = "ce8d97545f5f214f394ddaff086e1ca0ae90db67f6b498e7e49afd9e75868333"
EXPECTED_ONTOLOGY_LABELS_SHA256 = "ed645739cf01ade16042c532b25a6114e50e8dc530e0ceb3d205d8e7f1e1fc86"

ENDPOINTS = ("any_readmission", "ap_specific_readmission")
PROBABILITY_FIELDS = {
    "any_readmission": "p_any_readmission_calibrated",
    "ap_specific_readmission": "p_ap_specific_readmission_calibrated",
}
COVERAGES = (0.8, 0.9, 0.95)
SUBGROUPS = ("sex", "age_group")

# These names are the production artifact directory produced by the locked
# common-variable fine-tuning/calibration workflow.  The builder does not
# guess alternate files: an incomplete directory must fail closed.
ARTIFACT_FILES = {
    "checkpoint": "best_checkpoint.pt",
    "finetune_manifest": "finetune_manifest.json",
    "operating_manifest": "transformer_operating_point_manifest.json",
    "calibrators": "transformer_binary_calibrators.joblib",
    "thresholds": "transformer_operating_thresholds_2021A.json",
    "hierarchy_arrays": "hierarchy_arrays.npz",
    "hierarchy_vocabularies": "hierarchy_vocabularies.json",
    "dx_vocabulary": "diagnosis_vocabulary_state.json",
    "pr_vocabulary": "procedure_vocabulary_state.json",
}
ARTIFACT_ROLES = tuple(ARTIFACT_FILES)
EXPECTED_ARTIFACT_SHA256 = {
    "checkpoint": "e4680d697f303574ab946e6f7e6fcdc32409a46ca8e603d1f264ed8aa5c62efe",
    "finetune_manifest": "13c5271a70df4761764e2a681d5e1cd0db1d66d82cc4e544268c72dc39e52018",
    "operating_manifest": "d3ac1fc03e339a1ffadd0dd34cdbb77bd1eba56fae3d6ee96c68ee7994a62cf5",
    "calibrators": "0932551972498609f48c73496b9965a426612167bb487799a9531a9633299130",
    "thresholds": "69e6186e9349764e8b0010bde358de056ad424ada08037aad83a8960be405323",
    "hierarchy_arrays": "edb1f44b782fd4a9a70f7167007c14eccb7facbfe5706c3098ac995567778bc7",
    "hierarchy_vocabularies": "0708ecfeb75f004a3522195cc67b3eaa1f2ce078b76db14dd891fd540eedcaf5",
    "dx_vocabulary": "b890639fea65b18da16c5bd98801eeb5f3c09510fb098fa0b90ab2efdab5c74d",
    "pr_vocabulary": "bc478bd8729adc33915468a4452a87474590baa8e52770bad02d21065e07e9bb",
}
ROLES = ARTIFACT_ROLES + (
    "common_variable_model_lock",
    "conformal_registry",
    "adapter",
    "stage2_pra_lock",
    "pra_2022_lock",
    "ontology_labels",
    "conformal_binding",
)


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(path: Path, label: str) -> Path:
    """Resolve a path without allowing a symlink as the supplied object."""
    raw = Path(path)
    try:
        # Check every existing component.  Checking only the leaf would let a
        # path such as ``trusted_dir\link_outside\artifact`` escape the
        # caller's intended root while the leaf itself is not a symlink.
        current = raw
        while True:
            if current.is_symlink():
                raise RuntimeError(f"symlink input is not permitted: {label}")
            parent = current.parent
            if parent == current:
                break
            current = parent
        result = raw.resolve(strict=False)
    except OSError as exc:
        raise RuntimeError(f"cannot resolve {label}: {raw}") from exc
    return result


def _within(path: Path, root: Path, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"symlink escape: {label}") from exc
    return path


def identity(path: Path, label: str, *, root: Path | None = None) -> dict[str, Any]:
    """Return a canonical path/byte/hash identity after strict input checks."""
    resolved = _resolved(Path(path), label)
    if root is not None:
        resolved = _within(resolved, root, label)
    if not resolved.is_file():
        raise RuntimeError(f"missing input artifact: {label}: {resolved}")
    try:
        size = resolved.stat().st_size
        # A one-byte read proves that the first binary artifact is readable;
        # the checkpoint itself is intentionally not deserialized here.
        with resolved.open("rb") as handle:
            handle.read(1)
        digest = sha256(resolved)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"unreadable input artifact: {label}: {resolved}") from exc
    return {"path": str(resolved), "bytes": size, "sha256": digest}


def _registered(identity_value: Mapping[str, Any], expected_sha256: str, label: str) -> None:
    """Fail closed unless an input is the registered production artifact."""
    if identity_value.get("sha256") != expected_sha256:
        raise RuntimeError(f"registered production identity mismatch: {label}")


def _directory(path: Path, label: str) -> Path:
    resolved = _resolved(Path(path), label)
    if not resolved.is_dir():
        raise RuntimeError(f"missing input directory: {label}: {resolved}")
    return resolved


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON object: {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid JSON object: {label}")
    return value


def _false(value: Mapping[str, Any], key: str, label: str) -> None:
    if value.get(key) is not False:
        raise RuntimeError(f"{label}.{key} must be false")


def validate_method(path: Path) -> dict[str, Any]:
    """Validate method metadata without returning its archive source to output."""
    method = read_json(path, "method lock")
    source = method.get("source")
    if method.get("status") != METHOD_STATUS or not isinstance(source, Mapping):
        raise RuntimeError("method lock status")
    members = source.get("required_members")
    identities = source.get("required_member_identities")
    if (not isinstance(members, list) or len(members) != 4
            or not isinstance(identities, Mapping) or set(members) != set(identities)):
        raise RuntimeError("method lock member roster")
    if (isinstance(source.get("archive_bytes"), bool)
            or not isinstance(source.get("archive_bytes"), int)
            or not isinstance(source.get("archive_sha256"), str)):
        raise RuntimeError("method lock archive identity")
    if tuple(method.get("planned_readmission_under_shifted_dates", {}).get("annual_rule_sets", ())) != YEARS:
        raise RuntimeError("method lock five-year PRA contract")
    if method.get("prediction_anchor_and_history", {}).get("calendar_year_embedding") != "DISABLED":
        raise RuntimeError("method lock calendar-year contract")
    return method


def validate_runtime(path: Path) -> dict[str, Any]:
    """Hash and syntax-check the selected v7 runtime, without importing it."""
    result = identity(path, "runtime")
    try:
        source = Path(result["path"]).read_text(encoding="utf-8")
        ast.parse(source, filename=result["path"])
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RuntimeError("runtime is not a readable Python module") from exc
    return result


def _validate_code_module(path: Path, function: str) -> None:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RuntimeError("adapter is not a readable Python module") from exc
    definitions = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if function not in definitions:
        raise RuntimeError(f"adapter callable missing: {function}")


def _validate_model_artifacts(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Perform cheap manifest linkage checks; v7 prearchive repeats full checks."""
    finetune = read_json(Path(artifacts["finetune_manifest"]["path"]), "finetune manifest")
    if finetune.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022" or finetune.get("prediction_partition") != "2021A":
        raise RuntimeError("finetune manifest partition/status")
    _false(finetune, "2021B_accessed", "finetune manifest")
    _false(finetune, "year_2022_accessed", "finetune manifest")
    if finetune.get("ablations", {}).get(MODEL_ID) is not True:
        raise RuntimeError("finetune manifest common-variable model")
    checkpoint_declared = finetune.get("checkpoint")
    if not isinstance(checkpoint_declared, Mapping) or checkpoint_declared.get("sha256") != artifacts["checkpoint"]["sha256"]:
        raise RuntimeError("finetune/checkpoint identity mismatch")
    if finetune.get("model_config", {}).get("use_year_version") is not False:
        raise RuntimeError("finetune manifest calendar-year configuration")
    static = finetune.get("static_preprocessor")
    if not isinstance(static, Mapping) or not static.get("numeric_columns") or not static.get("categorical_columns"):
        raise RuntimeError("finetune manifest static preprocessor")

    operating = read_json(Path(artifacts["operating_manifest"]["path"]), "operating-point manifest")
    if operating.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A" or operating.get("selection_partition") != "2021A":
        raise RuntimeError("operating-point manifest partition/status")
    _false(operating, "2021B_accessed", "operating-point manifest")
    _false(operating, "year_2022_accessed", "operating-point manifest")
    if operating.get("source", {}).get("manifest_sha256") != artifacts["finetune_manifest"]["sha256"]:
        raise RuntimeError("operating/finetune identity mismatch")
    declared = operating.get("artifacts", {})
    for filename, role in (("transformer_binary_calibrators.joblib", "calibrators"), ("transformer_operating_thresholds_2021A.json", "thresholds")):
        item = declared.get(filename)
        if not isinstance(item, Mapping) or item.get("sha256") != artifacts[role]["sha256"]:
            raise RuntimeError(f"operating/{role} identity mismatch")
    return {"finetune": finetune, "operating": operating}


def _read_thresholds(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    payload = read_json(Path(artifact["path"]), "2021A operating thresholds")
    if set(payload) != set(ENDPOINTS):
        raise RuntimeError("2021A threshold endpoint roster")
    endpoint_specs: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        item = payload.get(endpoint)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"2021A threshold entry: {endpoint}")
        field = item.get("probability_column")
        if field != PROBABILITY_FIELDS[endpoint]:
            raise RuntimeError(f"2021A probability field: {endpoint}")
        value = item.get("threshold")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 < float(value) < 1:
            raise RuntimeError(f"2021A threshold value: {endpoint}")
        if item.get("rule") != "fixed 20% capacity on 2021A; threshold transported unchanged":
            raise RuntimeError(f"2021A threshold rule: {endpoint}")
        # Only the fields needed by v7 are copied.  No performance rows or
        # calibration summaries from the production JSON enter the spec.
        endpoint_specs[endpoint] = {
            "probability_field": field,
            "threshold": float(value),
            "source": artifact,
            "partition": "2021A",
        }
    return endpoint_specs


CONFORMAL_CANONICAL_SUFFIXES = {
    "conformal_manifest": "stage6_2021B/conformal_2021B/conformal_calibrator.json",
    "conformal_sets": "stage6_2021B/conformal_2021B/conformal_sets_2021B.parquet",
}


def _conformal_paths(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_lock = identity(root / "common_variable_model_lock_2021A.json", "common-variable model lock", root=root)
    registry = identity(root / "common_variable_2021b_conformal_registry.json", "conformal registry", root=root)
    binding_identity = identity(root / "mimic_conformal_binding.json", "MIMIC conformal binding", root=root)
    manifest = identity(root.joinpath(*CONFORMAL_CANONICAL_SUFFIXES["conformal_manifest"].split("/")), "conformal manifest", root=root)
    sets = identity(root.joinpath(*CONFORMAL_CANONICAL_SUFFIXES["conformal_sets"].split("/")), "conformal sets", root=root)
    return model_lock, registry, binding_identity, manifest, sets


def _validate_conformal(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_lock, registry, binding_identity, expected_manifest, expected_sets = _conformal_paths(root)
    binding = read_json(Path(binding_identity["path"]), "MIMIC conformal binding")
    if (binding.get("status") != "PASS_MIMIC_CONFORMAL_PROJECTION_PRE_DATA"
            or binding.get("calibration_partition") != "2021B only"
            or binding.get("model_or_threshold_selection_on_2021B") is not False
            or binding.get("year_2022_accessed") is not False
            or binding.get("allowed_mondrian_dimensions") != list(SUBGROUPS)):
        raise RuntimeError("MIMIC conformal binding partition/access seal")
    if sorted(float(x) for x in binding.get("nominal_coverages", ())) != list(COVERAGES):
        raise RuntimeError("MIMIC conformal coverage roster")
    mondrian = binding.get("mondrian")
    global_q = binding.get("global")
    if not isinstance(global_q, Mapping) or not isinstance(mondrian, Mapping) or set(mondrian) != set(SUBGROUPS):
        raise RuntimeError("MIMIC conformal subgroup roster")
    for coverage in ("80", "90", "95"):
        entry = global_q.get(coverage)
        if not isinstance(entry, Mapping) or isinstance(entry.get("q"), bool) or not isinstance(entry.get("q"), (int, float)) or not math.isfinite(float(entry["q"])) or not 0 <= float(entry["q"]) <= 1:
            raise RuntimeError("MIMIC global conformal q roster")
        for dimension in SUBGROUPS:
            groups = mondrian[dimension].get(coverage) if isinstance(mondrian[dimension], Mapping) else None
            if not isinstance(groups, Mapping) or not groups:
                raise RuntimeError(f"MIMIC {dimension} conformal q roster")
            for group, item in groups.items():
                if not isinstance(group, str) or not group or not isinstance(item, Mapping):
                    raise RuntimeError(f"MIMIC {dimension} conformal group")
                q = item.get("q")
                if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(float(q)) or not 0 <= float(q) <= 1:
                    raise RuntimeError(f"MIMIC {dimension} conformal q")
    serial = stable(binding).lower()
    if any(term in serial for term in ("payer", "insurance", "zip_income", "zip-income", "hospital_structure")):
        raise RuntimeError("prohibited conformal dimension")
    # The frozen record retains immutable remote provenance.  Local execution
    # resolves only the canonical Stage 6 copies beneath the supplied root.
    # A foreign path need not exist, but its suffix and identity must still
    # exactly describe the local canonical file.
    for key, expected in (("conformal_manifest", expected_manifest), ("conformal_sets", expected_sets)):
        nested = binding.get(key)
        suffix = CONFORMAL_CANONICAL_SUFFIXES[key]
        recorded_path = nested.get("path") if isinstance(nested, Mapping) else None
        normalized = recorded_path.replace("\\", "/") if isinstance(recorded_path, str) else ""
        if not normalized.endswith("/" + suffix):
            raise RuntimeError(f"MIMIC conformal provenance suffix: {key}")
        if (not isinstance(nested.get("bytes"), int)
                or not isinstance(nested.get("sha256"), str)
                or nested["bytes"] != expected["bytes"]
                or nested["sha256"] != expected["sha256"]):
            raise RuntimeError(f"MIMIC conformal nested identity: {key}")
    return model_lock, registry, binding_identity


def _validate_pra(path: Path, label: str, expected_sha: str, *, year: int, sidecars: tuple[str, ...]) -> dict[str, Any]:
    lock = identity(path, label)
    if lock["sha256"] != expected_sha:
        raise RuntimeError(f"{label} hash does not match frozen production lock")
    payload = read_json(Path(lock["path"]), label)
    if year == 2022:
        if payload.get("status") != "PASS_2022_PRA_LOCKED_PRE_TEST_ACCESS" or payload.get("calendar_year") != 2022 or payload.get("nrd_2022_accessed") is not False:
            raise RuntimeError("2022 PRA lock semantics")
    elif payload.get("status") != "PASS" or tuple(payload.get("years_locked", ())) != YEARS[:4]:
        raise RuntimeError("2018-2021 PRA lock semantics")
    for filename in sidecars:
        identity(Path(lock["path"]).parent / filename, f"{label} sidecar")
    return lock


def _immutable_write(path: Path, content: bytes) -> None:
    raw = Path(path)
    _resolved(raw, "output spec")
    if raw.is_symlink() or raw.exists():
        raise RuntimeError(f"immutable output exists: {raw}")
    destination = raw.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"immutable output exists: {destination}")
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RuntimeError(f"immutable output exists: {destination}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_spec(method_lock: Path, runtime: Path, adapter: Path, artifact_dir: Path,
               conformal_output_root: Path, stage2_lock: Path, pra_2022_lock: Path,
               ontology_labels: Path, output_spec: Path) -> dict[str, Any]:
    """Build and atomically publish one v7 binding spec without data access."""
    method_identity = identity(Path(method_lock), "method lock")
    _registered(method_identity, EXPECTED_METHOD_LOCK_SHA256, "method lock")
    method = validate_method(Path(method_identity["path"]))
    runtime_identity = validate_runtime(Path(runtime))
    _registered(runtime_identity, EXPECTED_RUNTIME_SHA256, "v7 runtime")

    artifact_root = _directory(Path(artifact_dir), "artifact-dir")
    artifacts = {
        role: identity(artifact_root / filename, role, root=artifact_root)
        for role, filename in ARTIFACT_FILES.items()
    }
    for role, artifact in artifacts.items():
        _registered(artifact, EXPECTED_ARTIFACT_SHA256[role], role)
    _validate_model_artifacts(artifacts)
    thresholds = _read_thresholds(artifacts["thresholds"])

    adapter_path = _resolved(Path(adapter), "adapter")
    adapter_identity = identity(adapter_path, "adapter")
    _registered(adapter_identity, EXPECTED_ADAPTER_SHA256, "inference adapter")
    _validate_code_module(adapter_path, "predict_common_variable_episode")

    conformal_root = _directory(Path(conformal_output_root), "conformal-output-root")
    model_lock, registry, conformal_binding = _validate_conformal(conformal_root)
    old_pra = _validate_pra(Path(stage2_lock), "stage2 PRA lock", OLD_PRA_SHA256, year=2021, sidecars=("annual_algorithm_code_sets.parquet", "annual_ccs_mapping.parquet"))
    new_pra = _validate_pra(Path(pra_2022_lock), "2022 PRA lock", NEW_PRA_SHA256, year=2022, sidecars=("annual_algorithm_code_sets_2022.parquet", "annual_ccs_mapping_2022.parquet"))
    labels = identity(Path(ontology_labels), "ontology labels")
    _registered(labels, EXPECTED_ONTOLOGY_LABELS_SHA256, "ontology labels")
    expected_label_sha = method.get("upstream_method_bindings", {}).get("ontology_icd_ccsr_labels_sha256")
    if isinstance(expected_label_sha, str) and labels["sha256"] != expected_label_sha:
        raise RuntimeError("ontology labels do not match method lock")

    # The explicit v7 role roster is intentionally duplicated here so a
    # partial binding cannot silently reach the v7 freezer.
    bindings: dict[str, Any] = {
        **artifacts,
        "common_variable_model_lock": model_lock,
        "conformal_registry": registry,
        "adapter": adapter_identity,
        "stage2_pra_lock": old_pra,
        "pra_2022_lock": new_pra,
        "ontology_labels": labels,
        "conformal_binding": conformal_binding,
    }
    if set(bindings) != set(ROLES):
        raise RuntimeError("incomplete v7 binding role roster")
    inference_adapter = {
        "model_id": MODEL_ID,
        "module": adapter_path.stem,
        "function": "predict_common_variable_episode",
        "mimic_fit": False,
        "mimic_recalibration": False,
        "mimic_threshold_optimization": False,
        "vocabulary_growth": False,
        **artifacts,
        "conformal_binding": conformal_binding,
    }
    contract = {
        "projection": MODEL_ID,
        "calibration_partition": "2021A",
        "conformal_partition": "2021B",
        "mimic_fit": False,
        "mimic_recalibration": False,
        "mimic_threshold_optimization": False,
        "vocabulary_growth": False,
    }
    evaluation = {
        "endpoints": thresholds,
        "coverages": list(COVERAGES),
        "subgroups": list(SUBGROUPS),
        "uncertainty": {"bootstrap_unit": "subject_id", "replicates": 1000},
    }
    spec = {
        # v7's freezer consumes the four fields below.  The runtime identity is
        # retained as audit metadata so a spec cannot be mistaken for another
        # runtime; v7 still rechecks its own source at execution time.
        "method_lock": method_identity,
        "runtime": runtime_identity,
        "contract": contract,
        "bindings": bindings,
        "inference_adapter": inference_adapter,
        "evaluation": evaluation,
    }
    serialized = stable(spec)
    archive_hint = str(method.get("source", {}).get("archive", ""))
    if archive_hint and archive_hint in serialized:
        raise RuntimeError("MIMIC archive path leaked into binding spec")
    _immutable_write(Path(output_spec), serialized.encode("utf-8"))
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-lock", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--conformal-output-root", type=Path, required=True)
    parser.add_argument("--stage2-lock", type=Path, required=True)
    parser.add_argument("--pra-2022-lock", type=Path, required=True)
    parser.add_argument("--ontology-labels", type=Path, required=True)
    parser.add_argument("--output-spec", type=Path, required=True)
    args = parser.parse_args()
    print(stable(build_spec(args.method_lock, args.runtime, args.adapter, args.artifact_dir,
                             args.conformal_output_root, args.stage2_lock, args.pra_2022_lock,
                             args.ontology_labels, args.output_spec)))


if __name__ == "__main__":
    main()
