#!/usr/bin/env python3
"""Fail-closed MIMIC inference for the frozen ``common_variable_only`` model.

The public entry point is :func:`predict_common_variable_episode`.  Its second
argument is the immutable execution lock and must contain an
``inference_adapter`` mapping with the following identity-bound artifacts:
``checkpoint``, ``finetune_manifest``, ``operating_manifest``, ``calibrators``,
``thresholds``, ``hierarchy_arrays``, ``hierarchy_vocabularies``, ``dx_vocabulary``,
``pr_vocabulary``, and ``conformal_binding``.  Every artifact identity is a
``{path, bytes, sha256}`` mapping.  This deliberately makes a partially bound
runtime unusable rather than guessing locations or contracts.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch

STAGE5 = Path(__file__).resolve().parents[1] / "stage5"
if str(STAGE5) not in sys.path:
    sys.path.insert(0, str(STAGE5))

from claims_transformer import ClaimsTransformer, ClaimsTransformerConfig  # noqa: E402
from finetune_ap_transformer_v2 import (  # noqa: E402
    COMMON_VARIABLE_CATEGORICAL_COLUMNS, COMMON_VARIABLE_NUMERIC_COLUMNS,
    StaticPreprocessor, timing_bucket,
)
from freeze_transformer_calibration import apply_calibrator  # noqa: E402


MODEL_ID = "common_variable_only"
EXPECTED_COMMON_VARIABLE_CHECKPOINT_SHA256 = "e4680d697f303574ab946e6f7e6fcdc32409a46ca8e603d1f264ed8aa5c62efe"
EXPECTED_COMMON_VARIABLE_FINETUNE_MANIFEST_SHA256 = "13c5271a70df4761764e2a681d5e1cd0db1d66d82cc4e544268c72dc39e52018"
LEAVES = ("none", "ap", "biliary", "sepsis_or_organ", "other")
ENDPOINTS = ("any_readmission", "ap_specific_readmission")
CONFORMAL_CANONICAL_SUFFIXES = {
    "conformal_manifest": "stage6_2021B/conformal_2021B/conformal_calibrator.json",
    "conformal_sets": "stage6_2021B/conformal_2021B/conformal_sets_2021B.parquet",
}
REQUIRED_ARTIFACTS = (
    "checkpoint", "finetune_manifest", "operating_manifest", "calibrators", "thresholds",
    "hierarchy_arrays", "hierarchy_vocabularies", "dx_vocabulary", "pr_vocabulary",
    "conformal_binding",
)
REQUIRED_FEATURES = (
    "AGE", "FEMALE", "LOS", "I10_NDX", "I10_NPR", "dx_tokens", "pr_tokens", "prday",
    "prior_dx_tokens_180d", "prior_pr_tokens_180d", "oov_audit",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _identity(value: object, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Missing {label} identity")
    raw = value.get("path", value.get("file"))
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"Missing {label} identity path")
    path = Path(raw).resolve()
    if not path.is_file() or not isinstance(value.get("sha256"), str):
        raise RuntimeError(f"Invalid {label} identity")
    if int(value.get("bytes", -1)) != path.stat().st_size or value["sha256"] != sha256(path):
        raise RuntimeError(f"{label} identity mismatch")
    return path


class _CurrentPlatformPathUnpickler(pickle.Unpickler):
    """Deserialize Linux/Windows ``pathlib`` payloads without global mutation.

    A Stage 5 checkpoint can contain a bookkeeping ``PosixPath`` serialized on
    Linux.  Python intentionally refuses to instantiate that concrete class on
    Windows.  Only checkpoint unpickling uses this class; raw checkpoint bytes
    are hash-checked before this point and every model/config/state validation
    remains unchanged afterwards.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
            return Path
        return super().find_class(module, name)


class _CurrentPlatformPathPickleModule:
    """Narrow ``torch.load`` pickle-module facade for cross-platform Paths."""

    Unpickler = _CurrentPlatformPathUnpickler
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    dump = staticmethod(pickle.dump)

    @staticmethod
    def load(handle: Any, **kwargs: Any) -> Any:
        return _CurrentPlatformPathUnpickler(handle, **kwargs).load()


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load only a hash-validated checkpoint with isolated path compatibility."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False,
                            pickle_module=_CurrentPlatformPathPickleModule)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint payload must be a mapping")
    return checkpoint


def _adapter_lock(execution_lock: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(execution_lock, Mapping):
        raise RuntimeError("Execution lock must be a mapping")
    adapter = execution_lock.get("inference_adapter")
    if not isinstance(adapter, Mapping) or adapter.get("model_id") != MODEL_ID:
        raise RuntimeError("Execution lock is not bound to common_variable_only adapter")
    if adapter.get("mimic_fit") is not False or adapter.get("mimic_recalibration") is not False or adapter.get("mimic_threshold_optimization") is not False or adapter.get("vocabulary_growth") is not False:
        raise RuntimeError("Execution lock does not prohibit MIMIC fitting/tuning/vocabulary growth")
    if set(REQUIRED_ARTIFACTS) - set(adapter):
        raise RuntimeError("Execution lock lacks required adapter artifacts")
    return adapter


def _load_hierarchy(arrays_path: Path, vocab_path: Path, dx_path: Path, pr_path: Path) -> tuple[int, torch.Tensor, torch.Tensor]:
    try:
        arrays = np.load(arrays_path, allow_pickle=False)
        vocab = _json(vocab_path, "hierarchy vocabularies")
        dx = _json(dx_path, "diagnosis vocabulary").get("token_to_id")
        pr = _json(pr_path, "procedure vocabulary").get("token_to_id")
    except (OSError, ValueError) as exc:
        raise RuntimeError("Unreadable frozen hierarchy/vocabulary artifact") from exc
    if not isinstance(dx, Mapping) or not isinstance(pr, Mapping) or not isinstance(vocab, Mapping):
        raise RuntimeError("Frozen hierarchy/vocabulary contract is malformed")
    required = {"dx_category", "pr_category", "dx_domain", "pr_domain"}
    if not required.issubset(set(arrays.files)):
        raise RuntimeError("Frozen hierarchy arrays are incomplete")
    dx_size, pr_size = len(dx), len(pr)
    if dx_size < 4 or pr_size < 4 or len(arrays["dx_category"]) != dx_size or len(arrays["pr_category"]) != pr_size:
        raise RuntimeError("Frozen hierarchy and vocabulary cardinalities disagree")
    for key in ("diagnosis_category", "procedure_category", "diagnosis_domain", "procedure_domain"):
        # Production hierarchy vocabularies are token-to-id JSON objects, not
        # arrays; cardinality, rather than JSON container type, is the frozen
        # contract used by Stage 5's load_unified_hierarchy.
        if not isinstance(vocab.get(key), Mapping) or not vocab[key]:
            raise RuntimeError("Frozen hierarchy vocabulary is incomplete")
    dx_cat_n, dx_dom_n = len(vocab["diagnosis_category"]), len(vocab["diagnosis_domain"])
    pr_cat = np.where(np.asarray(arrays["pr_category"]) > 0, np.asarray(arrays["pr_category"]) + dx_cat_n - 1, 0)
    pr_dom = np.where(np.asarray(arrays["pr_domain"]) > 0, np.asarray(arrays["pr_domain"]) + dx_dom_n - 1, 0)
    category = torch.from_numpy(np.concatenate([arrays["dx_category"], pr_cat]).astype(np.int64))
    domain = torch.from_numpy(np.concatenate([arrays["dx_domain"], pr_dom]).astype(np.int64))
    return dx_size, category, domain


def _canonical_conformal_file(binding_path: Path, recorded: object, suffix: str, label: str) -> Path:
    """Resolve a relocated Stage 6 file only under the local binding's parent.

    The frozen binding records its producer's provenance path, which can be an
    inaccessible foreign absolute path after a restricted local copy.  That
    provenance remains an immutable part of the identity contract: it must name
    the expected canonical suffix and retain the exact byte count and SHA-256.
    The actual file is always resolved from the local binding directory, with
    every symlink on that route rejected before it can redirect the artifact.
    """
    if not isinstance(recorded, Mapping) or not isinstance(recorded.get("path"), str):
        raise RuntimeError(f"MIMIC conformal nested identity: {label}")
    normalized = str(recorded["path"]).replace("\\", "/")
    if not normalized.endswith("/" + suffix):
        raise RuntimeError(f"MIMIC conformal provenance suffix: {label}")
    if binding_path.is_symlink():
        raise RuntimeError("MIMIC conformal symlink: binding")
    root = binding_path.parent
    target = root.joinpath(*suffix.split("/"))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:  # Defensive even though suffix is fixed above.
        raise RuntimeError(f"MIMIC conformal canonical escape: {label}") from exc
    cursor = root
    if cursor.is_symlink():
        raise RuntimeError("MIMIC conformal symlink: binding root")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"MIMIC conformal symlink: {label}")
    if not target.is_file():
        raise RuntimeError(f"MIMIC conformal nested identity: {label}")
    if not isinstance(recorded.get("bytes"), int) or not isinstance(recorded.get("sha256"), str):
        raise RuntimeError(f"MIMIC conformal nested identity: {label}")
    if target.stat().st_size != recorded["bytes"] or sha256(target) != recorded["sha256"]:
        raise RuntimeError(f"MIMIC conformal nested identity: {label}")
    return target.resolve()


def _validate_conformal(path: Path) -> dict[str, Any]:
    binding = _json(path, "MIMIC conformal binding")
    if (binding.get("status") != "PASS_MIMIC_CONFORMAL_PROJECTION_PRE_DATA"
            or binding.get("calibration_partition") != "2021B only"
            or binding.get("model_or_threshold_selection_on_2021B") is not False
            or binding.get("year_2022_accessed") is not False
            or binding.get("allowed_mondrian_dimensions") != ["sex", "age_group"]):
        raise RuntimeError("MIMIC conformal binding is not a sealed 2021B projection")
    if set(binding.get("mondrian", {})) != {"sex", "age_group"} or not isinstance(binding.get("global"), Mapping):
        raise RuntimeError("MIMIC conformal binding has unsupported subgroup dimensions")
    if sorted(float(x) for x in binding.get("nominal_coverages", [])) != [0.8, 0.9, 0.95]:
        raise RuntimeError("MIMIC conformal binding lacks 80/90/95 coverages")
    for name, suffix in CONFORMAL_CANONICAL_SUFFIXES.items():
        _canonical_conformal_file(path, binding.get(name), suffix, name)
    for source in (binding["global"], binding["mondrian"]["sex"], binding["mondrian"]["age_group"]):
        if not isinstance(source, Mapping):
            raise RuntimeError("Malformed conformal source")
        for coverage in ("80", "90", "95"):
            entry = source.get(coverage)
            if not isinstance(entry, Mapping):
                raise RuntimeError("Conformal q roster is incomplete")
            # Global has {q}; Mondrian has {group: {q, ...}}.
            if source is binding["global"] and not isinstance(entry.get("q"), (int, float)):
                raise RuntimeError("Global conformal q is absent")
    return binding


@dataclass
class _LoadedAdapter:
    model: ClaimsTransformer
    static: StaticPreprocessor
    dx_size: int
    max_tokens: int
    calibrators: dict[str, Any]
    thresholds: dict[str, float]
    conformal: dict[str, Any]


_CACHE: dict[str, _LoadedAdapter] = {}


def clear_model_cache() -> None:
    """Clear process-local loaded checkpoints (primarily for test/process teardown)."""
    _CACHE.clear()


def _load(execution_lock: Mapping[str, Any]) -> _LoadedAdapter:
    adapter = _adapter_lock(execution_lock)
    identity_paths = {name: _identity(adapter[name], name) for name in REQUIRED_ARTIFACTS}
    if (sha256(identity_paths["checkpoint"]) != EXPECTED_COMMON_VARIABLE_CHECKPOINT_SHA256
            or sha256(identity_paths["finetune_manifest"]) != EXPECTED_COMMON_VARIABLE_FINETUNE_MANIFEST_SHA256):
        raise RuntimeError("Execution lock does not bind the registered common-variable artifact identities")
    cache_key = "|".join(str(adapter[name]["sha256"]) for name in REQUIRED_ARTIFACTS)
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    finetune = _json(identity_paths["finetune_manifest"], "finetune manifest")
    if (finetune.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022"
            or finetune.get("prediction_partition") != "2021A"
            or finetune.get("2021B_accessed") is not False or finetune.get("year_2022_accessed") is not False
            or finetune.get("ablations", {}).get("common_variable_only") is not True):
        raise RuntimeError("Finetune manifest is not the sealed common-variable model")
    if finetune.get("checkpoint", {}).get("sha256") != sha256(identity_paths["checkpoint"]):
        raise RuntimeError("Checkpoint does not bind to finetune manifest")
    checkpoint = _load_checkpoint(identity_paths["checkpoint"])
    if not isinstance(checkpoint.get("model_config"), Mapping) or not isinstance(checkpoint.get("model_state"), Mapping):
        raise RuntimeError("Checkpoint has no complete model payload")
    if checkpoint.get("training_contract") != finetune.get("training_contract"):
        raise RuntimeError("Checkpoint training contract differs from finetune manifest")
    model_config = dict(checkpoint["model_config"])
    if model_config.get("use_year_version") is not False:
        raise RuntimeError("common_variable_only checkpoint must disable calendar-year embedding")
    static = StaticPreprocessor.from_dict(checkpoint.get("static_preprocessor", {}))
    if not static.numeric_columns or not static.categorical_columns or int(model_config.get("static_dim", -1)) != static.dimension:
        raise RuntimeError("Checkpoint static preprocessor/schema mismatch")
    if (tuple(static.numeric_columns) != COMMON_VARIABLE_NUMERIC_COLUMNS
            or tuple(static.categorical_columns) != COMMON_VARIABLE_CATEGORICAL_COLUMNS
            or finetune.get("model_config") != model_config
            or finetune.get("static_preprocessor") != static.to_dict()
            or finetune.get("static_feature_set") != {"numeric": list(static.numeric_columns), "categorical": list(static.categorical_columns), "dimension": static.dimension}):
        raise RuntimeError("Finetune manifest does not bind the frozen common-variable schema")
    dx_size, category, domain = _load_hierarchy(identity_paths["hierarchy_arrays"], identity_paths["hierarchy_vocabularies"], identity_paths["dx_vocabulary"], identity_paths["pr_vocabulary"])
    config = ClaimsTransformerConfig(**model_config)
    if config.unified_vocab_size != len(category) or config.category_vocab_size <= int(category.max()) or config.domain_vocab_size <= int(domain.max()):
        raise RuntimeError("Hierarchy arrays/model configuration shape mismatch")
    model = ClaimsTransformer(config, category, domain)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError("Checkpoint state_dict is incompatible with frozen model") from exc
    model.eval()
    operating = _json(identity_paths["operating_manifest"], "operating-point manifest")
    if (operating.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A"
            or operating.get("selection_partition") != "2021A" or operating.get("2021B_accessed") is not False
            or operating.get("year_2022_accessed") is not False
            or operating.get("source", {}).get("manifest_sha256") != sha256(identity_paths["finetune_manifest"])
            or operating.get("source", {}).get("sha256") != finetune.get("prediction", {}).get("sha256")):
        raise RuntimeError("Operating point is not bound to the sealed common-variable model")
    for name, path_key in (("transformer_binary_calibrators.joblib", "calibrators"), ("transformer_operating_thresholds_2021A.json", "thresholds")):
        declared = operating.get("artifacts", {}).get(name, {})
        if declared.get("sha256") != sha256(identity_paths[path_key]) or int(declared.get("bytes", -1)) != identity_paths[path_key].stat().st_size:
            raise RuntimeError("Operating-point artifact identity mismatch")
    calibrators = joblib.load(identity_paths["calibrators"])
    if not isinstance(calibrators, dict) or calibrators.get("version") != 1 or set(calibrators.get("endpoints", {})) != set(ENDPOINTS):
        raise RuntimeError("Frozen calibrator roster is malformed")
    thresholds_payload = _json(identity_paths["thresholds"], "operating thresholds")
    thresholds: dict[str, float] = {}
    for endpoint in ENDPOINTS:
        item = thresholds_payload.get(endpoint)
        if not isinstance(item, Mapping) or item.get("probability_column") != f"p_{endpoint}_calibrated" or item.get("rule") != "fixed 20% capacity on 2021A; threshold transported unchanged":
            raise RuntimeError("Frozen threshold contract is malformed")
        value = item.get("threshold")
        if not isinstance(value, (int, float)) or not 0 < float(value) < 1:
            raise RuntimeError("Frozen threshold is invalid")
        thresholds[endpoint] = float(value)
    raw_conformal_binding = adapter["conformal_binding"].get("path")
    if not isinstance(raw_conformal_binding, str) or not raw_conformal_binding:
        raise RuntimeError("Missing conformal binding identity path")
    conformal_binding_path = Path(raw_conformal_binding)
    if conformal_binding_path.is_symlink():
        raise RuntimeError("MIMIC conformal symlink: binding")
    if conformal_binding_path.resolve() != identity_paths["conformal_binding"]:
        raise RuntimeError("MIMIC conformal binding identity path mismatch")
    conformal = _validate_conformal(conformal_binding_path)
    result = _LoadedAdapter(model, static, dx_size, int(finetune["training_contract"]["max_tokens"]), calibrators, thresholds, conformal)
    _CACHE[cache_key] = result
    return result


def _tokens(value: object, label: str, maximum: int) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"{label} must be a token list")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)) or not 0 <= int(item) < maximum:
            raise RuntimeError(f"{label} contains out-of-vocabulary token")
        out.append(int(item))
    return out


def _example(feature: Mapping[str, Any], loaded: _LoadedAdapter) -> tuple[dict[str, torch.Tensor], float, str, str]:
    if not isinstance(feature, Mapping) or set(REQUIRED_FEATURES) - set(feature):
        raise RuntimeError("MIMIC feature payload is incomplete for the frozen model")
    if any(key in feature for key in ("year", "calendar_year", "year_index")):
        raise RuntimeError("Calendar year is prohibited from common-variable MIMIC inference")
    absent_static = set((*loaded.static.numeric_columns, *loaded.static.categorical_columns)) - set(feature)
    if absent_static:
        raise RuntimeError(f"MIMIC feature payload lacks frozen static fields: {sorted(absent_static)}")
    age = feature["AGE"]
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(float(age)) or not 18 <= float(age) <= 120:
        raise RuntimeError("AGE must be finite and within the locked adult range")
    female = feature["FEMALE"]
    if female not in (0, 1, 0.0, 1.0):
        raise RuntimeError("FEMALE must be frozen binary coding 0/1")
    audit = feature["oov_audit"]
    if not isinstance(audit, Mapping) or set(("diagnosis", "procedure", "total")) - set(audit):
        raise RuntimeError("MIMIC OOV audit is incomplete")
    if any(isinstance(audit[k], bool) or not isinstance(audit[k], (int, np.integer)) or int(audit[k]) < 0 for k in ("diagnosis", "procedure", "total")) or int(audit["total"]) != int(audit["diagnosis"]) + int(audit["procedure"]):
        raise RuntimeError("MIMIC OOV audit is invalid")
    current_dx = [x for x in _tokens(feature["dx_tokens"], "dx_tokens", loaded.dx_size) if x > 3]
    current_pr = [x for x in _tokens(feature["pr_tokens"], "pr_tokens", loaded.model.config.unified_vocab_size - loaded.dx_size) if x > 3]
    prior_dx = [x for x in _tokens(feature["prior_dx_tokens_180d"], "prior_dx_tokens_180d", loaded.dx_size) if x > 3]
    prior_pr = [x for x in _tokens(feature["prior_pr_tokens_180d"], "prior_pr_tokens_180d", loaded.model.config.unified_vocab_size - loaded.dx_size) if x > 3]
    prday = feature["prday"]
    if not isinstance(prday, (list, tuple)):
        raise RuntimeError("prday must be a list")
    current = [(x, 0, 0, 0) for x in current_dx]
    current.extend((x + loaded.dx_size, 1, timing_bucket(prday[i] if i < len(prday) else None, feature["LOS"]), 0) for i, x in enumerate(current_pr))
    prior = [(x, 0, 0, 1) for x in prior_dx] + [(x + loaded.dx_size, 1, 0, 1) for x in prior_pr]
    current_cap = min(len(current), loaded.max_tokens - (1 if prior else 0))
    chosen = current[:current_cap] + prior[:loaded.max_tokens - current_cap]
    if not chosen:
        chosen = [(3, 0, 0, 0)]
    static_frame = pd.DataFrame([{name: feature.get(name) for name in (*loaded.static.numeric_columns, *loaded.static.categorical_columns)}])
    static = loaded.static.transform(static_frame)
    n = len(chosen)
    inputs = {
        "token_ids": torch.tensor([[x[0] for x in chosen]], dtype=torch.long),
        "token_type": torch.tensor([[x[1] for x in chosen]], dtype=torch.long),
        "timing_bucket": torch.tensor([[x[2] for x in chosen]], dtype=torch.long),
        "encounter_index": torch.tensor([[x[3] for x in chosen]], dtype=torch.long),
        # This tensor is structurally required by ClaimsTransformer, but the sealed
        # config disables its embedding; no MIMIC calendar value is encoded here.
        "year_index": torch.zeros((1, n), dtype=torch.long),
        "token_mask": torch.ones((1, n), dtype=torch.bool),
        "static_features": torch.from_numpy(static),
        "admission_time_static": torch.zeros((1, 1), dtype=torch.float32),
    }
    sex = str(int(float(female)))
    years = float(age)
    age_group = "18-44" if years < 45 else "45-64" if years < 65 else "65-74" if years < 75 else "75+"
    return inputs, years, sex, age_group


def _q(source: Mapping[str, Any], coverage: str, group: str | None = None) -> float:
    entry = source.get(coverage)
    if group is not None:
        if isinstance(entry, Mapping) and isinstance(entry.get(group), Mapping):
            entry = entry[group]
        else:
            return math.nan
    if not isinstance(entry, Mapping) or not isinstance(entry.get("q"), (int, float)) or not 0 <= float(entry["q"]) <= 1:
        raise RuntimeError("Invalid frozen conformal q")
    return float(entry["q"])


def _set(probability: np.ndarray, q: float) -> list[str]:
    included = probability >= 1.0 - q
    if not included.any():
        included[int(probability.argmax())] = True
    return [leaf for leaf, yes in zip(LEAVES, included) if bool(yes)]


def predict_common_variable_episode(feature: Mapping[str, Any], execution_lock: Mapping[str, Any]) -> dict[str, Any]:
    """Run one frozen common-variable MIMIC episode without any fitting or tuning.

    Feature contract: required fields are ``AGE,FEMALE,LOS,I10_NDX,I10_NPR``,
    current/prior dx+pr token arrays, ``prday``, all static-preprocessor fields,
    and ``oov_audit={diagnosis,procedure,total}``.  The caller must already map
    unseen raw codes to token 3 and record that in the audit; raw vocabularies
    are never extended here.  ``year``, ``calendar_year``, and ``year_index``
    are deliberately rejected.
    """
    loaded = _load(execution_lock)
    inputs, _age, sex, age_group = _example(feature, loaded)
    with torch.no_grad():
        output = loaded.model(**inputs, return_mlm=False)
    leaves = torch.softmax(output["leaf_logits"], dim=1).cpu().numpy()[0]
    raw_any, raw_ap = float(leaves[1:].sum()), float(leaves[1])
    raw = {"any_readmission": raw_any, "ap_specific_readmission": raw_ap}
    calibrated: dict[str, float] = {}
    decisions: dict[str, bool] = {}
    for endpoint in ENDPOINTS:
        item = loaded.calibrators["endpoints"][endpoint]
        if not isinstance(item, Mapping) or not isinstance(item.get("method"), str):
            raise RuntimeError("Frozen calibrator endpoint is malformed")
        value = float(apply_calibrator(item["method"], item.get("model"), np.asarray([raw[endpoint]], dtype=np.float64))[0])
        calibrated[endpoint] = value
        decisions[endpoint] = bool(value >= loaded.thresholds[endpoint])
    conformal: dict[str, Any] = {}
    conflict = (int(leaves.argmax()) == 0 and decisions["any_readmission"]) or (int(leaves.argmax()) > 0 and not decisions["any_readmission"])
    for coverage in ("80", "90", "95"):
        global_set = _set(leaves.copy(), _q(loaded.conformal["global"], coverage))
        records: dict[str, dict[str, Any]] = {"global": {"set": global_set, "abstain": bool(len(global_set) != 1 or conflict)}}
        for dimension, group in (("sex", sex), ("age_group", age_group)):
            q = _q(loaded.conformal["mondrian"][dimension], coverage, group)
            if math.isnan(q):
                q = _q(loaded.conformal["global"], coverage)
            risk_set = _set(leaves.copy(), q)
            records[dimension] = {"group": group, "set": risk_set, "abstain": bool(len(risk_set) != 1 or conflict)}
        # Top-level values retain the v3 runtime's existing conformal contract.
        conformal[coverage] = {"set": global_set, "abstain": records["global"]["abstain"], **records}
    return {
        "p_any_readmission": raw_any, "p_leaf_ap": raw_ap,
        "p_any_readmission_calibrated": calibrated["any_readmission"],
        "p_ap_specific_readmission_calibrated": calibrated["ap_specific_readmission"],
        "decisions": decisions,
        "thresholds": dict(loaded.thresholds),
        "conformal": conformal,
        "model_id": MODEL_ID,
        "calendar_year_used": False,
    }


# The stable alias simplifies execution-lock adapters that conventionally name
# their callable ``predict``.
predict = predict_common_variable_episode

