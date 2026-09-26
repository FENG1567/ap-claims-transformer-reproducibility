#!/usr/bin/env python3
"""Self-contained, fail-closed restricted MIMIC-IV transfer runtime (v7).

The program has exactly four public CLI modes: method validation, immutable
execution-lock freezing, bound validation, and a formal run.  All lock and
scientific gates run before a ZIP is opened.  In particular a binary torch
checkpoint is identity-checked only; JSON parsing is reserved for manifests.
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, importlib.util, io, json, math, os, random, shutil, sys, tempfile, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
try:
    import numpy as _np
except Exception:  # pragma: no cover - constrained runtimes use scalar fallback
    _np = None

METHOD_STATUS = "LOCKED_METHOD_PRE_FORMAL_MIMIC_ANALYSIS_PENDING_MODEL_BINDINGS"
EXECUTION_STATUS = "FULLY_BOUND_PRE_DATA_EXECUTION_LOCK_V7"
MODEL_ID = "common_variable_only"
MODEL_LOCK_STATUS = "LOCKED_ON_2021A_PRE_2021B_PRE_2022"
REGISTRY_STATUS = "PASS_COMMON_VARIABLE_ONLY_2021B_CONFORMAL_PRE_MIMIC"
TRANSFER_STATUS = "PASS_MIMIC_CONFORMAL_PROJECTION_PRE_DATA"
YEARS = (2018, 2019, 2020, 2021, 2022)
# Model output labels are deliberately never renamed in the model.  The
# reporting labels are the method-lock ontology and are translated only here.
MODEL_LEAVES = ("none", "ap", "biliary", "sepsis_or_organ", "other")
LEAVES = ("none", "AP_specific", "biliary", "sepsis_or_acute_organ_dysfunction", "other")
REPORT_TO_MODEL = {"none": "none", "AP_specific": "ap", "biliary": "biliary", "sepsis_or_acute_organ_dysfunction": "sepsis_or_organ", "other": "other"}
MODEL_TO_REPORT = {v: k for k, v in REPORT_TO_MODEL.items()}
OLD_PRA_SHA = "e1e682fea14da07e5b9ac531be13859781b7f8ad277bfdc1d6e73509aca64fa1"
NEW_PRA_SHA = "8ad5391afb31c72aa54b41875cac0d458b05182fe02f3140ae464683b8855010"
# The actual restricted archive is never used by tests.  A lowered bootstrap
# count is categorically unavailable when this registered archive is involved.
REAL_MIMIC_ARCHIVE_SHA = "0dd448e341f489483194daafd884607d6f194648c6bf54eed56d1e2cd2af9e09"
PROHIBITED = {"PAY1", "ZIPINC_QRTL", "DISCWT", "CCR_NRD", "WAGEINDEX", "APRDRG", "HOSP_BEDSIZE", "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH"}
ARTIFACTS = ("checkpoint", "finetune_manifest", "operating_manifest", "calibrators", "thresholds", "hierarchy_arrays", "hierarchy_vocabularies", "dx_vocabulary", "pr_vocabulary", "conformal_binding")
ROLES = ARTIFACTS + ("common_variable_model_lock", "conformal_registry", "adapter", "stage2_pra_lock", "pra_2022_lock", "ontology_labels")

def stable(x: Any) -> str: return json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()
def json_obj(path: Path) -> dict[str, Any]:
    try: x = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc: raise RuntimeError("JSON object required: " + str(path)) from exc
    if not isinstance(x, dict): raise RuntimeError("JSON object required")
    return x
def identity(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str): raise RuntimeError("malformed binding: " + label)
    p = Path(str(value["path"])).resolve()
    if not p.is_file() or value.get("sha256") != sha(p) or int(value.get("bytes", -1)) != p.stat().st_size: raise RuntimeError("binding identity: " + label)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
def _canonical_conformal_file(binding_path: Path, recorded: object, suffix: str, label: str) -> dict[str, Any]:
    """Resolve Stage 6 data from the local binding, never its remote provenance.

    The conformal binding intentionally preserves the producer's immutable
    provenance path.  A local restricted copy therefore cannot require that
    foreign absolute path to exist.  It may only use the canonical file below
    the local binding and must prove that the provenance still names exactly
    that canonical suffix and exact bytes.  Symlink traversal is rejected so a
    local relocation cannot silently redirect a bound Stage 6 artifact.
    """
    if not isinstance(recorded, Mapping) or not isinstance(recorded.get("path"), str):
        raise RuntimeError("MIMIC conformal nested identity: " + label)
    normalized = str(recorded["path"]).replace("\\", "/")
    if not normalized.endswith("/" + suffix):
        raise RuntimeError("MIMIC conformal provenance suffix: " + label)
    root = binding_path.parent
    target = root.joinpath(*suffix.split("/"))
    try:
        rel = target.relative_to(root)
    except ValueError as exc:  # Defensive even though suffix is a constant.
        raise RuntimeError("MIMIC conformal canonical escape: " + label) from exc
    cursor = root
    if cursor.is_symlink():
        raise RuntimeError("MIMIC conformal symlink: binding root")
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError("MIMIC conformal symlink: " + label)
    if not target.is_file():
        raise RuntimeError("MIMIC conformal nested identity: " + label)
    if not isinstance(recorded.get("bytes"), int) or not isinstance(recorded.get("sha256"), str):
        raise RuntimeError("MIMIC conformal nested identity: " + label)
    actual = {"path": str(target.resolve()), "bytes": target.stat().st_size, "sha256": sha(target)}
    if actual["bytes"] != recorded["bytes"] or actual["sha256"] != recorded["sha256"]:
        raise RuntimeError("MIMIC conformal nested identity: " + label)
    return actual
def require_false(x: Mapping[str, Any], key: str, label: str) -> None:
    if x.get(key) is not False: raise RuntimeError(label + "." + key)
def link(x: object, key: str, b: Mapping[str, Any], label: str) -> None:
    if not isinstance(x, Mapping) or not isinstance(x.get(key), Mapping) or x[key].get("sha256") != b["sha256"]: raise RuntimeError(label + " linkage: " + key)
def code(x: Any) -> str: return "" if x is None else str(x).strip().upper().replace(".", "")
def norm(x: Any) -> str: return " ".join(str(x or "").upper().split())
def date(x: Any, field: str) -> datetime:
    s = str(x or "").strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try: return datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt)
        except ValueError: pass
    raise ValueError("invalid " + field)
def transfer_location(x: Any) -> bool:
    x = norm(x); return "ACUTE HOSPITAL" in x or ("TRANSFER" in x and "HOSPITAL" in x)

def validate_method(path: Path) -> dict[str, Any]:
    m = json_obj(path); s = m.get("source")
    if m.get("status") != METHOD_STATUS or not isinstance(s, Mapping): raise RuntimeError("method status")
    members, ids = s.get("required_members"), s.get("required_member_identities")
    if not isinstance(members, list) or not isinstance(ids, Mapping) or set(members) != set(ids) or len(members) != 4: raise RuntimeError("method member identities")
    if not isinstance(s.get("archive_bytes"), int) or not isinstance(s.get("archive_sha256"), str): raise RuntimeError("method archive identity")
    if tuple(m.get("planned_readmission_under_shifted_dates", {}).get("annual_rule_sets", ())) != YEARS: raise RuntimeError("five-year PRA method lock")
    if m.get("prediction_anchor_and_history", {}).get("calendar_year_embedding") != "DISABLED": raise RuntimeError("calendar-year method lock")
    return m
def validate_contract(x: object) -> dict[str, Any]:
    wanted = {"projection": MODEL_ID, "calibration_partition": "2021A", "conformal_partition": "2021B", "mimic_fit": False, "mimic_recalibration": False, "mimic_threshold_optimization": False, "vocabulary_growth": False}
    if not isinstance(x, Mapping) or any(x.get(k) != v for k, v in wanted.items()): raise RuntimeError("transfer contract")
    return dict(x)
def validate_evaluation(x: object) -> dict[str, Any]:
    if not isinstance(x, Mapping) or tuple(x.get("coverages", ())) != (.8, .9, .95) or tuple(x.get("subgroups", ())) != ("sex", "age_group") or not isinstance(x.get("endpoints"), Mapping) or not x["endpoints"]: raise RuntimeError("evaluation lock")
    for value in x["endpoints"].values():
        if not isinstance(value, Mapping) or not isinstance(value.get("probability_field"), str) or not isinstance(value.get("threshold"), (int, float)) or not 0 < float(value["threshold"]) < 1: raise RuntimeError("endpoint lock")
    return dict(x)

def validate_semantics(b: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate real production manifests.  Never JSON-parse the checkpoint."""
    f, o, ml, cr, cb = (json_obj(Path(b[n]["path"])) for n in ("finetune_manifest", "operating_manifest", "common_variable_model_lock", "conformal_registry", "conformal_binding"))
    if f.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022" or f.get("prediction_partition") != "2021A": raise RuntimeError("finetune seal")
    require_false(f, "2021B_accessed", "finetune"); require_false(f, "year_2022_accessed", "finetune")
    if f.get("ablations", {}).get(MODEL_ID) is not True or f.get("model_config", {}).get("use_year_version") is not False or f.get("checkpoint", {}).get("sha256") != b["checkpoint"]["sha256"]: raise RuntimeError("common-variable finetune/checkpoint binding")
    static = f.get("static_preprocessor")
    if not isinstance(static, Mapping) or not static.get("numeric_columns") or not static.get("categorical_columns") or any(x in stable(static).upper() for x in PROHIBITED): raise RuntimeError("static common-variable preprocessor")
    if o.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A" or o.get("selection_partition") != "2021A": raise RuntimeError("operating seal")
    require_false(o, "2021B_accessed", "operating"); require_false(o, "year_2022_accessed", "operating")
    if o.get("source", {}).get("manifest_sha256") != b["finetune_manifest"]["sha256"]: raise RuntimeError("operating/finetune linkage")
    for name, role in (("transformer_binary_calibrators.joblib", "calibrators"), ("transformer_operating_thresholds_2021A.json", "thresholds")):
        if o.get("artifacts", {}).get(name, {}).get("sha256") != b[role]["sha256"]: raise RuntimeError("operating artifact linkage: " + role)
    if ml.get("status") != MODEL_LOCK_STATUS or ml.get("selection_partition") != "2021A" or ml.get("selected_id") != MODEL_ID or ml.get("selection_mode") != "pre_specified_common_variable_only_no_metric_selection": raise RuntimeError("model lock")
    require_false(ml, "2021B_accessed", "model lock"); require_false(ml, "year_2022_accessed", "model lock")
    if ml.get("selected_model", {}).get("configuration") != {"common_variable_only": True}: raise RuntimeError("model projection")
    for x, k, role in ((ml.get("selected_model", {}), "checkpoint", "checkpoint"), (ml.get("selected_model", {}), "finetune_manifest", "finetune_manifest"), (ml.get("operating_point", {}), "manifest", "operating_manifest"), (ml.get("operating_point", {}), "binary_calibrators", "calibrators"), (ml.get("operating_point", {}), "thresholds", "thresholds")):
        link(x, k, b[role], "model lock")
    if cr.get("status") != REGISTRY_STATUS or cr.get("model_id") != MODEL_ID or cr.get("selection_partition") != "2021A" or cr.get("calibration_partition") != "2021B only" or cr.get("model_or_threshold_selection_on_2021B") is not False or cr.get("year_2022_accessed") is not False: raise RuntimeError("conformal registry")
    link(cr, "model_lock", b["common_variable_model_lock"], "conformal registry"); link(cr, "mimic_transfer_conformal_binding", b["conformal_binding"], "conformal registry")
    if cb.get("status") != TRANSFER_STATUS or cb.get("calibration_partition") != "2021B only" or cb.get("model_or_threshold_selection_on_2021B") is not False or cb.get("year_2022_accessed") is not False or cb.get("allowed_mondrian_dimensions") != ["sex", "age_group"] or set(cb.get("mondrian", {})) != {"sex", "age_group"} or not isinstance(cb.get("global"), Mapping): raise RuntimeError("MIMIC conformal binding")
    if "payer" in stable(cb).lower() or "zip_income_quartile" in stable(cb).lower(): raise RuntimeError("forbidden conformal dimensions")
    for cov in ("80", "90", "95"):
        if not isinstance(cb["global"].get(cov), Mapping) or not isinstance(cb["global"][cov].get("q"), (int, float)): raise RuntimeError("global conformal roster")
    suffixes = {
        "conformal_manifest": "stage6_2021B/conformal_2021B/conformal_calibrator.json",
        "conformal_sets": "stage6_2021B/conformal_2021B/conformal_sets_2021B.parquet",
    }
    binding_path = Path(b["conformal_binding"]["path"])
    for name, suffix in suffixes.items():
        registry_record = cr.get("stage6", {}).get(name) if isinstance(cr.get("stage6"), Mapping) else None
        if not isinstance(cb.get(name), Mapping) or not isinstance(registry_record, Mapping):
            raise RuntimeError("stage6 conformal linkage")
        # The registry and binding must preserve the same immutable nested
        # provenance record; only the local canonical target is relocated.
        for field in ("path", "bytes", "sha256"):
            if cb[name].get(field) != registry_record.get(field):
                raise RuntimeError("stage6 conformal linkage")
        _canonical_conformal_file(binding_path, cb[name], suffix, name)

def validate_pra(b: Mapping[str, Mapping[str, Any]]) -> tuple[Path, Path]:
    old, new = Path(b["stage2_pra_lock"]["path"]), Path(b["pra_2022_lock"]["path"])
    if b["stage2_pra_lock"]["sha256"] != OLD_PRA_SHA or b["pra_2022_lock"]["sha256"] != NEW_PRA_SHA: raise RuntimeError("PRA lock hashes")
    a, z = json_obj(old), json_obj(new)
    if a.get("status") != "PASS" or tuple(a.get("years_locked", ())) != YEARS[:4] or z.get("status") != "PASS_2022_PRA_LOCKED_PRE_TEST_ACCESS" or z.get("calendar_year") != 2022 or z.get("nrd_2022_accessed") is not False: raise RuntimeError("PRA lock semantics")
    for p in (old.parent / "annual_algorithm_code_sets.parquet", old.parent / "annual_ccs_mapping.parquet", new.parent / "annual_algorithm_code_sets_2022.parquet", new.parent / "annual_ccs_mapping_2022.parquet"):
        if not p.is_file(): raise RuntimeError("PRA artifact missing")
    return old, new

def _adapter_lock(b: Mapping[str, Mapping[str, Any]], a: object) -> dict[str, Any]:
    if not isinstance(a, Mapping) or a.get("model_id") != MODEL_ID or a.get("function") != "predict_common_variable_episode" or not isinstance(a.get("module"), str): raise RuntimeError("inference adapter contract")
    for k in ("mimic_fit", "mimic_recalibration", "mimic_threshold_optimization", "vocabulary_growth"):
        if a.get(k) is not False: raise RuntimeError("inference adapter prohibition: " + k)
    if set(ARTIFACTS) - set(a): raise RuntimeError("inference adapter artifact roster")
    for role in ARTIFACTS:
        if not isinstance(a[role], Mapping) or a[role].get("sha256") != b[role]["sha256"]: raise RuntimeError("inference adapter/top-level mismatch: " + role)
    return {"model_id": MODEL_ID, "module": str(a["module"]), "function": str(a["function"]), **{k: b[k] for k in ARTIFACTS}, **{k: False for k in ("mimic_fit", "mimic_recalibration", "mimic_threshold_optimization", "vocabulary_growth")}}

def prearchive(method_path: Path, execution_path: Path) -> dict[str, Any]:
    m = validate_method(method_path); e = json_obj(execution_path)
    if e.get("status") != EXECUTION_STATUS or e.get("method_lock", {}).get("sha256") != sha(method_path): raise RuntimeError("execution/method lock")
    validate_contract(e.get("contract")); raw = e.get("bindings")
    if not isinstance(raw, Mapping) or set(raw) != set(ROLES): raise RuntimeError("complete v7 bindings")
    b = {k: identity(raw[k], k) for k in ROLES}; ev = validate_evaluation(e.get("evaluation"))
    a = _adapter_lock(b, e.get("inference_adapter"))
    if e.get("adapter", {}).get("sha256") != b["adapter"]["sha256"]: raise RuntimeError("adapter identity")
    validate_semantics(b); old, new = validate_pra(b)
    return {"method": m, "execution": e, "bindings": b, "adapter": a, "evaluation": ev, "stage2": old, "pra2022": new}

def immutable(path: Path, data: bytes) -> None:
    path = Path(path).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise RuntimeError("immutable output exists")
    fd, temp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
def freeze(method: Path, spec: Path, out: Path) -> dict[str, Any]:
    validate_method(method); s = json_obj(spec); c = validate_contract(s.get("contract")); raw = s.get("bindings")
    if not isinstance(raw, Mapping) or set(raw) != set(ROLES): raise RuntimeError("complete v7 bindings")
    b = {k: identity(raw[k], k) for k in ROLES}; ev = validate_evaluation(s.get("evaluation")); a = _adapter_lock(b, s.get("inference_adapter"))
    # Freezing is itself a no-data gate: do not publish a lock that cannot pass
    # the real manifest/PRA contract later.
    validate_semantics(b); validate_pra(b)
    result = {"execution_lock_version": 7, "status": EXECUTION_STATUS, "method_lock": {"path": str(Path(method).resolve()), "bytes": Path(method).stat().st_size, "sha256": sha(method)}, "contract": c, "bindings": b, "adapter": {"path": b["adapter"]["path"], "bytes": b["adapter"]["bytes"], "sha256": b["adapter"]["sha256"]}, "inference_adapter": a, "evaluation": ev}
    immutable(out, stable(result).encode("utf-8")); return result
def validate_only(method: Path, execution: Path | None = None) -> dict[str, Any]:
    m = validate_method(method)
    if execution is None: return {"status": "PASS_VALIDATE_ONLY_NO_ARCHIVE_ACCESS", "archive_opened": False, "method_lock_sha256": sha(method), "pending_model_bindings": m.get("model_and_calibration", {}).get("required_pending_bindings", [])}
    prearchive(method, execution); return {"status": "PASS_VALIDATE_ONLY_NO_ARCHIVE_ACCESS", "archive_opened": False, "method_lock_sha256": sha(method), "execution_lock_sha256": sha(execution)}

def verify_archive(path: Path, method: Mapping[str, Any]) -> dict[str, Any]:
    s = method["source"]
    if not path.is_file() or path.stat().st_size != s["archive_bytes"] or sha(path) != s["archive_sha256"]: raise RuntimeError("archive identity")
    got: dict[str, Any] = {}
    with zipfile.ZipFile(path) as z:
        for name, exp in s["required_member_identities"].items():
            info = z.getinfo(name); h = hashlib.sha256()
            with z.open(info) as f:
                for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
            if info.file_size != exp["bytes"] or info.compress_size != exp["compressed_bytes"] or f"{info.CRC:08x}".lower() != str(exp["crc32"]).lower() or h.hexdigest() != exp["sha256"]: raise RuntimeError("archive member identity: " + name)
            got[name] = {"bytes": info.file_size, "compressed_bytes": info.compress_size, "crc32": f"{info.CRC:08x}", "sha256": h.hexdigest()}
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": s["archive_sha256"], "members": got}
def tables(path: Path, method: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with zipfile.ZipFile(path) as z:
        for name in method["source"]["required_members"]:
            with z.open(name) as raw, gzip.GzipFile(fileobj=raw) as gz: out[Path(name).name.replace(".csv.gz", "")] = list(csv.DictReader(io.TextIOWrapper(gz, encoding="utf-8", newline="")))
    if set(out) != {"admissions", "patients", "diagnoses_icd", "procedures_icd"}: raise RuntimeError("locked table roster")
    return out

def valid_admissions(admissions: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail closed per row, preserving valid episodes and aggregate-only audit.

    The method lock requires valid identifiers and admission/discharge times.
    Rows violating that input contract cannot safely contribute to episodes, but
    one malformed row must not invalidate an otherwise valid restricted cohort.
    Reasons are deliberately mutually exclusive in the listed precedence order
    so that the total is independently auditable without row-level disclosure.
    """
    reasons = {"missing_ids": 0, "invalid_or_missing_timestamps": 0, "discharge_before_admit": 0}
    good: list[dict[str, Any]] = []
    for raw in admissions:
        r = dict(raw)
        if not str(r.get("subject_id") or "").strip() or not str(r.get("hadm_id") or "").strip():
            reasons["missing_ids"] += 1
            continue
        try:
            r["_a"] = date(r.get("admittime"), "admittime")
            r["_d"] = date(r.get("dischtime"), "dischtime")
        except ValueError:
            reasons["invalid_or_missing_timestamps"] += 1
            continue
        if r["_d"] < r["_a"]:
            reasons["discharge_before_admit"] += 1
            continue
        good.append(r)
    excluded = sum(reasons.values())
    audit = {"input_rows": len(admissions), "valid_rows": len(good), "excluded_total": excluded, "reason_counts": reasons}
    if audit["input_rows"] != audit["valid_rows"] + audit["excluded_total"]:
        raise RuntimeError("admission data-quality audit accounting")
    return good, audit

def collapse(admissions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prepared, _ = valid_admissions(admissions)
    for raw in prepared:
        r = dict(raw)
        by[str(r["subject_id"])].append(r)
    result: list[dict[str, Any]] = []
    for sid, rs in by.items():
        rs.sort(key=lambda r: (r["_a"], r["_d"], str(r["hadm_id"]))); chain: list[dict[str, Any]] = []
        for r in rs:
            if not chain: chain = [r]; continue
            end, prev = max(x["_d"] for x in chain), max(chain, key=lambda x: (x["_d"], str(x["hadm_id"])))
            if r["_a"] <= end or (r["_a"] - end <= timedelta(hours=24) and (transfer_location(prev.get("discharge_location")) or transfer_location(r.get("admission_location")))): chain.append(r)
            else: result.append(episode(sid, chain)); chain = [r]
        if chain: result.append(episode(sid, chain))
    return sorted(result, key=lambda e: (e["subject_id"], e["start"], e["hadm_ids"]))
def episode(sid: str, xs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cs = sorted((dict(x) for x in xs), key=lambda r: (r["_a"], r["_d"], str(r["hadm_id"])))
    return {"subject_id": sid, "components": cs, "hadm_ids": tuple(str(x["hadm_id"]) for x in cs), "start": min(x["_a"] for x in cs), "end": max(x["_d"] for x in cs)}
def _index_by_hadm(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    indexed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[str(row.get("hadm_id"))].append(row)
    return indexed

def _candidate_rows(rows: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], hadms: Sequence[str]):
    if isinstance(rows, Mapping):
        for hadm in hadms:
            yield from rows.get(str(hadm), ())
    else:
        yield from rows

def principal(dx: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], e: Mapping[str, Any]) -> dict[str, Any] | None:
    rank = {h: i for i, h in enumerate(e["hadm_ids"])}; rows = [dict(r) for r in _candidate_rows(dx, e["hadm_ids"]) if str(r.get("hadm_id")) in rank and str(r.get("seq_num", "")).strip() == "1"]
    return min(rows, key=lambda r: (rank[str(r["hadm_id"])], code(r.get("icd_code")))) if rows else None
def eligible(admissions: Sequence[Mapping[str, Any]], patients: Sequence[Mapping[str, Any]], dx: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    eps, people, ans = collapse(admissions), {str(p["subject_id"]): dict(p) for p in patients}, []
    for e in eps:
        p = people.get(e["subject_id"])
        if not p: continue
        anchors = [c for c in e["components"] if (q := principal(dx, {**e, "hadm_ids": (str(c["hadm_id"]),)})) and str(q.get("icd_version")).strip() == "10" and code(q.get("icd_code")).startswith("K85")]
        if not anchors: continue
        anchor = min(anchors, key=lambda c: (c["_a"], str(c["hadm_id"])))
        try: age = int(float(p.get("anchor_age"))) + anchor["_a"].year - int(float(p.get("anchor_year")))
        except (TypeError, ValueError): continue
        # The frozen static preprocessor uses binary FEMALE.  Unknown sex is
        # not silently imputed: this is an explicit fail-closed exclusion.
        if norm(p.get("gender")) not in ("F", "M"): continue
        if not 18 <= age <= 120 or any(str(c.get("hospital_expire_flag", "")).strip() not in ("0", "0.0") for c in e["components"]): continue
        try: deaths = [date(x, "death") for x in [p.get("dod"), *[c.get("deathtime") for c in e["components"]]] if x]
        except ValueError: continue
        if any(d <= e["end"] for d in deaths): continue
        ans.append({**e, "index_anchor_component": anchor, "age": age})
    return ans, eps, people
def dedupe(rows: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], hadms: Sequence[str], *, procedure: bool = False, end: datetime | None = None) -> list[tuple[tuple[str, str, str, str], dict[str, Any], datetime | None]]:
    wanted, seen, out = set(hadms), set(), []
    for raw in _candidate_rows(rows, hadms):
        if str(raw.get("hadm_id")) not in wanted: continue
        r, c, rawdate = dict(raw), code(raw.get("icd_code")), str(raw.get("chartdate") or "").strip()
        if not c: continue
        parsed = None
        if procedure and rawdate:
            try: parsed = date(rawdate, "chartdate")
            except ValueError: parsed = None
            if parsed is not None and end is not None and parsed > end: continue
        key = (c, str(r.get("icd_version") or ""), str(r.get("seq_num") or ""), rawdate)
        if key not in seen: seen.add(key); out.append((key, r, parsed))
    return sorted(out, key=lambda x: x[0])
def vocab(path: Path) -> dict[str, int]:
    x = json_obj(path); x = x.get("token_to_id", x)
    if not isinstance(x, Mapping): raise RuntimeError("vocabulary")
    return {code(k): int(v) for k, v in x.items()}
def map_vocab(xs: Sequence[str], v: Mapping[str, int]) -> tuple[list[int], int]:
    out = [int(v.get(x, 3)) for x in xs]; return out, sum(x not in v for x in xs)
def features(index: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]], people: Mapping[str, Mapping[str, Any]], dx: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], pr: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], dxv: Mapping[str, int], prv: Mapping[str, int], cause, episodes_by_subject: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    subject_eps = list(episodes_by_subject.get(index["subject_id"], ())) if episodes_by_subject is not None else [e for e in episodes if e["subject_id"] == index["subject_id"]]
    prior = [e for e in subject_eps if e["end"] < index["start"]]; hist = [e for e in prior if index["start"] - e["end"] <= timedelta(days=180)]
    current_dx, current_pr = dedupe(dx, index["hadm_ids"]), dedupe(pr, index["hadm_ids"], procedure=True, end=index["end"])
    hids = [h for e in hist for h in e["hadm_ids"]]; prior_dx, prior_pr = dedupe(dx, hids), dedupe(pr, hids, procedure=True, end=index["start"])
    starts = {str(c["hadm_id"]): c["_a"] for c in index["components"]}; sex = norm(people[index["subject_id"]].get("gender")); female = 1 if sex == "F" else 0 if sex == "M" else None
    d, do = map_vocab([x[0][0] for x in current_dx], dxv); q, qo = map_vocab([x[0][0] for x in current_pr], prv); pd, pdo = map_vocab([x[0][0] for x in prior_dx], dxv); pp, ppo = map_vocab([x[0][0] for x in prior_pr], prv)
    earliest = min(e["start"] for e in subject_eps); causes = [cause(e) for e in hist]
    f = {"AGE": index["age"], "FEMALE": female, "LOS": (index["end"] - index["index_anchor_component"]["_a"]).total_seconds() / 86400, "I10_NDX": len(current_dx), "I10_NPR": len(current_pr), "dx_tokens": d, "pr_tokens": q, "prday": [(x[2].date() - starts[str(x[1]["hadm_id"])].date()).days if x[2] else None for x in current_pr], "prior_dx_tokens_180d": pd, "prior_pr_tokens_180d": pp, "prior_count_ytd": sum(e["end"].year == index["start"].year for e in prior), "prior_count_30d": sum(index["start"] - e["end"] <= timedelta(days=30) for e in prior), "prior_count_90d": sum(index["start"] - e["end"] <= timedelta(days=90) for e in prior), "prior_count_180d": len(hist), "prior_ed_count_180d": sum(any("EMERGENCY" in norm(c.get("admission_location")) or "EMERGENCY" in norm(c.get("admission_type")) for c in e["components"]) for e in hist), "prior_nonelective_count_180d": sum(any(norm(c.get("admission_type")) != "ELECTIVE" for c in e["components"]) for e in hist), "prior_ap_count_180d": causes.count("AP_specific"), "prior_biliary_count_180d": causes.count("biliary"), "prior_sepsis_or_organ_count_180d": causes.count("sepsis_or_acute_organ_dysfunction"), "prior_los_sum_180d": sum((e["end"] - e["start"]).total_seconds() / 86400 for e in hist), "days_since_prior_discharge": (index["start"] - max(e["end"] for e in prior)).total_seconds() / 86400 if prior else None, "history_30d_fully_observable": int(index["start"] - earliest >= timedelta(days=30)), "history_90d_fully_observable": int(index["start"] - earliest >= timedelta(days=90)), "history_180d_fully_observable": int(index["start"] - earliest >= timedelta(days=180)), "oov_audit": {"diagnosis": do + pdo, "procedure": qo + ppo, "total": do + pdo + qo + ppo}}
    if set(f) & PROHIBITED or any(k in f for k in ("year", "calendar_year", "year_index")): raise RuntimeError("feature contract")
    return f
def validate_feature_contract(feature: Mapping[str, Any], execution: Mapping[str, Any]) -> None:
    """Bind every static-model input to the frozen common-variable schema."""
    static = json_obj(Path(execution["bindings"]["finetune_manifest"]["path"])).get("static_preprocessor", {})
    names = set(static.get("numeric_columns", ())) | set(static.get("categorical_columns", ()))
    required = {"AGE", "FEMALE", "LOS", "I10_NDX", "I10_NPR", "dx_tokens", "pr_tokens", "prday", "prior_dx_tokens_180d", "prior_pr_tokens_180d", "oov_audit"}
    if not required.issubset(feature) or not names.issubset(feature) or feature.get("FEMALE") not in (0, 1): raise RuntimeError("frozen common-variable feature schema")
    if any(k in feature for k in ("year", "calendar_year", "year_index")) or set(feature) & PROHIBITED: raise RuntimeError("calendar/prohibited feature")

def classifier(dx: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], pr: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], old: Path, new: Path):
    source = Path(__file__).resolve().parents[1] / "stage2" / "planned_readmission.py"; spec = importlib.util.spec_from_file_location("locked_planned_readmission_v7", source)
    if spec is None or spec.loader is None: raise RuntimeError("frozen PRA adapter import")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); C = getattr(module, "PlannedReadmissionClassifier", None)
    if C is None: raise RuntimeError("frozen PlannedReadmissionClassifier absent")
    older = C.from_stage2_lock(old.parent.parent); newer = C(new.parent / "annual_algorithm_code_sets_2022.parquet", new.parent / "annual_ccs_mapping_2022.parquet")
    def f(e: Mapping[str, Any]) -> dict[int, str]:
        p = principal(dx, e)
        if not p: return {y: "algorithm_unknown" for y in YEARS}
        procedures = [x[0][0] for x in dedupe(pr, e["hadm_ids"], procedure=True, end=e["end"])]
        return {**{y: older.classify(y, code(p.get("icd_code")), procedures).status for y in YEARS[:4]}, 2022: newer.classify(2022, code(p.get("icd_code")), procedures).status}
    return f
def consensus(x: Mapping[int, str]) -> str:
    return "stable_planned" if set(x) == set(YEARS) and all(x[y] == "planned" for y in YEARS) else "stable_unplanned" if set(x) == set(YEARS) and all(x[y] == "unplanned" for y in YEARS) else "algorithm_unknown"
def hierarchy(path: Path):
    labels: dict[str, set[str]] = {"AP_specific": set(), "biliary": set(), "sepsis_or_acute_organ_dysfunction": set()}
    with path.open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h):
            for leaf, node in (("AP_specific", "readmission.ap_specific"), ("biliary", "readmission.biliary_event"), ("sepsis_or_acute_organ_dysfunction", "readmission.sepsis_or_acute_organ_dysfunction")):
                if r.get("label_node") == node: labels[leaf].add(code(r.get("code")))
    def cause(dx: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], e: Mapping[str, Any]) -> str | None:
        p = principal(dx, e)
        if not p or str(p.get("icd_version")).strip() != "10": return None
        c = code(p.get("icd_code"))
        return next((leaf for leaf in ("AP_specific", "biliary", "sepsis_or_acute_organ_dysfunction") if c in labels[leaf]), "other")
    return cause
def label(index: Mapping[str, Any], eps: Sequence[Mapping[str, Any]], classify, cause, episodes_by_subject: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    unknown = False
    subject_eps = episodes_by_subject.get(index["subject_id"], ()) if episodes_by_subject is not None else (e for e in eps if e["subject_id"] == index["subject_id"])
    for e in sorted((e for e in subject_eps if e["start"] > index["end"] and e["start"] - index["end"] <= timedelta(days=30)), key=lambda e: (e["start"], e["hadm_ids"])):
        state = consensus(classify(e))
        if state == "stable_planned": continue
        if state == "algorithm_unknown": unknown = True; continue
        if unknown: return {"label_status": "algorithm_unknown", "any_readmission": None, "leaf": None, "bounds": {"unknown_as_planned": 0, "unknown_as_unplanned": 1}}
        c = cause(e); return {"label_status": "qualified" if c in LEAVES else "cause_unknown", "any_readmission": 1, "leaf": c if c in LEAVES else None, "bounds": {"unknown_as_planned": 1, "unknown_as_unplanned": 1}}
    return {"label_status": "algorithm_unknown", "any_readmission": None, "leaf": None, "bounds": {"unknown_as_planned": 0, "unknown_as_unplanned": 1}} if unknown else {"label_status": "none", "any_readmission": 0, "leaf": "none", "bounds": {"unknown_as_planned": 0, "unknown_as_unplanned": 0}}

def load_adapter(x: Mapping[str, Any]):
    a, p = x["adapter"], Path(x["bindings"]["adapter"]["path"]); spec = importlib.util.spec_from_file_location("locked_mimic_inference_adapter_v7", p)
    if spec is None or spec.loader is None: raise RuntimeError("adapter import")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); fn = getattr(m, a["function"], None)
    if not callable(fn): raise RuntimeError("adapter callable")
    return lambda feature: fn(dict(feature), x["execution"])
def _auc(y: list[int], p: list[float]):
    a, b = [v for z, v in zip(y, p) if z], [v for z, v in zip(y, p) if not z]
    return None if not a or not b else sum((i > j) + .5 * (i == j) for i in a for j in b) / (len(a) * len(b))
def _auprc(y: list[int], p: list[float]):
    if not sum(y): return None
    positives = 0; total = 0.0
    for rank, (_, outcome) in enumerate(sorted(zip(p, y), reverse=True), 1):
        positives += outcome
        if outcome: total += positives / rank
    return total / sum(y)
def _calibration(y: list[int], p: list[float]) -> tuple[float | None, float | None]:
    """Unpenalized logistic calibration; undefined rather than invented."""
    if not sum(y) or sum(y) == len(y): return None, None
    q = [min(1 - 1e-12, max(1e-12, x)) for x in p]
    if len(set(q)) < 2: return None, None
    if _np is not None:
        # Keep the same 40-step Newton contract while moving the repeated
        # row-wise arithmetic into compiled array operations.  This is the
        # dominant path in 1,000 subject-cluster bootstrap replicates.
        z = _np.log(_np.asarray(q, dtype=float) / (1.0 - _np.asarray(q, dtype=float)))
        yy = _np.asarray(y, dtype=float); b0, b1 = 0.0, 1.0
        for _ in range(40):
            eta = _np.clip(b0 + b1 * z, -35.0, 35.0); fitted = 1.0 / (1.0 + _np.exp(-eta)); w = fitted * (1.0 - fitted)
            h00 = float(w.sum()); h01 = float(_np.dot(w, z)); h11 = float(_np.dot(w, z * z)); det = h00 * h11 - h01 * h01
            if det <= 1e-12: return None, None
            residual = yy - fitted; g0 = float(residual.sum()); g1 = float(_np.dot(residual, z)); d0 = (h11*g0-h01*g1)/det; d1 = (h00*g1-h01*g0)/det; b0 += d0; b1 += d1
            if max(abs(d0), abs(d1)) < 1e-8: return b0, b1
        return None, None
    logit = [math.log(x / (1 - x)) for x in q]; b0, b1 = 0.0, 1.0
    for _ in range(40):
        fitted = [1 / (1 + math.exp(-max(-35, min(35, b0 + b1 * z)))) for z in logit]
        w = [x * (1 - x) for x in fitted]; h00 = sum(w); h01 = sum(a*b for a, b in zip(w, logit)); h11 = sum(a*b*b for a, b in zip(w, logit)); det = h00*h11-h01*h01
        if det <= 1e-12: return None, None
        g0 = sum(a-b for a, b in zip(y, fitted)); g1 = sum((a-b)*z for a, b, z in zip(y, fitted, logit)); d0 = (h11*g0-h01*g1)/det; d1 = (h00*g1-h01*g0)/det; b0 += d0; b1 += d1
        if max(abs(d0), abs(d1)) < 1e-8: return b0, b1
    return None, None
def metrics(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any], label_field: str) -> dict[str, Any]:
    rows = [r for r in rows if r["label"].get(label_field) in (0, 1)]; y = [r["label"][label_field] for r in rows]; p = [float(r["prediction"][spec["probability_field"]]) for r in rows]
    return _metrics_arrays(y, p, spec)
def _metrics_arrays(y: Sequence[int], p: Sequence[float], spec: Mapping[str, Any]) -> dict[str, Any]:
    if not y: return {"n": 0, "status": "undefined"}
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in p): raise RuntimeError("invalid probability")
    threshold = float(spec["threshold"]); pred = [x >= threshold for x in p]; tp, fp = sum(a and b for a, b in zip(pred, y)), sum(a and not b for a, b in zip(pred, y)); ev, ne = sum(y), len(y) - sum(y); odds = threshold / (1 - threshold); q = [min(1 - 1e-12, max(1e-12, x)) for x in p]; intercept, slope = _calibration(y, p)
    event_gate = "confirmatory" if ev >= 100 and ne >= 100 else "exploratory_instability_warning"
    return {"n": len(y), "events": ev, "non_events": ne, "status": "ci_eligible" if min(ev, ne) >= 20 else "descriptive_only_event_gate", "prevalence": ev / len(y), "auroc": _auc(y, p), "auprc": _auprc(y, p), "brier": sum((a - b) ** 2 for a, b in zip(y, p)) / len(y), "log_loss": -sum(a*math.log(b)+(1-a)*math.log(1-b) for a, b in zip(y, q)) / len(y), "calibration_intercept": intercept, "calibration_slope": slope, "calibration_gate": event_gate, "dca_gate": event_gate, "decision_curve": {"threshold": threshold, "model_net_benefit": tp / len(y) - fp / len(y) * odds, "treat_all_net_benefit": ev / len(y) - ne / len(y) * odds, "treat_none_net_benefit": 0.0}}
def _conformal_metrics(rows: Sequence[Mapping[str, Any]], coverage: str, dimension: str | None) -> dict[str, Any]:
    """Score the corresponding frozen global/sex/age risk set, never a copy."""
    selected = [r for r in rows if r["label"].get("leaf") in LEAVES]
    included, abstained = [], []
    for r in selected:
        record = r["prediction"]["conformal"][coverage] if dimension is None else r["prediction"]["conformal"][coverage].get(dimension)
        if not isinstance(record, Mapping) or not isinstance(record.get("set"), list) or not isinstance(record.get("abstain"), bool): raise RuntimeError("missing conformal risk set")
        if dimension is not None and str(record.get("group")) != str(r[dimension]): raise RuntimeError("conformal subgroup key mismatch")
        risk_set = set(record["set"])
        if not risk_set.issubset(MODEL_LEAVES): raise RuntimeError("adapter conformal model leaf contract")
        included.append(REPORT_TO_MODEL[r["label"]["leaf"]] in risk_set); abstained.append(record["abstain"])
    if not selected: return {"n": 0, "status": "undefined"}
    target_events = sum(r["label"]["leaf"] != "none" for r in selected)
    out = {"n": len(selected), "target_events": target_events, "marginal_coverage": sum(included) / len(included), "abstention_rate": sum(abstained) / len(abstained), "event_gate": "confirmatory" if target_events >= 100 else "exploratory" if target_events >= 50 else "descriptive_only"}
    if dimension is not None:
        by: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for r, ok, abstain in zip(selected, included, abstained): by[str(r[dimension])].append((ok, abstain))
        out["groups"] = {g: {"n": len(v), "target_events": sum(r["label"]["leaf"] != "none" for r in selected if str(r[dimension]) == g), "coverage": sum(x[0] for x in v) / len(v), "abstention_rate": sum(x[1] for x in v) / len(v)} for g, v in sorted(by.items())}
        for value in out["groups"].values():
            value["event_gate"] = "confirmatory" if value["target_events"] >= 100 else "exploratory" if value["target_events"] >= 50 else "descriptive_only"
    return out
def _conformal_metrics_indexed(rows: Sequence[Mapping[str, Any]], indices: Sequence[int], coverage: str, dimension: str | None) -> dict[str, Any]:
    """Index-based equivalent of _conformal_metrics without row copies."""
    selected = [i for i in indices if rows[i]["label"].get("leaf") in LEAVES]
    included, abstained = [], []
    for i in selected:
        r = rows[i]; record = r["prediction"]["conformal"][coverage] if dimension is None else r["prediction"]["conformal"][coverage].get(dimension)
        if not isinstance(record, Mapping) or not isinstance(record.get("set"), list) or not isinstance(record.get("abstain"), bool): raise RuntimeError("missing conformal risk set")
        if dimension is not None and str(record.get("group")) != str(r[dimension]): raise RuntimeError("conformal subgroup key mismatch")
        risk_set = set(record["set"])
        if not risk_set.issubset(MODEL_LEAVES): raise RuntimeError("adapter conformal model leaf contract")
        included.append(REPORT_TO_MODEL[r["label"]["leaf"]] in risk_set); abstained.append(record["abstain"])
    if not selected: return {"n": 0, "status": "undefined"}
    target_events = sum(rows[i]["label"]["leaf"] != "none" for i in selected)
    out = {"n": len(selected), "target_events": target_events, "marginal_coverage": sum(included) / len(included), "abstention_rate": sum(abstained) / len(abstained), "event_gate": "confirmatory" if target_events >= 100 else "exploratory" if target_events >= 50 else "descriptive_only"}
    if dimension is not None:
        by: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for i, ok, abstain in zip(selected, included, abstained): by[str(rows[i][dimension])].append((ok, abstain))
        out["groups"] = {g: {"n": len(v), "target_events": sum(rows[i]["label"]["leaf"] != "none" for i in selected if str(rows[i][dimension]) == g), "coverage": sum(x[0] for x in v) / len(v), "abstention_rate": sum(x[1] for x in v) / len(v)} for g, v in sorted(by.items())}
        for value in out["groups"].values(): value["event_gate"] = "confirmatory" if value["target_events"] >= 100 else "exploratory" if value["target_events"] >= 50 else "descriptive_only"
    return out
def evaluate(rows: Sequence[Mapping[str, Any]], ev: Mapping[str, Any], reps: int) -> dict[str, Any]:
    if reps < 1: raise RuntimeError("bootstrap replicates")
    # Bootstrap previously copied every selected row and nested label mapping on
    # every replicate.  Keep the exact subject-cluster draw order, but operate
    # on integer row indices and primitive arrays to bound transient memory.
    rr = list(rows)
    def one_indices(indices):
        endpoints = {}
        for name, spec in ev["endpoints"].items():
            field = spec.get("label_field", name); prob = spec["probability_field"]
            yp = [(r["label"].get(field), float(r["prediction"][prob])) for r in (rr[i] for i in indices)]
            yp = [(int(y), p) for y, p in yp if y in (0, 1)]
            endpoints[name] = _metrics_arrays([y for y, _ in yp], [p for _, p in yp], spec)
        bounds = {}
        for name, value in (("all_algorithm_unknown_planned", 0), ("all_algorithm_unknown_unplanned", 1)):
            altered = {}
            for n, s in ev["endpoints"].items():
                field = s.get("label_field", n); prob = s["probability_field"]
                yp = []
                for i in indices:
                    r = rr[i]; y = value if field == "any_readmission" and r["label"].get("label_status") == "algorithm_unknown" else r["label"].get(field)
                    if y in (0, 1): yp.append((int(y), float(r["prediction"][prob])))
                altered[n] = _metrics_arrays([y for y, _ in yp], [p for _, p in yp], s)
            bounds[name] = altered
        conformal = {cov: {"global": _conformal_metrics_indexed(rr, indices, cov, None), "sex": _conformal_metrics_indexed(rr, indices, cov, "sex"), "age_group": _conformal_metrics_indexed(rr, indices, cov, "age_group")} for cov in ("80", "90", "95")}
        return {"endpoints": endpoints, "algorithm_unknown_extreme_bounds": bounds, "conformal": conformal}
    base = one_indices(list(range(len(rr)))); by: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rr): by[str(r["subject_id"])].append(i)
    rng, values, subjects = random.Random(20260918), defaultdict(list), sorted(by)
    def scalars(value: object, prefix: str = ""):
        if isinstance(value, Mapping):
            for key, item in value.items(): yield from scalars(item, prefix + ("." if prefix else "") + str(key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value): yield prefix, float(value)
    for _ in range(reps):
        x = one_indices([i for _ in subjects for i in by[rng.choice(subjects)]])
        # Includes endpoint metrics, unknown extreme bounds, and every defined
        # global/sex/age conformal coverage and abstention estimand.
        for name, value in scalars(x): values[name].append(value)
    ci = {k: {"defined_replicates": len(v), "percentile_95_ci": [sorted(v)[int(.025 * (len(v)-1))], sorted(v)[int(.975 * (len(v)-1))]]} for k, v in values.items() if v}
    return {**base, "subject_cluster_bootstrap": {"unit": "subject_id", "replicates": reps, "estimands": ci}}
def publish(out: Path, rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    if out.exists(): raise RuntimeError("refuse overwrite publication")
    stage = Path(tempfile.mkdtemp(prefix=".mimic-v7-", dir=out.parent))
    try:
        (stage / "restricted").mkdir(); (stage / "public").mkdir(); rp = stage / "restricted" / "predictions.jsonl"; rp.write_text("".join(stable(r).strip() + "\n" for r in rows), encoding="utf-8")
        schema = {"top_level": sorted({key for row in rows for key in row}), "features": sorted({key for row in rows for key in row.get("features", {})}), "prediction": sorted({key for row in rows for key in row.get("prediction", {})}), "label": sorted({key for row in rows for key in row.get("label", {})})}
        (stage / "restricted" / "manifest.json").write_text(stable({**manifest, "restricted_prediction_sha256": sha(rp), "row_count": len(rows), "restricted_row_field_schema": schema}), encoding="utf-8")
        n, events = len(rows), sum(r["label"].get("any_readmission") == 1 for r in rows); (stage / "public" / "aggregate.json").write_text(stable({"status": "PUBLIC_AGGREGATE_ONLY", "row_level_released": False, "n": "SUPPRESSED_N_LE_10" if n <= 10 else n, "events": "SUPPRESSED_N_LE_10" if events <= 10 else events, "same_system_only": True}), encoding="utf-8")
        os.replace(stage, out)
    except Exception: shutil.rmtree(stage, ignore_errors=True); raise
def _bootstrap_replicates(method: Mapping[str, Any], requested: int | None) -> int:
    if requested is None: return 1000
    if requested < 1 or os.environ.get("AP_CLAIMS_TEST_ONLY") != "1": raise RuntimeError("test-only bootstrap requires AP_CLAIMS_TEST_ONLY=1")
    # This checks only the already-read method lock, not the archive itself.
    if method["source"].get("archive_sha256") == REAL_MIMIC_ARCHIVE_SHA: raise RuntimeError("test-only bootstrap forbidden for the registered real MIMIC archive")
    return requested
def run(method: Path, execution: Path, archive: Path, out: Path, reps: int = 1000, *, test_only_bootstrap_replicates: int | None = None) -> dict[str, Any]:
    if reps != 1000: raise RuntimeError("formal production run requires exactly 1000 subject bootstrap replicates")
    x = prearchive(method, execution)  # No archive handle exists before this line returns.
    reps = _bootstrap_replicates(x["method"], test_only_bootstrap_replicates)
    archive_id = verify_archive(archive, x["method"]); t = tables(archive, x["method"]); dxv, prv = vocab(Path(x["bindings"]["dx_vocabulary"]["path"])), vocab(Path(x["bindings"]["pr_vocabulary"]["path"]))
    dx_index, pr_index = _index_by_hadm(t["diagnoses_icd"]), _index_by_hadm(t["procedures_icd"])
    admissions, admission_audit = valid_admissions(t["admissions"])
    idx, eps, people = eligible(admissions, t["patients"], dx_index); eps_by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode_row in eps: eps_by_subject[str(episode_row["subject_id"])].append(episode_row)
    predict, classify, cause = load_adapter(x), classifier(dx_index, pr_index, x["stage2"], x["pra2022"]), hierarchy(Path(x["bindings"]["ontology_labels"]["path"]))
    def infer_one(e: Mapping[str, Any]) -> dict[str, Any]:
        f = features(e, eps, people, dx_index, pr_index, dxv, prv, lambda z: cause(dx_index, z), eps_by_subject); validate_feature_contract(f, x["execution"]); p = dict(predict(f))
        if any(q["probability_field"] not in p for q in x["evaluation"]["endpoints"].values()) or any(str(int(c * 100)) not in p.get("conformal", {}) for c in x["evaluation"]["coverages"]): raise RuntimeError("adapter locked output")
        lab = label(e, eps, classify, lambda z: cause(dx_index, z), eps_by_subject); lab.update({"AP_specific": int(lab.get("leaf") == "AP_specific") if lab.get("any_readmission") is not None else None, "biliary": int(lab.get("leaf") == "biliary") if lab.get("any_readmission") is not None else None, "sepsis_or_acute_organ_dysfunction": int(lab.get("leaf") == "sepsis_or_acute_organ_dysfunction") if lab.get("any_readmission") is not None else None, "prolonged_los": int(f["LOS"] > 7), "in_hospital_death": 0})
        age_group = "18-44" if e["age"] < 45 else "45-64" if e["age"] < 65 else "65-74" if e["age"] < 75 else "75+"
        return {"subject_id": e["subject_id"], "hadm_ids": list(e["hadm_ids"]), "anchor": e["end"].isoformat(), "sex": str(int(f["FEMALE"])), "sex_report": "female" if f["FEMALE"] else "male", "age_group": age_group, "features": f, "prediction": p, "label": lab}
    rows = []
    if idx:
        # Warm the adapter's read-only model cache before concurrent calls.
        rows.append(infer_one(idx[0]))
        if len(idx) > 1:
            with ThreadPoolExecutor(max_workers=min(12, len(idx) - 1)) as pool:
                rows.extend(pool.map(infer_one, idx[1:]))
    result = evaluate(rows, x["evaluation"], reps); manifest = {"status": "PASS_RESTRICTED_LOCKED_MIMIC_TRANSFER_V7", "created_at": datetime.now(timezone.utc).isoformat(), "method_lock_sha256": sha(method), "execution_lock_sha256": sha(execution), "archive_identity": archive_id, "runtime_sha256": sha(Path(__file__)), "binding_identities": x["bindings"], "model_id": MODEL_ID, "model_artifact_identities": {key: x["bindings"][key] for key in ARTIFACTS}, "adapter_code_identity": x["bindings"]["adapter"], "inference_adapter": x["adapter"], "evaluation": result, "admission_data_quality_audit": admission_audit, "mimic_fit": False, "mimic_recalibration": False, "vocabulary_growth": False, "public_release": "aggregate_only"}; publish(out, rows, manifest); return {"status": manifest["status"], "output_dir": str(out.resolve()), "rows": len(rows), "admission_data_quality_audit": admission_audit}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--method-lock", type=Path, required=True); p.add_argument("--validate-only", action="store_true"); p.add_argument("--freeze-execution-lock", action="store_true"); p.add_argument("--bindings", type=Path); p.add_argument("--execution-lock-out", type=Path); p.add_argument("--execution-lock", type=Path); p.add_argument("--archive", type=Path); p.add_argument("--output-dir", type=Path); p.add_argument("--test-only-bootstrap-replicates", type=int)
    a = p.parse_args()
    if a.validate_only:
        if a.freeze_execution_lock or a.archive or a.bindings or a.execution_lock_out or a.output_dir: p.error("validate-only forbids archive/freeze/output arguments")
        print(stable(validate_only(a.method_lock, a.execution_lock))); return
    if a.freeze_execution_lock:
        if not a.bindings or not a.execution_lock_out or a.execution_lock or a.archive or a.output_dir or a.test_only_bootstrap_replicates is not None: p.error("freeze requires bindings and execution-lock-out only")
        print(stable(freeze(a.method_lock, a.bindings, a.execution_lock_out))); return
    if not a.execution_lock or not a.archive or not a.output_dir: p.error("formal run requires execution-lock, archive and output-dir")
    print(stable(run(a.method_lock, a.execution_lock, a.archive, a.output_dir, test_only_bootstrap_replicates=a.test_only_bootstrap_replicates)))
if __name__ == "__main__": main()
