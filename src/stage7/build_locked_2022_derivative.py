#!/usr/bin/env python3
"""Build the one immutable, minimum 2022 AP model/evaluation derivative.

This program is deliberately not an ETL framework.  It accepts exactly one
already-built 2022 AP episode parquet and its same-year history parquet, and
can only copy the frozen projection named by a SHA256-locked derivative
specification.  It neither learns nor changes an ontology, token vocabulary,
threshold, imputation rule, normalisation state, or eligibility rule.

Lock order is intentional: :func:`validate_preflight` verifies the immutable
pre-2022 unlock lock, its sidecar, this exact script path/byte count/SHA256,
and (for a production run) the frozen projection specification *before any
2022 input is opened*.  ``--validate-only`` performs only that preflight, so
it is safe to run while 2022 remains sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "stage7_locked_2022_minimum_derivative_v1"
OUTCOME_KEYS = (
    "any_readmission",
    "ap_specific_readmission",
    "biliary_event",
    "sepsis_or_organ_complication",
    "high_cost",
    "prolonged_los",
    "in_hospital_death",
)
IDENTITY_COLUMNS = (
    "encounter_hash",
    "patient_hash",
    "hospital_hash",
    "NRD_STRATUM",
    "DISCWT",
    "AGE",
    "FEMALE",
    "PAY1",
    "ZIPINC_QRTL",
)
HISTORY_KEYS = ("encounter_hash", "patient_hash", "analysis_partition")
# Superset of the frozen Stage5 Transformer static inputs.  Structural
# ablations may use a subset, but the sealed derivative must make every input
# of the selected frozen run available without ever substituting an outcome.
STAGE5_STATIC_INPUTS = (
    "AGE", "LOS", "I10_NDX", "I10_NPR", "CCR_NRD", "WAGEINDEX",
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d",
    "prior_max_severity_180d", "prior_max_mortality_risk_180d", "days_since_prior_discharge",
    "APRDRG_Severity", "APRDRG_Risk_Mortality", "AWEEKEND", "DMONTH", "ELECTIVE",
    "FEMALE", "HCUP_ED", "PAY1", "PL_NCHS", "RESIDENT", "ZIPINC_QRTL", "HOSP_BEDSIZE",
    "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH", "history_30d_fully_observable",
    "history_90d_fully_observable", "history_180d_fully_observable",
)


def sha256(path: Path) -> str:
    """Return a streaming SHA256 for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Missing required file: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def _listed_identity(lock: dict[str, Any], path: Path) -> bool:
    """True only for an exact frozen path, length, and digest."""
    resolved = path.resolve()
    entry = lock.get("frozen_code", {}).get(str(resolved))
    return bool(
        isinstance(entry, dict)
        and resolved.is_file()
        and entry.get("sha256") == sha256(resolved)
        and int(entry.get("bytes", -1)) == resolved.stat().st_size
    )


def validate_preflight(unlock_lock: Path, frozen_spec: Path | None = None) -> dict[str, Any]:
    """Validate lock/self identity without inspecting an episode/history input."""
    unlock_lock = unlock_lock.resolve()
    if not unlock_lock.is_file():
        raise RuntimeError("Missing pre-2022 unlock lock")
    sidecar = unlock_lock.with_suffix(unlock_lock.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("Missing immutable pre-2022 unlock SHA256 sidecar")
    expected_sidecar = f"{sha256(unlock_lock)}  {unlock_lock.name}"
    if sidecar.read_text(encoding="ascii").strip() != expected_sidecar:
        raise RuntimeError("Pre-2022 unlock lock SHA256 sidecar mismatch")
    lock = read_json_object(unlock_lock)
    if (
        lock.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
        or lock.get("sealed_test_year") != 2022
        or lock.get("2022_access_before_lock") is not False
    ):
        raise RuntimeError("Pre-2022 unlock lock is not authorized for one 2022 primary evaluation")
    script = Path(__file__).resolve()
    if not _listed_identity(lock, script):
        raise RuntimeError("Derivative script path/bytes/SHA256 are not frozen in the unlock lock")
    if frozen_spec is not None and not _listed_identity(lock, frozen_spec.resolve()):
        raise RuntimeError("Frozen 2022 derivative specification is not hash-locked in the unlock lock")
    return lock


def _strings(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and (allow_empty or item) for item in value):
        raise RuntimeError(f"Frozen derivative spec requires a list of non-empty strings: {name}")
    if len(value) != len(set(value)):
        raise RuntimeError(f"Frozen derivative spec has duplicate fields: {name}")
    return tuple(value)


def validate_derivative_spec(spec_path: Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen copy-only projection and its no-adaptation contract."""
    spec_path = spec_path.resolve()
    if not _listed_identity(lock, spec_path):
        raise RuntimeError("Frozen 2022 derivative specification is not hash-locked in the unlock lock")
    spec = read_json_object(spec_path)
    if spec.get("status") != "FROZEN_2022_DERIVATIVE_SPEC" or spec.get("sealed_test_year") != 2022:
        raise RuntimeError("Derivative specification is not frozen for the 2022 temporal test")
    if spec.get("test_partition") != "test":
        raise RuntimeError("Derivative specification must freeze test_partition='test'")
    if spec.get("data_dependent_adaptation") is not False:
        raise RuntimeError("Derivative specification must explicitly prohibit data-dependent adaptation")

    episode_columns = _strings(spec.get("episode_columns"), "episode_columns")
    history_columns = _strings(spec.get("history_columns"), "history_columns")
    token_columns = _strings(spec.get("token_sequence_columns"), "token_sequence_columns")
    static_columns = _strings(spec.get("static_columns"), "static_columns")
    for name in ("year_column", "partition_column", "eligibility_column", "leaf_column"):
        if not isinstance(spec.get(name), str) or not spec[name]:
            raise RuntimeError(f"Derivative specification missing {name}")
    if spec.get("year_column") != "year" or spec.get("partition_column") != "analysis_partition":
        raise RuntimeError("Derivative specification must retain the fixed year and partition fields")
    if spec.get("eligibility_value") is not True:
        raise RuntimeError("Derivative specification must freeze primary eligibility to true")
    outcomes = spec.get("outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != set(OUTCOME_KEYS):
        raise RuntimeError("Derivative specification must name exactly the seven frozen outcome labels")
    if not all(isinstance(column, str) and column for column in outcomes.values()):
        raise RuntimeError("Derivative specification has an invalid outcome source column")
    if len(set(outcomes.values())) != len(outcomes):
        raise RuntimeError("Derivative specification reuses an outcome source column")
    protected_episode_fields = set(IDENTITY_COLUMNS) | {
        spec["year_column"], spec["partition_column"], spec["eligibility_column"], spec["leaf_column"],
    } | set(token_columns) | set(static_columns)
    if protected_episode_fields & set(outcomes.values()):
        raise RuntimeError("Outcome source fields may not overlap frozen prediction/evaluation inputs")
    if tuple(history_columns[: len(HISTORY_KEYS)]) != HISTORY_KEYS:
        raise RuntimeError("History projection must begin with encounter_hash, patient_hash, analysis_partition")
    if any(column not in episode_columns for column in IDENTITY_COLUMNS):
        raise RuntimeError("Episode projection lacks required identity/evaluation fields")
    required_episode = set(IDENTITY_COLUMNS) | {
        spec["year_column"], spec["partition_column"], spec["eligibility_column"], spec["leaf_column"],
    } | set(token_columns) | set(static_columns) | set(outcomes.values())
    if set(episode_columns) != required_episode:
        raise RuntimeError("Episode projection is not the exact frozen minimum field set")
    if any(column in episode_columns for column in history_columns[len(HISTORY_KEYS):]):
        raise RuntimeError("History features must be supplied only by the history parquet")
    if not token_columns:
        raise RuntimeError("Derivative specification must retain frozen token/sequence inputs")
    available_inputs = set(IDENTITY_COLUMNS) | set(static_columns) | set(history_columns[len(HISTORY_KEYS):])
    missing_stage5_inputs = set(STAGE5_STATIC_INPUTS) - available_inputs
    if missing_stage5_inputs:
        raise RuntimeError(f"Derivative specification lacks frozen Stage5 static inputs: {sorted(missing_stage5_inputs)}")
    if set(STAGE5_STATIC_INPUTS) & set(outcomes.values()):
        raise RuntimeError("Frozen Stage5 static input contract illegally includes an outcome label")
    return spec


def _binary(frame: pd.DataFrame, columns: tuple[str, ...], *, allow_missing: set[str] | None = None) -> None:
    allow_missing = allow_missing or set()
    for column in columns:
        value = frame[column]
        if column in allow_missing:
            value = value.dropna()
        if value.isna().any() or not value.isin((0, 1, False, True)).all():
            raise RuntimeError(f"Frozen outcome label is not fully observed binary: {column}")


def _nonempty_key(frame: pd.DataFrame, column: str, source: str) -> None:
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        raise RuntimeError(f"{source} has missing {column}")


def _validate_inputs(episodes: pd.DataFrame, history: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Apply only frozen structural/eligibility checks and a strict 1:1 join."""
    year_column, partition_column = spec["year_column"], spec["partition_column"]
    for column in ("encounter_hash", "patient_hash", "hospital_hash"):
        _nonempty_key(episodes, column, "episode parquet")
    for column in ("encounter_hash", "patient_hash"):
        _nonempty_key(history, column, "history parquet")
    if episodes["encounter_hash"].duplicated().any():
        raise RuntimeError("Episode parquet has duplicate encounter_hash")
    if history["encounter_hash"].duplicated().any():
        raise RuntimeError("History parquet has duplicate encounter_hash")
    years = pd.to_numeric(episodes[year_column], errors="coerce")
    if years.isna().any() or not years.eq(2022).all():
        raise RuntimeError("Episode parquet contains non-2022 or mixed years")
    if episodes[partition_column].isna().any() or not episodes[partition_column].astype(str).eq("test").all():
        raise RuntimeError("Episode parquet contains non-test or mixed partitions")
    if history["analysis_partition"].isna().any() or not history["analysis_partition"].astype(str).eq("test").all():
        raise RuntimeError("History parquet contains non-test or mixed partitions")
    eligibility = episodes[spec["eligibility_column"]]
    if eligibility.isna().any() or not eligibility.isin((True, 1)).all():
        raise RuntimeError("Episode parquet violates the frozen primary eligibility rule")
    _binary(episodes, tuple(spec["outcomes"].values()), allow_missing={spec["outcomes"]["high_cost"]})
    leaf = pd.to_numeric(episodes[spec["leaf_column"]], errors="coerce")
    if leaf.isna().any() or not leaf.isin((0, 1, 2, 3, 4)).all():
        raise RuntimeError("Episode parquet has an invalid frozen primary leaf")
    any_label = episodes[spec["outcomes"]["any_readmission"]].astype(bool)
    if not np.array_equal(any_label.to_numpy(), (leaf.to_numpy(dtype=np.int8) > 0)):
        raise RuntimeError("Frozen any-readmission label and primary leaf are inconsistent")
    cause_sources = tuple(spec["outcomes"][key] for key in (
        "ap_specific_readmission", "biliary_event", "sepsis_or_organ_complication",
    ))
    cause_sum = episodes.loc[:, cause_sources].astype(np.int8).sum(axis=1).to_numpy()
    if (cause_sum > 1).any():
        raise RuntimeError("Frozen cause-specific outcome labels are not mutually exclusive")
    if not np.array_equal(cause_sum, np.isin(leaf.to_numpy(dtype=np.int8), (1, 2, 3)).astype(np.int64)):
        raise RuntimeError("Frozen cause-specific labels and primary leaf are inconsistent")
    merged = episodes.merge(
        history,
        on="encounter_hash",
        how="left",
        suffixes=("", "__history"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("Episode/history join has missing encounter_hash matches")
    if not (merged["patient_hash"] == merged["patient_hash__history"]).all():
        raise RuntimeError("Episode/history join has patient_hash mismatch")
    if not merged["analysis_partition__history"].astype(str).eq("test").all():
        raise RuntimeError("Episode/history join has partition mismatch")
    return merged.drop(columns=["_merge", "patient_hash__history", "analysis_partition__history"])


def _output_columns(spec: dict[str, Any]) -> list[str]:
    """The exact minimal frozen output projection, with no convenience fields."""
    return list(dict.fromkeys((
        "analysis_partition", *IDENTITY_COLUMNS,
        spec["year_column"], spec["leaf_column"],
        *spec["outcomes"].values(), *spec["token_sequence_columns"],
        *spec["static_columns"], *spec["history_columns"][len(HISTORY_KEYS):],
    )))


def _prepare_output(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    partials = list(output_dir.parent.glob(output_dir.name + ".partial*")) if output_dir.parent.exists() else []
    if output_dir.exists() or partials:
        raise RuntimeError("2022 derivative output or partial output already exists and is immutable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))


def _publish_derivative(frame: pd.DataFrame, temporary: Path, output_dir: Path,
                        episode_path: Path, history_path: Path, unlock_path: Path,
                        spec_path: Path) -> dict[str, Any]:
    derivative_path = temporary / "locked_2022_derivative.parquet"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, derivative_path, compression="zstd", compression_level=6,
                   row_group_size=131072, use_dictionary=True, write_statistics=True)
    output_identity = identity(derivative_path)
    manifest = {
        "status": "PASS_LOCKED_2022_MINIMUM_DERIVATIVE",
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "2022_accessed": True,
        "one_shot": True,
        "data_dependent_adaptation": False,
        "row_count": int(len(frame)),
        "unique_encounter_hash": int(frame["encounter_hash"].nunique()),
        "field_list": list(frame.columns),
        "input_identity": {"episodes": identity(episode_path), "history": identity(history_path)},
        "unlock_lock": identity(unlock_path),
        "frozen_derivative_spec": identity(spec_path),
        "derivative": {"file": derivative_path.name, "bytes": output_identity["bytes"], "sha256": output_identity["sha256"]},
        "no_outcome_counts_or_small_cells_logged": True,
        "high_cost_missingness_preserved_as_mask": True,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(stable_json(manifest), encoding="utf-8")
    manifest_digest = sha256(manifest_path)
    (temporary / "manifest.json.sha256").write_text(f"{manifest_digest}  manifest.json\n", encoding="ascii")
    os.replace(temporary, output_dir.resolve())
    return {**manifest, "manifest_sha256": manifest_digest, "output": str(output_dir.resolve())}


def build_locked_2022_derivative(episodes_2022: Path, history_2022: Path, unlock_lock: Path,
                                 frozen_spec: Path, output_dir: Path, *, threads: int = 8) -> dict[str, Any]:
    """Open 2022 exactly once after lock-first validation and publish atomically."""
    if not 1 <= threads <= 8:
        raise RuntimeError("--threads must be between 1 and 8")
    # No operation on either 2022 input occurs before these checks return.
    lock = validate_preflight(unlock_lock, frozen_spec)
    spec = validate_derivative_spec(frozen_spec, lock)
    temporary = _prepare_output(output_dir)
    try:
        # These checks/read calls are deliberately after every pre-2022 lock gate.
        episodes_2022 = episodes_2022.resolve()
        history_2022 = history_2022.resolve()
        if not episodes_2022.is_file() or not history_2022.is_file():
            raise RuntimeError("Both explicit 2022 episode and history parquet inputs are required")
        pa.set_cpu_count(threads)
        episodes = pq.read_table(episodes_2022, columns=list(spec["episode_columns"])).to_pandas()
        history = pq.read_table(history_2022, columns=list(spec["history_columns"])).to_pandas()
        merged = _validate_inputs(episodes, history, spec)
        output_columns = _output_columns(spec)
        if any(column not in merged.columns for column in output_columns):
            raise RuntimeError("Frozen output projection is absent after the one-to-one join")
        derivative = merged.loc[:, output_columns].copy()
        derivative.insert(0, "analysis_year", 2022)
        # ``analysis_partition`` is retained from the episode source and must already be test.
        if list(derivative.columns).count("analysis_year") != 1:
            raise RuntimeError("Invalid duplicate standardized analysis_year output")
        return _publish_derivative(derivative, temporary, output_dir, episodes_2022, history_2022,
                                   unlock_lock.resolve(), frozen_spec.resolve())
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlock-lock", type=Path, required=True)
    parser.add_argument("--frozen-derivative-spec", type=Path)
    parser.add_argument("--episodes-2022", type=Path)
    parser.add_argument("--history-2022", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        if any(value is not None for value in (args.episodes_2022, args.history_2022, args.output_dir)):
            raise SystemExit("--validate-only accepts no 2022 input or output arguments")
        # A spec, when supplied, is hash-checked but never treated as data.
        result = validate_preflight(args.unlock_lock, args.frozen_derivative_spec)
        print(stable_json({"status": "PASS_PRE_2022_DERIVATIVE_PREFLIGHT", "sealed_test_year": result["sealed_test_year"]}))
        return
    if args.frozen_derivative_spec is None or args.episodes_2022 is None or args.history_2022 is None or args.output_dir is None:
        raise SystemExit("Production run requires --frozen-derivative-spec, --episodes-2022, --history-2022, and --output-dir")
    result = build_locked_2022_derivative(
        args.episodes_2022, args.history_2022, args.unlock_lock, args.frozen_derivative_spec,
        args.output_dir, threads=args.threads,
    )
    print(stable_json(result))


if __name__ == "__main__":
    main()
