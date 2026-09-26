#!/usr/bin/env python3
"""Freeze global and Mondrian hierarchical conformal risk-set calibrators on 2021B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LEAVES = ("none", "ap", "biliary", "sepsis_or_organ", "other")
PROBABILITY_COLUMNS = tuple(f"p_leaf_{x}" for x in LEAVES)
MONDRIAN_DIMENSIONS = ("sex", "age_group", "payer", "zip_income_quartile")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def validate_operating_point(directory: Path) -> dict[str, object]:
    directory = directory.resolve()
    manifest_path = directory / "transformer_operating_point_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A"
            or manifest.get("selection_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError("Transformer operating point is not sealed on 2021A")
    artifacts = manifest.get("artifacts", {})
    threshold_path = directory / "transformer_operating_thresholds_2021A.json"
    identity = artifacts.get(threshold_path.name, {})
    if (not threshold_path.is_file() or identity.get("sha256") != sha256(threshold_path)
            or int(identity.get("bytes", -1)) != threshold_path.stat().st_size):
        raise RuntimeError("Transformer operating-threshold artifact identity mismatch")
    thresholds = read_json(threshold_path)
    endpoint = thresholds.get("any_readmission", {})
    probability_column = endpoint.get("probability_column")
    threshold = endpoint.get("threshold")
    if probability_column != "p_any_readmission_calibrated" or not isinstance(threshold, (int, float)):
        raise RuntimeError("Missing locked calibrated any-readmission operating point")
    return {
        "probability_column": probability_column,
        "threshold": float(threshold),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "threshold_file": str(threshold_path),
        "threshold_sha256": sha256(threshold_path),
    }


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        raise ValueError("No calibration scores")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - alpha))))
    return float(np.partition(scores, rank - 1)[rank - 1])


def normalize_probabilities(frame: pd.DataFrame) -> np.ndarray:
    p = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid class probabilities")
    row_sum = p.sum(axis=1)
    if (row_sum <= 0).any():
        raise ValueError("Probability row with nonpositive mass")
    p = p / row_sum[:, None]
    return p


def make_groups(frame: pd.DataFrame) -> dict[str, pd.Series]:
    age = pd.cut(frame["AGE"], bins=[17, 44, 64, 74, np.inf], labels=["18-44", "45-64", "65-74", "75+"])
    return {
        "sex": frame["FEMALE"].astype("Int64").astype(str),
        "age_group": age.astype(str),
        "payer": frame["PAY1"].astype("Int64").astype(str),
        "zip_income_quartile": frame["ZIPINC_QRTL"].astype("Int64").astype(str),
    }


def risk_sets(probabilities: np.ndarray, q: float) -> np.ndarray:
    sets = probabilities >= (1.0 - q)
    # Including argmax only enlarges the set, so finite-sample coverage cannot decrease.
    empty = ~sets.any(axis=1)
    if empty.any():
        sets[np.flatnonzero(empty), probabilities[empty].argmax(axis=1)] = True
    return sets


def mondrian_risk_sets(
    probabilities: np.ndarray,
    scores: np.ndarray,
    groups: pd.Series | np.ndarray,
    alpha: float,
    global_q: float,
    min_rows: int,
) -> tuple[np.ndarray, dict[str, dict[str, float | int | str]]]:
    """Apply a group-specific conformal q to every row in a Mondrian dimension.

    Groups below ``min_rows`` intentionally reuse the already-computed global q.
    The returned entries are the exact audit records later written to the
    calibrator manifest; in particular, the status distinguishes calibration
    from the small-group fallback.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    if len(scores) != len(probabilities):
        raise ValueError("scores and probabilities must have the same number of rows")
    values = pd.Series(groups, copy=False).astype(str)
    if len(values) != len(probabilities):
        raise ValueError("groups and probabilities must have the same number of rows")

    sets = np.empty_like(probabilities, dtype=bool)
    entries: dict[str, dict[str, float | int | str]] = {}
    for group_value in sorted(pd.unique(values).tolist(), key=str):
        group_value = str(group_value)
        mask = values.eq(group_value).to_numpy()
        n_rows = int(mask.sum())
        if n_rows >= min_rows:
            q = finite_sample_quantile(scores[mask], alpha)
            status = "CALIBRATED"
        else:
            q = float(global_q)
            status = "GLOBAL_FALLBACK_SMALL_CALIBRATION_GROUP"
        sets[mask] = risk_sets(probabilities[mask], q)
        entries[group_value] = {"n": n_rows, "q": float(q), "status": status}
    return sets, entries


def bitmask(sets: np.ndarray) -> np.ndarray:
    weights = (1 << np.arange(sets.shape[1], dtype=np.uint8)).astype(np.uint8)
    return (sets.astype(np.uint8) * weights).sum(axis=1).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-2021b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transformer-operating-point-dir", type=Path, required=True)
    parser.add_argument("--min-mondrian-rows", type=int, default=200)
    args = parser.parse_args()
    source = args.predictions_2021b.resolve()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    operating_point = validate_operating_point(args.transformer_operating_point_dir)
    binary_probability_column = str(operating_point["probability_column"])
    any_readmission_threshold = float(operating_point["threshold"])
    required = ["year", "analysis_partition", "encounter_hash", "patient_hash", "readmission_leaf",
                "AGE", "FEMALE", "PAY1", "ZIPINC_QRTL", binary_probability_column, *PROBABILITY_COLUMNS]
    frame = pq.read_table(source, columns=required).to_pandas()
    if set(frame["year"].unique()) != {2021} or set(frame["analysis_partition"].unique()) != {"2021B"}:
        raise RuntimeError("Conformal input is not exclusively 2021B")
    if frame["encounter_hash"].duplicated().any():
        raise RuntimeError("Duplicate conformal encounters")
    y = frame["readmission_leaf"].to_numpy(dtype=np.int64)
    if (y < 0).any() or (y >= len(LEAVES)).any():
        raise RuntimeError("Invalid hierarchical leaf label")
    p = normalize_probabilities(frame)
    scores = 1.0 - p[np.arange(len(y)), y]
    groups = make_groups(frame)
    if tuple(groups) != MONDRIAN_DIMENSIONS:
        raise RuntimeError("Mondrian dimensions are not stable")
    alphas = (0.10, 0.20, 0.05)
    calibration = {"global": {}, "mondrian": {}}
    output = frame[["year", "encounter_hash", "patient_hash"]].copy()
    output_fields = {"global": {}, "mondrian": {}}
    for dimension, values in groups.items():
        output[f"group_{dimension}"] = values.astype(str).to_numpy()
        output_fields["mondrian"][dimension] = {"group_column": f"group_{dimension}", "levels": {}}
    for alpha in alphas:
        label = str(int(round((1 - alpha) * 100)))
        q = finite_sample_quantile(scores, alpha)
        calibration["global"][label] = {"alpha": alpha, "n": len(scores), "q": q}
        sets = risk_sets(p, q)
        output[f"leaf_set_mask_{label}"] = bitmask(sets)
        output[f"leaf_set_size_{label}"] = sets.sum(axis=1).astype(np.int8)
        output[f"includes_readmission_parent_{label}"] = sets[:, 1:].any(axis=1)
        output[f"includes_root_{label}"] = True
        output[f"ancestor_closed_{label}"] = True
        if label == "90":
            singleton = sets.sum(axis=1) == 1
            predicted_leaf = sets.argmax(axis=1)
            conflict = ((predicted_leaf == 0) & (frame[binary_probability_column].to_numpy() >= any_readmission_threshold)) | (
                (predicted_leaf > 0) & (frame[binary_probability_column].to_numpy() < any_readmission_threshold)
            )
            output["abstain_90"] = ~(singleton & ~conflict)
            output["binary_hierarchy_conflict_90"] = conflict
        for dimension, values in groups.items():
            mondrian_sets, entries = mondrian_risk_sets(
                p, scores, values, alpha, global_q=q, min_rows=args.min_mondrian_rows
            )
            for group_value, entry in entries.items():
                mask = values.astype(str).eq(group_value).to_numpy()
                entry["events_any_readmission"] = int((y[mask] > 0).sum())
            calibration["mondrian"].setdefault(dimension, {})[label] = entries
            output[f"leaf_set_mask_{dimension}_{label}"] = bitmask(mondrian_sets)
            output[f"leaf_set_size_{dimension}_{label}"] = mondrian_sets.sum(axis=1).astype(np.int8)
            output[f"includes_readmission_parent_{dimension}_{label}"] = mondrian_sets[:, 1:].any(axis=1)
            output[f"includes_root_{dimension}_{label}"] = True
            output[f"ancestor_closed_{dimension}_{label}"] = True
            output_fields["mondrian"][dimension]["levels"][label] = {
                "leaf_set_mask": f"leaf_set_mask_{dimension}_{label}",
                "leaf_set_size": f"leaf_set_size_{dimension}_{label}",
                "includes_readmission_parent": f"includes_readmission_parent_{dimension}_{label}",
                "includes_root": f"includes_root_{dimension}_{label}",
                "ancestor_closed": f"ancestor_closed_{dimension}_{label}",
                "q_source": "group q when calibration rows meet min_mondrian_rows; otherwise global q",
            }
        output_fields["global"][label] = {
            "leaf_set_mask": f"leaf_set_mask_{label}",
            "leaf_set_size": f"leaf_set_size_{label}",
            "includes_readmission_parent": f"includes_readmission_parent_{label}",
            "includes_root": f"includes_root_{label}",
            "ancestor_closed": f"ancestor_closed_{label}",
            "q_source": "global finite_sample_quantile over all 2021B rows",
        }

    prediction_path = out / "conformal_sets_2021B.parquet"
    pq.write_table(pa.Table.from_pandas(output, preserve_index=False), prediction_path, compression="zstd", compression_level=6)
    calibrator_path = out / "conformal_calibrator.json"
    manifest = {
        "status": "PASS_LOCKED_PRE_2022", "method": "split conformal class-probability threshold",
        "hierarchy": {"root": "any index stay", "parent": "readmission", "leaves": list(LEAVES)},
        "ancestor_closure": True, "calibration_partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False,
        "nominal_coverages": [0.90, 0.80, 0.95], "calibration": calibration,
        "risk_set_outputs": output_fields,
        "mondrian_fallback": {
            "min_calibration_rows": args.min_mondrian_rows,
            "rule": "Groups below min_calibration_rows use the corresponding global q; all groups retain an output row.",
        },
        "abstention_rule": "90% leaf set not singleton or binary/leaf hierarchy conflict",
        "any_readmission_threshold_from_2021A": any_readmission_threshold,
        "any_readmission_probability_column": binary_probability_column,
        "operating_point_source": operating_point,
        "confirmatory_subgroup_event_gate": ">=100 events; 50-99 exploratory; <50 descriptive (applied only at evaluation)",
        "source": {"file": source.name, "bytes": source.stat().st_size, "sha256": sha256(source)},
        "artifact": {"file": prediction_path.name, "bytes": prediction_path.stat().st_size, "sha256": sha256(prediction_path)},
        "year_2022_accessed": False,
    }
    calibrator_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
