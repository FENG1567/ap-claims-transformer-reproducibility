#!/usr/bin/env python3
"""Bootstrap evaluation-only calibration intervals for corrected 2022 predictions.

This additive, post-unblinding script estimates calibration intercept and slope
from already frozen predictions.  It neither changes predictions nor fits a
new prediction model, calibrator, threshold, ontology, or endpoint mapping.
Only aggregate output is emitted.  Run it only in the authorised NRD runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit


SEED = 20260921
EXPECTED_EPISODES = 119114
OUTCOMES = {
    "any_readmission": "outcome_any_readmission",
    "ap_specific_readmission": "outcome_ap_specific_readmission",
}
MODELS = ("joint_main", "lightgbm")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Input file is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _validate_vector(name: str, values: np.ndarray, positive: bool = False) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or not len(out) or not np.isfinite(out).all():
        raise RuntimeError(f"{name} must be a non-empty finite vector")
    if positive and np.any(out <= 0):
        raise RuntimeError(f"{name} must be positive")
    return out


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted logistic recalibration *assessment* (intercept, slope).

    The logistic fit is the conventional calibration intercept/slope diagnostic.
    It is not applied back to the predictions.  A strict finite, non-separated
    fit is required because silently reporting a partial bootstrap would
    understate uncertainty.
    """
    values = calibration_intercept_slope_matrix(y, p, weights[:, None])
    return float(values[0, 0]), float(values[1, 0])


def calibration_intercept_slope_matrix(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray, max_iter: int = 50,
) -> np.ndarray:
    """Vectorised Newton fits for one or more weighted bootstrap replicates."""
    y = _validate_vector("outcome", y)
    if not np.all((y == 0) | (y == 1)) or y.min() == y.max():
        raise RuntimeError("Outcome must contain both binary classes")
    p = _validate_vector("prediction", p)
    if len(p) != len(y) or np.any((p <= 0) | (p >= 1)):
        raise RuntimeError("Predictions must be finite probabilities strictly between zero and one")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != len(y) or not w.shape[1] or not np.isfinite(w).all() or np.any(w < 0):
        raise RuntimeError("Bootstrap weights must be a finite non-negative n-by-replicate matrix")
    if np.any(w.sum(axis=0) <= 0):
        raise RuntimeError("A bootstrap replicate has zero total weight")

    logit_p = np.log(p / (1.0 - p))[:, None]
    b0 = np.zeros(w.shape[1], dtype=float)
    b1 = np.ones(w.shape[1], dtype=float)
    converged = np.zeros(w.shape[1], dtype=bool)
    for _ in range(max_iter):
        eta = np.clip(b0[None, :] + logit_p * b1[None, :], -35.0, 35.0)
        mu = expit(eta)
        variance = np.maximum(mu * (1.0 - mu), 1e-12)
        residual = y[:, None] - mu
        g0 = np.sum(w * residual, axis=0)
        g1 = np.sum(w * residual * logit_p, axis=0)
        h00 = np.sum(w * variance, axis=0)
        h01 = np.sum(w * variance * logit_p, axis=0)
        h11 = np.sum(w * variance * logit_p * logit_p, axis=0)
        determinant = h00 * h11 - h01 * h01
        if np.any(~np.isfinite(determinant)) or np.any(determinant <= 1e-12):
            raise RuntimeError("Calibration intercept/slope is unidentified in a bootstrap replicate")
        d0 = (h11 * g0 - h01 * g1) / determinant
        d1 = (-h01 * g0 + h00 * g1) / determinant
        b0 += d0
        b1 += d1
        converged |= np.maximum(np.abs(d0), np.abs(d1)) < 1e-9
        if converged.all():
            break
    if not converged.all() or not np.isfinite(b0).all() or not np.isfinite(b1).all():
        raise RuntimeError("Calibration Newton iteration did not converge for every replicate")
    return np.vstack([b0, b1])


def make_hospital_within_stratum_bootstrap(
    frame: pd.DataFrame, reps: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return hospital-PSU multiplicities sampled independently within stratum."""
    if reps < 1:
        raise ValueError("Bootstrap replicates must be positive")
    required = {"NRD_STRATUM", "hospital_hash"}
    if missing := required - set(frame.columns):
        raise RuntimeError(f"Predictions lack bootstrap columns: {sorted(missing)}")
    psu = frame[["NRD_STRATUM", "hospital_hash"]].astype(str).drop_duplicates()
    psu = psu.sort_values(["NRD_STRATUM", "hospital_hash"], kind="mergesort").reset_index(drop=True)
    if psu.empty or (psu.NRD_STRATUM == "").any() or (psu.hospital_hash == "").any():
        raise RuntimeError("NRD bootstrap strata or hospital PSU identifiers are invalid")
    key_to_id = {(row.NRD_STRATUM, row.hospital_hash): i for i, row in psu.iterrows()}
    row_gid = np.fromiter(
        (key_to_id[(str(s), str(h))] for s, h in zip(frame.NRD_STRATUM, frame.hospital_hash)),
        dtype=np.int32, count=len(frame),
    )
    rng = np.random.default_rng(seed)
    multiplicity = np.zeros((reps, len(psu)), dtype=np.uint16)
    for _, indices in psu.groupby("NRD_STRATUM", sort=False).groups.items():
        ids = np.asarray(list(indices), dtype=np.int32)
        if len(ids) == 1:
            multiplicity[:, ids[0]] = 1
        else:
            multiplicity[:, ids] = rng.multinomial(
                len(ids), np.repeat(1.0 / len(ids), len(ids)), size=reps,
            )
    audit = {"psu_count": int(len(psu)), "stratum_count": int(psu.NRD_STRATUM.nunique())}
    return multiplicity, row_gid, audit


def calibration_bootstrap(
    y: np.ndarray, p: np.ndarray, base_weights: np.ndarray, multiplicity: np.ndarray,
    row_gid: np.ndarray, batch: int = 12,
) -> np.ndarray:
    """Paired PSU bootstrap estimates, rows = replicate and columns = intercept/slope."""
    y = _validate_vector("outcome", y)
    p = _validate_vector("prediction", p)
    base_weights = _validate_vector("DISCWT", base_weights, positive=True)
    if len(y) != len(p) or len(y) != len(base_weights) or len(row_gid) != len(y):
        raise RuntimeError("Calibration inputs have inconsistent lengths")
    if batch < 1:
        raise ValueError("batch must be positive")
    result = np.full((len(multiplicity), 2), np.nan, dtype=float)
    for start in range(0, len(multiplicity), batch):
        stop = min(start + batch, len(multiplicity))
        sampled_weights = multiplicity[start:stop, row_gid].T.astype(float) * base_weights[:, None]
        values = calibration_intercept_slope_matrix(y, p, sampled_weights)
        result[start:stop, :] = values.T
    if not np.isfinite(result).all():
        raise RuntimeError("A bootstrap calibration estimate is non-finite")
    return result


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    values = _validate_vector("bootstrap values", values)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _load_and_validate_inputs(predictions: Path, evaluation_spec: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        frame = pq.read_table(predictions).to_pandas()
    except Exception as exc:  # pyarrow uses several concrete exception classes.
        raise RuntimeError("Corrected 2022 predictions parquet is unreadable") from exc
    try:
        spec = json.loads(evaluation_spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Evaluation specification is unreadable") from exc
    if not isinstance(spec, dict) or not isinstance(spec.get("models"), dict):
        raise RuntimeError("Evaluation specification lacks its frozen model roster")
    if set(MODELS) - set(spec["models"]):
        raise RuntimeError("Evaluation specification does not bind both requested frozen models")
    required = {"NRD_STRATUM", "hospital_hash", "DISCWT", *OUTCOMES.values()}
    for outcome in OUTCOMES:
        for model in MODELS:
            required.add(f"p_cal__{outcome}__{model}")
    if missing := required - set(frame.columns):
        raise RuntimeError(f"Corrected predictions lack required columns: {sorted(missing)}")
    if len(frame) != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected corrected 2022 cohort of {EXPECTED_EPISODES:,} episodes; found {len(frame):,}")
    if frame[["NRD_STRATUM", "hospital_hash"]].isna().any().any():
        raise RuntimeError("Corrected predictions have missing bootstrap design variables")
    _validate_vector("DISCWT", pd.to_numeric(frame.DISCWT, errors="coerce").to_numpy(float), positive=True)
    for outcome, ycol in OUTCOMES.items():
        y = pd.to_numeric(frame[ycol], errors="coerce").to_numpy(float)
        if not np.isfinite(y).all() or not np.all((y == 0) | (y == 1)) or y.min() == y.max():
            raise RuntimeError(f"Invalid binary outcome in corrected predictions: {outcome}")
        for model in MODELS:
            p = pd.to_numeric(frame[f"p_cal__{outcome}__{model}"], errors="coerce").to_numpy(float)
            if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
                raise RuntimeError(f"Invalid frozen probability column: {outcome}/{model}")
    return frame, spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True,
                        help="corrected 2022 frozen prediction parquet")
    parser.add_argument("--evaluation-spec", type=Path, required=True,
                        help="immutable evaluation specification bound to the frozen model roster")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=1000,
                        help="hospital-PSU-within-NRD_STRATUM replicates (default: 1000)")
    parser.add_argument("--batch", type=int, default=12,
                        help="internal vectorisation batch; it does not alter the bootstrap draws")
    args = parser.parse_args()
    if args.reps < 1000:
        raise SystemExit("At least 1000 bootstrap replicates are required for the formal calibration intervals")
    if args.batch < 1:
        raise SystemExit("--batch must be positive")
    output = args.output_dir.resolve()
    if output.exists() or list(output.parent.glob(output.name + ".partial-*")):
        raise SystemExit("Refusing to overwrite output or collide with a prior partial output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        prediction_identity = file_identity(args.predictions)
        spec_identity = file_identity(args.evaluation_spec)
        frame, _spec = _load_and_validate_inputs(args.predictions, args.evaluation_spec)
        base_weights = pd.to_numeric(frame.DISCWT, errors="coerce").to_numpy(float)
        multiplicity, row_gid, bootstrap_audit = make_hospital_within_stratum_bootstrap(frame, args.reps, SEED)
        rows: list[dict[str, Any]] = []
        for outcome, ycol in OUTCOMES.items():
            y = pd.to_numeric(frame[ycol], errors="coerce").to_numpy(np.int8)
            for model in MODELS:
                p = pd.to_numeric(frame[f"p_cal__{outcome}__{model}"], errors="coerce").to_numpy(float)
                intercept, slope = calibration_intercept_slope(y, p, base_weights)
                boot = calibration_bootstrap(y, p, base_weights, multiplicity, row_gid, args.batch)
                intercept_low, intercept_high = percentile_ci(boot[:, 0])
                slope_low, slope_high = percentile_ci(boot[:, 1])
                rows.append({
                    "outcome": outcome,
                    "model": model,
                    "episode_count": int(len(frame)),
                    "event_count": int(y.sum()),
                    "calibration_intercept": intercept,
                    "calibration_intercept_ci_low": intercept_low,
                    "calibration_intercept_ci_high": intercept_high,
                    "calibration_slope": slope,
                    "calibration_slope_ci_low": slope_low,
                    "calibration_slope_ci_high": slope_high,
                    "bootstrap_scheme": "hospital PSU within NRD_STRATUM",
                    "bootstrap_replicates": args.reps,
                    "seed": SEED,
                    "assessment_only_no_recalibration": True,
                })
        pd.DataFrame(rows).to_csv(temp / "calibration_intercept_slope_bootstrap_ci.csv", index=False)
        summary = {
            "status": "PASS_POST_UNBLINDING_2022_CALIBRATION_INTERVALS",
            "post_unblinding": True,
            "assessment_only_no_recalibration": True,
            "frozen_predictions_modified": False,
            "models": list(MODELS),
            "outcomes": list(OUTCOMES),
            "episode_count": int(len(frame)),
            "bootstrap_scheme": "hospital PSU within NRD_STRATUM",
            "bootstrap_replicates": args.reps,
            "seed": SEED,
            "bootstrap_design": bootstrap_audit,
            "inputs": {"predictions": prediction_identity, "evaluation_spec": spec_identity},
            "limitations": [
                "The 2022 analysis is post-unblinding and cannot restore an untouched sealed test.",
                "Calibration intercept and slope are evaluation diagnostics only; no recalibration was fitted back to predictions.",
                "No 2022 result was used to change model architecture, features, calibration, thresholds, ontology, or endpoint mappings.",
            ],
        }
        (temp / "calibration_bootstrap_summary.json").write_text(stable_json(summary), encoding="utf-8")
        manifest = {
            "status": summary["status"],
            "input_hashes": summary["inputs"],
            "files": [],
        }
        for path in sorted(temp.glob("*")):
            if path.is_file():
                manifest["files"].append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        (temp / "manifest.json").write_text(stable_json(manifest), encoding="utf-8")
        os.replace(temp, output)
        print(stable_json({"status": summary["status"], "output": str(output), "files": len(manifest["files"])}))
    except Exception:
        # Preserve the uniquely named partial directory for a failure audit;
        # a subsequent invocation with the same requested output is rejected.
        raise


if __name__ == "__main__":
    main()
