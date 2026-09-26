#!/usr/bin/env python3
"""Freeze 2018--2020 all-admission auxiliary labels before pretraining.

The output is a development-only contract consumed by the pretraining runner.
It never reads 2021 or 2022 and never imputes missing outcome labels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


YEARS = (2018, 2019, 2020)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values):
        raise ValueError("no valid values for weighted quantile")
    order = np.argsort(values, kind="mergesort")
    ordered_values, ordered_weights = values[order], weights[order]
    cutoff = probability * float(ordered_weights.sum(dtype=np.float64))
    index = int(np.searchsorted(np.cumsum(ordered_weights, dtype=np.float64), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def bounded_pos_weight(positive: int, valid: int, ceiling: float = 20.0) -> float:
    if positive <= 0 or positive >= valid:
        raise ValueError(f"degenerate auxiliary label: {positive=}, {valid=}")
    return float(min(ceiling, (valid - positive) / positive))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500_000)
    args = parser.parse_args()
    root, manifest_path, output = args.root.resolve(), args.input_manifest.resolve(), args.output.resolve()
    manifest_rows = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) >= 4 and fields[1].endswith("admissions_model.parquet"):
                manifest_rows[fields[1].replace("\\", "/")] = {"bytes": int(fields[2]), "sha256": fields[3]}

    cost_parts: list[np.ndarray] = []
    cost_weight_parts: list[np.ndarray] = []
    los_count: dict[int, int] = defaultdict(int)
    los_weight: dict[int, float] = defaultdict(float)
    age_sum = age_sum_sq = 0.0
    age_n = 0
    death_valid = death_positive = 0
    total_rows = 0
    inputs = {}

    for year in YEARS:
        path = root / f"data/nrd/year={year}/admissions_model.parquet"
        rel = f"data/nrd/year={year}/admissions_model.parquet"
        registered = manifest_rows.get(rel)
        if registered is None:
            raise RuntimeError(f"input is absent from frozen manifest: {rel}")
        if path.stat().st_size != registered["bytes"]:
            raise RuntimeError(f"input size differs from frozen manifest: {rel}")
        inputs[rel] = registered
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=args.batch_size,
            columns=["cost_2021_usd", "DISCWT", "LOS", "DIED", "AGE"],
        ):
            total_rows += batch.num_rows
            columns = {name: np.asarray(batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False), dtype=np.float64)
                       for name in batch.schema.names}
            weight = columns["DISCWT"]
            cost = columns["cost_2021_usd"]
            valid_cost = np.isfinite(cost) & (cost > 0) & np.isfinite(weight) & (weight > 0)
            cost_parts.append(cost[valid_cost].astype(np.float32, copy=False))
            cost_weight_parts.append(weight[valid_cost].astype(np.float32, copy=False))

            los = columns["LOS"]
            valid_los = np.isfinite(los) & (los >= 0) & np.isfinite(weight) & (weight > 0)
            if valid_los.any():
                los_integer = los[valid_los].astype(np.int32)
                for value in np.unique(los_integer):
                    selected = los_integer == value
                    los_count[int(value)] += int(selected.sum())
                    los_weight[int(value)] += float(weight[valid_los][selected].sum(dtype=np.float64))

            died = columns["DIED"]
            valid_death = np.isfinite(died) & np.isin(died, (0, 1))
            death_valid += int(valid_death.sum())
            death_positive += int((died[valid_death] == 1).sum())

            age = columns["AGE"]
            valid_age = np.isfinite(age) & (age >= 0) & (age <= 120)
            age_values = age[valid_age]
            age_n += int(len(age_values))
            age_sum += float(age_values.sum(dtype=np.float64))
            age_sum_sq += float(np.square(age_values).sum(dtype=np.float64))

    costs = np.concatenate(cost_parts).astype(np.float64, copy=False)
    cost_weights = np.concatenate(cost_weight_parts).astype(np.float64, copy=False)
    cost_threshold = weighted_quantile(costs, cost_weights, 0.90)
    high_cost_positive = int((costs > cost_threshold).sum())
    cost_valid = int(len(costs))
    del cost_parts, cost_weight_parts, costs, cost_weights
    gc.collect()

    prolonged_valid = int(sum(los_count.values()))
    prolonged_positive = int(sum(count for value, count in los_count.items() if value > 7))
    ordered_los = sorted(los_weight)
    target = 0.90 * sum(los_weight.values())
    running = 0.0
    los_weighted_p90 = None
    for value in ordered_los:
        running += los_weight[value]
        if running >= target:
            los_weighted_p90 = float(value)
            break
    if los_weighted_p90 is None:
        raise RuntimeError("could not determine LOS weighted quantile")
    age_mean = age_sum / age_n
    age_variance = max(0.0, age_sum_sq / age_n - age_mean * age_mean)

    spec = {
        "status": "LOCKED_DEVELOPMENT_ONLY_BEFORE_2021B_AND_2022",
        "development_years": list(YEARS),
        "all_admission_rows": total_rows,
        "task_role": "auxiliary representation learning; not prospective bedside performance",
        "prediction_anchor": "admission day 0",
        "feature_policy": {
            "allowed": "admission-time static fields and current procedures with valid PRDAY <= 0",
            "forbidden": "current-stay diagnoses, PRDAY > 0 or missing/invalid, final LOS/cost/death/disposition",
        },
        "high_cost": {
            "definition": "cost_2021_usd strictly greater than all-admission DISCWT-weighted p90",
            "threshold_2021_usd": cost_threshold,
            "valid_n": cost_valid,
            "positive_n": high_cost_positive,
            "positive_weight_capped": bounded_pos_weight(high_cost_positive, cost_valid),
        },
        "prolonged_los": {
            "definition": "LOS > 7 days",
            "weighted_p90_days_descriptive": los_weighted_p90,
            "valid_n": prolonged_valid,
            "positive_n": prolonged_positive,
            "positive_weight_capped": bounded_pos_weight(prolonged_positive, prolonged_valid),
        },
        "death": {
            "definition": "DIED == 1",
            "valid_n": death_valid,
            "positive_n": death_positive,
            "positive_weight_capped": bounded_pos_weight(death_positive, death_valid),
        },
        "age_scaling": {
            "mean": age_mean,
            "scale": math.sqrt(age_variance) if age_variance > 1e-12 else 1.0,
            "valid_n": age_n,
        },
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256(manifest_path),
        "inputs": inputs,
        "year_2021_accessed": False,
        "year_2022_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
