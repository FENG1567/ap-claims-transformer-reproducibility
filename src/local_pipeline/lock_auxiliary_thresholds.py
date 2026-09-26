from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import duckdb
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(os.environ.get("AP_CLAIMS_WORKDIR", str(REPOSITORY_ROOT / "workdir"))) / "compact"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "workdir" / "stage3_build" / "auxiliary_thresholds.json"
YEARS = [2018, 2019, 2020]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
        raise ValueError("values and weights must be aligned one-dimensional arrays")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        raise ValueError("no valid weighted observations")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    threshold = q * weights.sum()
    index = int(np.searchsorted(np.cumsum(weights), threshold, side="left"))
    return float(values[min(index, len(values) - 1)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.root / f"year={year}" / "ap_episodes.parquet" for year in YEARS]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("All 2018-2020 AP episode files are required")
    con = duckdb.connect()
    union = " UNION ALL ".join(
        f"SELECT * FROM read_parquet('{path.as_posix()}')" for path in paths
    )
    frame = con.execute(
        f"""
        SELECT cost_2021_usd, cast(LOS AS DOUBLE) AS los, cast(DISCWT AS DOUBLE) AS weight
        FROM ({union})
        WHERE index_primary_base_eligible
          AND principal_ap
          AND DISCWT > 0
        """
    ).fetchdf()
    cost = frame.loc[frame.cost_2021_usd.notna()]
    los = frame.loc[frame.los.notna() & (frame.los >= 0)]
    weighted_cost = weighted_quantile(
        cost.cost_2021_usd.to_numpy(), cost.weight.to_numpy(), 0.90
    )
    unweighted_cost = float(np.quantile(cost.cost_2021_usd.to_numpy(), 0.90))
    weighted_los = weighted_quantile(los.los.to_numpy(), los.weight.to_numpy(), 0.90)
    unweighted_los = float(np.quantile(los.los.to_numpy(), 0.90))
    payload = {
        "status": "LOCKED_BEFORE_2021_EVALUATION_AND_2022_ACCESS",
        "development_years": YEARS,
        "eligibility": "principal AP and index_primary_base_eligible; future outcome status not used",
        "high_cost_primary": {
            "definition": "cost_2021_usd strictly greater than weighted_p90_usd",
            "weighted_p90_usd": weighted_cost,
            "valid_n": int(len(cost)),
            "weight_sum": float(cost.weight.sum()),
        },
        "high_cost_sensitivity": {
            "definition": "cost_2021_usd strictly greater than unweighted_p90_usd",
            "unweighted_p90_usd": unweighted_cost,
        },
        "prolonged_los_primary": {
            "definition": "LOS > 7 days",
        },
        "prolonged_los_sensitivity": {
            "definition": "LOS strictly greater than development weighted_p90_days",
            "weighted_p90_days": weighted_los,
            "unweighted_p90_days": unweighted_los,
            "valid_n": int(len(los)),
            "weight_sum": float(los.weight.sum()),
        },
        "input_sha256": {str(path): sha256(path) for path in paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
