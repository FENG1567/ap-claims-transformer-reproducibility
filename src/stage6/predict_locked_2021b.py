#!/usr/bin/env python3
"""Generate the selected Transformer's one permitted 2021B prediction file.

The program requires a completed 2021A model-selection lock and a hash-bound
2021A calibration/operating-point lock.  It accepts only 2021B and explicitly
cannot read 2022.  The output contains raw hierarchical probabilities plus the
two frozen binary recalibrations needed by conformal abstention and downstream
clinical-utility evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

STAGE5 = Path(__file__).resolve().parents[1] / "stage5"
if str(STAGE5) not in sys.path:
    sys.path.insert(0, str(STAGE5))

from claims_transformer import ClaimsTransformer, ClaimsTransformerConfig  # noqa: E402
from finetune_ap_transformer_v2 import (  # noqa: E402
    HISTORY_COLUMNS,
    OUTPUT_CONTEXT_COLUMNS,
    REQUIRED_EPISODE_COLUMNS,
    ExampleDataset,
    StaticPreprocessor,
    collate_examples,
    load_unified_hierarchy,
    make_examples,
    predict,
    require_unique_columns,
    unique_preserving_order,
)
from freeze_transformer_calibration import apply_calibrator  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def validate_model_lock(lock_path: Path, selected_model_dir: Path) -> dict[str, Any]:
    lock = read_json(lock_path)
    if (lock.get("status") != "LOCKED_ON_2021A_PRE_2021B_PRE_2022"
            or lock.get("selection_partition") != "2021A"
            or lock.get("2021B_accessed") is not False
            or lock.get("year_2022_accessed") is not False):
        raise RuntimeError("Transformer model selection is not sealed before 2021B/2022")
    selected_id = str(lock.get("selected_id", ""))
    if not selected_id or selected_model_dir.resolve().name != selected_id:
        raise RuntimeError("Selected model directory does not match model_selection_lock selected_id")
    return lock


def validate_selected_model(selected_model_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = selected_model_dir / "finetune_manifest.json"
    checkpoint_path = selected_model_dir / "best_checkpoint.pt"
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022"
            or manifest.get("prediction_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError("Selected Transformer manifest is not sealed")
    if manifest.get("checkpoint", {}).get("sha256") != sha256(checkpoint_path):
        raise RuntimeError("Selected Transformer checkpoint hash mismatch")
    return manifest, checkpoint_path


def validate_operating_point(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "transformer_operating_point_manifest.json"
    calibrator_path = directory / "transformer_binary_calibrators.joblib"
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A"
            or manifest.get("selection_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError("Transformer calibration is not sealed on 2021A")
    identity = manifest.get("artifacts", {}).get(calibrator_path.name, {})
    if (not calibrator_path.is_file() or identity.get("sha256") != sha256(calibrator_path)
            or int(identity.get("bytes", -1)) != calibrator_path.stat().st_size):
        raise RuntimeError("Transformer calibrator artifact identity mismatch")
    bundle = joblib.load(calibrator_path)
    if (not isinstance(bundle, dict) or bundle.get("version") != 1
            or set(bundle.get("endpoints", {})) != {"any_readmission", "ap_specific_readmission"}):
        raise RuntimeError("Transformer calibrator bundle has an invalid contract")
    return manifest, bundle


def _available_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def read_2021b(root: Path, history_dir: Path,
               static_columns: Iterable[str]) -> pd.DataFrame:
    episode_path = root / "data" / "nrd" / "year=2021" / "ap_episodes.parquet"
    history_path = history_dir / "ap_history_2021.parquet"
    available = _available_columns(episode_path)
    needed = unique_preserving_order((*REQUIRED_EPISODE_COLUMNS,
                                     *(c for c in (*static_columns, *OUTPUT_CONTEXT_COLUMNS) if c in available)))
    frame = pq.read_table(
        episode_path,
        columns=needed,
        filters=[("primary_analysis_eligible", "=", True), ("analysis_partition", "=", "2021B")],
    ).to_pandas()
    require_unique_columns(frame, "2021B episode partition")
    if (frame.empty or set(frame["year"].astype(int)) != {2021}
            or set(frame["analysis_partition"]) != {"2021B"}):
        raise RuntimeError("2021B partition gate failed")
    history_columns = unique_preserving_order(("encounter_hash", "patient_hash", "analysis_partition", *HISTORY_COLUMNS))
    history = pq.read_table(
        history_path,
        columns=history_columns,
        filters=[("analysis_partition", "=", "2021B")],
    ).to_pandas()
    require_unique_columns(history, "2021B history partition")
    if history["encounter_hash"].duplicated().any():
        raise RuntimeError("Duplicate 2021B history keys")
    merged = frame.merge(history.drop(columns=["patient_hash", "analysis_partition"]),
                         on="encounter_hash", how="left", validate="one_to_one")
    require_unique_columns(merged, "2021B merged partition")
    if merged[list(HISTORY_COLUMNS)].isna().all(axis=1).any():
        raise RuntimeError("Missing 2021B history join")
    return merged


def completed_output(output_dir: Path, model_hash: str,
                     operating_manifest_hash: str) -> dict[str, Any] | None:
    manifest_path = output_dir / "prediction_manifest.json"
    prediction_path = output_dir / "predictions_2021B.parquet"
    if not manifest_path.exists() and not prediction_path.exists():
        return None
    if not manifest_path.is_file() or not prediction_path.is_file():
        raise RuntimeError("Partial 2021B prediction output exists; refuse overwrite")
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_PREDICTIONS_2021B_ONLY"
            or manifest.get("partition") != "2021B"
            or manifest.get("year_2022_accessed") is not False
            or manifest.get("model", {}).get("sha256") != model_hash
            or manifest.get("operating_point", {}).get("manifest_sha256") != operating_manifest_hash
            or manifest.get("artifact", {}).get("sha256") != sha256(prediction_path)):
        raise RuntimeError("Existing 2021B output fails identity or partition validation")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--model-selection-lock", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--operating-point-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or not 0 <= args.workers <= 8 or args.batch_size < 1:
        raise SystemExit("Invalid resource arguments")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    torch.set_num_threads(args.threads)

    root = args.root.resolve()
    history = args.history_dir.resolve()
    hierarchy = args.hierarchy_dir.resolve()
    selected_dir = args.selected_model_dir.resolve()
    operating_dir = args.operating_point_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = validate_model_lock(args.model_selection_lock.resolve(), selected_dir)
    selected_manifest, checkpoint_path = validate_selected_model(selected_dir)
    operating_manifest, calibrator_bundle = validate_operating_point(operating_dir)
    operating_manifest_hash = sha256(operating_dir / "transformer_operating_point_manifest.json")
    existing = completed_output(output_dir, sha256(checkpoint_path), operating_manifest_hash)
    if existing is not None:
        print(stable_json(existing))
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("training_contract") != selected_manifest.get("training_contract"):
        raise RuntimeError("Best checkpoint training contract differs from selected manifest")
    static = StaticPreprocessor.from_dict(checkpoint["static_preprocessor"])
    frame = read_2021b(root, history, (*static.numeric_columns, *static.categorical_columns))
    static_values = static.transform(frame)
    bundle = load_unified_hierarchy(root, hierarchy)
    config = ClaimsTransformerConfig(**checkpoint["model_config"])
    model = ClaimsTransformer(config, bundle["token_to_category"], bundle["token_to_domain"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    max_tokens = int(selected_manifest["training_contract"]["max_tokens"])
    examples = make_examples(frame, bundle["dx_size"], max_tokens, static_values)
    loader = DataLoader(ExampleDataset(examples), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_examples)
    predictions = predict(model, loader, device, True)
    if set(predictions["analysis_partition"]) != {"2021B"} or set(predictions["year"].astype(int)) != {2021}:
        raise RuntimeError("Generated prediction partition is not exclusively 2021B")
    endpoint_columns = {
        "any_readmission": ("p_any_readmission", "p_any_readmission_calibrated"),
        "ap_specific_readmission": ("p_leaf_ap", "p_ap_specific_readmission_calibrated"),
    }
    for endpoint, (raw_column, calibrated_column) in endpoint_columns.items():
        item = calibrator_bundle["endpoints"][endpoint]
        predictions[calibrated_column] = apply_calibrator(
            str(item["method"]), item["model"], predictions[raw_column].to_numpy(dtype=np.float64)
        )
    prediction_path = output_dir / "predictions_2021B.parquet"
    pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), prediction_path,
                   compression="zstd", compression_level=6)
    manifest = {
        "status": "PASS_PREDICTIONS_2021B_ONLY",
        "partition": "2021B",
        "rows": int(len(predictions)),
        "events": {
            "any_readmission": int(predictions["any_unplanned_readmission_30d"].astype(int).sum()),
            "ap_specific_readmission": int((predictions["readmission_leaf"].astype(int) == 1).sum()),
        },
        "model_selection_lock": {"file": str(args.model_selection_lock.resolve()),
                                 "sha256": sha256(args.model_selection_lock.resolve()),
                                 "selected_id": lock["selected_id"]},
        "model": {"file": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size,
                  "sha256": sha256(checkpoint_path)},
        "operating_point": {"manifest": str(operating_dir / "transformer_operating_point_manifest.json"),
                            "manifest_sha256": operating_manifest_hash,
                            "status": operating_manifest["status"]},
        "artifact": {"file": prediction_path.name, "bytes": prediction_path.stat().st_size,
                     "sha256": sha256(prediction_path)},
        "model_or_threshold_selection_on_2021B": False,
        "year_2022_accessed": False,
    }
    (output_dir / "prediction_manifest.json").write_text(stable_json(manifest), encoding="utf-8")
    print(stable_json(manifest))


if __name__ == "__main__":
    main()
