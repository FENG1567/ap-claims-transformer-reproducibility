from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_verifier_passes() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_release.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RELEASE_VERIFY_PASS" in completed.stdout


def test_expected_public_results_are_present() -> None:
    expected = {
        "primary_absolute_metrics_ci.csv",
        "co_primary_simultaneous_inference.csv",
        "calibration_intercept_slope_bootstrap_ci.csv",
        "mimic_common_baseline_overall_metrics_ci.csv",
        "mimic_common_baseline_paired_differences_ci.csv",
    }
    actual = {p.name for p in (ROOT / "results" / "public_source_data").glob("*.csv")}
    assert expected <= actual


def test_public_tables_have_no_direct_identifiers() -> None:
    forbidden = {"subject_id", "hadm_id", "patient_id", "encounter_id", "hospital_id", "mrn"}
    for path in (ROOT / "results" / "public_source_data").glob("*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = {item.strip().lower() for item in next(reader)}
        assert not (header & forbidden), path.name


def test_formal_mimic_runtime_only() -> None:
    folder = ROOT / "src" / "mimic_transfer"
    runtimes = sorted(p.name for p in folder.glob("run_locked_mimic_transfer*.py"))
    assert runtimes == ["run_locked_mimic_transfer_v7.py"]
    assert (folder / "common_variable_inference_adapter_v2.py").is_file()
    assert (folder / "build_mimic_transfer_execution_spec_v3.py").is_file()

