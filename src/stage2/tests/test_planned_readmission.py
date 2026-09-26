from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from work.stage2.planned_readmission import PlannedReadmissionClassifier, normalize_code


@pytest.fixture()
def classifier(tmp_path: Path) -> PlannedReadmissionClassifier:
    code_sets = pd.DataFrame(
        [
            [2021, "PR.1", "PCS", "PR1"],
            [2021, "PR.2", "CM", "DX2"],
            [2021, "PR.3", "PCS", "PR3"],
            [2021, "PR.4", "CM", "DX4"],
        ],
        columns=["calendar_year", "table", "code_system", "code"],
    )
    mapping = pd.DataFrame(
        [
            [2021, "CM", "DX0"],
            [2021, "CM", "DX2"],
            [2021, "CM", "DX4"],
            [2021, "PCS", "PR0"],
            [2021, "PCS", "PR1"],
            [2021, "PCS", "PR3"],
        ],
        columns=["calendar_year", "code_system", "code"],
    )
    code_path = tmp_path / "codes.parquet"
    map_path = tmp_path / "map.parquet"
    code_sets.to_parquet(code_path, index=False)
    mapping.to_parquet(map_path, index=False)
    return PlannedReadmissionClassifier(code_path, map_path)


def test_code_normalization() -> None:
    assert normalize_code(" a12.34 ") == "A1234"
    assert normalize_code(None) == ""


def test_always_planned_procedure_overrides_acute_diagnosis(
    classifier: PlannedReadmissionClassifier,
) -> None:
    result = classifier.classify(2021, "DX4", ["PR1"])
    assert result.status == "planned"
    assert result.pr1 and result.pr4


def test_always_planned_principal_diagnosis(
    classifier: PlannedReadmissionClassifier,
) -> None:
    result = classifier.classify(2021, "DX2", ["PR0"])
    assert result.status == "planned"
    assert result.pr2


def test_potentially_planned_requires_nonacute_diagnosis(
    classifier: PlannedReadmissionClassifier,
) -> None:
    assert classifier.classify(2021, "DX0", ["PR3"]).status == "planned"
    assert classifier.classify(2021, "DX4", ["PR3"]).status == "unplanned"


def test_no_planned_trigger_is_unplanned(
    classifier: PlannedReadmissionClassifier,
) -> None:
    assert classifier.classify(2021, "DX0", ["PR0"]).status == "unplanned"


def test_unmapped_or_missing_input_is_unknown(
    classifier: PlannedReadmissionClassifier,
) -> None:
    assert classifier.classify(2021, "UNKNOWN", ["PR0"]).status == "algorithm_unknown"
    assert classifier.classify(2021, "DX0", ["UNKNOWN"]).status == "algorithm_unknown"
    assert classifier.classify(2021, None, ["PR0"]).status == "algorithm_unknown"


def test_sealed_or_unsupported_year_rejected(
    classifier: PlannedReadmissionClassifier,
) -> None:
    with pytest.raises(ValueError):
        classifier.classify(2022, "DX0", ["PR0"])
