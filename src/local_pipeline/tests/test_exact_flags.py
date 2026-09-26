from __future__ import annotations

import numpy as np
import pandas as pd

from work.local_pipeline.extract_2021_exact_flags import (
    hash_series,
    lookup_mask,
    normalize_code,
    normalize_factorized,
    prefix_mask,
)


def test_normalize_code() -> None:
    assert normalize_code(" k85.10 ") == "K8510"
    assert normalize_code(None) == ""


def test_factorized_membership_preserves_shape() -> None:
    frame = pd.DataFrame([["K85.10", "A41.9"], [None, "K80.00"]])
    codes, values = normalize_factorized(frame)
    assert codes.shape == frame.shape
    assert prefix_mask(codes, values, "K85").tolist() == [[True, False], [False, False]]
    target = {"A419", "K8000"}
    assert lookup_mask(codes, values, target).tolist() == [[False, True], [False, True]]


def test_hash_is_year_scoped_and_deterministic() -> None:
    series = pd.Series([1, 2, 1])
    first = hash_series(2021, series)
    second = hash_series(2021, series)
    other_year = hash_series(2020, series)
    assert np.array_equal(first, second)
    assert first[0] == first[2]
    assert not np.array_equal(first, other_year)
