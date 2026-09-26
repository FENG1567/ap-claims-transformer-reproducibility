from __future__ import annotations

import numpy as np

from work.local_pipeline.lock_auxiliary_thresholds import weighted_quantile


def test_weighted_quantile_respects_weights() -> None:
    values = np.array([1.0, 2.0, 10.0])
    weights = np.array([1.0, 1.0, 8.0])
    assert weighted_quantile(values, weights, 0.5) == 10.0


def test_weighted_quantile_ignores_invalid_rows() -> None:
    values = np.array([1.0, np.nan, 3.0])
    weights = np.array([1.0, 10.0, 0.0])
    assert weighted_quantile(values, weights, 0.9) == 1.0
