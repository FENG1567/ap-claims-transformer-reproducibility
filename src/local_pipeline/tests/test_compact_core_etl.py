from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


MODULE_PATH = Path(__file__).parents[1] / "compact_core_etl.py"
spec = importlib.util.spec_from_file_location("compact_core_etl", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_vocabulary_freeze_maps_unseen_to_oov() -> None:
    vocab = module.Vocabulary()
    train = pd.DataFrame([["K850", "K851", None]])
    encoded_train = vocab.encode(train, update=True)
    assert encoded_train.min() >= 3
    before = dict(vocab.token_to_id)
    later = pd.DataFrame([["K850", "NEWCODE", None]])
    encoded_later = vocab.encode(later, update=False)
    assert encoded_later[0, 0] == before["K850"]
    assert encoded_later[0, 1] == module.RESERVED["[OOV]"]
    assert encoded_later[0, 2] == module.RESERVED["[MISSING]"]
    assert vocab.token_to_id == before
    assert vocab.counts["K850"] == 2
    assert vocab.counts["NEWCODE"] == 1
    assert vocab.counts["[MISSING]"] == 2
    assert vocab.oov_counts["NEWCODE"] == 1


def test_year_salted_hash_is_stable_and_year_specific() -> None:
    series = pd.Series(["abc", "abc", "def"])
    first = module.hash_series(2018, series)
    second = module.hash_series(2018, series)
    next_year = module.hash_series(2019, series)
    assert (first == second).all()
    assert first[0] == first[1]
    assert not (first == next_year).any()


def test_vocabulary_roundtrip_binds_processed_years(tmp_path: Path) -> None:
    vocab = module.Vocabulary()
    vocab.encode(pd.DataFrame([["K850", "K850", "K851"]]), update=True)
    path = tmp_path / "vocabulary.json"
    vocab.save(path, frozen_after_year=2020, processed_years=[2018])
    restored = module.Vocabulary.load(path, expected_processed_years=[2018])
    assert restored.token_to_id == vocab.token_to_id
    assert restored.counts == vocab.counts
    with pytest.raises(RuntimeError, match="Vocabulary/progress year mismatch"):
        module.Vocabulary.load(path, expected_processed_years=[2018, 2019])


def test_resume_rejects_orphaned_completed_parquet(tmp_path: Path) -> None:
    out = tmp_path / "compact"
    year_dir = out / "year=2018"
    year_dir.mkdir(parents=True)
    pq.write_table(pa.table({"x": [1]}), year_dir / "core_compact.parquet")
    with pytest.raises(RuntimeError, match="Orphaned completed Parquet"):
        module.load_resume_state(out)
