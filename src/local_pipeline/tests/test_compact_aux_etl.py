from __future__ import annotations

from work.local_pipeline.compact_aux_etl import GROUP_CONFIG, validate_spec


def test_annual_auxiliary_schemas_have_required_fields() -> None:
    for year in (2018, 2019, 2020, 2021):
        for group in ("HOSPITAL", "SEVERITY"):
            source_order = validate_spec(year, group)
            assert set(source_order) == set(GROUP_CONFIG[group]["columns"])


def test_2020_hospital_source_order_is_preserved() -> None:
    source_order = validate_spec(2020, "HOSPITAL")
    assert source_order.index("H_CONTRL") > source_order.index("HOSP_UR_TEACH")
