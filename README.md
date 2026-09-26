# AP Claims Transformer reproducibility package

This repository accompanies the manuscript **"Claims Transformer versus LightGBM for 30-day readmission after acute pancreatitis: frozen temporal validation and restricted cross-database transport."** It contains the analysis code, frozen non-patient configuration files, disclosure-safe aggregate source data, final figures, tests, and release manifests needed to inspect the computational workflow.

## Reproducibility boundary

The public repository can reproduce the released aggregate checks and inspect the released figures. Re-running model development or the formal 2022 NRD and MIMIC-IV evaluations requires separately licensed data and, for exact inference, the frozen model artifacts described in `docs/RESTRICTED_DATA.md`.

- NRD 2018-2020: development.
- NRD 2021A: model selection, operating thresholds, and probability calibration.
- NRD 2021B: conformal calibration only.
- NRD 2022 and MIMIC-IV: evaluation only; no tuning or recalibration.
- Public results are aggregate-only. No patient-, encounter-, hospital-, prediction-, or bootstrap-replicate rows are included.
- Cells subject to the locked small-cell rule are suppressed.

## Repository map

| Path | Purpose |
|---|---|
| `src/local_pipeline/` | Claims extraction and episode construction code. |
| `src/stage2/` | Frozen ontology and planned-readmission lock builders. |
| `src/stage4/` | Baseline and history-model code. |
| `src/stage5/` | Claims Transformer, pretraining, fine-tuning, calibration, and operating-point code. |
| `src/stage6/` | 2021B conformal calibration and frozen prediction code. |
| `src/stage7/` | Locked 2022 temporal evaluation and documented technical-amendment code. |
| `src/mimic_transfer/` | Formal v7 restricted-transfer runtime, adapter, execution-spec builder, and tests. |
| `src/revision_analyses/` | Additive 2022 and MIMIC analyses used in the IJMI revision. |
| `src/reporting/` | Revision input-provenance note. |
| `configs/` | Frozen, non-patient method locks and mappings. |
| `results/public_source_data/` | Disclosure-safe aggregate source data used by the final paper figures and tables. |
| `results/figures/` | Final PDF/SVG figures. |
| `scripts/` | Public release verification entry point. |
| `manifests/` | Release inventory, provenance, and SHA-256 checksums. |

## Quick start

Python 3.12 is recommended.

```bash
conda env create -f environment.yml
conda activate ap-claims-transformer
python scripts/verify_release.py
pytest -q tests/test_release_smoke.py
```

## Full restricted rerun

The full workflow cannot be run from a public clone alone because NRD and MIMIC-IV are controlled-access datasets and the frozen model bundle is not redistributed. Obtain access independently, retain the original archive layout, and follow `docs/DATA_ACCESS.md` and `docs/REPRODUCIBILITY.md`. Never commit raw data, row-level derived data, model predictions, credentials, or access tokens.

## Verification

`python scripts/verify_release.py` checks the release manifest, SHA-256 values, absolute-path leakage, common credential patterns, forbidden row-level filenames, and basic public-table suppression conditions. `pytest -q tests/test_release_smoke.py` performs an independent public-package smoke test.

## Citation

Repository: https://github.com/FENG1567/ap-claims-transformer-reproducibility

The verified public release is tagged `v1.0.0`. See `CITATION.cff` for the preferred citation. Add the journal DOI and an archival software DOI if they are assigned later.

## Licence

The release is prepared under the MIT License for repository code. Third-party data, ontology mappings, and controlled-access datasets remain governed by their original licences and data-use agreements.
