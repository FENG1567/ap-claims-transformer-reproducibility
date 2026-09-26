# Result provenance

The public result candidate is the IJMI revision-v4 candidate frozen on 2026-09-22. `results/public_source_data/` is copied from its aggregate-only `source_data` directory. `results/figures/` contains the corresponding PDF and SVG figures.

Key boundaries:

- Corrected 2022 temporal cohort: 119,114 eligible episodes; 16,727 any-readmission events; 8,110 AP-specific events.
- Joint Transformer weighted AUPRC: 0.364 for any readmission and 0.226 for AP-specific readmission.
- LightGBM weighted AUPRC: 0.369 and 0.230.
- Joint Transformer minus LightGBM AUPRC: -0.00529 and -0.00357.
- Formal MIMIC transfer: 1,810 pre-gate episodes and 1,786 analysed episodes.
- The MIMIC endpoint is 30-day same-system readmission, not complete readmission capture.

The release supports a positive validation/transportability boundary but not a Transformer-superiority claim. The frozen representation retained moderate discrimination, while LightGBM had a small but reproducible AUPRC advantage in the 2022 temporal test.

`manifests/PUBLIC_RELEASE_MANIFEST.tsv` records every released file's role, origin class, redistribution status, size, and SHA-256. `manifests/SOURCE_PROVENANCE.tsv` maps copied scientific code and data back to their private-project source path without exposing machine-specific absolute paths.

