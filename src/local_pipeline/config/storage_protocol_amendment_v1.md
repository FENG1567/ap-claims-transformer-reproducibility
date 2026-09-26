# Storage protocol amendment v1

## Material Passport

- Project: knowledge-augmented claims representation model for cause-specific readmission after acute pancreatitis
- Change date: 2026-09-12, before any project-specific 2022 outcome access
- Change class: engineering/storage implementation only
- Scientific endpoints, cohort definitions, comparison hierarchy and temporal test year: unchanged

## Reason for amendment

An early planning note used a conservative requirement of at least 1 TB
of server-side writable space.  Direct inspection showed that the authorized
server account has a 500 GB quota and approximately 134 GB free, whereas the
local D: volume has approximately 938 GB free.  The original 1 TB figure was a
safety allowance for keeping expanded CSV files, multiple ETL generations and
many checkpoints; it was not a scientific requirement.

## Frozen low-footprint implementation

1. Read the encrypted NRD archives locally with interactive 7-Zip; passwords
   must not appear in commands, scripts, logs, Git or process arguments.
2. Never persist all five years of expanded CSV files.  Process one stream or
   one annual temporary extraction at a time.
3. Encode diagnosis and procedure strings as deterministic integer token IDs;
   store PRDAY as a compact integer vector and structured fields with the
   narrowest lossless type.
4. Store Arrow/Parquet shards with ZSTD compression and bounded row groups.
5. Construct the active vocabulary from 2018–2021 only.  Keep 2022 unseen codes
   as OOV after the vocabulary lock.
6. Upload 2018–2020 all-stay pretraining shards, the 2018–2021 AP modeling
   cohort, frozen vocabularies, schemas and aggregate QC.  Do not upload a
   second copy of the raw archives already verified on the server.
7. Use local D: only for encrypted-source-derived working data.  Patient-level
   data must not enter Git or public services.
8. Use server `/tmp` only for regenerable scratch and the project home for
   durable artifacts.  Do not rely on `/tmp` as the sole copy of a result.
9. Retain the selected checkpoint, the latest recovery checkpoint and at most
   one prior safety checkpoint during full training.
10. Do not unlock or summarize 2022 outcomes until analysis, baseline and model
    locks are present and their hashes match.

## Empirical storage gate

Before full ETL, a representative pilot must record rows, compressed bytes per
row, throughput and peak working memory.  Full ETL may proceed only when:

- local D: has at least 600 GB free at start;
- projected local peak is no more than 70% of available local space;
- projected durable server upload is no more than 85 GB;
- at least 25 GB server quota remains after the projected upload; and
- all pilot schema and losslessness checks pass.

If a projection fails, the pipeline must stop before full extraction.  Changing
the storage gate does not authorize feature selection using outcomes or any
2022 test-set inspection.

