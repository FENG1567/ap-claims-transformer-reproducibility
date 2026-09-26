# Stage 2 gate

- Status: **PASS**
- Candidate: `analysis_lock.json` SHA256 `d87e13255781cfc52623757abb5546c771053b554419a62a887605ae11ee1b86`
- Ontology rows: 168; unique ICD codes: 168
- Official-codebook claim failures: 0
- Missing default CCSR mappings: 0 (explicit hierarchy-missing token required)
- Planned-readmission lock: CMS PRA v4.0, annual-aligned 2018–2021, SHA256 `e1e682fea14da07e5b9ac531be13859781b7f8ad277bfdc1d6e73509aca64fa1`
- 2022 outcome access: none

The ICD/CCSR ontology, hierarchy, timing matrix, leakage blacklist, SAP, decision log, and annual-aligned CMS Planned Readmission Algorithm v4.0 code sets are frozen. All valid 2018–2021 ICD-10-CM/PCS catalog codes are covered by the corresponding Yale/CORE-modified CCS maps. The main outcome may be named 30-day unplanned readmission when the implementation tests and observed-code mapping gate pass. The 2022 test outcome remains sealed; its predeclared annual code alignment is implemented only at Stage 7.
