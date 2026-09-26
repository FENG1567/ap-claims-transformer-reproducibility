# Restricted and excluded artifacts

The following materials are intentionally absent from this public package.

| Material | Reason | Reproduction route |
|---|---|---|
| NRD raw files, row-level derivatives, predictions, and replicate-level bootstrap output | HCUP licence and disclosure restrictions | Obtain NRD directly from HCUP and rerun locally. |
| MIMIC-IV raw files, row-level episode tables, predictions, and identifiers | PhysioNet credentialed-access agreement | Obtain MIMIC-IV 3.1 through PhysioNet and rerun locally. |
| Frozen Transformer checkpoint (approximately 457 MB) | GitHub size limit and conservative controlled-data derivative boundary | Recreate from licensed NRD data, or deposit separately only after institutional and HCUP review. |
| Fitted calibrator binary and row-derived preprocessing states | Conservative controlled-data derivative boundary | Recreate in the licensed environment. |
| Credentials, tokens, SSH keys, local usernames, and machine-specific absolute paths | Security and portability | Supply through interactive login or local environment configuration; never commit them. |
| Internal failed runs and temporary directories | Not required for scientific rerun and may confuse the formal release | Retain in the private archive. |

The repository is therefore a **transparent public reconstruction package**, not a redistribution of controlled data or a self-contained copy of the trained model. Exact numerical reproduction of restricted analyses requires the same licensed source data, frozen model bundle, and locked software environment.

