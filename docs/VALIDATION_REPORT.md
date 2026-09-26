# Release validation evidence

Validation date: 2026-09-27

## Checks performed

| Check | Result |
|---|---|
| Python syntax compilation for `src/`, `scripts/`, and `tests/` | PASS |
| JSON parsing for all released JSON files | PASS (16 files) |
| Public release verifier | PASS |
| Public smoke tests | PASS (4 tests) |
| Direct-identifier column scan | PASS |
| Credential/private-key pattern scan | PASS |
| Machine-specific user/server path scan | PASS |
| GitHub 100 MB single-file limit | PASS |

The manifest and checksum verifier must be rerun after any file is modified.
