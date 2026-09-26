#!/usr/bin/env python3
"""Validate the disclosure-safe AP Claims Transformer public release."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "PUBLIC_RELEASE_MANIFEST.tsv"
CHECKSUMS = ROOT / "manifests" / "SHA256SUMS.txt"

TEXT_SUFFIXES = {
    ".cff", ".csv", ".json", ".md", ".ps1", ".py", ".r", ".sh",
    ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt", ".dbc", ".dbf", ".dta", ".duckdb", ".gz", ".joblib",
    ".npz", ".pkl", ".pt", ".pth", ".sas7bdat", ".sqlite", ".zip",
}
DIRECT_IDENTIFIERS = {
    "subject_id", "hadm_id", "patient_id", "encounter_id", "hospital_id",
    "mrn", "name", "address", "phone", "email", "date_of_birth", "dob",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "password assignment": re.compile(r"(?i)\bpassword\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
}
ABSOLUTE_PATH_PATTERNS = {
    "Windows user path": re.compile(r"(?i)[A-Z]:\\Users\\"),
    "Linux home path": re.compile(r"(?<![A-Za-z0-9_])/home/[^\s'\"]+"),
    "macOS user path": re.compile(r"(?<![A-Za-z0-9_])/Users/[^\s'\"]+"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_for_release() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
        and p.name != "SHA256SUMS.txt"
    )


def validate_manifest(errors: list[str]) -> None:
    if not MANIFEST.exists():
        errors.append("Missing manifests/PUBLIC_RELEASE_MANIFEST.tsv")
        return
    rows: dict[str, dict[str, str]] = {}
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"path", "bytes", "sha256", "role", "redistribution"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            errors.append("Release manifest has an invalid header")
            return
        for row in reader:
            rows[row["path"]] = row

    expected = {
        p.relative_to(ROOT).as_posix(): p
        for p in files_for_release()
        if p != MANIFEST
    }
    if set(rows) != set(expected):
        missing = sorted(set(expected) - set(rows))
        extra = sorted(set(rows) - set(expected))
        errors.append(f"Manifest membership mismatch; missing={missing}, extra={extra}")
    for relative, path in expected.items():
        row = rows.get(relative)
        if not row:
            continue
        if int(row["bytes"]) != path.stat().st_size:
            errors.append(f"Manifest byte count mismatch: {relative}")
        if row["sha256"].lower() != sha256(path):
            errors.append(f"Manifest SHA-256 mismatch: {relative}")


def validate_checksum_file(errors: list[str]) -> None:
    if not CHECKSUMS.exists():
        errors.append("Missing manifests/SHA256SUMS.txt")
        return
    rows: dict[str, str] = {}
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"Invalid checksum line: {line}")
            continue
        rows[relative] = digest.lower()
    expected = {
        p.relative_to(ROOT).as_posix(): p
        for p in files_for_release()
        if p != CHECKSUMS
    }
    if set(rows) != set(expected):
        errors.append("SHA256SUMS membership does not match the release")
    for relative, path in expected.items():
        if rows.get(relative) != sha256(path):
            errors.append(f"SHA256SUMS mismatch: {relative}")


def validate_files(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f"File exceeds GitHub's 100 MB limit: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Forbidden controlled/binary artifact: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 25 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if relative != "scripts/verify_release.py":
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"Potential {label} in {relative}")
            for label, pattern in ABSOLUTE_PATH_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"Machine-specific {label} in {relative}")


def validate_public_csv(errors: list[str]) -> None:
    data_dir = ROOT / "results" / "public_source_data"
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        errors.append("No public aggregate source-data CSV files found")
        return
    for path in csv_files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                errors.append(f"Missing CSV header: {path.name}")
                continue
            columns = {c.strip().lower() for c in reader.fieldnames}
            leaked = sorted(columns & DIRECT_IDENTIFIERS)
            if leaked:
                errors.append(f"Direct identifier columns in {path.name}: {leaked}")
            for row_number, row in enumerate(reader, start=2):
                for column, value in row.items():
                    key = (column or "").strip().lower()
                    if not value or not re.search(r"(^n$|count|events|non_events|sample_size)", key):
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if 0 < number <= 10:
                        errors.append(
                            f"Unsuppressed small count in {path.name}:{row_number} column {column}"
                        )


def main() -> int:
    errors: list[str] = []
    validate_manifest(errors)
    validate_checksum_file(errors)
    validate_files(errors)
    validate_public_csv(errors)
    if errors:
        print("RELEASE_VERIFY_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"RELEASE_VERIFY_PASS files={len(files_for_release())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
