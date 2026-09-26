from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path


PUBLIC_FILE_API = "https://qualitynet.cms.gov/publicgateway/public/files/{file_id}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def with_retries(operation, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2**attempt)


def request_json(url: str) -> dict[str, object]:
    def operation() -> dict[str, object]:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    return with_retries(operation)


def get_total_bytes(url: str) -> int:
    def operation() -> int:
        request = urllib.request.Request(
            url,
            headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
            if not match:
                raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
            return int(match.group(1))

    return with_retries(operation)


def download_part(url: str, start: int, end: int, path: Path) -> tuple[int, int, Path]:
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return start, end, path
    temporary = path.with_suffix(path.suffix + ".partial")
    def operation() -> None:
        temporary.unlink(missing_ok=True)
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
            if response.status != 206:
                raise RuntimeError(f"Range {start}-{end} returned HTTP {response.status}")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)

    with_retries(operation)
    if temporary.stat().st_size != expected:
        raise RuntimeError(
            f"Range {start}-{end} length {temporary.stat().st_size} != {expected}"
        )
    os.replace(temporary, path)
    return start, end, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{24}", args.file_id):
        raise SystemExit("QualityNet file ID must be 24 lowercase hexadecimal characters")
    if args.workers < 1 or args.workers > 16:
        raise SystemExit("workers must be between 1 and 16")
    if args.chunk_mib < 1 or args.chunk_mib > 64:
        raise SystemExit("chunk-mib must be between 1 and 64")

    metadata = request_json(PUBLIC_FILE_API.format(file_id=args.file_id))
    signed_url = str(metadata["downloadUrl"])
    total = get_total_bytes(signed_url)
    chunk_size = args.chunk_mib * 1024 * 1024
    part_dir = args.output.with_suffix(args.output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + chunk_size - 1, total - 1))
        for start in range(0, total, chunk_size)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_part,
                signed_url,
                start,
                end,
                part_dir / f"part-{index:05d}.bin",
            )
            for index, (start, end) in enumerate(ranges)
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            start, end, _ = future.result()
            print(f"PART_OK {completed}/{len(futures)} bytes={start}-{end}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("wb") as out:
        for index, (start, end) in enumerate(ranges):
            part = part_dir / f"part-{index:05d}.bin"
            expected = end - start + 1
            if part.stat().st_size != expected:
                raise RuntimeError(f"Part validation failed: {part}")
            with part.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    out.write(block)
    if temporary.stat().st_size != total:
        raise RuntimeError(f"Assembled length {temporary.stat().st_size} != {total}")
    os.replace(temporary, args.output)
    shutil.rmtree(part_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "file_id": args.file_id,
                "file_name": metadata.get("fileName"),
                "bytes": total,
                "sha256": sha256(args.output),
                "workers": args.workers,
                "chunk_mib": args.chunk_mib,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
