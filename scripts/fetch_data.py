"""Fetch the ULB credit-card fraud dataset to data/raw/ (gitignored) with integrity checks.

Requires the Kaggle CLI (`pip install kaggle`) and credentials in ~/.kaggle/kaggle.json.
Verifies the exact row count and, once pinned, the SHA-256 of creditcard.csv so every
machine trains on byte-identical data.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

DATASET = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"
EXPECTED_ROWS = 284_807  # data rows, excluding header

# Pinned after the first verified fetch; None means "print the hash so it can be pinned".
EXPECTED_SHA256: str | None = None

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_data_rows(path: Path) -> int:
    with path.open(newline="") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1  # minus header


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / CSV_NAME

    if not csv_path.exists():
        print(f"downloading {DATASET} -> {RAW_DIR}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"],
            check=True,
        )
    else:
        print(f"{csv_path} already present; verifying")

    rows = count_data_rows(csv_path)
    if rows != EXPECTED_ROWS:
        print(f"FAIL: expected {EXPECTED_ROWS} rows, got {rows}", file=sys.stderr)
        return 1

    digest = sha256_of(csv_path)
    if EXPECTED_SHA256 is None:
        print(f"row count OK ({rows}); sha256 not yet pinned — pin this: {digest}")
    elif digest != EXPECTED_SHA256:
        print(f"FAIL: sha256 mismatch\n  expected {EXPECTED_SHA256}\n  got      {digest}",
              file=sys.stderr)
        return 1
    else:
        print(f"row count OK ({rows}); sha256 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
