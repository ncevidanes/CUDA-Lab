#!/usr/bin/env python3

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate CUDA-Lab vector-add input dataset."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-rows", type=int, default=16384)
    parser.add_argument("--expected-sha256")

    args = parser.parse_args()

    path = Path(args.input)

    ok = True
    rows = 0
    max_semantic_error = 0.0

    print(f"DATASET_PATH={path}")

    if not path.is_file():
        print("DATASET_FILE_GATE=FAIL")
        print("DATASET_VALIDATION_GATE=FAIL")
        return 1

    print("DATASET_FILE_GATE=PASS")

    observed_sha = sha256_file(path)

    print(f"DATASET_SHA256={observed_sha}")

    if args.expected_sha256:
        print(f"EXPECTED_DATASET_SHA256={args.expected_sha256}")

        if observed_sha == args.expected_sha256:
            print("DATASET_SHA256_GATE=PASS")
        else:
            print("DATASET_SHA256_GATE=FAIL")
            ok = False

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames == ["index", "a", "b"]:
            print("DATASET_HEADER_GATE=PASS")
        else:
            print("DATASET_HEADER_GATE=FAIL")
            print(f"DATASET_HEADER={reader.fieldnames}")
            ok = False

        for expected_index, row in enumerate(reader):
            try:
                index = int(row["index"])
                a = float(row["a"])
                b = float(row["b"])

                error_a = abs(a - float(expected_index))
                error_b = abs(b - float(2 * expected_index))

                max_semantic_error = max(
                    max_semantic_error,
                    error_a,
                    error_b,
                )

                if index != expected_index:
                    ok = False

                if error_a != 0.0 or error_b != 0.0:
                    ok = False

                rows += 1

            except Exception as exc:
                print(f"DATASET_EXCEPTION={exc!r}")
                ok = False
                break

    print(f"DATASET_ROWS={rows}")
    print(f"EXPECTED_ROWS={args.expected_rows}")
    print(f"DATASET_MAX_SEMANTIC_ERROR={max_semantic_error}")

    if rows == args.expected_rows:
        print("DATASET_ROW_COUNT_GATE=PASS")
    else:
        print("DATASET_ROW_COUNT_GATE=FAIL")
        ok = False

    if max_semantic_error == 0.0:
        print("DATASET_SEMANTIC_GATE=PASS")
    else:
        print("DATASET_SEMANTIC_GATE=FAIL")
        ok = False

    print(
        "DATASET_VALIDATION_GATE="
        + ("PASS" if ok else "FAIL")
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
