#!/usr/bin/env python3

import argparse
import csv
import hashlib
from itertools import zip_longest
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate CUDA-Lab vector-add output."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-rows", type=int, default=16384)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-output-sha256")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    ok = True
    rows = 0
    max_error = 0.0

    if input_path.is_file():
        print("RESULT_INPUT_FILE_GATE=PASS")
    else:
        print("RESULT_INPUT_FILE_GATE=FAIL")
        ok = False

    if output_path.is_file():
        print("RESULT_OUTPUT_FILE_GATE=PASS")
    else:
        print("RESULT_OUTPUT_FILE_GATE=FAIL")
        ok = False

    if not ok:
        print("RESULT_VALIDATION_GATE=FAIL")
        return 1

    input_sha = sha256_file(input_path)
    output_sha = sha256_file(output_path)

    print(f"RESULT_INPUT_SHA256={input_sha}")
    print(f"RESULT_OUTPUT_SHA256={output_sha}")

    if args.expected_input_sha256:
        if input_sha == args.expected_input_sha256:
            print("RESULT_INPUT_SHA256_GATE=PASS")
        else:
            print("RESULT_INPUT_SHA256_GATE=FAIL")
            ok = False

    if args.expected_output_sha256:
        if output_sha == args.expected_output_sha256:
            print("RESULT_OUTPUT_SHA256_GATE=PASS")
        else:
            print("RESULT_OUTPUT_SHA256_GATE=FAIL")
            ok = False

    with input_path.open(newline="", encoding="utf-8") as fi, \
         output_path.open(newline="", encoding="utf-8") as fo:

        input_reader = csv.DictReader(fi)
        output_reader = csv.DictReader(fo)

        if input_reader.fieldnames == ["index", "a", "b"]:
            print("RESULT_INPUT_HEADER_GATE=PASS")
        else:
            print("RESULT_INPUT_HEADER_GATE=FAIL")
            ok = False

        if output_reader.fieldnames == ["index", "a", "b", "c"]:
            print("RESULT_OUTPUT_HEADER_GATE=PASS")
        else:
            print("RESULT_OUTPUT_HEADER_GATE=FAIL")
            ok = False

        for expected_index, pair in enumerate(
            zip_longest(input_reader, output_reader)
        ):
            input_row, output_row = pair

            if input_row is None or output_row is None:
                ok = False
                print("RESULT_ROW_PAIRING_GATE=FAIL")
                break

            try:
                input_index = int(input_row["index"])
                output_index = int(output_row["index"])

                input_a = float(input_row["a"])
                input_b = float(input_row["b"])

                output_a = float(output_row["a"])
                output_b = float(output_row["b"])
                output_c = float(output_row["c"])

                expected_c = input_a + input_b
                error = abs(output_c - expected_c)

                max_error = max(max_error, error)

                if input_index != expected_index:
                    ok = False

                if output_index != expected_index:
                    ok = False

                if output_a != input_a:
                    ok = False

                if output_b != input_b:
                    ok = False

                if output_c != expected_c:
                    ok = False

                if output_c != float(3 * expected_index):
                    ok = False

                rows += 1

            except Exception as exc:
                print(f"RESULT_EXCEPTION={exc!r}")
                ok = False
                break

    print(f"RESULT_VALIDATED_ROWS={rows}")
    print(f"EXPECTED_ROWS={args.expected_rows}")
    print(f"RESULT_MAX_ERROR={max_error}")

    if rows == args.expected_rows:
        print("RESULT_ROW_COUNT_GATE=PASS")
    else:
        print("RESULT_ROW_COUNT_GATE=FAIL")
        ok = False

    if max_error == 0.0:
        print("RESULT_NUMERICAL_GATE=PASS")
    else:
        print("RESULT_NUMERICAL_GATE=FAIL")
        ok = False

    print(
        "RESULT_VALIDATION_GATE="
        + ("PASS" if ok else "FAIL")
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
