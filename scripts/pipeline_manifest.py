#!/usr/bin/env python3

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def command_output(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return ""

    text = result.stdout.strip()

    if not text:
        text = result.stderr.strip()

    return text


def first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def parse_key_value_log(path: Path):
    values = {}

    if not path.is_file():
        return values

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():

        line = raw_line.strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        if key and re.fullmatch(r"[A-Z0-9_]+", key):
            values[key] = value

    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate CUDA-Lab provenance manifest."
    )

    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--dataset-validation-log", required=True)
    parser.add_argument("--result-validation-log", required=True)
    parser.add_argument("--manifest", required=True)

    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-output-sha256")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    run_log_path = Path(args.run_log).resolve()
    dataset_log_path = Path(
        args.dataset_validation_log
    ).resolve()
    result_log_path = Path(
        args.result_validation_log
    ).resolve()

    manifest_path = Path(args.manifest).resolve()

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_values = parse_key_value_log(run_log_path)
    dataset_values = parse_key_value_log(dataset_log_path)
    result_values = parse_key_value_log(result_log_path)

    git_commit = command_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
    )

    git_branch = command_output(
        ["git", "branch", "--show-current"],
        cwd=repo_root,
    )

    git_status = command_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
    )

    nvcc_full = command_output(["nvcc", "--version"])
    cmake_full = command_output(["cmake", "--version"])
    compiler_full = command_output(["c++", "--version"])

    gpu_query = command_output([
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ])

    gpu_name = ""
    compute_capability = ""
    driver_version = ""

    if gpu_query:
        parts = [
            item.strip()
            for item in gpu_query.splitlines()[0].split(",")
        ]

        if len(parts) >= 3:
            gpu_name = parts[0]
            compute_capability = parts[1]
            driver_version = parts[2]

    nvcc_release = ""

    match = re.search(
        r"release\s+([0-9.]+)",
        nvcc_full,
    )

    if match:
        nvcc_release = match.group(1)

    input_sha = (
        sha256_file(input_path)
        if input_path.is_file()
        else ""
    )

    output_sha = (
        sha256_file(output_path)
        if output_path.is_file()
        else ""
    )

    checks = {
        "input_exists": input_path.is_file(),
        "output_exists": output_path.is_file(),
        "git_worktree_clean": git_status == "",
        "run_gate_pass":
            run_values.get("VECTOR_ADD_FILE_GATE") == "PASS",
        "dataset_validation_pass":
            dataset_values.get("DATASET_VALIDATION_GATE") == "PASS",
        "result_validation_pass":
            result_values.get("RESULT_VALIDATION_GATE") == "PASS",
    }

    if args.expected_commit:
        checks["git_commit_identity"] = (
            git_commit == args.expected_commit
        )

    if args.expected_input_sha256:
        checks["input_sha256_identity"] = (
            input_sha == args.expected_input_sha256
        )

    if args.expected_output_sha256:
        checks["output_sha256_identity"] = (
            output_sha == args.expected_output_sha256
        )

    try:
        max_error = float(
            run_values.get("MAX_ERROR", "nan")
        )
    except ValueError:
        max_error = float("nan")

    checks["cuda_max_error_zero"] = (
        max_error == 0.0
    )

    overall_pass = all(checks.values())

    manifest = {
        "schema_version": "1.0",
        "pipeline": "vector_add_file",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat().replace("+00:00", "Z"),

        "software": {
            "repository": "ncevidanes/CUDA-Lab",
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_worktree_clean": git_status == "",
        },

        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "gpu_name": gpu_name,
            "compute_capability": compute_capability,
            "nvidia_driver_version": driver_version,
            "cuda_toolkit_version": nvcc_release,
            "nvcc": first_line(
                nvcc_full.splitlines()[-1]
                if nvcc_full
                else ""
            ),
            "cmake": first_line(cmake_full),
            "cxx_compiler": first_line(compiler_full),
        },

        "input": {
            "path": str(input_path),
            "size_bytes":
                input_path.stat().st_size
                if input_path.is_file()
                else None,
            "sha256": input_sha,
        },

        "output": {
            "path": str(output_path),
            "size_bytes":
                output_path.stat().st_size
                if output_path.is_file()
                else None,
            "sha256": output_sha,
        },

        "cuda_execution": {
            "gpu_name":
                run_values.get("GPU_NAME", gpu_name),
            "elements":
                run_values.get("ELEMENTS"),
            "threads_per_block":
                run_values.get("THREADS_PER_BLOCK"),
            "blocks":
                run_values.get("BLOCKS"),
            "max_error":
                run_values.get("MAX_ERROR"),
            "execution_gate":
                run_values.get("VECTOR_ADD_FILE_GATE"),
        },

        "validation": {
            "dataset_gate":
                dataset_values.get(
                    "DATASET_VALIDATION_GATE"
                ),
            "result_gate":
                result_values.get(
                    "RESULT_VALIDATION_GATE"
                ),
            "result_validated_rows":
                result_values.get(
                    "RESULT_VALIDATED_ROWS"
                ),
            "result_max_error":
                result_values.get(
                    "RESULT_MAX_ERROR"
                ),
        },

        "expected_identity": {
            "git_commit": args.expected_commit,
            "input_sha256": args.expected_input_sha256,
            "output_sha256": args.expected_output_sha256,
        },

        "checks": checks,

        "overall_gate":
            "PASS" if overall_pass else "FAIL",
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"MANIFEST_PATH={manifest_path}")
    print(f"MANIFEST_GIT_COMMIT={git_commit}")
    print(f"MANIFEST_GPU_NAME={gpu_name}")
    print(f"MANIFEST_COMPUTE_CAPABILITY={compute_capability}")
    print(f"MANIFEST_INPUT_SHA256={input_sha}")
    print(f"MANIFEST_OUTPUT_SHA256={output_sha}")
    print(
        "MANIFEST_GATE="
        + ("PASS" if overall_pass else "FAIL")
    )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
