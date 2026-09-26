#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/CUDA-Lab}"
BUILD_DIR="${ROOT_CUDA_BUILD_DIR:-/content/cuda-lab-build}"
ENV_PREFIX="${ROOT_ENV_PREFIX:-/content/root-cuda-env}"
DATASET_NAME="mc21_14TeV_Epos_100k"
STAGE_DIR="${ROOT_STAGE_DIR:-/content/root-data/${DATASET_NAME}}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_RESULTS="${LOCAL_RESULTS:-/content/root-results/${RUN_ID}}"

DRIVE_ROOT="/content/drive/MyDrive"
EXPECTED_DATASET_DIR="$DRIVE_ROOT/Doutorado/Doutorado/Orientação Luciano/BackUp_linux_V02/readingHits/datasets/$DATASET_NAME"
DATASET_DIR="${ROOT_DATASET_DIR:-}"

if [[ -z "$DATASET_DIR" && -d "$EXPECTED_DATASET_DIR/raw" ]]; then
  DATASET_DIR="$EXPECTED_DATASET_DIR"
fi
if [[ -z "$DATASET_DIR" ]]; then
  DATASET_DIR="$(find "$DRIVE_ROOT" -type d -name "$DATASET_NAME" -print -quit 2>/dev/null || true)"
fi

if [[ -z "$DATASET_DIR" || ! -d "$DATASET_DIR/raw" ]]; then
  echo "DRIVE_DATASET_DISCOVERY_GATE=FAIL"
  exit 1
fi

echo "DRIVE_DATASET_DIR=$DATASET_DIR"
echo "DRIVE_DATASET_DISCOVERY_GATE=PASS"

mkdir -p "$STAGE_DIR/raw" "$LOCAL_RESULTS"

echo "=== STAGE DRIVE -> COLAB LOCAL DISK ==="
cp -u "$DATASET_DIR"/raw/combined_mc21_14TeV_Epos_part*.root "$STAGE_DIR/raw/"
cp -u "$DATASET_DIR"/manifests/sha256.txt "$STAGE_DIR/sha256.txt"

LOCAL_COUNT="$(find "$STAGE_DIR/raw" -maxdepth 1 -type f -name '*.root' | wc -l)"
echo "STAGED_ROOT_COUNT=$LOCAL_COUNT"
[[ "$LOCAL_COUNT" -eq 8 ]] || { echo "STAGE_FILE_COUNT_GATE=FAIL"; exit 1; }
echo "STAGE_FILE_COUNT_GATE=PASS"

(
  cd "$STAGE_DIR/raw"
  sha256sum -c "$STAGE_DIR/sha256.txt"
)
echo "STAGE_SHA256_GATE=PASS"

INPUT_LIST="$STAGE_DIR/input_files.txt"
find "$STAGE_DIR/raw" -maxdepth 1 -type f -name 'combined_mc21_14TeV_Epos_part*.root' \
  | sort -V > "$INPUT_LIST"

echo "=== BUILD CHECK ==="
BIN="$BUILD_DIR/projects/03-root-cuda/root-cuda-multifile/root_hits_energy"
[[ -x "$BIN" ]] || { echo "BINARY_GATE=FAIL"; exit 1; }
echo "BINARY_GATE=PASS"

echo "=== MICROMAMBA CHECK ==="
MM_BIN="${MICROMAMBA_BIN:-}"
if [[ -z "$MM_BIN" ]]; then
  MM_BIN="$(command -v micromamba 2>/dev/null || true)"
fi
if [[ -z "$MM_BIN" && -x /content/bin/micromamba ]]; then
  MM_BIN="/content/bin/micromamba"
fi
if [[ -z "$MM_BIN" || ! -x "$MM_BIN" ]]; then
  echo "MICROMAMBA_GATE=FAIL"
  exit 1
fi
echo "MICROMAMBA_BIN=$MM_BIN"
echo "MICROMAMBA_GATE=PASS"

echo "=== FULL 100K RUN ==="
"$MM_BIN" run -p "$ENV_PREFIX" "$BIN" \
  --input-list "$INPUT_LIST" \
  --output-dir "$LOCAL_RESULTS" \
  --tree CollectionTree \
  --batch-events "${BATCH_EVENTS:-512}" \
  --threads "${THREADS:-256}" \
  | tee "$LOCAL_RESULTS/run.log"

grep -q '^RESULT_GATE=PASS$' "$LOCAL_RESULTS/run.log"
echo "FULL_RUN_GATE=PASS"

echo "=== ROOT RESULT ARTIFACT CHECK ==="
"$MM_BIN" run -p "$ENV_PREFIX" python - "$LOCAL_RESULTS/results.root" <<'PY'
import os
import sys
import ROOT

path = sys.argv[1]
if not os.path.isfile(path):
    print("RESULT_ROOT_FILE_GATE=FAIL reason=missing_file")
    raise SystemExit(1)

f = ROOT.TFile.Open(path, "READ")
if not f or f.IsZombie():
    print("RESULT_ROOT_FILE_GATE=FAIL reason=zombie_file")
    raise SystemExit(1)

tree = f.Get("EventDetectorMetrics")
gate = f.Get("RESULT_GATE")
entries = int(tree.GetEntries()) if tree else -1
gate_value = gate.GetTitle() if gate else "MISSING"

print(f"RESULT_ROOT_ENTRIES={entries}")
print(f"RESULT_ROOT_NAMED_GATE={gate_value}")

if entries != 500000:
    print("RESULT_ROOT_TREE_GATE=FAIL")
    raise SystemExit(1)
print("RESULT_ROOT_TREE_GATE=PASS")

if gate_value != "PASS":
    print("RESULT_ROOT_NAMED_GATE_CHECK=FAIL")
    raise SystemExit(1)
print("RESULT_ROOT_NAMED_GATE_CHECK=PASS")
print("RESULT_ROOT_GATE=PASS")
f.Close()
PY

DRIVE_RESULTS="$DATASET_DIR/results/$RUN_ID"
mkdir -p "$DRIVE_RESULTS"
cp -a "$LOCAL_RESULTS"/. "$DRIVE_RESULTS"/

echo "RESULT_COPY_GATE=PASS"
echo "LOCAL_RESULTS=$LOCAL_RESULTS"
echo "DRIVE_RESULTS=$DRIVE_RESULTS"
echo "ROOT_CUDA_100K_GATE=PASS"
