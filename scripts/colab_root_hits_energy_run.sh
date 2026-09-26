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
DATASET_DIR="${ROOT_DATASET_DIR:-}"

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

echo "=== FULL 100K RUN ==="
micromamba run -p "$ENV_PREFIX" "$BIN" \
  --input-list "$INPUT_LIST" \
  --output-dir "$LOCAL_RESULTS" \
  --tree CollectionTree \
  --batch-events "${BATCH_EVENTS:-512}" \
  --threads "${THREADS:-256}" \
  | tee "$LOCAL_RESULTS/run.log"

grep -q '^RESULT_GATE=PASS$' "$LOCAL_RESULTS/run.log"
echo "FULL_RUN_GATE=PASS"

DRIVE_RESULTS="$DATASET_DIR/results/$RUN_ID"
mkdir -p "$DRIVE_RESULTS"
cp -a "$LOCAL_RESULTS"/. "$DRIVE_RESULTS"/

echo "RESULT_COPY_GATE=PASS"
echo "LOCAL_RESULTS=$LOCAL_RESULTS"
echo "DRIVE_RESULTS=$DRIVE_RESULTS"
echo "ROOT_CUDA_100K_GATE=PASS"
