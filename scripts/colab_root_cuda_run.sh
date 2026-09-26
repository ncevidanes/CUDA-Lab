#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${ROOT_CUDA_BUILD_DIR:-/content/cuda-lab-build}"
BIN="${ROOT_CUDA_BINARY:-$BUILD_DIR/projects/03-root-cuda/root-cuda-multifile/root_cuda_multifile}"

INPUT_DIR="${ROOT_INPUT_DIR:-/content/drive/MyDrive/CUDA-Lab/root-input}"
DRIVE_OUTPUT_BASE="${ROOT_DRIVE_OUTPUT_BASE:-/content/drive/MyDrive/CUDA-Lab/root-results}"
LOCAL_OUTPUT_BASE="${ROOT_LOCAL_OUTPUT_BASE:-/content/cuda-root-results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_OUTPUT="$LOCAL_OUTPUT_BASE/$RUN_ID"
DRIVE_OUTPUT="$DRIVE_OUTPUT_BASE/$RUN_ID"
INPUT_LIST="$LOCAL_OUTPUT/input_files.txt"
LOG="$LOCAL_OUTPUT/run.log"

TREE_NAME="${ROOT_TREE_NAME:-Events}"
TRUTH_BRANCH="${ROOT_TRUTH_BRANCH:-x_true}"
RECO_BRANCH="${ROOT_RECO_BRANCH:-x_hat}"
ENERGY_BRANCH="${ROOT_ENERGY_BRANCH:-energy}"
BATCH_ELEMENTS="${ROOT_BATCH_ELEMENTS:-1048576}"
CUDA_THREADS="${ROOT_CUDA_THREADS:-256}"
HIST_BINS="${ROOT_HIST_BINS:-200}"
HIST_MIN="${ROOT_HIST_MIN:-0}"
HIST_MAX="${ROOT_HIST_MAX:-5000}"

mkdir -p "$LOCAL_OUTPUT" "$DRIVE_OUTPUT"

if [[ ! -x "$BIN" ]]; then
  echo "BINARY_GATE=FAIL path=$BIN" >&2
  exit 20
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "INPUT_DIR_GATE=FAIL path=$INPUT_DIR" >&2
  exit 21
fi

find "$INPUT_DIR" -type f -name '*.root' -print | sort > "$INPUT_LIST"
FILE_COUNT="$(wc -l < "$INPUT_LIST")"
if [[ "$FILE_COUNT" -eq 0 ]]; then
  echo "INPUT_FILE_GATE=FAIL count=0 path=$INPUT_DIR" >&2
  exit 22
fi

echo "==================================================" | tee "$LOG"
echo " CUDA LAB — ROOT + CUDA MULTI-FILE COLAB RUN" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "RUN_ID=$RUN_ID" | tee -a "$LOG"
echo "INPUT_DIR=$INPUT_DIR" | tee -a "$LOG"
echo "INPUT_FILE_COUNT=$FILE_COUNT" | tee -a "$LOG"
echo "LOCAL_OUTPUT=$LOCAL_OUTPUT" | tee -a "$LOG"
echo "DRIVE_OUTPUT=$DRIVE_OUTPUT" | tee -a "$LOG"

ARGS=(
  --input-list "$INPUT_LIST"
  --output-dir "$LOCAL_OUTPUT"
  --tree "$TREE_NAME"
  --truth "$TRUTH_BRANCH"
  --reco "$RECO_BRANCH"
  --batch-elements "$BATCH_ELEMENTS"
  --threads "$CUDA_THREADS"
  --hist-bins "$HIST_BINS"
  --hist-min "$HIST_MIN"
  --hist-max "$HIST_MAX"
)
if [[ -n "$ENERGY_BRANCH" ]]; then
  ARGS+=(--energy "$ENERGY_BRANCH")
else
  ARGS+=(--no-energy)
fi

set +e
"$BIN" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

rsync -a "$LOCAL_OUTPUT/" "$DRIVE_OUTPUT/"

if [[ "$STATUS" -ne 0 ]]; then
  echo "COLAB_PIPELINE_GATE=FAIL exit=$STATUS" | tee -a "$LOG"
  rsync -a "$LOCAL_OUTPUT/" "$DRIVE_OUTPUT/"
  exit "$STATUS"
fi

echo "COLAB_LOCAL_OUTPUT_GATE=PASS path=$LOCAL_OUTPUT" | tee -a "$LOG"
echo "DRIVE_OUTPUT_GATE=PASS path=$DRIVE_OUTPUT" | tee -a "$LOG"
echo "COLAB_PIPELINE_GATE=PASS" | tee -a "$LOG"
rsync -a "$LOCAL_OUTPUT/" "$DRIVE_OUTPUT/"
