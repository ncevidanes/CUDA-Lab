#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${ROOT_ENV_PREFIX:-/content/root-cuda-env}"
SMOKE_INPUT="${SMOKE_INPUT:-/content/root-cuda-smoke-input}"

bash "$REPO_ROOT/scripts/colab_root_cuda_bootstrap.sh"
rm -rf "$SMOKE_INPUT"
mkdir -p "$SMOKE_INPUT"
micromamba run -p "$ENV_PREFIX" python \
  "$REPO_ROOT/projects/03-root-cuda/root-cuda-multifile/tools/generate_test_root.py" \
  --output-dir "$SMOKE_INPUT" --files 4 --events 1000 --seed 9512

ROOT_INPUT_DIR="$SMOKE_INPUT" \
ROOT_DRIVE_OUTPUT_BASE="${ROOT_DRIVE_OUTPUT_BASE:-/content/drive/MyDrive/CUDA-Lab/root-smoke-results}" \
RUN_ID="smoke-$(date -u +%Y%m%dT%H%M%SZ)" \
  bash "$REPO_ROOT/scripts/colab_root_cuda_run.sh"
