#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${ROOT_ENV_PREFIX:-/content/root-cuda-env}"
BUILD_DIR="${ROOT_CUDA_BUILD_DIR:-/content/cuda-lab-build}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU_GATE=FAIL reason=nvidia-smi_not_found" >&2
  exit 10
fi
nvidia-smi >/dev/null

echo "GPU_GATE=PASS"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || true

if ! command -v micromamba >/dev/null 2>&1; then
  mkdir -p /content/bin
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /content/bin --strip-components=1 bin/micromamba
  export PATH="/content/bin:$PATH"
fi

if [[ ! -x "$ENV_PREFIX/bin/root-config" ]]; then
  micromamba create -y -p "$ENV_PREFIX" -c conda-forge \
    'root>=6.36,<6.41' cmake ninja python
fi

ROOT_VERSION="$(micromamba run -p "$ENV_PREFIX" root-config --version)"
echo "ROOT_VERSION=$ROOT_VERSION"

CUDA_ARCH="${CUDA_ARCH:-}"
if [[ -z "$CUDA_ARCH" ]]; then
  CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '. ' || true)"
fi
if [[ -z "$CUDA_ARCH" ]]; then
  CUDA_ARCH=75
fi

echo "CUDA_ARCH=$CUDA_ARCH"

rm -rf "$BUILD_DIR"
micromamba run -p "$ENV_PREFIX" cmake \
  -S "$REPO_ROOT" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCUDA_LAB_ENABLE_ROOT_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH="$ENV_PREFIX"

micromamba run -p "$ENV_PREFIX" cmake --build "$BUILD_DIR" --target root_cuda_multifile -j2

BIN="$BUILD_DIR/projects/03-root-cuda/root-cuda-multifile/root_cuda_multifile"
[[ -x "$BIN" ]] || { echo "BUILD_GATE=FAIL" >&2; exit 11; }

echo "ROOT_CUDA_BINARY=$BIN"
echo "BUILD_GATE=PASS"
