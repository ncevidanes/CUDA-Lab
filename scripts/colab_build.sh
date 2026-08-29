#!/usr/bin/env bash

set +e
set +u
set +o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"

echo "======================================================"
echo " CUDA LAB — COLAB BUILD"
echo "======================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA_SMI_GATE=PASS"
else
    echo "NVIDIA_SMI_GATE=FAIL"
fi

if command -v nvcc >/dev/null 2>&1; then
    echo "NVCC_GATE=PASS"
    nvcc --version | tail -1
else
    echo "NVCC_GATE=FAIL"
fi

GPU_NAME="$(
    nvidia-smi \
        --query-gpu=name \
        --format=csv,noheader \
        2>/dev/null |
    head -1
)"

COMPUTE_CAP="$(
    nvidia-smi \
        --query-gpu=compute_cap \
        --format=csv,noheader \
        2>/dev/null |
    head -1 |
    tr -d ' '
)"

CUDA_ARCH="${COMPUTE_CAP/.}"

echo "GPU_NAME=$GPU_NAME"
echo "COMPUTE_CAPABILITY=$COMPUTE_CAP"
echo "CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"

cmake \
    -S "$ROOT" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

CONFIG_RC=$?

if [ "$CONFIG_RC" -eq 0 ]; then
    echo "CMAKE_CONFIGURE_GATE=PASS"
else
    echo "CMAKE_CONFIGURE_GATE=FAIL"
fi

if [ "$CONFIG_RC" -eq 0 ]; then
    cmake --build "$BUILD_DIR" --parallel
    BUILD_RC=$?
else
    BUILD_RC=1
fi

if [ "$BUILD_RC" -eq 0 ]; then
    echo "CMAKE_BUILD_GATE=PASS"
else
    echo "CMAKE_BUILD_GATE=FAIL"
fi

if [ "$BUILD_RC" -eq 0 ]; then
    ctest \
        --test-dir "$BUILD_DIR" \
        --output-on-failure

    TEST_RC=$?
else
    TEST_RC=1
fi

if [ "$TEST_RC" -eq 0 ]; then
    echo "CTEST_GATE=PASS"
else
    echo "CTEST_GATE=FAIL"
fi

echo "======================================================"
echo " CUDA LAB — COLAB BUILD FINALIZADO"
echo "======================================================"
