#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-fortran}"

cmake     -S .     -B "${BUILD_DIR}"     -DCMAKE_BUILD_TYPE=Release     -DCUDA_LAB_ENABLE_FORTRAN=ON

cmake --build "${BUILD_DIR}" -j"$(nproc)"

ctest     --test-dir "${BUILD_DIR}"     --output-on-failure     -R '^FORTRAN_'
