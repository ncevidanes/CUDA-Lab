# ROOT + CUDA multi-file pipeline

This project integrates ROOT I/O with CUDA processing for multiple `.root` files.

## Input contract

Default schema:

- `TTree`: `Events`
- `x_true`: `std::vector<float>`
- `x_hat`: `std::vector<float>`
- `energy`: `std::vector<float>` (optional)

All names are configurable at runtime.

The vectors are flattened into pinned host buffers in batches. ROOT remains responsible for file I/O and deserialization; CUDA only sees plain numeric buffers.

## Calculations

For every batch, both CPU and GPU calculate:

- residual squared: `(x_hat - x_true)^2`;
- global RMSE via reduction;
- energy sum (when the energy branch exists);
- energy histogram (CPU loop versus CUDA `atomicAdd`).

GPU stages are timed independently:

- H2D;
- kernels/reductions/histogram;
- D2H.

ROOT read time and CPU compute time are also recorded.

## Outputs

The executable writes:

- `summary.csv`;
- `per_file.csv`;
- `histogram.csv`;
- `run_manifest.json`;
- `results.root` with CPU/GPU histograms and a `FileMetrics` tree.

`RESULT_GATE=PASS` requires CPU/GPU RMSE agreement, energy-sum agreement and exact histogram equality for all successfully processed files; any file-level `FAIL` fails the run. Unreadable files or files without the requested tree/required branches are reported as `SKIP` and preserved in the audit table.

## Build

The repository top-level option is intentionally OFF by default so existing CUDA-only CI does not require ROOT.

```bash
cmake -S . -B build \
  -DCUDA_LAB_ENABLE_ROOT_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --target root_cuda_multifile -j2
```

## Colab execution

After Google Drive is mounted in Colab:

```bash
bash scripts/colab_root_cuda_bootstrap.sh
bash scripts/colab_root_cuda_run.sh
```

Default Drive locations:

```text
input : /content/drive/MyDrive/CUDA-Lab/root-input
output: /content/drive/MyDrive/CUDA-Lab/root-results/<RUN_ID>
```

The run is first written to `/content/cuda-root-results/<RUN_ID>` and then mirrored to Drive.

### Branch overrides

```bash
ROOT_TREE_NAME=CollectionTree \
ROOT_TRUTH_BRANCH=x_true \
ROOT_RECO_BRANCH=x_hat \
ROOT_ENERGY_BRANCH=energy \
ROOT_BATCH_ELEMENTS=2097152 \
ROOT_CUDA_THREADS=256 \
  bash scripts/colab_root_cuda_run.sh
```

To disable the energy branch:

```bash
ROOT_ENERGY_BRANCH='' bash scripts/colab_root_cuda_run.sh
```

## External HD -> Drive staging

On the Ubuntu machine where the external HD is connected:

```bash
bash scripts/stage_root_hd_to_drive.sh
```

It uses the already-configured `gdrive-cuda:` rclone remote by default. If `EXTERNAL_ROOT_DIR` is not set, the script searches the usual Linux removable-media mount points and selects the directory containing the largest number of `.root` files.

For an explicit source:

```bash
EXTERNAL_ROOT_DIR=/media/$USER/HD/Dataset \
  bash scripts/stage_root_hd_to_drive.sh
```

The transfer is checksum-aware and never deletes the source or remote files.

## Smoke test

In Colab, with Drive mounted:

```bash
bash scripts/colab_root_cuda_smoke.sh
```

The smoke test generates four synthetic ROOT files, builds the ROOT+CUDA target, processes all files and writes the result bundle to Drive.
