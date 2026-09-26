# ROOT + CUDA multi-file pipeline

This project keeps ROOT responsible for `.root` I/O/deserialization on the CPU and sends plain numeric buffers to CUDA.

Two executables are kept deliberately:

- `root_cuda_multifile`: the original generic/synthetic teaching prototype (`Events`, `x_true`, `x_hat`, optional `energy`);
- `root_hits_energy`: the real ATLAS-HITS 100k pipeline for `CollectionTree`.

## Real dataset contract — `mc21_14TeV_Epos_100k`

Validated dataset:

- 8 ROOT files: `combined_mc21_14TeV_Epos_part1.root` ... `part8.root`;
- 100,000 entries total;
- tree: `CollectionTree`;
- identical schema in all files;
- energy branches:
  - `TileCalHit_energy`;
  - `LArHitEMB_energy`;
  - `LArHitEMEC_energy`;
  - `LArHitHEC_energy`;
  - `LArHitFCAL_energy`;
- event identifier: `EventNumber`.

Each energy branch is read as an array of `double`. Events are batched and flattened into one energy vector plus segment offsets. A segment is one `(event, calorimeter subsystem)` pair.

## CPU/GPU calculation

For every segment, CPU and GPU independently calculate:

- number of hits;
- sum of hit energies;
- maximum hit energy.

The CUDA implementation launches one block per segment. Threads walk the segment and cooperate through shared memory to reduce the sum and maximum. The output is validated segment-by-segment against the CPU reference.

A 100k run requires:

- exactly 100,000 events;
- exactly 500,000 event×subdetector segments;
- all segment comparisons PASS;
- no non-finite energy values.

## Timing

The run records separately:

- ROOT read + flatten/data-adapter time;
- CPU compute time;
- pageable-to-pinned host packing time;
- H2D transfer;
- CUDA kernel;
- D2H transfer.

This allows both compute-only and transfer-inclusive comparisons.

## Outputs

`root_hits_energy` writes:

- `summary.csv`;
- `per_file.csv`;
- `per_subdetector.csv`;
- `results.root` containing `EventDetectorMetrics` and `RESULT_GATE`;
- `run.log` when launched by the Colab script.

## Colab pipeline

```text
Google Drive/raw
      ↓
Colab local disk (SHA-256 verified)
      ↓
ROOT / CPU
      ↓
flattened buffers + offsets
      ↓
CUDA segmented reduction
      ↓
CPU/GPU numerical gates
      ↓
Colab results
      ↓
Google Drive/results/<RUN_ID>
```

Run from `notebooks/colab/root_hits_energy_100k.ipynb`, or manually after mounting Drive:

```bash
bash scripts/colab_root_cuda_bootstrap.sh
bash scripts/colab_root_hits_energy_run.sh
```

The top-level CMake option remains OFF by default so the CUDA-only CI does not require ROOT:

```bash
cmake -S . -B build \
  -DCUDA_LAB_ENABLE_ROOT_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --target root_hits_energy -j2
```
