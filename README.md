# CUDA-Lab

Laboratório progressivo para estudo e desenvolvimento em CUDA C++.

## Arquitetura

O projeto separa código, dados e execução:

- GitHub: código-fonte, CMake, scripts e documentação.
- Ubuntu local: desenvolvimento e armazenamento principal.
- Google Drive: transporte de datasets e resultados.
- Google Colab: backend NVIDIA/CUDA.

## Roadmap

1. CUDA kernels
2. GPU memory
3. Thrust
4. CUB
5. cuRAND
6. cuBLAS
7. cuSPARSE
8. cuFFT
9. cuSOLVER

## Primeiro projeto

projects/01-kernels/cuda-smoke

O programa executa uma soma vetorial em GPU e valida
numericamente o resultado.

## Build no Colab

Comando:

    bash scripts/colab_build.sh

A execução é aprovada quando aparecem:

    CUDA_SMOKE_GATE=PASS
    CTEST_GATE=PASS

## Dados

Datasets e resultados produzidos durante as execuções não são
versionados pelo Git.

O fluxo de dados utiliza Google Drive e rclone.
