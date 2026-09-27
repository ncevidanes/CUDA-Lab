# CUDA-Lab

Laboratório progressivo para estudo e desenvolvimento em CUDA C++, agora com
uma trilha complementar de Modern Fortran para HPC científico.

## Arquitetura

O projeto separa código, dados e execução:

- GitHub: código-fonte, CMake, scripts e documentação.
- Ubuntu local: desenvolvimento e armazenamento principal.
- Google Drive: transporte de datasets e resultados.
- Google Colab: backend NVIDIA/CUDA.

## Trilha CUDA

A sequência pedagógica principal continua em CUDA C++:

1. Thread -> elemento
2. Shared memory + reduction
3. Memory access / coalescing
4. Atomics
5. Scan / prefix sum
6. Thrust / CUB
7. Streams
8. Pinned memory + async pipeline
9. AoS x SoA
10. Profiling / occupancy
11. Precisão / reprodutibilidade

A integração ROOT-CUDA é tratada como aplicação controlada desses conceitos.

## Trilha Fortran / HPC

Fortran foi incorporado como trilha paralela, sem substituir ROOT/C++ ou CUDA
C++.

```text
ROOT / C++
    |
    v
buffers planos
    |
    +--> CPU C++
    +--> CPU Fortran
    +--> CUDA C++
    \--> CUDA Fortran
```

Estado inicial:

1. F01 - Modern Fortran: implementado
2. F02 - ISO_C_BINDING C++ <-> Fortran: implementado
3. F03 - Benchmark C++ x Fortran: implementado
4. F04 - OpenMP Fortran: planejado
5. F05 - OpenACC: planejado
6. F06 - CUDA Fortran: planejado
7. F07 - CUDA C++ x CUDA Fortran: planejado
8. F08 - ROOT -> C++ -> Fortran: planejado
9. F09 - ROOT -> CUDA C++ / CUDA Fortran: planejado
10. F10 - Benchmark científico e reprodutibilidade: planejado

Detalhes: `fortran/README.md`.

## Build no Colab

CUDA-Lab tradicional:

```bash
bash scripts/colab_build.sh
```

Trilha Fortran, quando houver compilador Fortran disponível:

```bash
bash scripts/fortran_build.sh
```

O suporte Fortran é autodetectado. Se nenhum compilador Fortran estiver
disponível, os alvos Fortran são ignorados e a compilação CUDA continua
normalmente.

## Dados

Datasets e resultados produzidos durante as execuções não são versionados pelo
Git.

O fluxo de dados utiliza Google Drive e rclone.
