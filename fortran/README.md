# Fortran / HPC Track

Trilha complementar do CUDA-Lab para computação científica com Modern Fortran,
interoperabilidade C/C++ e, posteriormente, programação acelerada em GPU.

A trilha principal CUDA C++ continua independente. Fortran entra como uma
segunda camada computacional, sem substituir ROOT/C++ nem os Warm-ups CUDA.

## Arquitetura

```text
ROOT / C++
    |
    v
buffers planos
    |
    +--> CPU C++
    +--> CPU Fortran
    +--> CUDA C++
    \--> CUDA Fortran   (fase futura)
```

## Roadmap

| Etapa | Tema | Estado |
| --- | --- | --- |
| F01 | Modern Fortran: arrays, tipos e redução | Implementado |
| F02 | ISO_C_BINDING: C++ <-> Fortran | Implementado |
| F03 | Benchmark C++ x Fortran | Implementado |
| F04 | OpenMP Fortran | Planejado |
| F05 | OpenACC | Planejado |
| F06 | CUDA Fortran | Planejado |
| F07 | CUDA C++ x CUDA Fortran | Planejado |
| F08 | ROOT -> C++ -> Fortran | Planejado |
| F09 | ROOT -> CUDA C++ / CUDA Fortran | Planejado |
| F10 | Benchmark científico e reprodutibilidade | Planejado |

## Build

O suporte Fortran é autodetectado pelo CMake. Em um ambiente com CUDA e
compilador Fortran:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_LAB_ENABLE_FORTRAN=ON
cmake --build build -j
ctest --test-dir build --output-on-failure -R '^FORTRAN_'
```

Ou:

```bash
bash scripts/fortran_build.sh
```

Se nenhum compilador Fortran for encontrado, os alvos desta trilha são
ignorados e o CUDA-Lab continua compilando normalmente.

## CUDA Fortran

CUDA Fortran será introduzido apenas em F06. Essa etapa exigirá o NVIDIA HPC
SDK / `nvfortran`. O primeiro marco usa somente Fortran padrão + ISO_C_BINDING
para manter a base portátil e testável com GNU Fortran.
