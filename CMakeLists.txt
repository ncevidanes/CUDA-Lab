cmake_minimum_required(VERSION 3.24)

project(
    CUDALab
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CTest)
include(CheckLanguage)

option(
    CUDA_LAB_ENABLE_FORTRAN
    "Build the complementary Fortran/HPC track when a Fortran compiler is available"
    ON
)

add_subdirectory(
    projects/01-kernels/cuda-smoke
)

add_subdirectory(
    projects/01-kernels/vector-add-file
)

add_subdirectory(
    projects/02-memory/cuda-memcpy-baseline
)

if(CUDA_LAB_ENABLE_FORTRAN)
    check_language(Fortran)

    if(CMAKE_Fortran_COMPILER)
        enable_language(Fortran)

        set(CMAKE_Fortran_STANDARD 2008)
        set(CMAKE_Fortran_STANDARD_REQUIRED ON)

        message(
            STATUS
            "Fortran/HPC track enabled with compiler: ${CMAKE_Fortran_COMPILER}"
        )

        add_subdirectory(fortran)
    else()
        message(
            STATUS
            "Fortran compiler not found; Fortran/HPC track will be skipped"
        )
    endif()
endif()
