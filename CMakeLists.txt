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

add_subdirectory(
    projects/01-kernels/cuda-smoke
)

add_subdirectory(
    projects/01-kernels/vector-add-file
)
