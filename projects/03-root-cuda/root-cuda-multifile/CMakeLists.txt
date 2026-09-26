find_package(ROOT REQUIRED COMPONENTS Core RIO Tree Hist)

add_executable(
    root_cuda_multifile
    src/root_cuda_multifile.cu
)

target_link_libraries(
    root_cuda_multifile
    PRIVATE
        ROOT::Core
        ROOT::RIO
        ROOT::Tree
        ROOT::Hist
)

set_target_properties(
    root_cuda_multifile
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
