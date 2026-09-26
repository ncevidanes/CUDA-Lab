find_package(ROOT REQUIRED COMPONENTS Core RIO Tree Hist)

add_executable(
    root_cuda_multifile
    src/root_cuda_multifile.cu
)

add_executable(
    root_hits_energy
    src/root_hits_energy.cu
)

target_include_directories(
    root_hits_energy
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach(target root_cuda_multifile root_hits_energy)
    target_link_libraries(
        ${target}
        PRIVATE
            ROOT::Core
            ROOT::RIO
            ROOT::Tree
            ROOT::Hist
    )

    set_target_properties(
        ${target}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()
