add_executable(
    cuda_smoke
    src/cuda_smoke.cu
)

target_compile_options(
    cuda_smoke
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O2>
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
)

if(BUILD_TESTING)

    add_test(
        NAME cuda_smoke
        COMMAND cuda_smoke
    )

    set_tests_properties(
        cuda_smoke
        PROPERTIES
        PASS_REGULAR_EXPRESSION
        "CUDA_SMOKE_GATE=PASS"
    )

endif()
