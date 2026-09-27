add_library(
    fortran_numeric
    STATIC
    src/fortran_numeric.f90
)

target_include_directories(
    fortran_numeric
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(
    fortran_f02_interop
    src/cpp_driver.cpp
)

target_link_libraries(
    fortran_f02_interop
    PRIVATE
        fortran_numeric
)

add_test(
    NAME FORTRAN_F02_C_INTEROP
    COMMAND fortran_f02_interop
)
