add_executable(
    fortran_f01_arrays
    src/modern_fortran_arrays.f90
)

add_test(
    NAME FORTRAN_F01_MODERN_ARRAYS
    COMMAND fortran_f01_arrays
)
