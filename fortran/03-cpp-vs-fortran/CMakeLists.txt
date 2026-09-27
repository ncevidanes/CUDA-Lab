add_executable(
    fortran_f03_cpp_vs_fortran
    src/cpp_vs_fortran.cpp
)

target_link_libraries(
    fortran_f03_cpp_vs_fortran
    PRIVATE
        fortran_numeric
)

add_test(
    NAME FORTRAN_F03_CPP_VS_FORTRAN
    COMMAND fortran_f03_cpp_vs_fortran 262144 3
)
