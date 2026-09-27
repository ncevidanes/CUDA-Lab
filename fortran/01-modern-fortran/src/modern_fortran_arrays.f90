program modern_fortran_arrays
    use, intrinsic :: iso_fortran_env, only : real64, int64
    implicit none

    integer(int64), parameter :: n = 1000000_int64
    real(real64), allocatable :: x(:)
    real(real64) :: computed_sum, expected_sum, abs_error
    integer(int64) :: i

    allocate(x(n))

    do concurrent (i = 1_int64:n)
        x(i) = real(i, real64)
    end do

    computed_sum = sum(x)
    expected_sum = real(n, real64) * real(n + 1_int64, real64) / 2.0_real64
    abs_error = abs(computed_sum - expected_sum)

    print '(A,I0)', 'ELEMENTS=', n
    print '(A,ES24.16)', 'FORTRAN_SUM=', computed_sum
    print '(A,ES24.16)', 'EXPECTED_SUM=', expected_sum
    print '(A,ES12.4)', 'ABS_ERROR=', abs_error

    if (abs_error <= 1.0e-6_real64) then
        print '(A)', 'FORTRAN_F01_GATE=PASS'
    else
        print '(A)', 'FORTRAN_F01_GATE=FAIL'
        error stop 1
    end if
end program modern_fortran_arrays
