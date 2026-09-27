module cuda_lab_fortran_api
    use, intrinsic :: iso_c_binding, only : c_double, c_size_t
    implicit none
    private

    public :: fortran_sum, fortran_scale

contains

    function fortran_sum(x, n) result(total) bind(C, name='fortran_sum')
        real(c_double), intent(in) :: x(*)
        integer(c_size_t), value, intent(in) :: n
        real(c_double) :: total
        integer(c_size_t) :: i

        total = 0.0_c_double

        do i = 1_c_size_t, n
            total = total + x(i)
        end do
    end function fortran_sum

    subroutine fortran_scale(x, n, alpha) bind(C, name='fortran_scale')
        real(c_double), intent(inout) :: x(*)
        integer(c_size_t), value, intent(in) :: n
        real(c_double), value, intent(in) :: alpha
        integer(c_size_t) :: i

        do i = 1_c_size_t, n
            x(i) = alpha * x(i)
        end do
    end subroutine fortran_scale

end module cuda_lab_fortran_api
