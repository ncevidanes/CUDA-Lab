#include "fortran_numeric.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    constexpr std::size_t n = 1024;
    std::vector<double> values(n);
    std::iota(values.begin(), values.end(), 1.0);

    const double expected_sum =
        static_cast<double>(n) * static_cast<double>(n + 1) / 2.0;

    const double sum_before =
        fortran_sum(values.data(), values.size());

    fortran_scale(values.data(), values.size(), 0.5);

    const double sum_after =
        fortran_sum(values.data(), values.size());

    const double error_before =
        std::abs(sum_before - expected_sum);

    const double error_after =
        std::abs(sum_after - expected_sum * 0.5);

    std::cout << "ELEMENTS=" << n << '\n';
    std::cout << "SUM_BEFORE=" << sum_before << '\n';
    std::cout << "SUM_AFTER=" << sum_after << '\n';
    std::cout << "ABS_ERROR_BEFORE=" << error_before << '\n';
    std::cout << "ABS_ERROR_AFTER=" << error_after << '\n';

    if (error_before <= 1.0e-12 && error_after <= 1.0e-12) {
        std::cout << "FORTRAN_F02_GATE=PASS\n";
        return 0;
    }

    std::cout << "FORTRAN_F02_GATE=FAIL\n";
    return 1;
}
