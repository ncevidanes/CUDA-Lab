#include "fortran_numeric.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

double cpp_sum(const double* data, std::size_t n) {
    double total = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        total += data[i];
    }

    return total;
}

template <typename Fn>
double benchmark_ms(Fn&& fn, int repetitions, double& result) {
    using clock = std::chrono::steady_clock;

    double best_ms = std::numeric_limits<double>::infinity();

    for (int rep = 0; rep < repetitions; ++rep) {
        const auto start = clock::now();
        result = fn();
        const auto stop = clock::now();

        const std::chrono::duration<double, std::milli> elapsed =
            stop - start;

        best_ms = std::min(best_ms, elapsed.count());
    }

    return best_ms;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t n = 1u << 20;
    int repetitions = 10;

    if (argc > 1) {
        n = static_cast<std::size_t>(std::stoull(argv[1]));
    }

    if (argc > 2) {
        repetitions = std::stoi(argv[2]);
    }

    if (n == 0 || repetitions <= 0) {
        std::cerr
            << "Usage: fortran_f03_cpp_vs_fortran "
            << "[elements>0] [repetitions>0]\n";
        return 2;
    }

    std::vector<double> values(n);

    for (std::size_t i = 0; i < n; ++i) {
        values[i] =
            1.0 + static_cast<double>(i % 1024) * 1.0e-6;
    }

    const double reference =
        std::accumulate(values.begin(), values.end(), 0.0);

    double cpp_result = 0.0;
    double fortran_result = 0.0;

    const double cpp_ms = benchmark_ms(
        [&] {
            return cpp_sum(values.data(), values.size());
        },
        repetitions,
        cpp_result
    );

    const double fortran_ms = benchmark_ms(
        [&] {
            return fortran_sum(values.data(), values.size());
        },
        repetitions,
        fortran_result
    );

    const double cpp_error =
        std::abs(cpp_result - reference);

    const double fortran_error =
        std::abs(fortran_result - reference);

    const double tolerance =
        1.0e-9 * std::max(1.0, std::abs(reference));

    std::cout << std::setprecision(12);
    std::cout << "ELEMENTS=" << n << '\n';
    std::cout << "REPETITIONS=" << repetitions << '\n';
    std::cout << "CPP_MS=" << cpp_ms << '\n';
    std::cout << "FORTRAN_MS=" << fortran_ms << '\n';
    std::cout << "CPP_SUM=" << cpp_result << '\n';
    std::cout << "FORTRAN_SUM=" << fortran_result << '\n';
    std::cout << "REFERENCE_SUM=" << reference << '\n';
    std::cout << "CPP_ABS_ERROR=" << cpp_error << '\n';
    std::cout << "FORTRAN_ABS_ERROR=" << fortran_error << '\n';

    if (cpp_error <= tolerance &&
        fortran_error <= tolerance) {
        std::cout << "FORTRAN_F03_GATE=PASS\n";
        return 0;
    }

    std::cout << "FORTRAN_F03_GATE=FAIL\n";
    return 1;
}
