#pragma once

#include <cstddef>

extern "C" {

double fortran_sum(const double* x, std::size_t n);

void fortran_scale(double* x, std::size_t n, double alpha);

}
