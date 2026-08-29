#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool cuda_check(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr
        << "CUDA_ERROR_OPERATION=" << operation << '\n'
        << "CUDA_ERROR=" << cudaGetErrorString(status) << '\n';

    return false;
}

__global__
void vector_add(
    const float* a,
    const float* b,
    float* c,
    std::size_t n)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

}

int main()
{
    cudaDeviceProp prop{};

    if (!cuda_check(
            cudaGetDeviceProperties(&prop, 0),
            "cudaGetDeviceProperties")) {
        return EXIT_FAILURE;
    }

    constexpr std::size_t N = 1U << 20;
    constexpr std::size_t BYTES = N * sizeof(float);

    std::vector<float> h_a(N, 1.5F);
    std::vector<float> h_b(N, 2.5F);
    std::vector<float> h_c(N, 0.0F);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    bool ok = true;

    ok = ok && cuda_check(
        cudaMalloc(&d_a, BYTES),
        "cudaMalloc(a)");

    ok = ok && cuda_check(
        cudaMalloc(&d_b, BYTES),
        "cudaMalloc(b)");

    ok = ok && cuda_check(
        cudaMalloc(&d_c, BYTES),
        "cudaMalloc(c)");

    if (!ok) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return EXIT_FAILURE;
    }

    ok = ok && cuda_check(
        cudaMemcpy(
            d_a,
            h_a.data(),
            BYTES,
            cudaMemcpyHostToDevice),
        "cudaMemcpy(a)");

    ok = ok && cuda_check(
        cudaMemcpy(
            d_b,
            h_b.data(),
            BYTES,
            cudaMemcpyHostToDevice),
        "cudaMemcpy(b)");

    constexpr int THREADS = 256;

    const int blocks =
        static_cast<int>(
            (N + THREADS - 1) / THREADS);

    if (ok) {
        vector_add<<<blocks, THREADS>>>(
            d_a,
            d_b,
            d_c,
            N);

        ok = ok && cuda_check(
            cudaGetLastError(),
            "kernel_launch");

        ok = ok && cuda_check(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize");
    }

    if (ok) {
        ok = ok && cuda_check(
            cudaMemcpy(
                h_c.data(),
                d_c,
                BYTES,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(c)");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    if (!ok) {
        std::cout << "CUDA_SMOKE_GATE=FAIL\n";
        return EXIT_FAILURE;
    }

    float max_error = 0.0F;

    for (std::size_t i = 0; i < N; ++i) {
        const float expected =
            h_a[i] + h_b[i];

        max_error =
            std::max(
                max_error,
                std::fabs(h_c[i] - expected));
    }

    std::cout
        << "GPU_NAME=" << prop.name << '\n'
        << "ELEMENTS=" << N << '\n'
        << "THREADS_PER_BLOCK=" << THREADS << '\n'
        << "BLOCKS=" << blocks << '\n'
        << "MAX_ERROR=" << max_error << '\n';

    if (max_error == 0.0F) {
        std::cout << "CUDA_SMOKE_GATE=PASS\n";
        return EXIT_SUCCESS;
    }

    std::cout << "CUDA_SMOKE_GATE=FAIL\n";

    return EXIT_FAILURE;
}
