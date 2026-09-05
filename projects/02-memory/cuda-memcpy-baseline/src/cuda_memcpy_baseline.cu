#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>


namespace
{


bool cuda_check(
    cudaError_t status,
    const char* operation)
{
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr
        << "CUDA_ERROR_OPERATION="
        << operation
        << '\n'
        << "CUDA_ERROR="
        << cudaGetErrorString(status)
        << '\n';

    return false;
}


struct Arguments
{
    std::string output;
    int iterations = 20;
};


bool parse_arguments(
    int argc,
    char** argv,
    Arguments& args)
{
    for (int i = 1; i < argc; ++i) {

        const std::string argument = argv[i];

        if (
            argument == "--output"
            && i + 1 < argc
        ) {
            args.output = argv[++i];
        }

        else if (
            argument == "--iterations"
            && i + 1 < argc
        ) {
            try {
                args.iterations =
                    std::stoi(argv[++i]);
            }
            catch (...) {
                return false;
            }
        }

        else {
            return false;
        }
    }

    return
        !args.output.empty()
        && args.iterations > 0;
}


double average(
    const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }

    return
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        )
        / static_cast<double>(
            values.size()
        );
}


double minimum(
    const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }

    return *std::min_element(
        values.begin(),
        values.end()
    );
}


double bandwidth_gb_per_s(
    std::size_t bytes,
    double milliseconds)
{
    if (milliseconds <= 0.0) {
        return 0.0;
    }

    const double seconds =
        milliseconds / 1000.0;

    return
        static_cast<double>(bytes)
        / seconds
        / 1.0e9;
}


template <typename Function>
bool measure_transfer(
    int iterations,
    Function function,
    std::vector<double>& times_ms)
{
    times_ms.clear();
    times_ms.reserve(
        static_cast<std::size_t>(iterations)
    );

    for (int iteration = 0;
         iteration < iterations;
         ++iteration) {

        const auto start =
            std::chrono::steady_clock::now();

        if (!function()) {
            return false;
        }

        /*
         * We deliberately synchronize after every transfer.
         *
         * This makes this benchmark an application-visible
         * synchronous baseline, rather than measuring only
         * the enqueue cost of the CUDA Runtime call.
         */
        if (!cuda_check(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize")) {
            return false;
        }

        const auto stop =
            std::chrono::steady_clock::now();

        const double elapsed_ms =
            std::chrono::duration<
                double,
                std::milli
            >(stop - start).count();

        times_ms.push_back(elapsed_ms);
    }

    return true;
}


} // namespace


int main(
    int argc,
    char** argv)
{
    Arguments args;

    if (!parse_arguments(
            argc,
            argv,
            args)) {

        std::cerr
            << "USAGE=cuda_memcpy_baseline "
            << "--output FILE.csv "
            << "[--iterations N]\n"
            << "CUDA_MEMCPY_BASELINE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }


    cudaDeviceProp properties{};

    if (!cuda_check(
            cudaGetDeviceProperties(
                &properties,
                0
            ),
            "cudaGetDeviceProperties")) {

        std::cout
            << "CUDA_MEMCPY_BASELINE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }


    std::size_t free_device_bytes = 0;
    std::size_t total_device_bytes = 0;

    if (!cuda_check(
            cudaMemGetInfo(
                &free_device_bytes,
                &total_device_bytes
            ),
            "cudaMemGetInfo")) {

        std::cout
            << "CUDA_MEMCPY_BASELINE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }


    const std::vector<std::size_t> sizes_mib = {
        1,
        4,
        16,
        64,
        256
    };


    const std::filesystem::path output_path(
        args.output
    );

    if (output_path.has_parent_path()) {

        std::error_code error;

        std::filesystem::create_directories(
            output_path.parent_path(),
            error
        );

        if (error) {

            std::cerr
                << "OUTPUT_DIRECTORY_ERROR="
                << error.message()
                << '\n'
                << "CUDA_MEMCPY_BASELINE_GATE=FAIL\n";

            return EXIT_FAILURE;
        }
    }


    std::ofstream output(args.output);

    if (!output) {

        std::cerr
            << "OUTPUT_OPEN_GATE=FAIL\n"
            << "CUDA_MEMCPY_BASELINE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }


    output
        << "size_mib,"
        << "bytes,"
        << "iterations,"
        << "h2d_avg_ms,"
        << "h2d_min_ms,"
        << "h2d_GB_per_s,"
        << "d2h_avg_ms,"
        << "d2h_min_ms,"
        << "d2h_GB_per_s,"
        << "validation\n";


    std::cout
        << "GPU_NAME="
        << properties.name
        << '\n'
        << "HOST_MEMORY_TYPE=PAGEABLE\n"
        << "TRANSFER_API=cudaMemcpy\n"
        << "TRANSFER_MODE=SYNCHRONOUS_BASELINE\n"
        << "ITERATIONS="
        << args.iterations
        << '\n'
        << "DEVICE_MEMORY_TOTAL_BYTES="
        << total_device_bytes
        << '\n'
        << "DEVICE_MEMORY_FREE_BYTES="
        << free_device_bytes
        << '\n';


    bool overall_ok = true;


    for (const std::size_t mib : sizes_mib) {

        const std::size_t bytes =
            mib
            * 1024ULL
            * 1024ULL;

        const std::size_t elements =
            bytes / sizeof(std::uint32_t);


        std::vector<std::uint32_t> host_source(
            elements
        );

        std::vector<std::uint32_t> host_destination(
            elements,
            0U
        );


        for (std::size_t i = 0;
             i < elements;
             ++i) {

            host_source[i] =
                static_cast<std::uint32_t>(
                    i
                )
                ^ 0xA5A5A5A5U;
        }


        std::uint32_t* device_buffer = nullptr;


        if (!cuda_check(
                cudaMalloc(
                    reinterpret_cast<void**>(
                        &device_buffer
                    ),
                    bytes
                ),
                "cudaMalloc")) {

            overall_ok = false;
            break;
        }


        bool size_ok = true;


        /*
         * Warm-up transfers are not included in the
         * reported measurements.
         */
        for (int warmup = 0;
             warmup < 2;
             ++warmup) {

            size_ok = size_ok && cuda_check(
                cudaMemcpy(
                    device_buffer,
                    host_source.data(),
                    bytes,
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy_H2D_warmup"
            );

            size_ok = size_ok && cuda_check(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize_H2D_warmup"
            );

            size_ok = size_ok && cuda_check(
                cudaMemcpy(
                    host_destination.data(),
                    device_buffer,
                    bytes,
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy_D2H_warmup"
            );

            size_ok = size_ok && cuda_check(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize_D2H_warmup"
            );

            if (!size_ok) {
                break;
            }
        }


        std::vector<double> h2d_times;
        std::vector<double> d2h_times;


        if (size_ok) {

            size_ok = measure_transfer(
                args.iterations,

                [&]() {
                    return cuda_check(
                        cudaMemcpy(
                            device_buffer,
                            host_source.data(),
                            bytes,
                            cudaMemcpyHostToDevice
                        ),
                        "cudaMemcpy_H2D"
                    );
                },

                h2d_times
            );
        }


        if (size_ok) {

            size_ok = measure_transfer(
                args.iterations,

                [&]() {
                    return cuda_check(
                        cudaMemcpy(
                            host_destination.data(),
                            device_buffer,
                            bytes,
                            cudaMemcpyDeviceToHost
                        ),
                        "cudaMemcpy_D2H"
                    );
                },

                d2h_times
            );
        }


        bool validation_ok = false;

        if (size_ok) {

            validation_ok =
                host_source
                == host_destination;
        }


        const double h2d_avg =
            average(h2d_times);

        const double h2d_min =
            minimum(h2d_times);

        const double d2h_avg =
            average(d2h_times);

        const double d2h_min =
            minimum(d2h_times);

        const double h2d_bandwidth =
            bandwidth_gb_per_s(
                bytes,
                h2d_avg
            );

        const double d2h_bandwidth =
            bandwidth_gb_per_s(
                bytes,
                d2h_avg
            );


        output
            << mib
            << ','
            << bytes
            << ','
            << args.iterations
            << ','
            << std::fixed
            << std::setprecision(6)
            << h2d_avg
            << ','
            << h2d_min
            << ','
            << h2d_bandwidth
            << ','
            << d2h_avg
            << ','
            << d2h_min
            << ','
            << d2h_bandwidth
            << ','
            << (
                validation_ok
                ? "PASS"
                : "FAIL"
            )
            << '\n';


        std::cout
            << "SIZE_MIB="
            << mib
            << " H2D_AVG_MS="
            << std::fixed
            << std::setprecision(6)
            << h2d_avg
            << " H2D_GB_PER_S="
            << h2d_bandwidth
            << " D2H_AVG_MS="
            << d2h_avg
            << " D2H_GB_PER_S="
            << d2h_bandwidth
            << " VALIDATION="
            << (
                validation_ok
                ? "PASS"
                : "FAIL"
            )
            << '\n';


        if (!size_ok || !validation_ok) {
            overall_ok = false;
        }


        cudaFree(device_buffer);


        if (!overall_ok) {
            break;
        }
    }


    output.flush();


    if (!output) {
        overall_ok = false;
    }


    if (overall_ok) {

        std::cout
            << "CUDA_MEMCPY_BASELINE_GATE=PASS\n";

        return EXIT_SUCCESS;
    }


    std::cout
        << "CUDA_MEMCPY_BASELINE_GATE=FAIL\n";

    return EXIT_FAILURE;
}
