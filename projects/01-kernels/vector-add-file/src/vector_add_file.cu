#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

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

struct Arguments
{
    std::string input;
    std::string output;
};

bool parse_arguments(
    int argc,
    char** argv,
    Arguments& args)
{
    for (int i = 1; i < argc; ++i) {

        const std::string arg = argv[i];

        if (arg == "--input" && i + 1 < argc) {
            args.input = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            args.output = argv[++i];
        }
        else {
            std::cerr
                << "ARGUMENT_ERROR=" << arg << '\n';
            return false;
        }
    }

    return
        !args.input.empty()
        && !args.output.empty();
}

bool read_input(
    const std::string& path,
    std::vector<std::size_t>& indices,
    std::vector<float>& a,
    std::vector<float>& b)
{
    std::ifstream input(path);

    if (!input) {
        std::cerr
            << "INPUT_OPEN_GATE=FAIL\n";
        return false;
    }

    std::string line;

    if (!std::getline(input, line)) {
        std::cerr
            << "INPUT_HEADER_GATE=FAIL\n";
        return false;
    }

    if (line != "index,a,b") {
        std::cerr
            << "INPUT_HEADER_GATE=FAIL\n"
            << "INPUT_HEADER=" << line << '\n';
        return false;
    }

    std::size_t expected_index = 0;

    while (std::getline(input, line)) {

        if (line.empty()) {
            continue;
        }

        std::stringstream stream(line);

        std::string index_text;
        std::string a_text;
        std::string b_text;

        if (!std::getline(stream, index_text, ',') ||
            !std::getline(stream, a_text, ',') ||
            !std::getline(stream, b_text, ',')) {

            std::cerr
                << "CSV_PARSE_GATE=FAIL\n"
                << "CSV_LINE=" << line << '\n';

            return false;
        }

        try {

            const std::size_t index =
                static_cast<std::size_t>(
                    std::stoull(index_text));

            const float value_a =
                std::stof(a_text);

            const float value_b =
                std::stof(b_text);

            if (index != expected_index) {
                std::cerr
                    << "INDEX_SEQUENCE_GATE=FAIL\n"
                    << "EXPECTED_INDEX="
                    << expected_index << '\n'
                    << "OBSERVED_INDEX="
                    << index << '\n';

                return false;
            }

            indices.push_back(index);
            a.push_back(value_a);
            b.push_back(value_b);

            ++expected_index;
        }
        catch (const std::exception& e) {

            std::cerr
                << "CSV_CONVERSION_GATE=FAIL\n"
                << "CSV_ERROR=" << e.what() << '\n';

            return false;
        }
    }

    return
        !indices.empty()
        && indices.size() == a.size()
        && a.size() == b.size();
}

bool write_output(
    const std::string& path,
    const std::vector<std::size_t>& indices,
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c)
{
    const std::filesystem::path output_path(path);

    if (output_path.has_parent_path()) {
        std::error_code ec;

        std::filesystem::create_directories(
            output_path.parent_path(),
            ec);

        if (ec) {
            std::cerr
                << "OUTPUT_DIRECTORY_GATE=FAIL\n"
                << "OUTPUT_DIRECTORY_ERROR="
                << ec.message() << '\n';

            return false;
        }
    }

    std::ofstream output(path);

    if (!output) {
        std::cerr
            << "OUTPUT_OPEN_GATE=FAIL\n";
        return false;
    }

    output
        << "index,a,b,c\n"
        << std::fixed
        << std::setprecision(1);

    for (std::size_t i = 0; i < c.size(); ++i) {
        output
            << indices[i] << ','
            << a[i] << ','
            << b[i] << ','
            << c[i] << '\n';
    }

    return static_cast<bool>(output);
}

}

int main(int argc, char** argv)
{
    Arguments args;

    if (!parse_arguments(argc, argv, args)) {

        std::cerr
            << "USAGE=vector_add_file "
            << "--input INPUT.csv "
            << "--output OUTPUT.csv\n"
            << "VECTOR_ADD_FILE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }

    std::vector<std::size_t> indices;
    std::vector<float> h_a;
    std::vector<float> h_b;

    if (!read_input(
            args.input,
            indices,
            h_a,
            h_b)) {

        std::cout
            << "VECTOR_ADD_FILE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }

    const std::size_t n = h_a.size();

    std::vector<float> h_c(n, 0.0F);

    const std::size_t bytes =
        n * sizeof(float);

    cudaDeviceProp prop{};

    if (!cuda_check(
            cudaGetDeviceProperties(&prop, 0),
            "cudaGetDeviceProperties")) {

        std::cout
            << "VECTOR_ADD_FILE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    bool ok = true;

    ok = ok && cuda_check(
        cudaMalloc(&d_a, bytes),
        "cudaMalloc(a)");

    ok = ok && cuda_check(
        cudaMalloc(&d_b, bytes),
        "cudaMalloc(b)");

    ok = ok && cuda_check(
        cudaMalloc(&d_c, bytes),
        "cudaMalloc(c)");

    if (ok) {

        ok = ok && cuda_check(
            cudaMemcpy(
                d_a,
                h_a.data(),
                bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(a)");

        ok = ok && cuda_check(
            cudaMemcpy(
                d_b,
                h_b.data(),
                bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(b)");
    }

    constexpr int threads = 256;

    const int blocks =
        static_cast<int>(
            (n + threads - 1) / threads);

    if (ok) {

        vector_add<<<blocks, threads>>>(
            d_a,
            d_b,
            d_c,
            n);

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
                bytes,
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(c)");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    if (!ok) {

        std::cout
            << "VECTOR_ADD_FILE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }

    float max_error = 0.0F;

    for (std::size_t i = 0; i < n; ++i) {

        const float expected =
            h_a[i] + h_b[i];

        max_error =
            std::max(
                max_error,
                std::fabs(h_c[i] - expected));
    }

    if (!write_output(
            args.output,
            indices,
            h_a,
            h_b,
            h_c)) {

        std::cout
            << "VECTOR_ADD_FILE_GATE=FAIL\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "GPU_NAME=" << prop.name << '\n'
        << "INPUT_FILE=" << args.input << '\n'
        << "OUTPUT_FILE=" << args.output << '\n'
        << "ELEMENTS=" << n << '\n'
        << "THREADS_PER_BLOCK=" << threads << '\n'
        << "BLOCKS=" << blocks << '\n'
        << "MAX_ERROR=" << max_error << '\n';

    if (max_error == 0.0F) {

        std::cout
            << "VECTOR_ADD_FILE_GATE=PASS\n";

        return EXIT_SUCCESS;
    }

    std::cout
        << "VECTOR_ADD_FILE_GATE=FAIL\n";

    return EXIT_FAILURE;
}
