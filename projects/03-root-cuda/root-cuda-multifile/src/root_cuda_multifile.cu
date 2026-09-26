#include <cuda_runtime.h>

#include <TFile.h>
#include <TH1D.h>
#include <TNamed.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::ostringstream oss__;                                           \
            oss__ << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": " \
                  << cudaGetErrorString(err__);                                 \
            throw std::runtime_error(oss__.str());                              \
        }                                                                       \
    } while (0)

struct Config {
    std::string input_list;
    std::string output_dir;
    std::string tree_name = "Events";
    std::string truth_branch = "x_true";
    std::string reco_branch = "x_hat";
    std::string energy_branch = "energy";
    std::size_t batch_elements = 1u << 20;
    int threads = 256;
    int hist_bins = 200;
    double hist_min = 0.0;
    double hist_max = 5000.0;
};

struct Timing {
    double root_read_ms = 0.0;
    double cpu_compute_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
};

struct Metrics {
    std::uint64_t events = 0;
    std::uint64_t elements = 0;
    long double cpu_sqsum = 0.0L;
    long double gpu_sqsum = 0.0L;
    long double cpu_energy_sum = 0.0L;
    long double gpu_energy_sum = 0.0L;
    Timing timing;
    std::vector<std::uint64_t> cpu_hist;
    std::vector<std::uint64_t> gpu_hist;
    bool has_energy = false;
};

struct FileResult {
    std::string path;
    Metrics metrics;
    std::string status = "PASS";
    std::string message;
};

__global__ void transform_metrics_kernel(const float* truth,
                                         const float* reco,
                                         const float* energy,
                                         double* squared_residual,
                                         double* energy_as_double,
                                         std::size_t n,
                                         bool has_energy) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double r =
        static_cast<double>(reco[i]) - static_cast<double>(truth[i]);
    squared_residual[i] = r * r;
    if (has_energy) {
        energy_as_double[i] = static_cast<double>(energy[i]);
    }
}

__global__ void reduce_sum_kernel(const double* input,
                                  double* output,
                                  std::size_t n) {
    extern __shared__ double sdata[];
    const unsigned int tid = threadIdx.x;
    const std::size_t base =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2;
    const std::size_t i0 = base + tid;
    const std::size_t i1 = i0 + blockDim.x;

    double v = 0.0;
    if (i0 < n) v += input[i0];
    if (i1 < n) v += input[i1];
    sdata[tid] = v;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void histogram_kernel(const float* values,
                                 unsigned long long* bins,
                                 std::size_t n,
                                 double xmin,
                                 double xmax,
                                 int nbins) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double x = static_cast<double>(values[i]);
    if (x < xmin || x >= xmax) return;

    const double fraction = (x - xmin) / (xmax - xmin);
    const int bin =
        static_cast<int>(fraction * static_cast<double>(nbins));

    if (bin >= 0 && bin < nbins) {
        atomicAdd(&bins[bin], 1ULL);
    }
}

class GpuWorkspace {
public:
    struct BatchResult {
        double sqsum = 0.0;
        double energy_sum = 0.0;
        double h2d_ms = 0.0;
        double kernel_ms = 0.0;
        double d2h_ms = 0.0;
        std::vector<std::uint64_t> hist;
    };

    GpuWorkspace(std::size_t capacity, int threads, int hist_bins)
        : capacity_(capacity),
          threads_(threads),
          hist_bins_(hist_bins) {
        if (capacity_ == 0) {
            throw std::runtime_error("batch capacity must be > 0");
        }
        if (threads_ <= 0 ||
            (threads_ & (threads_ - 1)) != 0 ||
            threads_ > 1024) {
            throw std::runtime_error(
                "--threads must be a power of two in [1, 1024]");
        }

        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&ev0_));
        CUDA_CHECK(cudaEventCreate(&ev1_));
        CUDA_CHECK(cudaEventCreate(&ev2_));
        CUDA_CHECK(cudaEventCreate(&ev3_));

        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&h_truth_),
            capacity_ * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&h_reco_),
            capacity_ * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&h_energy_),
            capacity_ * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&h_metric_sums_),
            2 * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&h_hist_),
            hist_bins_ * sizeof(unsigned long long)));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_truth_),
            capacity_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_reco_),
            capacity_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_energy_),
            capacity_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_sq_),
            capacity_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_energy_double_),
            capacity_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_metric_sums_),
            2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_hist_),
            hist_bins_ * sizeof(unsigned long long)));

        scratch_capacity_ =
            std::max<std::size_t>(1, reduce_blocks(capacity_));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_scratch_a_),
            scratch_capacity_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_scratch_b_),
            scratch_capacity_ * sizeof(double)));
    }

    ~GpuWorkspace() {
        if (d_scratch_b_) cudaFree(d_scratch_b_);
        if (d_scratch_a_) cudaFree(d_scratch_a_);
        if (d_hist_) cudaFree(d_hist_);
        if (d_metric_sums_) cudaFree(d_metric_sums_);
        if (d_energy_double_) cudaFree(d_energy_double_);
        if (d_sq_) cudaFree(d_sq_);
        if (d_energy_) cudaFree(d_energy_);
        if (d_reco_) cudaFree(d_reco_);
        if (d_truth_) cudaFree(d_truth_);

        if (h_hist_) cudaFreeHost(h_hist_);
        if (h_metric_sums_) cudaFreeHost(h_metric_sums_);
        if (h_energy_) cudaFreeHost(h_energy_);
        if (h_reco_) cudaFreeHost(h_reco_);
        if (h_truth_) cudaFreeHost(h_truth_);

        if (ev3_) cudaEventDestroy(ev3_);
        if (ev2_) cudaEventDestroy(ev2_);
        if (ev1_) cudaEventDestroy(ev1_);
        if (ev0_) cudaEventDestroy(ev0_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    float* h_truth() { return h_truth_; }
    float* h_reco() { return h_reco_; }
    float* h_energy() { return h_energy_; }

    BatchResult process(std::size_t n,
                        bool has_energy,
                        double hist_min,
                        double hist_max) {
        if (n == 0 || n > capacity_) {
            throw std::runtime_error("invalid GPU batch size");
        }

        CUDA_CHECK(cudaEventRecord(ev0_, stream_));

        CUDA_CHECK(cudaMemcpyAsync(
            d_truth_,
            h_truth_,
            n * sizeof(float),
            cudaMemcpyHostToDevice,
            stream_));
        CUDA_CHECK(cudaMemcpyAsync(
            d_reco_,
            h_reco_,
            n * sizeof(float),
            cudaMemcpyHostToDevice,
            stream_));
        if (has_energy) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_energy_,
                h_energy_,
                n * sizeof(float),
                cudaMemcpyHostToDevice,
                stream_));
        }

        CUDA_CHECK(cudaEventRecord(ev1_, stream_));

        const int blocks =
            static_cast<int>((n + threads_ - 1) / threads_);

        transform_metrics_kernel<<<blocks, threads_, 0, stream_>>>(
            d_truth_,
            d_reco_,
            d_energy_,
            d_sq_,
            d_energy_double_,
            n,
            has_energy);
        CUDA_CHECK(cudaGetLastError());

        const double* sq_final = reduce_sum(d_sq_, n);
        CUDA_CHECK(cudaMemcpyAsync(
            d_metric_sums_,
            sq_final,
            sizeof(double),
            cudaMemcpyDeviceToDevice,
            stream_));

        if (has_energy) {
            const double* e_final =
                reduce_sum(d_energy_double_, n);
            CUDA_CHECK(cudaMemcpyAsync(
                d_metric_sums_ + 1,
                e_final,
                sizeof(double),
                cudaMemcpyDeviceToDevice,
                stream_));

            CUDA_CHECK(cudaMemsetAsync(
                d_hist_,
                0,
                hist_bins_ * sizeof(unsigned long long),
                stream_));

            histogram_kernel<<<blocks, threads_, 0, stream_>>>(
                d_energy_,
                d_hist_,
                n,
                hist_min,
                hist_max,
                hist_bins_);
            CUDA_CHECK(cudaGetLastError());
        } else {
            CUDA_CHECK(cudaMemsetAsync(
                d_metric_sums_ + 1,
                0,
                sizeof(double),
                stream_));
            CUDA_CHECK(cudaMemsetAsync(
                d_hist_,
                0,
                hist_bins_ * sizeof(unsigned long long),
                stream_));
        }

        CUDA_CHECK(cudaEventRecord(ev2_, stream_));

        CUDA_CHECK(cudaMemcpyAsync(
            h_metric_sums_,
            d_metric_sums_,
            2 * sizeof(double),
            cudaMemcpyDeviceToHost,
            stream_));
        CUDA_CHECK(cudaMemcpyAsync(
            h_hist_,
            d_hist_,
            hist_bins_ * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            stream_));

        CUDA_CHECK(cudaEventRecord(ev3_, stream_));
        CUDA_CHECK(cudaEventSynchronize(ev3_));

        float h2d = 0.0f;
        float kernel = 0.0f;
        float d2h = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &h2d, ev0_, ev1_));
        CUDA_CHECK(cudaEventElapsedTime(
            &kernel, ev1_, ev2_));
        CUDA_CHECK(cudaEventElapsedTime(
            &d2h, ev2_, ev3_));

        BatchResult result;
        result.sqsum = h_metric_sums_[0];
        result.energy_sum = h_metric_sums_[1];
        result.h2d_ms = h2d;
        result.kernel_ms = kernel;
        result.d2h_ms = d2h;
        result.hist.assign(hist_bins_, 0);

        for (int i = 0; i < hist_bins_; ++i) {
            result.hist[i] =
                static_cast<std::uint64_t>(h_hist_[i]);
        }

        return result;
    }

private:
    std::size_t reduce_blocks(std::size_t n) const {
        return
            (n + static_cast<std::size_t>(threads_) * 2 - 1) /
            (static_cast<std::size_t>(threads_) * 2);
    }

    const double* reduce_sum(const double* input,
                             std::size_t n) {
        const double* current = input;
        std::size_t current_n = n;
        bool use_a = true;

        while (current_n > 1) {
            const std::size_t blocks =
                reduce_blocks(current_n);

            double* out =
                use_a ? d_scratch_a_ : d_scratch_b_;

            reduce_sum_kernel<<<
                static_cast<int>(blocks),
                threads_,
                threads_ * sizeof(double),
                stream_>>>(
                    current,
                    out,
                    current_n);
            CUDA_CHECK(cudaGetLastError());

            current = out;
            current_n = blocks;
            use_a = !use_a;
        }

        return current;
    }

    std::size_t capacity_ = 0;
    int threads_ = 256;
    int hist_bins_ = 0;
    std::size_t scratch_capacity_ = 0;

    cudaStream_t stream_{};
    cudaEvent_t ev0_{};
    cudaEvent_t ev1_{};
    cudaEvent_t ev2_{};
    cudaEvent_t ev3_{};

    float* h_truth_ = nullptr;
    float* h_reco_ = nullptr;
    float* h_energy_ = nullptr;
    double* h_metric_sums_ = nullptr;
    unsigned long long* h_hist_ = nullptr;

    float* d_truth_ = nullptr;
    float* d_reco_ = nullptr;
    float* d_energy_ = nullptr;
    double* d_sq_ = nullptr;
    double* d_energy_double_ = nullptr;
    double* d_metric_sums_ = nullptr;
    unsigned long long* d_hist_ = nullptr;
    double* d_scratch_a_ = nullptr;
    double* d_scratch_b_ = nullptr;
};

static void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0
        << " --input-list FILE --output-dir DIR [options]\n"
        << "Options:\n"
        << "  --tree NAME              TTree name (default Events)\n"
        << "  --truth NAME             std::vector<float> truth branch (default x_true)\n"
        << "  --reco NAME              std::vector<float> reconstructed branch (default x_hat)\n"
        << "  --energy NAME            optional std::vector<float> energy branch (default energy)\n"
        << "  --no-energy              disable energy sum/histogram\n"
        << "  --batch-elements N       flattened elements per batch (default 1048576)\n"
        << "  --threads N              CUDA threads/block, power of two (default 256)\n"
        << "  --hist-bins N            histogram bins (default 200)\n"
        << "  --hist-min X             histogram lower edge (default 0)\n"
        << "  --hist-max X             histogram upper edge (default 5000)\n";
}

static Config parse_args(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--input-list") {
            cfg.input_list = next("--input-list");
        } else if (arg == "--output-dir") {
            cfg.output_dir = next("--output-dir");
        } else if (arg == "--tree") {
            cfg.tree_name = next("--tree");
        } else if (arg == "--truth") {
            cfg.truth_branch = next("--truth");
        } else if (arg == "--reco") {
            cfg.reco_branch = next("--reco");
        } else if (arg == "--energy") {
            cfg.energy_branch = next("--energy");
        } else if (arg == "--no-energy") {
            cfg.energy_branch.clear();
        } else if (arg == "--batch-elements") {
            cfg.batch_elements =
                std::stoull(next("--batch-elements"));
        } else if (arg == "--threads") {
            cfg.threads =
                std::stoi(next("--threads"));
        } else if (arg == "--hist-bins") {
            cfg.hist_bins =
                std::stoi(next("--hist-bins"));
        } else if (arg == "--hist-min") {
            cfg.hist_min =
                std::stod(next("--hist-min"));
        } else if (arg == "--hist-max") {
            cfg.hist_max =
                std::stod(next("--hist-max"));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error(
                "unknown argument: " + arg);
        }
    }

    if (cfg.input_list.empty()) {
        throw std::runtime_error(
            "--input-list is required");
    }
    if (cfg.output_dir.empty()) {
        throw std::runtime_error(
            "--output-dir is required");
    }
    if (cfg.batch_elements == 0) {
        throw std::runtime_error(
            "--batch-elements must be > 0");
    }
    if (cfg.hist_bins <= 0) {
        throw std::runtime_error(
            "--hist-bins must be > 0");
    }
    if (!(cfg.hist_max > cfg.hist_min)) {
        throw std::runtime_error(
            "--hist-max must be > --hist-min");
    }

    return cfg;
}

static std::vector<std::string>
load_input_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(
            "cannot open input list: " + path);
    }

    std::vector<std::string> files;
    std::string line;

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        const auto first =
            line.find_first_not_of(" \t");
        if (first == std::string::npos ||
            line[first] == '#') {
            continue;
        }

        const auto last =
            line.find_last_not_of(" \t");
        files.push_back(
            line.substr(first, last - first + 1));
    }

    if (files.empty()) {
        throw std::runtime_error(
            "input list contains no files");
    }

    return files;
}

static int histogram_bin(float x,
                         const Config& cfg) {
    if (x < cfg.hist_min ||
        x >= cfg.hist_max) {
        return -1;
    }

    const double f =
        (static_cast<double>(x) - cfg.hist_min) /
        (cfg.hist_max - cfg.hist_min);

    const int b =
        static_cast<int>(f * cfg.hist_bins);

    return
        (b >= 0 && b < cfg.hist_bins) ? b : -1;
}

static void add_metrics(Metrics& dst,
                        const Metrics& src) {
    dst.events += src.events;
    dst.elements += src.elements;
    dst.cpu_sqsum += src.cpu_sqsum;
    dst.gpu_sqsum += src.gpu_sqsum;
    dst.cpu_energy_sum += src.cpu_energy_sum;
    dst.gpu_energy_sum += src.gpu_energy_sum;

    dst.timing.root_read_ms +=
        src.timing.root_read_ms;
    dst.timing.cpu_compute_ms +=
        src.timing.cpu_compute_ms;
    dst.timing.h2d_ms +=
        src.timing.h2d_ms;
    dst.timing.kernel_ms +=
        src.timing.kernel_ms;
    dst.timing.d2h_ms +=
        src.timing.d2h_ms;

    dst.has_energy =
        dst.has_energy || src.has_energy;

    if (dst.cpu_hist.empty()) {
        dst.cpu_hist.assign(
            src.cpu_hist.size(), 0);
    }
    if (dst.gpu_hist.empty()) {
        dst.gpu_hist.assign(
            src.gpu_hist.size(), 0);
    }

    for (std::size_t i = 0;
         i < src.cpu_hist.size();
         ++i) {
        dst.cpu_hist[i] +=
            src.cpu_hist[i];
    }
    for (std::size_t i = 0;
         i < src.gpu_hist.size();
         ++i) {
        dst.gpu_hist[i] +=
            src.gpu_hist[i];
    }
}

static double rmse(long double sqsum,
                   std::uint64_t n) {
    if (n == 0) {
        return std::numeric_limits<double>
            ::quiet_NaN();
    }

    return std::sqrt(
        static_cast<double>(
            sqsum /
            static_cast<long double>(n)));
}

static bool nearly_equal(double a,
                         double b,
                         double rel = 1e-5,
                         double abs = 1e-7) {
    const double diff =
        std::abs(a - b);

    return diff <= std::max(
        abs,
        rel * std::max(
            std::abs(a),
            std::abs(b)));
}

static bool metrics_gate(const Metrics& m) {
    if (m.elements == 0) {
        return false;
    }

    if (!nearly_equal(
            rmse(m.cpu_sqsum, m.elements),
            rmse(m.gpu_sqsum, m.elements),
            1e-5,
            1e-6)) {
        return false;
    }

    if (m.has_energy) {
        if (!nearly_equal(
                static_cast<double>(
                    m.cpu_energy_sum),
                static_cast<double>(
                    m.gpu_energy_sum),
                1e-5,
                1e-4)) {
            return false;
        }

        if (m.cpu_hist != m.gpu_hist) {
            return false;
        }
    }

    return true;
}

static void process_current_batch(
    GpuWorkspace& gpu,
    std::size_t count,
    bool has_energy,
    const Config& cfg,
    Metrics& metrics) {

    if (count == 0) {
        return;
    }

    const auto cpu_begin =
        std::chrono::steady_clock::now();

    long double cpu_sq = 0.0L;
    long double cpu_esum = 0.0L;
    std::vector<std::uint64_t>
        cpu_hist(cfg.hist_bins, 0);

    for (std::size_t i = 0;
         i < count;
         ++i) {
        const long double r =
            static_cast<long double>(
                gpu.h_reco()[i]) -
            static_cast<long double>(
                gpu.h_truth()[i]);

        cpu_sq += r * r;

        if (has_energy) {
            cpu_esum +=
                static_cast<long double>(
                    gpu.h_energy()[i]);

            const int b =
                histogram_bin(
                    gpu.h_energy()[i],
                    cfg);

            if (b >= 0) {
                ++cpu_hist[
                    static_cast<std::size_t>(b)];
            }
        }
    }

    const auto cpu_end =
        std::chrono::steady_clock::now();

    const auto gpu_result =
        gpu.process(
            count,
            has_energy,
            cfg.hist_min,
            cfg.hist_max);

    metrics.elements += count;
    metrics.cpu_sqsum += cpu_sq;
    metrics.gpu_sqsum +=
        gpu_result.sqsum;
    metrics.cpu_energy_sum +=
        cpu_esum;
    metrics.gpu_energy_sum +=
        gpu_result.energy_sum;

    metrics.timing.cpu_compute_ms +=
        std::chrono::duration<
            double,
            std::milli>(
                cpu_end - cpu_begin)
            .count();

    metrics.timing.h2d_ms +=
        gpu_result.h2d_ms;
    metrics.timing.kernel_ms +=
        gpu_result.kernel_ms;
    metrics.timing.d2h_ms +=
        gpu_result.d2h_ms;

    for (int i = 0;
         i < cfg.hist_bins;
         ++i) {
        metrics.cpu_hist[
            static_cast<std::size_t>(i)] +=
            cpu_hist[
                static_cast<std::size_t>(i)];

        metrics.gpu_hist[
            static_cast<std::size_t>(i)] +=
            gpu_result.hist[
                static_cast<std::size_t>(i)];
    }
}

static FileResult process_file(
    const std::string& path,
    const Config& cfg,
    GpuWorkspace& gpu) {

    FileResult result;
    result.path = path;
    result.metrics.cpu_hist.assign(
        cfg.hist_bins, 0);
    result.metrics.gpu_hist.assign(
        cfg.hist_bins, 0);
    result.metrics.has_energy =
        !cfg.energy_branch.empty();

    std::unique_ptr<TFile> file(
        TFile::Open(
            path.c_str(),
            "READ"));

    if (!file || file->IsZombie()) {
        result.status = "SKIP";
        result.message =
            "cannot open ROOT file";
        return result;
    }

    TTree* tree = nullptr;
    file->GetObject(
        cfg.tree_name.c_str(),
        tree);

    if (!tree) {
        result.status = "SKIP";
        result.message =
            "tree not found: " +
            cfg.tree_name;
        return result;
    }

    if (!tree->GetBranch(
            cfg.truth_branch.c_str()) ||
        !tree->GetBranch(
            cfg.reco_branch.c_str())) {
        result.status = "SKIP";
        result.message =
            "required truth/reco branch missing";
        return result;
    }

    const bool has_energy =
        !cfg.energy_branch.empty() &&
        tree->GetBranch(
            cfg.energy_branch.c_str());

    result.metrics.has_energy =
        has_energy;

    if (!cfg.energy_branch.empty() &&
        !has_energy) {
        std::cerr
            << "WARN file=" << path
            << " energy branch missing; "
            << "continuing without energy metrics\n";
    }

    TTreeReader reader(tree);
    TTreeReaderValue<std::vector<float>>
        truth(
            reader,
            cfg.truth_branch.c_str());
    TTreeReaderValue<std::vector<float>>
        reco(
            reader,
            cfg.reco_branch.c_str());

    std::unique_ptr<
        TTreeReaderValue<
            std::vector<float>>>
        energy;

    if (has_energy) {
        energy =
            std::make_unique<
                TTreeReaderValue<
                    std::vector<float>>>(
                        reader,
                        cfg.energy_branch.c_str());
    }

    std::size_t buffered = 0;

    while (true) {
        const auto read_begin =
            std::chrono::steady_clock::now();

        const bool has_entry =
            reader.Next();

        const auto read_end =
            std::chrono::steady_clock::now();

        result.metrics.timing.root_read_ms +=
            std::chrono::duration<
                double,
                std::milli>(
                    read_end - read_begin)
                .count();

        if (!has_entry) {
            break;
        }

        ++result.metrics.events;

        const auto& t = *truth;
        const auto& r = *reco;

        if (t.size() != r.size()) {
            result.status = "FAIL";
            result.message =
                "truth/reco vector size mismatch";
            return result;
        }

        if (has_energy &&
            (**energy).size() != t.size()) {
            result.status = "FAIL";
            result.message =
                "energy vector size mismatch";
            return result;
        }

        std::size_t pos = 0;

        while (pos < t.size()) {
            const std::size_t room =
                cfg.batch_elements - buffered;
            const std::size_t take =
                std::min(
                    room,
                    t.size() - pos);

            std::memcpy(
                gpu.h_truth() + buffered,
                t.data() + pos,
                take * sizeof(float));

            std::memcpy(
                gpu.h_reco() + buffered,
                r.data() + pos,
                take * sizeof(float));

            if (has_energy) {
                const auto& e = **energy;
                std::memcpy(
                    gpu.h_energy() + buffered,
                    e.data() + pos,
                    take * sizeof(float));
            }

            buffered += take;
            pos += take;

            if (buffered ==
                cfg.batch_elements) {
                process_current_batch(
                    gpu,
                    buffered,
                    has_energy,
                    cfg,
                    result.metrics);
                buffered = 0;
            }
        }
    }

    if (buffered > 0) {
        process_current_batch(
            gpu,
            buffered,
            has_energy,
            cfg,
            result.metrics);
    }

    if (result.metrics.elements == 0) {
        result.status = "FAIL";
        result.message =
            "no flattened elements read";
    } else if (!metrics_gate(
                   result.metrics)) {
        result.status = "FAIL";
        result.message =
            "CPU/GPU numerical gate failed";
    }

    return result;
}

static std::string csv_escape(
    const std::string& s) {

    if (s.find_first_of(",\"\n") ==
        std::string::npos) {
        return s;
    }

    std::string out = "\"";

    for (char c : s) {
        if (c == '\"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }

    out += '\"';
    return out;
}

static void write_outputs(
    const Config& cfg,
    const std::vector<FileResult>& files,
    const Metrics& global,
    bool overall_gate,
    const std::string& gpu_name,
    int cuda_driver,
    int cuda_runtime) {

    fs::create_directories(
        cfg.output_dir);

    {
        std::ofstream out(
            fs::path(cfg.output_dir) /
            "per_file.csv");

        out
            << "file,status,message,events,elements,"
            << "rmse_cpu,rmse_gpu,"
            << "energy_sum_cpu,energy_sum_gpu,"
            << "root_read_ms,cpu_compute_ms,"
            << "h2d_ms,kernel_ms,d2h_ms\n";

        out << std::setprecision(12);

        for (const auto& f : files) {
            const auto& m = f.metrics;

            out
                << csv_escape(f.path) << ','
                << f.status << ','
                << csv_escape(f.message) << ','
                << m.events << ','
                << m.elements << ','
                << rmse(
                    m.cpu_sqsum,
                    m.elements) << ','
                << rmse(
                    m.gpu_sqsum,
                    m.elements) << ','
                << static_cast<double>(
                    m.cpu_energy_sum) << ','
                << static_cast<double>(
                    m.gpu_energy_sum) << ','
                << m.timing.root_read_ms << ','
                << m.timing.cpu_compute_ms << ','
                << m.timing.h2d_ms << ','
                << m.timing.kernel_ms << ','
                << m.timing.d2h_ms
                << '\n';
        }
    }

    {
        std::ofstream out(
            fs::path(cfg.output_dir) /
            "summary.csv");

        out << "metric,value\n";
        out << std::setprecision(12);

        out
            << "files_total,"
            << files.size()
            << '\n';

        out
            << "files_pass,"
            << std::count_if(
                files.begin(),
                files.end(),
                [](const auto& f) {
                    return f.status == "PASS";
                })
            << '\n';

        out
            << "files_fail,"
            << std::count_if(
                files.begin(),
                files.end(),
                [](const auto& f) {
                    return f.status == "FAIL";
                })
            << '\n';

        out
            << "files_skip,"
            << std::count_if(
                files.begin(),
                files.end(),
                [](const auto& f) {
                    return f.status == "SKIP";
                })
            << '\n';

        out
            << "events,"
            << global.events
            << '\n';
        out
            << "elements,"
            << global.elements
            << '\n';
        out
            << "rmse_cpu,"
            << rmse(
                global.cpu_sqsum,
                global.elements)
            << '\n';
        out
            << "rmse_gpu,"
            << rmse(
                global.gpu_sqsum,
                global.elements)
            << '\n';
        out
            << "energy_sum_cpu,"
            << static_cast<double>(
                global.cpu_energy_sum)
            << '\n';
        out
            << "energy_sum_gpu,"
            << static_cast<double>(
                global.gpu_energy_sum)
            << '\n';
        out
            << "root_read_ms,"
            << global.timing.root_read_ms
            << '\n';
        out
            << "cpu_compute_ms,"
            << global.timing.cpu_compute_ms
            << '\n';
        out
            << "h2d_ms,"
            << global.timing.h2d_ms
            << '\n';
        out
            << "kernel_ms,"
            << global.timing.kernel_ms
            << '\n';
        out
            << "d2h_ms,"
            << global.timing.d2h_ms
            << '\n';
        out
            << "result_gate,"
            << (overall_gate
                    ? "PASS"
                    : "FAIL")
            << '\n';
    }

    {
        std::ofstream out(
            fs::path(cfg.output_dir) /
            "histogram.csv");

        out
            << "bin,low,high,"
            << "cpu_count,gpu_count\n";

        for (int i = 0;
             i < cfg.hist_bins;
             ++i) {
            const double low =
                cfg.hist_min +
                (cfg.hist_max -
                 cfg.hist_min) *
                    i /
                    cfg.hist_bins;

            const double high =
                cfg.hist_min +
                (cfg.hist_max -
                 cfg.hist_min) *
                    (i + 1) /
                    cfg.hist_bins;

            out
                << i << ','
                << low << ','
                << high << ','
                << global.cpu_hist[
                    static_cast<std::size_t>(i)]
                << ','
                << global.gpu_hist[
                    static_cast<std::size_t>(i)]
                << '\n';
        }
    }

    {
        std::ofstream out(
            fs::path(cfg.output_dir) /
            "run_manifest.json");

        out << "{\n";
        out
            << "  \"tree\": \""
            << cfg.tree_name
            << "\",\n";
        out
            << "  \"truth_branch\": \""
            << cfg.truth_branch
            << "\",\n";
        out
            << "  \"reco_branch\": \""
            << cfg.reco_branch
            << "\",\n";
        out
            << "  \"energy_branch\": \""
            << cfg.energy_branch
            << "\",\n";
        out
            << "  \"batch_elements\": "
            << cfg.batch_elements
            << ",\n";
        out
            << "  \"threads_per_block\": "
            << cfg.threads
            << ",\n";
        out
            << "  \"gpu_name\": \""
            << gpu_name
            << "\",\n";
        out
            << "  \"cuda_driver_version\": "
            << cuda_driver
            << ",\n";
        out
            << "  \"cuda_runtime_version\": "
            << cuda_runtime
            << ",\n";
        out
            << "  \"result_gate\": \""
            << (overall_gate
                    ? "PASS"
                    : "FAIL")
            << "\"\n";
        out << "}\n";
    }

    const fs::path root_path =
        fs::path(cfg.output_dir) /
        "results.root";

    TFile root_out(
        root_path.string().c_str(),
        "RECREATE");

    TH1D h_cpu(
        "energy_cpu",
        "Energy histogram CPU",
        cfg.hist_bins,
        cfg.hist_min,
        cfg.hist_max);

    TH1D h_gpu(
        "energy_gpu",
        "Energy histogram GPU",
        cfg.hist_bins,
        cfg.hist_min,
        cfg.hist_max);

    for (int i = 0;
         i < cfg.hist_bins;
         ++i) {
        h_cpu.SetBinContent(
            i + 1,
            static_cast<double>(
                global.cpu_hist[
                    static_cast<std::size_t>(i)]));

        h_gpu.SetBinContent(
            i + 1,
            static_cast<double>(
                global.gpu_hist[
                    static_cast<std::size_t>(i)]));
    }

    h_cpu.Write();
    h_gpu.Write();

    TTree metrics_tree(
        "FileMetrics",
        "Per-file CPU/GPU metrics");

    std::string file_path;
    std::string status;
    std::string message;

    ULong64_t events = 0;
    ULong64_t elements = 0;

    double rmse_cpu = 0.0;
    double rmse_gpu = 0.0;
    double energy_cpu = 0.0;
    double energy_gpu = 0.0;
    double root_read_ms = 0.0;
    double cpu_compute_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;

    metrics_tree.Branch(
        "file_path",
        &file_path);
    metrics_tree.Branch(
        "status",
        &status);
    metrics_tree.Branch(
        "message",
        &message);
    metrics_tree.Branch(
        "events",
        &events);
    metrics_tree.Branch(
        "elements",
        &elements);
    metrics_tree.Branch(
        "rmse_cpu",
        &rmse_cpu);
    metrics_tree.Branch(
        "rmse_gpu",
        &rmse_gpu);
    metrics_tree.Branch(
        "energy_sum_cpu",
        &energy_cpu);
    metrics_tree.Branch(
        "energy_sum_gpu",
        &energy_gpu);
    metrics_tree.Branch(
        "root_read_ms",
        &root_read_ms);
    metrics_tree.Branch(
        "cpu_compute_ms",
        &cpu_compute_ms);
    metrics_tree.Branch(
        "h2d_ms",
        &h2d_ms);
    metrics_tree.Branch(
        "kernel_ms",
        &kernel_ms);
    metrics_tree.Branch(
        "d2h_ms",
        &d2h_ms);

    for (const auto& f : files) {
        file_path = f.path;
        status = f.status;
        message = f.message;
        events = f.metrics.events;
        elements = f.metrics.elements;

        rmse_cpu =
            rmse(
                f.metrics.cpu_sqsum,
                f.metrics.elements);

        rmse_gpu =
            rmse(
                f.metrics.gpu_sqsum,
                f.metrics.elements);

        energy_cpu =
            static_cast<double>(
                f.metrics.cpu_energy_sum);

        energy_gpu =
            static_cast<double>(
                f.metrics.gpu_energy_sum);

        root_read_ms =
            f.metrics.timing.root_read_ms;
        cpu_compute_ms =
            f.metrics.timing.cpu_compute_ms;
        h2d_ms =
            f.metrics.timing.h2d_ms;
        kernel_ms =
            f.metrics.timing.kernel_ms;
        d2h_ms =
            f.metrics.timing.d2h_ms;

        metrics_tree.Fill();
    }

    metrics_tree.Write();

    TNamed gate_obj(
        "RESULT_GATE",
        overall_gate
            ? "PASS"
            : "FAIL");

    gate_obj.Write();
    root_out.Close();
}

int main(int argc, char** argv) {
    try {
        const Config cfg =
            parse_args(argc, argv);

        const auto files =
            load_input_list(
                cfg.input_list);

        fs::create_directories(
            cfg.output_dir);

        int device = 0;
        CUDA_CHECK(
            cudaGetDevice(&device));

        cudaDeviceProp prop{};
        CUDA_CHECK(
            cudaGetDeviceProperties(
                &prop,
                device));

        int driver_version = 0;
        int runtime_version = 0;

        CUDA_CHECK(
            cudaDriverGetVersion(
                &driver_version));

        CUDA_CHECK(
            cudaRuntimeGetVersion(
                &runtime_version));

        std::cout
            << "==================================================\n"
            << " CUDA LAB — ROOT + CUDA MULTI-FILE\n"
            << "==================================================\n"
            << "GPU_NAME=" << prop.name << '\n'
            << "CUDA_CC="
            << prop.major
            << '.'
            << prop.minor
            << '\n'
            << "INPUT_FILE_COUNT="
            << files.size()
            << '\n'
            << "BATCH_ELEMENTS="
            << cfg.batch_elements
            << '\n'
            << "THREADS="
            << cfg.threads
            << '\n';

        GpuWorkspace gpu(
            cfg.batch_elements,
            cfg.threads,
            cfg.hist_bins);

        std::vector<FileResult> results;
        results.reserve(
            files.size());

        Metrics global;
        global.cpu_hist.assign(
            cfg.hist_bins, 0);
        global.gpu_hist.assign(
            cfg.hist_bins, 0);

        for (std::size_t i = 0;
             i < files.size();
             ++i) {
            std::cout
                << "FILE_BEGIN="
                << (i + 1)
                << '/'
                << files.size()
                << " PATH="
                << files[i]
                << '\n';

            FileResult r =
                process_file(
                    files[i],
                    cfg,
                    gpu);

            std::cout
                << "FILE_STATUS="
                << r.status
                << " EVENTS="
                << r.metrics.events
                << " ELEMENTS="
                << r.metrics.elements
                << " RMSE_CPU="
                << rmse(
                    r.metrics.cpu_sqsum,
                    r.metrics.elements)
                << " RMSE_GPU="
                << rmse(
                    r.metrics.gpu_sqsum,
                    r.metrics.elements)
                << '\n';

            if (r.status == "PASS") {
                add_metrics(
                    global,
                    r.metrics);
            }

            results.push_back(
                std::move(r));
        }

        const bool gate =
            metrics_gate(global) &&
            std::none_of(
                results.begin(),
                results.end(),
                [](const auto& r) {
                    return r.status == "FAIL";
                });

        write_outputs(
            cfg,
            results,
            global,
            gate,
            prop.name,
            driver_version,
            runtime_version);

        std::cout
            << "GLOBAL_EVENTS="
            << global.events
            << '\n'
            << "GLOBAL_ELEMENTS="
            << global.elements
            << '\n'
            << "GLOBAL_RMSE_CPU="
            << rmse(
                global.cpu_sqsum,
                global.elements)
            << '\n'
            << "GLOBAL_RMSE_GPU="
            << rmse(
                global.gpu_sqsum,
                global.elements)
            << '\n'
            << "ROOT_READ_MS="
            << global.timing.root_read_ms
            << '\n'
            << "CPU_COMPUTE_MS="
            << global.timing.cpu_compute_ms
            << '\n'
            << "H2D_MS="
            << global.timing.h2d_ms
            << '\n'
            << "KERNEL_MS="
            << global.timing.kernel_ms
            << '\n'
            << "D2H_MS="
            << global.timing.d2h_ms
            << '\n'
            << "RESULT_GATE="
            << (gate ? "PASS" : "FAIL")
            << '\n';

        return gate ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr
            << "FATAL="
            << e.what()
            << '\n';
        return 1;
    }
}
