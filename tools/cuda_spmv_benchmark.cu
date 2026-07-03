/**
 * @file cuda_spmv_benchmark.cu
 * @brief CUDA SpMV benchmark and CPU/GPU accuracy-speed-memory comparison.
 *
 * Usage:
 *   cuda_spmv_benchmark.exe --matrix <path.csrb> [--cpu-threads N]
 *                           [--cpu-iters N] [--gpu-iters N] [--warmup N]
 *                           [--op spmv|spmvt|both]
 *                           [--out result.json]
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr uint32_t kCsrbMagic = 0x43535242u; // 'CSRB'

struct CsrbHeader {
    uint32_t magic = kCsrbMagic;
    uint32_t version = 1;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint64_t nnz = 0;
};

struct CSRHost {
    int original_rows = 0;
    int rows = 0;
    int cols = 0;
    uint64_t nnz = 0;
    std::vector<uint32_t> row_ptr;
    std::vector<uint32_t> col_idx;
    std::vector<float> values;
};

struct CompareStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    double l2_rel = 0.0;
};

struct OpBenchStats {
    bool enabled = false;
    double cpu_ms_avg = 0.0;
    double gpu_ms_avg = 0.0;
    double speedup = 0.0;
    double cpu_gflops = 0.0;
    double gpu_gflops = 0.0;
    CompareStats accuracy;
};

struct BenchResult {
    int rows = 0;
    int cols = 0;
    uint64_t nnz = 0;
    int cpu_threads = 1;
    int cpu_iters = 0;
    int gpu_iters = 0;
    int warmup_iters = 0;

    std::string op_mode = "spmv";
    OpBenchStats spmv;
    OpBenchStats spmvt;

    // Backward-compatible top-level aliases.
    double cpu_ms_avg = 0.0;
    double gpu_ms_avg = 0.0;
    double speedup = 0.0;
    double cpu_gflops = 0.0;
    double gpu_gflops = 0.0;
    CompareStats accuracy;

    uint64_t host_bytes = 0;
    uint64_t gpu_alloc_bytes = 0;
    uint64_t gpu_mem_delta_bytes = 0;
};

enum class OpMode {
    SpMV,
    SpMVT,
    Both
};

inline const char* opModeToString(OpMode mode) {
    switch (mode) {
    case OpMode::SpMV:
        return "spmv";
    case OpMode::SpMVT:
        return "spmvt";
    case OpMode::Both:
        return "both";
    default:
        return "spmv";
    }
}

inline void throwIf(bool cond, const std::string& msg) {
    if (cond) {
        throw std::runtime_error(msg);
    }
}

inline std::string bytesToHuman(uint64_t bytes) {
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << " " << units[unit];
    return oss.str();
}

inline void cudaCheck(cudaError_t err, const char* expr, const char* file,
                      int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error: " << expr << " @ " << file << ":" << line
            << " -> " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline void cusparseCheck(cusparseStatus_t st, const char* expr,
                          const char* file, int line) {
    if (st != CUSPARSE_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuSPARSE error: " << expr << " @ " << file << ":" << line
            << " -> status=" << static_cast<int>(st);
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) ::cudaCheck((expr), #expr, __FILE__, __LINE__)
#define CUSPARSE_CHECK(expr) ::cusparseCheck((expr), #expr, __FILE__, __LINE__)

CSRHost loadCsrb(const std::string& path, int max_rows) {
    std::ifstream fin(path, std::ios::binary);
    throwIf(!fin.is_open(), "Failed to open matrix: " + path);

    fin.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(fin.tellg());
    fin.seekg(0, std::ios::beg);

    CsrbHeader h{};
    fin.read(reinterpret_cast<char*>(&h), sizeof(h));
    throwIf(!fin.good(), "Failed to read CSRB header: " + path);
    throwIf(h.magic != kCsrbMagic, "Invalid CSRB magic: " + path);
    throwIf(h.version != 1,
            "Unsupported CSRB version (expect 1): " + std::to_string(h.version));

    const uint64_t expected_size =
        sizeof(CsrbHeader) +
        (static_cast<uint64_t>(h.rows) + 1) * sizeof(uint32_t) +
        static_cast<uint64_t>(h.nnz) * sizeof(uint32_t) +
        static_cast<uint64_t>(h.nnz) * sizeof(float);
    throwIf(file_size != expected_size,
            "CSRB file size mismatch. expected=" + std::to_string(expected_size) +
                ", actual=" + std::to_string(file_size));

    const int original_rows = static_cast<int>(h.rows);
    const int used_rows =
        (max_rows > 0) ? std::min(static_cast<int>(h.rows), max_rows)
                       : static_cast<int>(h.rows);
    throwIf(used_rows <= 0, "used_rows <= 0 after --max-rows");

    std::vector<uint32_t> full_row_ptr(static_cast<size_t>(h.rows) + 1, 0);
    fin.read(reinterpret_cast<char*>(full_row_ptr.data()),
             static_cast<std::streamsize>(full_row_ptr.size() * sizeof(uint32_t)));
    throwIf(!fin.good(), "Failed to read row_ptr from CSRB: " + path);

    const uint64_t used_nnz =
        static_cast<uint64_t>(full_row_ptr[static_cast<size_t>(used_rows)]);

    CSRHost mat;
    mat.original_rows = original_rows;
    mat.rows = used_rows;
    mat.cols = static_cast<int>(h.cols);
    mat.nnz = used_nnz;
    mat.row_ptr.assign(full_row_ptr.begin(),
                       full_row_ptr.begin() + static_cast<size_t>(used_rows) + 1);
    mat.col_idx.resize(static_cast<size_t>(used_nnz));
    mat.values.resize(static_cast<size_t>(used_nnz));

    const uint64_t col_offset =
        sizeof(CsrbHeader) + (static_cast<uint64_t>(h.rows) + 1) * sizeof(uint32_t);
    const uint64_t val_offset =
        col_offset + static_cast<uint64_t>(h.nnz) * sizeof(uint32_t);

    fin.seekg(static_cast<std::streamoff>(col_offset), std::ios::beg);
    fin.read(reinterpret_cast<char*>(mat.col_idx.data()),
             static_cast<std::streamsize>(mat.col_idx.size() * sizeof(uint32_t)));
    throwIf(!fin.good(), "Failed to read subset col_idx: " + path);

    fin.seekg(static_cast<std::streamoff>(val_offset), std::ios::beg);
    fin.read(reinterpret_cast<char*>(mat.values.data()),
             static_cast<std::streamsize>(mat.values.size() * sizeof(float)));
    throwIf(!fin.good(), "Failed to read subset values: " + path);

    throwIf(mat.row_ptr.empty() || mat.row_ptr.front() != 0, "Invalid row_ptr[0]");
    throwIf(mat.row_ptr.back() != mat.nnz,
            "Invalid row_ptr[used_rows] != used_nnz");
    for (int i = 0; i < mat.rows; ++i) {
        throwIf(mat.row_ptr[static_cast<size_t>(i)] >
                    mat.row_ptr[static_cast<size_t>(i + 1)],
                "row_ptr is not monotonic at row " + std::to_string(i));
    }
    for (uint64_t i = 0; i < mat.nnz; ++i) {
        throwIf(mat.col_idx[static_cast<size_t>(i)] >=
                    static_cast<uint32_t>(mat.cols),
                "col_idx out of range at nnz " + std::to_string(i));
    }

    return mat;
}

std::vector<double> makeInputX(int n) {
    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = std::sin(0.001 * i) + std::cos(0.003 * i);
    }
    return x;
}

void cpuSpmv(const CSRHost& A, const double* x, double* y, int threads) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int row = 0; row < A.rows; ++row) {
        const uint32_t r0 = A.row_ptr[static_cast<size_t>(row)];
        const uint32_t r1 = A.row_ptr[static_cast<size_t>(row + 1)];
        double sum = 0.0;
        for (uint32_t t = r0; t < r1; ++t) {
            const uint32_t col = A.col_idx[static_cast<size_t>(t)];
            const float a = A.values[static_cast<size_t>(t)];
            sum += static_cast<double>(a) * x[static_cast<size_t>(col)];
        }
        y[static_cast<size_t>(row)] = sum;
    }
}

void cpuSpmvT(const CSRHost& A, const double* x, double* y, int threads) {
    std::fill(y, y + static_cast<size_t>(A.cols), 0.0);
#ifdef _OPENMP
    const int actual_threads = std::max(1, std::min(threads, omp_get_max_threads()));
    constexpr size_t kMaxThreadLocalBytes = 512ull * 1024ull * 1024ull;
    constexpr size_t kMaxElems = kMaxThreadLocalBytes / sizeof(double);
    const bool use_threadlocal =
        (actual_threads > 1) &&
        (static_cast<size_t>(A.cols) <=
         (kMaxElems / static_cast<size_t>(actual_threads)));

    if (use_threadlocal) {
        std::vector<std::vector<double>> local(
            static_cast<size_t>(actual_threads),
            std::vector<double>(static_cast<size_t>(A.cols), 0.0));

#pragma omp parallel num_threads(actual_threads)
        {
            const int tid = omp_get_thread_num();
            auto& ly = local[static_cast<size_t>(tid)];
#pragma omp for schedule(static)
            for (int row = 0; row < A.rows; ++row) {
                const double xval = x[static_cast<size_t>(row)];
                const uint32_t r0 = A.row_ptr[static_cast<size_t>(row)];
                const uint32_t r1 = A.row_ptr[static_cast<size_t>(row + 1)];
                for (uint32_t t = r0; t < r1; ++t) {
                    ly[static_cast<size_t>(A.col_idx[static_cast<size_t>(t)])] +=
                        static_cast<double>(A.values[static_cast<size_t>(t)]) * xval;
                }
            }
        }

        for (int tid = 0; tid < actual_threads; ++tid) {
            const auto& ly = local[static_cast<size_t>(tid)];
            for (int col = 0; col < A.cols; ++col) {
                y[static_cast<size_t>(col)] += ly[static_cast<size_t>(col)];
            }
        }
        return;
    }

#pragma omp parallel for num_threads(actual_threads) schedule(static)
    for (int row = 0; row < A.rows; ++row) {
        const double xval = x[static_cast<size_t>(row)];
        const uint32_t r0 = A.row_ptr[static_cast<size_t>(row)];
        const uint32_t r1 = A.row_ptr[static_cast<size_t>(row + 1)];
        for (uint32_t t = r0; t < r1; ++t) {
            const size_t col = static_cast<size_t>(A.col_idx[static_cast<size_t>(t)]);
            const double inc = static_cast<double>(A.values[static_cast<size_t>(t)]) * xval;
#pragma omp atomic
            y[col] += inc;
        }
    }
#else
    (void)threads;
    for (int row = 0; row < A.rows; ++row) {
        const double xval = x[static_cast<size_t>(row)];
        const uint32_t r0 = A.row_ptr[static_cast<size_t>(row)];
        const uint32_t r1 = A.row_ptr[static_cast<size_t>(row + 1)];
        for (uint32_t t = r0; t < r1; ++t) {
            y[static_cast<size_t>(A.col_idx[static_cast<size_t>(t)])] +=
                static_cast<double>(A.values[static_cast<size_t>(t)]) * xval;
        }
    }
#endif
}

CompareStats compareVectors(const std::vector<double>& ref,
                           const std::vector<double>& pred) {
    throwIf(ref.size() != pred.size(), "Vector size mismatch in compareVectors");
    CompareStats st{};
    long double l2_ref = 0.0L;
    long double l2_diff = 0.0L;
    long double abs_sum = 0.0L;
    long double sq_sum = 0.0L;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::abs(ref[i] - pred[i]);
        st.max_abs = std::max(st.max_abs, d);
        abs_sum += static_cast<long double>(d);
        sq_sum += static_cast<long double>(d) * static_cast<long double>(d);
        l2_ref += static_cast<long double>(ref[i]) * static_cast<long double>(ref[i]);
        l2_diff += static_cast<long double>(ref[i] - pred[i]) *
                   static_cast<long double>(ref[i] - pred[i]);
    }
    const long double n = static_cast<long double>(ref.size());
    st.mean_abs = static_cast<double>(abs_sum / n);
    st.rmse = static_cast<double>(std::sqrt(sq_sum / n));
    if (l2_ref > 0.0L) {
        st.l2_rel = static_cast<double>(std::sqrt(l2_diff / l2_ref));
    } else {
        st.l2_rel = 0.0;
    }
    return st;
}

void writeJson(const std::string& path, const BenchResult& r) {
    std::ofstream out(path, std::ios::binary);
    throwIf(!out.is_open(), "Failed to open output JSON: " + path);

    auto writeOpStats = [&](const char* name, const OpBenchStats& op, bool comma) {
        out << "  \"" << name << "\": ";
        if (!op.enabled) {
            out << "null";
            if (comma) {
                out << ",";
            }
            out << "\n";
            return;
        }
        out << "{\n";
        out << "    \"cpu_ms_avg\": " << op.cpu_ms_avg << ",\n";
        out << "    \"gpu_ms_avg\": " << op.gpu_ms_avg << ",\n";
        out << "    \"speedup\": " << op.speedup << ",\n";
        out << "    \"cpu_gflops\": " << op.cpu_gflops << ",\n";
        out << "    \"gpu_gflops\": " << op.gpu_gflops << ",\n";
        out << "    \"accuracy\": {\n";
        out << "      \"max_abs\": " << op.accuracy.max_abs << ",\n";
        out << "      \"mean_abs\": " << op.accuracy.mean_abs << ",\n";
        out << "      \"rmse\": " << op.accuracy.rmse << ",\n";
        out << "      \"l2_rel\": " << op.accuracy.l2_rel << "\n";
        out << "    }\n";
        out << "  }";
        if (comma) {
            out << ",";
        }
        out << "\n";
    };

    out << std::fixed << std::setprecision(8);
    out << "{\n";
    out << "  \"rows\": " << r.rows << ",\n";
    out << "  \"cols\": " << r.cols << ",\n";
    out << "  \"nnz\": " << r.nnz << ",\n";
    out << "  \"cpu_threads\": " << r.cpu_threads << ",\n";
    out << "  \"cpu_iters\": " << r.cpu_iters << ",\n";
    out << "  \"gpu_iters\": " << r.gpu_iters << ",\n";
    out << "  \"warmup_iters\": " << r.warmup_iters << ",\n";
    out << "  \"op_mode\": \"" << r.op_mode << "\",\n";
    writeOpStats("spmv", r.spmv, true);
    writeOpStats("spmvt", r.spmvt, true);
    out << "  \"cpu_ms_avg\": " << r.cpu_ms_avg << ",\n";
    out << "  \"gpu_ms_avg\": " << r.gpu_ms_avg << ",\n";
    out << "  \"speedup\": " << r.speedup << ",\n";
    out << "  \"cpu_gflops\": " << r.cpu_gflops << ",\n";
    out << "  \"gpu_gflops\": " << r.gpu_gflops << ",\n";
    out << "  \"accuracy\": {\n";
    out << "    \"max_abs\": " << r.accuracy.max_abs << ",\n";
    out << "    \"mean_abs\": " << r.accuracy.mean_abs << ",\n";
    out << "    \"rmse\": " << r.accuracy.rmse << ",\n";
    out << "    \"l2_rel\": " << r.accuracy.l2_rel << "\n";
    out << "  },\n";
    out << "  \"host_bytes\": " << r.host_bytes << ",\n";
    out << "  \"gpu_alloc_bytes\": " << r.gpu_alloc_bytes << ",\n";
    out << "  \"gpu_mem_delta_bytes\": " << r.gpu_mem_delta_bytes << "\n";
    out << "}\n";
}

void printResult(const BenchResult& r) {
    auto printOp = [](const char* name, const OpBenchStats& op) {
        if (!op.enabled) {
            return;
        }
        std::cout << name << ":\n";
        std::cout << "  CPU  : " << std::fixed << std::setprecision(3)
                  << op.cpu_ms_avg << " ms/iter, " << op.cpu_gflops
                  << " GFLOPS\n";
        std::cout << "  GPU  : " << op.gpu_ms_avg << " ms/iter, "
                  << op.gpu_gflops << " GFLOPS\n";
        std::cout << "  Speedup (CPU/GPU): " << op.speedup << "x\n";
        std::cout << "  Accuracy: max_abs=" << std::setprecision(8)
                  << op.accuracy.max_abs << ", rmse=" << op.accuracy.rmse
                  << ", l2_rel=" << op.accuracy.l2_rel << "\n";
    };

    std::cout << "\n=== CUDA SpMV Benchmark Result ===\n";
    std::cout << "Matrix: rows=" << r.rows << ", cols=" << r.cols
              << ", nnz=" << r.nnz << "\n";
    std::cout << "Op mode: " << r.op_mode << "\n";
    printOp("SpMV (A*x)", r.spmv);
    printOp("SpMV^T (A^T*x)", r.spmvt);
    std::cout << "Host memory (estimate): " << bytesToHuman(r.host_bytes) << "\n";
    std::cout << "GPU  allocation:        " << bytesToHuman(r.gpu_alloc_bytes)
              << "\n";
    std::cout << "GPU  mem delta(runtime): " << bytesToHuman(r.gpu_mem_delta_bytes)
              << "\n";
}

struct Args {
    std::string matrix_path;
    int cpu_threads = 16;
    int cpu_iters = 5;
    int gpu_iters = 50;
    int warmup_iters = 5;
    int max_rows = 50000;
    OpMode op_mode = OpMode::Both;
    std::string out_json;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](const std::string& key) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + key);
            }
            return argv[++i];
        };
        if (k == "--matrix") {
            a.matrix_path = next(k);
        } else if (k == "--cpu-threads") {
            a.cpu_threads = std::max(1, std::stoi(next(k)));
        } else if (k == "--cpu-iters") {
            a.cpu_iters = std::max(1, std::stoi(next(k)));
        } else if (k == "--gpu-iters") {
            a.gpu_iters = std::max(1, std::stoi(next(k)));
        } else if (k == "--warmup") {
            a.warmup_iters = std::max(0, std::stoi(next(k)));
        } else if (k == "--max-rows") {
            a.max_rows = std::max(1, std::stoi(next(k)));
        } else if (k == "--op") {
            const std::string mode = next(k);
            if (mode == "spmv") {
                a.op_mode = OpMode::SpMV;
            } else if (mode == "spmvt" || mode == "transpose") {
                a.op_mode = OpMode::SpMVT;
            } else if (mode == "both") {
                a.op_mode = OpMode::Both;
            } else {
                throw std::runtime_error(
                    "Unknown --op mode, expected spmv|spmvt|both");
            }
        } else if (k == "--out") {
            a.out_json = next(k);
        } else if (k == "--help" || k == "-h") {
            std::cout
                << "Usage: cuda_spmv_benchmark.exe --matrix <path.csrb> "
                   "[--cpu-threads N] [--cpu-iters N] [--gpu-iters N] [--warmup N] "
                   "[--op spmv|spmvt|both] "
                   "[--max-rows N] [--out json]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + k);
        }
    }
    throwIf(a.matrix_path.empty(), "--matrix is required");
    return a;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);

        int dev_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&dev_count));
        throwIf(dev_count <= 0, "No CUDA device detected.");
        CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using GPU[0]: " << prop.name
                  << ", VRAM="
                  << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0))
                  << " GB\n";

        CSRHost A = loadCsrb(args.matrix_path, args.max_rows);
        const bool run_spmv =
            (args.op_mode == OpMode::SpMV || args.op_mode == OpMode::Both);
        const bool run_spmvt =
            (args.op_mode == OpMode::SpMVT || args.op_mode == OpMode::Both);

        std::vector<double> x_spmv;
        std::vector<double> y_spmv_cpu;
        std::vector<double> y_spmv_gpu;
        if (run_spmv) {
            x_spmv = makeInputX(A.cols);
            y_spmv_cpu.assign(static_cast<size_t>(A.rows), 0.0);
            y_spmv_gpu.assign(static_cast<size_t>(A.rows), 0.0);
        }

        std::vector<double> x_spmvt;
        std::vector<double> y_spmvt_cpu;
        std::vector<double> y_spmvt_gpu;
        if (run_spmvt) {
            x_spmvt = makeInputX(A.rows);
            y_spmvt_cpu.assign(static_cast<size_t>(A.cols), 0.0);
            y_spmvt_gpu.assign(static_cast<size_t>(A.cols), 0.0);
        }

        size_t free0 = 0, total0 = 0;
        CUDA_CHECK(cudaMemGetInfo(&free0, &total0));

        uint32_t* d_row = nullptr;
        uint32_t* d_col = nullptr;
        float* d_val = nullptr;
        double* d_x_spmv = nullptr;
        double* d_y_spmv = nullptr;
        double* d_x_spmvt = nullptr;
        double* d_y_spmvt = nullptr;
        void* d_buf = nullptr;
        size_t d_buf_bytes = 0;

        const size_t row_bytes = A.row_ptr.size() * sizeof(uint32_t);
        const size_t col_bytes = A.col_idx.size() * sizeof(uint32_t);
        const size_t val_bytes = A.values.size() * sizeof(float);
        const size_t x_spmv_bytes = run_spmv ? x_spmv.size() * sizeof(double) : 0;
        const size_t y_spmv_bytes = run_spmv ? y_spmv_gpu.size() * sizeof(double) : 0;
        const size_t x_spmvt_bytes = run_spmvt ? x_spmvt.size() * sizeof(double) : 0;
        const size_t y_spmvt_bytes = run_spmvt ? y_spmvt_gpu.size() * sizeof(double) : 0;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_row), row_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_col), col_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_val), val_bytes));
        if (run_spmv) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x_spmv),
                                  x_spmv_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y_spmv),
                                  y_spmv_bytes));
        }
        if (run_spmvt) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x_spmvt),
                                  x_spmvt_bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y_spmvt),
                                  y_spmvt_bytes));
        }

        CUDA_CHECK(cudaMemcpy(d_row, A.row_ptr.data(), row_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col, A.col_idx.data(), col_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val, A.values.data(), val_bytes,
                              cudaMemcpyHostToDevice));
        if (run_spmv) {
            CUDA_CHECK(cudaMemcpy(d_x_spmv, x_spmv.data(), x_spmv_bytes,
                                  cudaMemcpyHostToDevice));
        }
        if (run_spmvt) {
            CUDA_CHECK(cudaMemcpy(d_x_spmvt, x_spmvt.data(), x_spmvt_bytes,
                                  cudaMemcpyHostToDevice));
        }

        cusparseHandle_t handle = nullptr;
        CUSPARSE_CHECK(cusparseCreate(&handle));

        cusparseSpMatDescr_t matA = nullptr;
        CUSPARSE_CHECK(cusparseCreateCsr(
            &matA, static_cast<int64_t>(A.rows), static_cast<int64_t>(A.cols),
            static_cast<int64_t>(A.nnz), d_row, d_col, d_val, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        cusparseDnVecDescr_t vecX_spmv = nullptr;
        cusparseDnVecDescr_t vecY_spmv = nullptr;
        cusparseDnVecDescr_t vecX_spmvt = nullptr;
        cusparseDnVecDescr_t vecY_spmvt = nullptr;

        const double alpha = 1.0;
        const double beta = 0.0;

        if (run_spmv) {
            CUSPARSE_CHECK(cusparseCreateDnVec(
                &vecX_spmv, static_cast<int64_t>(A.cols), d_x_spmv, CUDA_R_64F));
            CUSPARSE_CHECK(cusparseCreateDnVec(
                &vecY_spmv, static_cast<int64_t>(A.rows), d_y_spmv, CUDA_R_64F));
            size_t bytes = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX_spmv,
                &beta, vecY_spmv, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
            d_buf_bytes = std::max(d_buf_bytes, bytes);
        }
        if (run_spmvt) {
            CUSPARSE_CHECK(cusparseCreateDnVec(
                &vecX_spmvt, static_cast<int64_t>(A.rows), d_x_spmvt, CUDA_R_64F));
            CUSPARSE_CHECK(cusparseCreateDnVec(
                &vecY_spmvt, static_cast<int64_t>(A.cols), d_y_spmvt, CUDA_R_64F));
            size_t bytes = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, matA, vecX_spmvt,
                &beta, vecY_spmvt, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
            d_buf_bytes = std::max(d_buf_bytes, bytes);
        }

        if (d_buf_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&d_buf, d_buf_bytes));
        }

        size_t free1 = 0, total1 = 0;
        CUDA_CHECK(cudaMemGetInfo(&free1, &total1));

        auto runGpuOp = [&](cusparseOperation_t op, cusparseDnVecDescr_t vx,
                            cusparseDnVecDescr_t vy, double* dy,
                            size_t y_bytes) -> double {
            for (int i = 0; i < args.warmup_iters; ++i) {
                CUDA_CHECK(cudaMemset(dy, 0, y_bytes));
                CUSPARSE_CHECK(cusparseSpMV(
                    handle, op, &alpha, matA, vx, &beta, vy, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t ev0 = nullptr;
            cudaEvent_t ev1 = nullptr;
            CUDA_CHECK(cudaEventCreate(&ev0));
            CUDA_CHECK(cudaEventCreate(&ev1));
            CUDA_CHECK(cudaEventRecord(ev0));
            for (int i = 0; i < args.gpu_iters; ++i) {
                CUDA_CHECK(cudaMemset(dy, 0, y_bytes));
                CUSPARSE_CHECK(cusparseSpMV(
                    handle, op, &alpha, matA, vx, &beta, vy, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
            }
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float gpu_ms_total = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_total, ev0, ev1));
            CUDA_CHECK(cudaEventDestroy(ev0));
            CUDA_CHECK(cudaEventDestroy(ev1));
            return static_cast<double>(gpu_ms_total) /
                   static_cast<double>(args.gpu_iters);
        };

        OpBenchStats spmv_stats{};
        OpBenchStats spmvt_stats{};
        const double flop_per_iter = 2.0 * static_cast<double>(A.nnz);

        if (run_spmv) {
            const auto cpu_begin = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < args.cpu_iters; ++i) {
                cpuSpmv(A, x_spmv.data(), y_spmv_cpu.data(), args.cpu_threads);
            }
            const auto cpu_end = std::chrono::high_resolution_clock::now();
            const double cpu_ms_total =
                std::chrono::duration<double, std::milli>(cpu_end - cpu_begin)
                    .count();
            const double cpu_ms_avg =
                cpu_ms_total / static_cast<double>(args.cpu_iters);

            const double gpu_ms_avg =
                runGpuOp(CUSPARSE_OPERATION_NON_TRANSPOSE, vecX_spmv, vecY_spmv,
                         d_y_spmv, y_spmv_bytes);
            CUDA_CHECK(cudaMemcpy(y_spmv_gpu.data(), d_y_spmv, y_spmv_bytes,
                                  cudaMemcpyDeviceToHost));

            spmv_stats.enabled = true;
            spmv_stats.cpu_ms_avg = cpu_ms_avg;
            spmv_stats.gpu_ms_avg = gpu_ms_avg;
            spmv_stats.speedup =
                (gpu_ms_avg > 0.0) ? (cpu_ms_avg / gpu_ms_avg) : 0.0;
            spmv_stats.cpu_gflops = (flop_per_iter / (cpu_ms_avg * 1e-3)) / 1e9;
            spmv_stats.gpu_gflops = (flop_per_iter / (gpu_ms_avg * 1e-3)) / 1e9;
            spmv_stats.accuracy = compareVectors(y_spmv_cpu, y_spmv_gpu);
        }

        if (run_spmvt) {
            const auto cpu_begin = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < args.cpu_iters; ++i) {
                cpuSpmvT(A, x_spmvt.data(), y_spmvt_cpu.data(), args.cpu_threads);
            }
            const auto cpu_end = std::chrono::high_resolution_clock::now();
            const double cpu_ms_total =
                std::chrono::duration<double, std::milli>(cpu_end - cpu_begin)
                    .count();
            const double cpu_ms_avg =
                cpu_ms_total / static_cast<double>(args.cpu_iters);

            const double gpu_ms_avg =
                runGpuOp(CUSPARSE_OPERATION_TRANSPOSE, vecX_spmvt, vecY_spmvt,
                         d_y_spmvt, y_spmvt_bytes);
            CUDA_CHECK(cudaMemcpy(y_spmvt_gpu.data(), d_y_spmvt, y_spmvt_bytes,
                                  cudaMemcpyDeviceToHost));

            spmvt_stats.enabled = true;
            spmvt_stats.cpu_ms_avg = cpu_ms_avg;
            spmvt_stats.gpu_ms_avg = gpu_ms_avg;
            spmvt_stats.speedup =
                (gpu_ms_avg > 0.0) ? (cpu_ms_avg / gpu_ms_avg) : 0.0;
            spmvt_stats.cpu_gflops = (flop_per_iter / (cpu_ms_avg * 1e-3)) / 1e9;
            spmvt_stats.gpu_gflops = (flop_per_iter / (gpu_ms_avg * 1e-3)) / 1e9;
            spmvt_stats.accuracy = compareVectors(y_spmvt_cpu, y_spmvt_gpu);
        }

        BenchResult result{};
        result.rows = A.rows;
        result.cols = A.cols;
        result.nnz = A.nnz;
        result.cpu_threads = args.cpu_threads;
        result.cpu_iters = args.cpu_iters;
        result.gpu_iters = args.gpu_iters;
        result.warmup_iters = args.warmup_iters;
        result.op_mode = opModeToString(args.op_mode);
        result.spmv = spmv_stats;
        result.spmvt = spmvt_stats;

        const OpBenchStats& primary = result.spmv.enabled ? result.spmv : result.spmvt;
        result.cpu_ms_avg = primary.cpu_ms_avg;
        result.gpu_ms_avg = primary.gpu_ms_avg;
        result.speedup = primary.speedup;
        result.cpu_gflops = primary.cpu_gflops;
        result.gpu_gflops = primary.gpu_gflops;
        result.accuracy = primary.accuracy;

        result.host_bytes = static_cast<uint64_t>(
            row_bytes + col_bytes + val_bytes + x_spmv_bytes + y_spmv_bytes * 2 +
            x_spmvt_bytes + y_spmvt_bytes * 2);
        result.gpu_alloc_bytes = static_cast<uint64_t>(
            row_bytes + col_bytes + val_bytes + x_spmv_bytes + y_spmv_bytes +
            x_spmvt_bytes + y_spmvt_bytes + d_buf_bytes);
        result.gpu_mem_delta_bytes =
            static_cast<uint64_t>(free0 > free1 ? (free0 - free1) : 0);

        printResult(result);
        if (A.rows < A.original_rows) {
            std::cout << "Note: benchmark used row subset " << A.rows << "/"
                      << A.original_rows << " (--max-rows).\n";
        }
        if (!args.out_json.empty()) {
            writeJson(args.out_json, result);
            std::cout << "Saved JSON: " << args.out_json << "\n";
        }

        if (vecX_spmv)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecX_spmv));
        if (vecY_spmv)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecY_spmv));
        if (vecX_spmvt)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecX_spmvt));
        if (vecY_spmvt)
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecY_spmvt));
        CUSPARSE_CHECK(cusparseDestroySpMat(matA));
        CUSPARSE_CHECK(cusparseDestroy(handle));

        if (d_buf)
            CUDA_CHECK(cudaFree(d_buf));
        if (d_y_spmvt)
            CUDA_CHECK(cudaFree(d_y_spmvt));
        if (d_x_spmvt)
            CUDA_CHECK(cudaFree(d_x_spmvt));
        if (d_y_spmv)
            CUDA_CHECK(cudaFree(d_y_spmv));
        if (d_x_spmv)
            CUDA_CHECK(cudaFree(d_x_spmv));
        CUDA_CHECK(cudaFree(d_val));
        CUDA_CHECK(cudaFree(d_col));
        CUDA_CHECK(cudaFree(d_row));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
