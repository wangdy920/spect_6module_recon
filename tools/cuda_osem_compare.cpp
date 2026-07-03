#include "app/BedOffsets.h"
#include "app/OSEMReconstructor.h"
#include "app/ProjectionOps.h"
#include "cuda/CudaConfig.h"
#include "io/CSRBuilder.h"
#include "sparse/SparseOpsBackend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {

struct CompareConfig {
    std::string matrixPath;
    int threads = 16;
    int iterations = 2;
    int subsets = 1;
    int warmupRuns = 1;
    int repeats = 3;
    int maxRows = 0;
    int bedCount = 1;
    int baseNx = 0;
    int baseNy = 0;
    int baseNz = 0;
    std::vector<int> bedOverlaps;
    std::string outputPath;
};

struct ProcessMemorySample {
    uint64_t workingSetBytes = 0;
    uint64_t peakWorkingSetBytes = 0;
};

struct TimingSummary {
    double msAvg = 0.0;
    double msMin = 0.0;
    double msMax = 0.0;
};

struct BackendSummary {
    TimingSummary timing;
    uint64_t workingSetBeforeBytes = 0;
    uint64_t workingSetAfterBytes = 0;
    uint64_t peakWorkingSetBytes = 0;
    uint64_t gpuFreeBeforeBytes = 0;
    uint64_t gpuFreeAfterWarmupBytes = 0;
    uint64_t gpuFreeAfterBytes = 0;
    uint64_t gpuTotalBytes = 0;
    uint64_t gpuPersistentDeltaBytes = 0;
    std::vector<double> finalImage;
};

struct CompareStats {
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    double rmse = 0.0;
    double l2Rel = 0.0;
};

struct CompareResult {
    int rows = 0;
    int cols = 0;
    uint64_t nnz = 0;
    int iterations = 0;
    int subsets = 0;
    int warmupRuns = 0;
    int repeats = 0;
    int threads = 0;
    int bedCount = 1;
    int baseNx = 0;
    int baseNy = 0;
    int baseNz = 0;
    int globalNz = 0;
    uint64_t hostBytesEstimate = 0;
    BackendSummary cpu;
    BackendSummary gpu;
    CompareStats accuracy;
    double speedup = 0.0;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string bytesToHuman(uint64_t bytes) {
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << ' ' << units[unit];
    return oss.str();
}

ProcessMemorySample getProcessMemorySample() {
    ProcessMemorySample sample;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters))) {
        sample.workingSetBytes = static_cast<uint64_t>(counters.WorkingSetSize);
        sample.peakWorkingSetBytes =
            static_cast<uint64_t>(counters.PeakWorkingSetSize);
    }
#endif
    return sample;
}

spect::io::CSRView truncateView(const spect::io::CSRView& source, int maxRows) {
    if (!source.valid()) {
        fail("Invalid CSR view.");
    }
    if (maxRows <= 0 || maxRows >= source.rows) {
        return source;
    }

    spect::io::CSRView truncated = source;
    truncated.rows = maxRows;
    truncated.nnz = source.row_ptr[static_cast<size_t>(maxRows)];
    return truncated;
}

std::vector<int> parseCsvInts(const std::string& raw) {
    std::vector<int> values;
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::stoi(item));
        }
    }
    return values;
}

void normalizeBedOverlaps(CompareConfig& config) {
    const int joints = std::max(0, config.bedCount - 1);
    if (joints == 0) {
        config.bedOverlaps.clear();
        return;
    }
    if (config.bedOverlaps.size() == 1 && joints > 1) {
        config.bedOverlaps.assign(static_cast<size_t>(joints),
                                  config.bedOverlaps.front());
    } else if (static_cast<int>(config.bedOverlaps.size()) < joints) {
        config.bedOverlaps.resize(static_cast<size_t>(joints), 0);
    } else if (static_cast<int>(config.bedOverlaps.size()) > joints) {
        config.bedOverlaps.resize(static_cast<size_t>(joints));
    }
}

void inferBaseDims(CompareConfig& config, int cols) {
    if (config.baseNx > 0 && config.baseNy > 0 && config.baseNz > 0) {
        if (config.baseNx * config.baseNy * config.baseNz != cols) {
            fail("base dimensions do not match matrix.cols");
        }
        return;
    }

    const double root = std::round(std::cbrt(static_cast<double>(cols)));
    const int side = static_cast<int>(root);
    if (side > 0 && side * side * side == cols) {
        config.baseNx = side;
        config.baseNy = side;
        config.baseNz = side;
        return;
    }

    fail("Unable to infer base dimensions from matrix.cols; pass --base-dim Nx,Ny,Nz.");
}

int computeGlobalNz(const CompareConfig& config) {
    return config.baseNz * config.bedCount -
           spect::app::computeTotalOverlap(config.bedCount, config.bedOverlaps);
}

std::vector<long long> makeBedOffsets(const CompareConfig& config) {
    const std::vector<int> bedZStarts =
        spect::app::computeBedZOffsets(config.bedCount, config.baseNz,
                                       config.bedOverlaps);
    const long long sliceSize =
        static_cast<long long>(config.baseNx) * static_cast<long long>(config.baseNy);
    std::vector<long long> offsets(static_cast<size_t>(config.bedCount), 0);
    for (int bed = 0; bed < config.bedCount; ++bed) {
        offsets[static_cast<size_t>(bed)] =
            static_cast<long long>(bedZStarts[static_cast<size_t>(bed)]) * sliceSize;
    }
    return offsets;
}

std::vector<double> makeTruthImage(int cols) {
    std::vector<double> truth(static_cast<size_t>(cols), 1.0);
    for (int col = 0; col < cols; ++col) {
        truth[static_cast<size_t>(col)] =
            0.75 + 0.2 * std::sin(0.013 * static_cast<double>(col)) +
            0.1 * std::cos(0.031 * static_cast<double>(col));
    }
    return truth;
}

std::vector<double> makeInitialImage(int cols) {
    return std::vector<double>(static_cast<size_t>(cols), 1.0);
}

CompareStats compareVectors(const std::vector<double>& lhs,
                            const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        fail("compareVectors size mismatch.");
    }

    CompareStats stats;
    double sumAbs = 0.0;
    double sumSq = 0.0;
    double refSq = 0.0;
    for (size_t index = 0; index < lhs.size(); ++index) {
        const double diff = lhs[index] - rhs[index];
        const double absDiff = std::abs(diff);
        stats.maxAbs = std::max(stats.maxAbs, absDiff);
        sumAbs += absDiff;
        sumSq += diff * diff;
        refSq += lhs[index] * lhs[index];
    }
    if (!lhs.empty()) {
        stats.meanAbs = sumAbs / static_cast<double>(lhs.size());
        stats.rmse = std::sqrt(sumSq / static_cast<double>(lhs.size()));
    }
    stats.l2Rel = (refSq > 0.0) ? std::sqrt(sumSq / refSq) : 0.0;
    return stats;
}

TimingSummary summarizeTimes(const std::vector<double>& elapsedMs) {
    if (elapsedMs.empty()) {
        return {};
    }

    TimingSummary summary;
    summary.msMin = *std::min_element(elapsedMs.begin(), elapsedMs.end());
    summary.msMax = *std::max_element(elapsedMs.begin(), elapsedMs.end());
    summary.msAvg = std::accumulate(elapsedMs.begin(), elapsedMs.end(), 0.0) /
                    static_cast<double>(elapsedMs.size());
    return summary;
}

std::vector<double> makeSyntheticProjection(const spect::io::CSRView& matrix,
                                            const std::vector<double>& truth,
                                            const spect::ReconOpts& opts,
                                            int threads) {
    const auto subsets =
        spect::make_simple_subsets(matrix.rows * std::max(1, opts.bedCount), 1);
    const std::vector<int>& subsetRows = subsets.front();

    spect::SparseSubsetContext ctx;
    ctx.bedCount = std::max(1, opts.bedCount);
    ctx.bedOffsetVoxels =
        opts.bedOffsetVoxels.empty() ? nullptr : &opts.bedOffsetVoxels;
    ctx.bedAttenFactors =
        opts.bedAttenFactors.empty() ? nullptr : &opts.bedAttenFactors;
    ctx.numAngles = opts.num_angles;

    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), threads);
    for (double& value : projections) {
        value = std::max(value, 1e-6);
    }
    return projections;
}

std::vector<double> computeDenom(const spect::io::CSRView& matrix,
                                 const spect::ReconOpts& opts, int globalN,
                                 int threads) {
    const auto subsets =
        spect::make_simple_subsets(matrix.rows * std::max(1, opts.bedCount), 1);
    return spect::computeSubsetSensitivity(matrix, subsets.front(), globalN,
                                           threads, opts);
}

void runSinglePass(const spect::io::CSRView& matrix,
                   const std::vector<std::vector<int>>& subsets,
                   const std::vector<double>& projections,
                   const std::vector<double>& denom,
                   const spect::ReconOpts& opts,
                   int iterations,
                   int nx,
                   int ny,
                   int nz,
                   std::vector<double>& image) {
    for (int iter = 0; iter < iterations; ++iter) {
        for (const auto& subsetRows : subsets) {
            if (subsetRows.empty()) {
                continue;
            }
            spect::OSEM_step(matrix, subsetRows, projections, image, denom,
                             nx, ny, nz, opts, nullptr, nullptr, 0.0,
                             nullptr, 1.0);
        }
    }
}

BackendSummary benchmarkBackend(const spect::io::CSRView& matrix,
                                const std::vector<std::vector<int>>& subsets,
                                const std::vector<double>& projections,
                                const std::vector<double>& denom,
                                const spect::ISparseOps& sparseOps,
                                const CompareConfig& config,
                                int globalNx,
                                int globalNy,
                                int globalNz,
                                bool captureGpuMemory) {
    BackendSummary summary;

    spect::ReconOpts opts;
    opts.threads = config.threads;
    opts.n_subsets = config.subsets;
    opts.bedCount = config.bedCount;
    if (config.bedCount > 1) {
        opts.bedOverlaps = config.bedOverlaps;
        opts.bedOffsetVoxels = makeBedOffsets(config);
    }
    opts.sparseOps = &sparseOps;

    summary.workingSetBeforeBytes = getProcessMemorySample().workingSetBytes;

    if (captureGpuMemory) {
        const auto gpuMem = spect::cuda::getMemoryInfo();
        summary.gpuFreeBeforeBytes = static_cast<uint64_t>(gpuMem.free);
        summary.gpuTotalBytes = static_cast<uint64_t>(gpuMem.total);
    }

    for (int warmup = 0; warmup < config.warmupRuns; ++warmup) {
        std::vector<double> image = makeInitialImage(static_cast<int>(denom.size()));
        runSinglePass(matrix, subsets, projections, denom, opts,
                      config.iterations, globalNx, globalNy, globalNz, image);
    }

    if (captureGpuMemory) {
        const auto gpuMem = spect::cuda::getMemoryInfo();
        summary.gpuFreeAfterWarmupBytes = static_cast<uint64_t>(gpuMem.free);
        summary.gpuPersistentDeltaBytes =
            summary.gpuFreeBeforeBytes - summary.gpuFreeAfterWarmupBytes;
    }

    std::vector<double> times;
    times.reserve(static_cast<size_t>(config.repeats));
    for (int repeat = 0; repeat < config.repeats; ++repeat) {
        std::vector<double> image = makeInitialImage(static_cast<int>(denom.size()));
        const auto start = std::chrono::steady_clock::now();
        runSinglePass(matrix, subsets, projections, denom, opts,
                      config.iterations, globalNx, globalNy, globalNz, image);
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(end - start);
        times.push_back(elapsed.count());
        summary.finalImage = std::move(image);
    }

    summary.timing = summarizeTimes(times);
    const ProcessMemorySample processAfter = getProcessMemorySample();
    summary.workingSetAfterBytes = processAfter.workingSetBytes;
    summary.peakWorkingSetBytes = processAfter.peakWorkingSetBytes;

    if (captureGpuMemory) {
        const auto gpuMem = spect::cuda::getMemoryInfo();
        summary.gpuFreeAfterBytes = static_cast<uint64_t>(gpuMem.free);
    }

    return summary;
}

uint64_t estimateHostBytes(const spect::io::CSRView& matrix,
                           int globalN,
                           int repeats) {
    const uint64_t matrixBytes =
        static_cast<uint64_t>(matrix.rows + 1) * sizeof(uint32_t) +
        static_cast<uint64_t>(matrix.nnz) * (sizeof(uint32_t) + sizeof(float));
    const uint64_t workBytes =
        static_cast<uint64_t>(globalN) * sizeof(double) * 4 +
        static_cast<uint64_t>(matrix.rows) * sizeof(double) +
        static_cast<uint64_t>(globalN) * sizeof(double) * 2 *
            static_cast<uint64_t>(std::max(1, repeats));
    return matrixBytes + workBytes;
}

void writeJsonString(std::ostream& os, const std::string& value) {
    os << '"';
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            os << '\\' << ch;
        } else if (ch == '\n') {
            os << "\\n";
        } else {
            os << ch;
        }
    }
    os << '"';
}

void writeJson(const CompareResult& result, const CompareConfig& config,
               std::ostream& os) {
    os << std::fixed << std::setprecision(6);
    os << "{\n";
    os << "  \"scope\": ";
    writeJsonString(os, "multibed-capable osem compare; psf/ctac disabled");
    os << ",\n";
    os << "  \"matrix_path\": ";
    writeJsonString(os, config.matrixPath);
    os << ",\n";
    os << "  \"rows\": " << result.rows << ",\n";
    os << "  \"cols\": " << result.cols << ",\n";
    os << "  \"nnz\": " << result.nnz << ",\n";
    os << "  \"iterations\": " << result.iterations << ",\n";
    os << "  \"subsets\": " << result.subsets << ",\n";
    os << "  \"bed_count\": " << result.bedCount << ",\n";
    os << "  \"base_dim\": [" << result.baseNx << ", " << result.baseNy
       << ", " << result.baseNz << "],\n";
    os << "  \"global_nz\": " << result.globalNz << ",\n";
    os << "  \"warmup_runs\": " << result.warmupRuns << ",\n";
    os << "  \"repeats\": " << result.repeats << ",\n";
    os << "  \"threads\": " << result.threads << ",\n";
    os << "  \"host_bytes_estimate\": " << result.hostBytesEstimate << ",\n";
    os << "  \"cpu\": {\n";
    os << "    \"ms_avg\": " << result.cpu.timing.msAvg << ",\n";
    os << "    \"ms_min\": " << result.cpu.timing.msMin << ",\n";
    os << "    \"ms_max\": " << result.cpu.timing.msMax << ",\n";
    os << "    \"working_set_before_bytes\": " << result.cpu.workingSetBeforeBytes << ",\n";
    os << "    \"working_set_after_bytes\": " << result.cpu.workingSetAfterBytes << ",\n";
    os << "    \"peak_working_set_bytes\": " << result.cpu.peakWorkingSetBytes << "\n";
    os << "  },\n";
    os << "  \"gpu\": {\n";
    os << "    \"ms_avg\": " << result.gpu.timing.msAvg << ",\n";
    os << "    \"ms_min\": " << result.gpu.timing.msMin << ",\n";
    os << "    \"ms_max\": " << result.gpu.timing.msMax << ",\n";
    os << "    \"working_set_before_bytes\": " << result.gpu.workingSetBeforeBytes << ",\n";
    os << "    \"working_set_after_bytes\": " << result.gpu.workingSetAfterBytes << ",\n";
    os << "    \"peak_working_set_bytes\": " << result.gpu.peakWorkingSetBytes << ",\n";
    os << "    \"gpu_free_before_bytes\": " << result.gpu.gpuFreeBeforeBytes << ",\n";
    os << "    \"gpu_free_after_warmup_bytes\": " << result.gpu.gpuFreeAfterWarmupBytes << ",\n";
    os << "    \"gpu_free_after_bytes\": " << result.gpu.gpuFreeAfterBytes << ",\n";
    os << "    \"gpu_total_bytes\": " << result.gpu.gpuTotalBytes << ",\n";
    os << "    \"gpu_persistent_delta_bytes\": " << result.gpu.gpuPersistentDeltaBytes << "\n";
    os << "  },\n";
    os << "  \"accuracy\": {\n";
    os << "    \"max_abs\": " << result.accuracy.maxAbs << ",\n";
    os << "    \"mean_abs\": " << result.accuracy.meanAbs << ",\n";
    os << "    \"rmse\": " << result.accuracy.rmse << ",\n";
    os << "    \"l2_rel\": " << result.accuracy.l2Rel << "\n";
    os << "  },\n";
    os << "  \"speedup\": " << result.speedup << "\n";
    os << "}\n";
}

void printUsage() {
    std::cout
        << "Usage: spect_cuda_osem_compare.exe --matrix <path.csrb> [options]\n"
        << "  --threads <N>      CPU threads for CPU baseline and shared OSEM setup\n"
        << "  --iters <N>        OSEM iterations per run (default: 2)\n"
        << "  --subsets <N>      OSEM subsets (default: 1)\n"
        << "  --bed-count <N>    Cooperative bed count (default: 1)\n"
        << "  --bed-overlaps a,b Overlap list in Z slices when bed-count > 1\n"
        << "  --base-dim x,y,z   Base single-bed volume dimensions\n"
        << "  --warmup <N>       Warmup runs before timing (default: 1)\n"
        << "  --repeats <N>      Timed runs per backend (default: 3)\n"
        << "  --max-rows <N>     Limit matrix rows for smoke/perf runs\n"
        << "  --out <path.json>  Write JSON result\n";
}

CompareConfig parseArgs(int argc, char** argv) {
    CompareConfig config;

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        const auto requireValue = [&](const char* name) -> std::string {
            if (index + 1 >= argc) {
                fail(std::string("Missing value for ") + name);
            }
            return argv[++index];
        };

        if (arg == "--matrix") {
            config.matrixPath = requireValue("--matrix");
        } else if (arg == "--threads") {
            config.threads = std::stoi(requireValue("--threads"));
        } else if (arg == "--iters") {
            config.iterations = std::stoi(requireValue("--iters"));
        } else if (arg == "--subsets") {
            config.subsets = std::stoi(requireValue("--subsets"));
        } else if (arg == "--bed-count") {
            config.bedCount = std::stoi(requireValue("--bed-count"));
        } else if (arg == "--bed-overlaps") {
            config.bedOverlaps = parseCsvInts(requireValue("--bed-overlaps"));
        } else if (arg == "--base-dim") {
            const std::vector<int> dims = parseCsvInts(requireValue("--base-dim"));
            if (dims.size() != 3) {
                fail("--base-dim expects Nx,Ny,Nz");
            }
            config.baseNx = dims[0];
            config.baseNy = dims[1];
            config.baseNz = dims[2];
        } else if (arg == "--warmup") {
            config.warmupRuns = std::stoi(requireValue("--warmup"));
        } else if (arg == "--repeats") {
            config.repeats = std::stoi(requireValue("--repeats"));
        } else if (arg == "--max-rows") {
            config.maxRows = std::stoi(requireValue("--max-rows"));
        } else if (arg == "--out") {
            config.outputPath = requireValue("--out");
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            std::exit(0);
        } else {
            fail("Unknown argument: " + arg);
        }
    }

    if (config.matrixPath.empty()) {
        fail("--matrix is required.");
    }
    if (config.threads <= 0 || config.iterations <= 0 || config.subsets <= 0 ||
        config.warmupRuns < 0 || config.repeats <= 0 || config.bedCount <= 0) {
        fail("threads/iters/subsets/repeats must be positive and warmup must be non-negative.");
    }
    if (config.bedCount > 1 && config.subsets > 1) {
        fail("multibed compare currently supports subsets=1 only.");
    }
    return config;
}

} // namespace

int main(int argc, char** argv) {
    try {
        CompareConfig config = parseArgs(argc, argv);

        if (!spect::isCudaSparseOpsAvailable()) {
            fail("CUDA sparse backend is not linked into this build.");
        }
        if (!spect::cuda::isCudaAvailable()) {
            fail("No CUDA device is available at runtime.");
        }

        spect::io::CSRMap matrixMap;
        matrixMap.open(config.matrixPath);
        const spect::io::CSRView matrix = truncateView(matrixMap.view(), config.maxRows);
        inferBaseDims(config, matrix.cols);
        normalizeBedOverlaps(config);
        const int globalNz = computeGlobalNz(config);
        const int globalN = config.baseNx * config.baseNy * globalNz;
        const auto subsets =
            spect::make_simple_subsets(matrix.rows * config.bedCount, config.subsets);
        if (subsets.empty()) {
            fail("Failed to create subsets for the selected matrix rows.");
        }

        spect::ReconOpts truthOpts;
        truthOpts.threads = config.threads;
        truthOpts.n_subsets = config.subsets;
        truthOpts.bedCount = config.bedCount;
        if (config.bedCount > 1) {
            truthOpts.bedOverlaps = config.bedOverlaps;
            truthOpts.bedOffsetVoxels = makeBedOffsets(config);
        }

        const std::vector<double> truth = makeTruthImage(globalN);
        const std::vector<double> projections =
            makeSyntheticProjection(matrix, truth, truthOpts, config.threads);
        const std::vector<double> denom =
            computeDenom(matrix, truthOpts, globalN, config.threads);

        CompareResult result;
        result.rows = matrix.rows;
        result.cols = matrix.cols;
        result.nnz = matrix.nnz;
        result.iterations = config.iterations;
        result.subsets = config.subsets;
        result.warmupRuns = config.warmupRuns;
        result.repeats = config.repeats;
        result.threads = config.threads;
        result.bedCount = config.bedCount;
        result.baseNx = config.baseNx;
        result.baseNy = config.baseNy;
        result.baseNz = config.baseNz;
        result.globalNz = globalNz;
        result.hostBytesEstimate = estimateHostBytes(matrix, globalN, config.repeats);

        result.cpu = benchmarkBackend(matrix, subsets, projections, denom,
                                      spect::getSparseOpsCPU(), config,
                                      config.baseNx, config.baseNy, globalNz,
                                      false);
        result.gpu = benchmarkBackend(matrix, subsets, projections, denom,
                                      spect::getSparseOpsCUDA(), config,
                                      config.baseNx, config.baseNy, globalNz,
                                      true);
        result.accuracy = compareVectors(result.cpu.finalImage, result.gpu.finalImage);
        result.speedup =
            (result.gpu.timing.msAvg > 0.0)
                ? (result.cpu.timing.msAvg / result.gpu.timing.msAvg)
                : 0.0;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "rows=" << result.rows << ", cols=" << result.cols
                  << ", nnz=" << result.nnz << ", beds=" << result.bedCount
                  << ", subsets=" << result.subsets << ", iters="
                  << result.iterations << "\n";
        std::cout << "CPU avg=" << result.cpu.timing.msAvg << " ms"
                  << ", GPU avg=" << result.gpu.timing.msAvg << " ms"
                  << ", speedup=" << result.speedup << "x\n";
        std::cout << "accuracy max_abs=" << result.accuracy.maxAbs
                  << ", rmse=" << result.accuracy.rmse
                  << ", l2_rel=" << result.accuracy.l2Rel << "\n";
        std::cout << "host_bytes_estimate=" << result.hostBytesEstimate << " ("
                  << bytesToHuman(result.hostBytesEstimate) << ")\n";
        std::cout << "gpu_persistent_delta_bytes="
                  << result.gpu.gpuPersistentDeltaBytes << " ("
                  << bytesToHuman(result.gpu.gpuPersistentDeltaBytes) << ")\n";

        if (!config.outputPath.empty()) {
            std::ofstream output(config.outputPath, std::ios::binary);
            if (!output) {
                fail("Failed to open output JSON path: " + config.outputPath);
            }
            writeJson(result, config, output);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[CUDA-OSEM-COMPARE] " << ex.what() << std::endl;
        return 1;
    }
}