#include "app/BedOffsets.h"
#include "app/ProjectionOps.h"
#include "app/ReconConfig.h"
#include "cuda/CudaConfig.h"
#include "cuda/ProjDomainChunked.h"
#include "io/CSRBuilder.h"
#include "sparse/SparseOpsCPU.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
    std::string matrixPath;
    int maxRows = 0;
    int iterations = 6;
    int bedCount = 0;
    int baseNx = 0;
    int baseNy = 0;
    int baseNz = 0;
    int threads = 1;
    bool warmup = true;
    uint64_t nnzBudget = 0;
    std::vector<int> bedOverlaps;
    std::string outputPath;
};

struct OwnedTinyIoMatrix {
    std::vector<uint32_t> rowPtr;
    std::vector<uint32_t> colIdx;
    std::vector<float> values;
    int rows = 0;
    int cols = 0;

    spect::io::CSRView view() const {
        spect::io::CSRView matrix{};
        matrix.rows = rows;
        matrix.cols = cols;
        matrix.nnz = values.size();
        matrix.row_ptr = rowPtr.data();
        matrix.col_idx = colIdx.data();
        matrix.val = values.data();
        return matrix;
    }
};

struct LoadedMatrix {
    OwnedTinyIoMatrix synthetic;
    std::unique_ptr<spect::io::CSRMap> mapped;
    spect::io::CSRView view{};
};

struct CompareStats {
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    double l2Rel = 0.0;
};

struct ProbeResult {
    std::string modeName;
    std::vector<double> image;
    std::vector<double> numer;
    spect::cuda::CudaChunkStageTimingSnapshot timing;
    CompareStats imageVsCpu;
    CompareStats numerVsCpu;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
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

bool parseBool(const std::string& raw) {
    if (raw == "1" || raw == "true" || raw == "on" || raw == "yes") {
        return true;
    }
    if (raw == "0" || raw == "false" || raw == "off" || raw == "no") {
        return false;
    }
    fail("invalid boolean value: " + raw);
}

void parseBaseDims(Config& config, const std::string& raw) {
    const std::vector<int> dims = parseCsvInts(raw);
    if (dims.size() != 3) {
        fail("base dimensions must be Nx,Ny,Nz");
    }
    config.baseNx = dims[0];
    config.baseNy = dims[1];
    config.baseNz = dims[2];
}

void parseArgs(int argc, char** argv, Config& config) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--matrix") {
            config.matrixPath = requireValue("--matrix");
        } else if (arg == "--max-rows") {
            config.maxRows = std::stoi(requireValue("--max-rows"));
        } else if (arg == "--iterations") {
            config.iterations = std::stoi(requireValue("--iterations"));
        } else if (arg == "--bed-count") {
            config.bedCount = std::stoi(requireValue("--bed-count"));
        } else if (arg == "--base-dim") {
            parseBaseDims(config, requireValue("--base-dim"));
        } else if (arg == "--bed-overlap") {
            config.bedOverlaps = parseCsvInts(requireValue("--bed-overlap"));
        } else if (arg == "--threads") {
            config.threads = std::stoi(requireValue("--threads"));
        } else if (arg == "--warmup") {
            config.warmup = parseBool(requireValue("--warmup"));
        } else if (arg == "--nnz-budget") {
            config.nnzBudget = static_cast<uint64_t>(std::stoull(requireValue("--nnz-budget")));
        } else if (arg == "--output") {
            config.outputPath = requireValue("--output");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: spect_cuda_chunk_probe [options]\n"
                << "  --matrix <path>           Optional CSRB path; omit to use synthetic matrix\n"
                << "  --max-rows <n>           Optional row truncation for mapped CSR\n"
                << "  --iterations <n>         Repeated EM iterations (default 6)\n"
                << "  --bed-count <n>          Bed count (default synthetic=2, mapped=4)\n"
                << "  --base-dim Nx,Ny,Nz      Required if matrix.cols is not a cube\n"
                << "  --bed-overlap a,b,c      Multi-bed overlap list\n"
                << "  --threads <n>            CPU helper threads (default 1)\n"
                << "  --warmup on|off          Warm prepared cache before timing (default on)\n"
                << "  --nnz-budget <n>         Force chunk budget override\n"
                << "  --output <path>          Optional summary file\n";
            std::exit(0);
        } else {
            fail("unknown argument: " + arg);
        }
    }
}

OwnedTinyIoMatrix makeSyntheticMatrix() {
    OwnedTinyIoMatrix matrix;
    matrix.rowPtr = {0, 2, 4, 6, 8};
    matrix.colIdx = {0, 1, 1, 2, 0, 2, 0, 1};
    matrix.values = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 1.0f, 1.0f, 4.0f};
    matrix.rows = 4;
    matrix.cols = 3;
    return matrix;
}

spect::io::CSRView truncateView(const spect::io::CSRView& source, int maxRows) {
    if (!source.valid()) {
        fail("invalid CSR view");
    }
    if (maxRows <= 0 || maxRows >= source.rows) {
        return source;
    }
    spect::io::CSRView truncated = source;
    truncated.rows = maxRows;
    truncated.nnz = source.row_ptr[static_cast<size_t>(maxRows)];
    return truncated;
}

LoadedMatrix loadMatrix(const Config& config) {
    LoadedMatrix loaded;
    if (config.matrixPath.empty()) {
        loaded.synthetic = makeSyntheticMatrix();
        loaded.view = loaded.synthetic.view();
        return loaded;
    }

    loaded.mapped = std::make_unique<spect::io::CSRMap>();
    loaded.mapped->open(config.matrixPath);
    loaded.view = truncateView(loaded.mapped->view(), config.maxRows);
    return loaded;
}

void inferBaseDims(Config& config, int cols) {
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

    fail("unable to infer base dimensions from matrix.cols");
}

void normalizeBedOverlaps(Config& config) {
    if (config.bedCount <= 1) {
        config.bedOverlaps.clear();
        return;
    }
    const int joints = config.bedCount - 1;
    if (config.bedOverlaps.empty()) {
        config.bedOverlaps.assign(static_cast<size_t>(joints), 0);
    } else if (config.bedOverlaps.size() == 1 && joints > 1) {
        config.bedOverlaps.assign(static_cast<size_t>(joints),
                                  config.bedOverlaps.front());
    } else if (static_cast<int>(config.bedOverlaps.size()) < joints) {
        config.bedOverlaps.resize(static_cast<size_t>(joints), 0);
    } else if (static_cast<int>(config.bedOverlaps.size()) > joints) {
        config.bedOverlaps.resize(static_cast<size_t>(joints));
    }
}

std::vector<long long> makeBedOffsets(const Config& config) {
    const std::vector<int> bedZStarts = spect::app::computeBedZOffsets(
        config.bedCount, config.baseNz, config.bedOverlaps);
    const long long sliceSize =
        static_cast<long long>(config.baseNx) * static_cast<long long>(config.baseNy);
    std::vector<long long> offsets(static_cast<size_t>(config.bedCount), 0);
    for (int bed = 0; bed < config.bedCount; ++bed) {
        offsets[static_cast<size_t>(bed)] =
            static_cast<long long>(bedZStarts[static_cast<size_t>(bed)]) * sliceSize;
    }
    return offsets;
}

int computeGlobalNz(const Config& config) {
    return config.baseNz * config.bedCount -
           spect::app::computeTotalOverlap(config.bedCount, config.bedOverlaps);
}

std::vector<double> makeTruthImage(size_t count) {
    std::vector<double> truth(count, 1.0);
    for (size_t i = 0; i < count; ++i) {
        truth[i] = 0.75 + 0.2 * std::sin(0.013 * static_cast<double>(i)) +
                   0.1 * std::cos(0.031 * static_cast<double>(i));
    }
    return truth;
}

std::vector<double> makeInitialImage(size_t count) {
    return std::vector<double>(count, 1.0);
}

CompareStats compareVectors(const std::vector<double>& lhs,
                            const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        fail("compareVectors size mismatch");
    }

    CompareStats stats;
    double sumAbs = 0.0;
    double sumDiffSq = 0.0;
    double sumRefSq = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const double diff = lhs[i] - rhs[i];
        const double absDiff = std::abs(diff);
        stats.maxAbs = std::max(stats.maxAbs, absDiff);
        sumAbs += absDiff;
        sumDiffSq += diff * diff;
        sumRefSq += lhs[i] * lhs[i];
    }
    if (!lhs.empty()) {
        stats.meanAbs = sumAbs / static_cast<double>(lhs.size());
    }
    stats.l2Rel = (sumRefSq > 0.0) ? std::sqrt(sumDiffSq / sumRefSq) : 0.0;
    return stats;
}

std::vector<double> computeCpuProjections(const spect::io::CSRView& matrix,
                                          const std::vector<double>& truth,
                                          const std::vector<long long>& bedOffsets,
                                          int bedCount) {
    const std::vector<int> subsetRows =
        spect::make_simple_subsets(matrix.rows * bedCount, 1).front();
    spect::SparseSubsetContext ctx;
    ctx.bedCount = bedCount;
    ctx.bedOffsetVoxels = &bedOffsets;
    std::vector<double> projections(subsetRows.size(), 0.0);
    spect::getSparseOpsCPU().spmvSubset(matrix, truth.data(), subsetRows, ctx,
                                        projections.data(), 1);
    return projections;
}

std::vector<double> computeCpuDenom(const spect::io::CSRView& matrix,
                                    const std::vector<long long>& bedOffsets,
                                    int bedCount, int globalN, int threads) {
    spect::ReconOpts optsCpu;
    optsCpu.threads = threads;
    optsCpu.bedCount = bedCount;
    optsCpu.n_subsets = 1;
    optsCpu.bedOffsetVoxels = bedOffsets;
    optsCpu.sparseOps = &spect::getSparseOpsCPU();
    const std::vector<int> subsetRows =
        spect::make_simple_subsets(matrix.rows * bedCount, 1).front();
    return spect::computeSubsetSensitivity(matrix, subsetRows, globalN, threads,
                                           optsCpu);
}

std::vector<double> runCpuRepeatedEm(const spect::io::CSRView& matrix,
                                     const std::vector<double>& projections,
                                     const std::vector<double>& initialImage,
                                     const std::vector<double>& denom,
                                     const std::vector<long long>& bedOffsets,
                                     int bedCount, int iterations, int threads,
                                     std::vector<double>* lastNumer) {
    const std::vector<int> subsetRows =
        spect::make_simple_subsets(matrix.rows * bedCount, 1).front();
    spect::SparseSubsetContext ctx;
    ctx.bedCount = bedCount;
    ctx.bedOffsetVoxels = &bedOffsets;
    std::vector<double> image = initialImage;
    std::vector<double> yhat(projections.size(), 0.0);
    std::vector<double> ratio(projections.size(), 0.0);
    std::vector<double> numer(initialImage.size(), 0.0);

    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(yhat.begin(), yhat.end(), 0.0);
        std::fill(numer.begin(), numer.end(), 0.0);
        spect::getSparseOpsCPU().spmvSubset(matrix, image.data(), subsetRows, ctx,
                                            yhat.data(), threads);
        for (size_t i = 0; i < ratio.size(); ++i) {
            ratio[i] = projections[i] / std::max(yhat[i], 1e-12);
        }
        spect::getSparseOpsCPU().spmvTSubset(matrix, ratio.data(), subsetRows, ctx,
                                             numer.data(), threads);
        for (size_t voxel = 0; voxel < image.size(); ++voxel) {
            if (denom[voxel] > 1e-12) {
                image[voxel] *= numer[voxel] / denom[voxel];
            }
        }
    }

    if (lastNumer != nullptr) {
        *lastNumer = numer;
    }
    return image;
}

ProbeResult runGpuRepeatedEm(const spect::io::CSRView& matrix,
                             const std::vector<double>& projections,
                             const std::vector<double>& initialImage,
                             const std::vector<double>& denom,
                             const std::vector<long long>& bedOffsets,
                             int bedCount, int iterations, bool warmup,
                             const std::vector<double>& cpuImage,
                             const std::vector<double>& cpuNumer) {
    if (warmup) {
        if (!spect::cuda::warmupFullSubsetChunkCacheCUDA(matrix, bedCount, nullptr,
                                                         0, 1)) {
            fail("chunk warmup failed for stable mode");
        }
    }

    std::vector<double> image = initialImage;
    std::vector<double> numer(initialImage.size(), 0.0);
    spect::cuda::resetCudaChunkStageTimingCUDA();
    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(numer.begin(), numer.end(), 0.0);
        if (!spect::cuda::runFullSubsetEmNumerCUDA(
                matrix, projections, image.data(), bedCount, &bedOffsets, nullptr,
                0, 1, numer.data())) {
            fail("runFullSubsetEmNumerCUDA returned false for stable mode");
        }
        for (size_t voxel = 0; voxel < image.size(); ++voxel) {
            if (denom[voxel] > 1e-12) {
                image[voxel] *= numer[voxel] / denom[voxel];
            }
        }
    }

    ProbeResult result;
    result.modeName = "stable";
    result.image = std::move(image);
    result.numer = std::move(numer);
    result.timing = spect::cuda::getCudaChunkStageTimingSnapshotCUDA();
    result.imageVsCpu = compareVectors(cpuImage, result.image);
    result.numerVsCpu = compareVectors(cpuNumer, result.numer);
    return result;
}

void printResult(std::ostream& os, const ProbeResult& result) {
    os << "mode=" << result.modeName << '\n';
    os << "  image.maxAbs=" << std::scientific << result.imageVsCpu.maxAbs
       << " image.meanAbs=" << result.imageVsCpu.meanAbs
       << " image.l2Rel=" << result.imageVsCpu.l2Rel << '\n';
    os << "  numer.maxAbs=" << result.numerVsCpu.maxAbs
       << " numer.meanAbs=" << result.numerVsCpu.meanAbs
       << " numer.l2Rel=" << result.numerVsCpu.l2Rel << '\n';
    os << std::fixed << std::setprecision(6);
    os << "  chunk_build=" << result.timing.chunkBuildSeconds
       << "s pinned_stage=" << result.timing.pinnedStageSeconds
       << "s h2d_chunk=" << result.timing.h2dChunkSeconds
       << "s upload_wait=" << result.timing.uploadWaitOnComputeSeconds
       << "s fused_bed_cycle=" << result.timing.fusedBedCycleSeconds
       << "s cusparse_fwd=" << result.timing.cusparseForwardSeconds
       << "s cusparse_fwd_host=" << result.timing.cusparseForwardHostSeconds
       << "s ratio=" << result.timing.ratioSeconds
       << "s transpose_chunk=" << result.timing.transposeChunkSeconds
       << "s cusparse_bwd=" << result.timing.cusparseBackwardSeconds
       << "s cusparse_bwd_host=" << result.timing.cusparseBackwardHostSeconds
       << "s\n";
    os << "  calls chunk_build=" << result.timing.chunkBuildCalls
       << " pinned_stage=" << result.timing.pinnedStageCalls
       << " h2d_chunk=" << result.timing.h2dChunkCalls
       << " upload_wait=" << result.timing.uploadWaitOnComputeCalls
       << " fused_bed_cycle=" << result.timing.fusedBedCycleCalls
       << " cusparse_fwd=" << result.timing.cusparseForwardCalls
       << " cusparse_fwd_host=" << result.timing.cusparseForwardHostCalls
       << " ratio=" << result.timing.ratioCalls
       << " transpose_chunk=" << result.timing.transposeChunkCalls
       << " cusparse_bwd=" << result.timing.cusparseBackwardCalls
       << " cusparse_bwd_host=" << result.timing.cusparseBackwardHostCalls
       << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (!spect::cuda::isCudaAvailable()) {
            fail("CUDA device is not available.");
        }

        Config config;
        parseArgs(argc, argv, config);
        if (config.iterations <= 0) {
            fail("iterations must be positive");
        }

        LoadedMatrix loaded = loadMatrix(config);
        if (!loaded.view.valid()) {
            fail("loaded CSR view is invalid");
        }
        if (config.bedCount <= 0) {
            config.bedCount = config.matrixPath.empty() ? 2 : 4;
        }
        inferBaseDims(config, loaded.view.cols);
        normalizeBedOverlaps(config);

        const int globalNz = computeGlobalNz(config);
        const size_t globalN = static_cast<size_t>(config.baseNx) *
                               static_cast<size_t>(config.baseNy) *
                               static_cast<size_t>(globalNz);
        const std::vector<long long> bedOffsets = makeBedOffsets(config);
        const std::vector<double> truth = makeTruthImage(globalN);
        const std::vector<double> initialImage = makeInitialImage(globalN);
        const std::vector<double> projections =
            computeCpuProjections(loaded.view, truth, bedOffsets, config.bedCount);
        const std::vector<double> denom =
            computeCpuDenom(loaded.view, bedOffsets, config.bedCount,
                            static_cast<int>(globalN), config.threads);

        if (config.nnzBudget > 0) {
            spect::cuda::setCudaChunkNnzBudgetOverrideForTestingCUDA(
                config.nnzBudget);
        }

        std::vector<double> cpuNumer;
        const std::vector<double> cpuImage = runCpuRepeatedEm(
            loaded.view, projections, initialImage, denom, bedOffsets,
            config.bedCount, config.iterations, config.threads, &cpuNumer);

        std::ostringstream summary;
        summary << "matrix.rows=" << loaded.view.rows
                << " matrix.cols=" << loaded.view.cols
                << " matrix.nnz=" << loaded.view.nnz << '\n';
        summary << "bedCount=" << config.bedCount
                << " globalN=" << globalN
                << " iterations=" << config.iterations
                << " warmup=" << (config.warmup ? "on" : "off") << '\n';
        summary << "runtime chunk_nnz_budget="
                << spect::cuda::getProjDomainChunkNnzBudgetCUDA()
                << " prepared_host_cache_cap_bytes="
                << spect::cuda::getCudaPreparedChunkHostCacheBudgetCapCUDA()
                << " resolved_chunk_build_threads="
                << spect::cuda::resolveCudaHostChunkBuildThreadsForTestingCUDA(
                       config.threads, static_cast<uint64_t>(loaded.view.nnz))
                << '\n';

        const ProbeResult result = runGpuRepeatedEm(
            loaded.view, projections, initialImage, denom, bedOffsets,
            config.bedCount, config.iterations, config.warmup, cpuImage,
            cpuNumer);
        printResult(summary, result);

        spect::cuda::setCudaChunkNnzBudgetOverrideForTestingCUDA(0);

        const std::string text = summary.str();
        std::cout << text;
        if (!config.outputPath.empty()) {
            std::ofstream out(config.outputPath, std::ios::binary);
            out << text;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "spect_cuda_chunk_probe failed: " << ex.what() << '\n';
        spect::cuda::setCudaChunkNnzBudgetOverrideForTestingCUDA(0);
        return 1;
    }
}
