#pragma once

#include "app/ProjectionOps.h"
#include "filters/Convolution.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spect {
namespace app {

struct MetricComputationResult {
    double likelihood = 0.0;
    const char* modelTag = "runtime";
};

namespace detail {

template <typename MatrixType>
inline double computeForwardProjectionValue(const MatrixType& matrix,
                                            int row,
                                            size_t colOffset,
                                            const std::vector<double>& image,
                                            const uint16_t* attPtr) {
    const double invScale = 1.0 / kAttenUint16Scale;
    double sum = 0.0;
    const size_t r0 = static_cast<size_t>(matrix.row_ptr[static_cast<size_t>(row)]);
    const size_t r1 = static_cast<size_t>(matrix.row_ptr[static_cast<size_t>(row) + 1]);
    for (size_t t = r0; t < r1; ++t) {
        double value = static_cast<double>(matrix.val[t]);
        if (attPtr) {
            value *= static_cast<double>(attPtr[matrix.col_idx[t]]) * invScale;
        }
        sum += value * image[static_cast<size_t>(matrix.col_idx[t]) + colOffset];
    }
    return sum;
}

} // namespace detail

template <typename MatrixType>
inline std::vector<double> computeExpectedProjectionRuntime(
    const MatrixType& matrix,
    const std::vector<double>& image,
    const ReconOpts& opts,
    int Nx,
    int Ny,
    int Nz,
    const PSFKernels* psfKernels,
    int threads,
    const char** modelTag = nullptr) {
    const int bedCount = std::max(1, opts.bedCount);
    const size_t expectedProjectionCount =
        static_cast<size_t>(matrix.rows) * static_cast<size_t>(bedCount);

    std::vector<double> blurredImage;
    const std::vector<double>* imageUsed = &image;
    if (needs_psf_blur(opts, psfKernels)) {
        apply_psf_forward(image, blurredImage, Nx, Ny, Nz, opts, psfKernels, threads);
        imageUsed = &blurredImage;
    }

    std::vector<double> predicted(expectedProjectionCount, 0.0);
    const int baseRows = matrix.rows;
    const int baseCols = matrix.cols;
    const size_t localVoxelCount = static_cast<size_t>(baseCols);
    const size_t raysPerAngle =
        (opts.num_angles > 0) ? (static_cast<size_t>(baseRows) / static_cast<size_t>(opts.num_angles)) : 0;

#pragma omp parallel for schedule(static) num_threads(threads)
    for (ptrdiff_t globalRow = 0; globalRow < static_cast<ptrdiff_t>(expectedProjectionCount);
         ++globalRow) {
        const int bed = (bedCount > 1) ? static_cast<int>(globalRow) / baseRows : 0;
        const int row = (bedCount > 1) ? static_cast<int>(globalRow) % baseRows
                                       : static_cast<int>(globalRow);
        const size_t colOffset = !opts.bedOffsetVoxels.empty()
                                     ? static_cast<size_t>(opts.bedOffsetVoxels[static_cast<size_t>(bed)])
                                     : static_cast<size_t>(bed) * localVoxelCount;

        const uint16_t* attPtr = nullptr;
        if (!opts.bedAttenFactors.empty()) {
            int angle = 0;
            if (raysPerAngle > 0) {
                angle = row / static_cast<int>(raysPerAngle);
            }
            if (angle >= opts.num_angles) {
                angle = opts.num_angles - 1;
            }
            attPtr = opts.bedAttenFactors[static_cast<size_t>(bed)].data() +
                     static_cast<size_t>(angle) * localVoxelCount;
        }

        predicted[static_cast<size_t>(globalRow)] =
            detail::computeForwardProjectionValue(matrix, row, colOffset, *imageUsed, attPtr);
    }

    const char* resolvedTag = "runtime";
    if (opts.psf.mode == PSFMode::ProjDomain) {
        resolvedTag = "runtime-projdomain";
        const size_t frameSize = static_cast<size_t>(opts.det_nz) * static_cast<size_t>(opts.det_nx);
        const bool geometryValid = psfKernels != nullptr && opts.det_nz > 0 && opts.det_nx > 0 &&
                                   opts.num_angles > 0 && frameSize > 0 &&
                                   frameSize * static_cast<size_t>(opts.num_angles) ==
                                       static_cast<size_t>(baseRows);
        if (geometryValid) {
            for (int bed = 0; bed < bedCount; ++bed) {
                const size_t bedOffset = static_cast<size_t>(bed) * static_cast<size_t>(baseRows);
                for (int angle = 0; angle < opts.num_angles; ++angle) {
                    const size_t frameOffset = bedOffset + static_cast<size_t>(angle) * frameSize;
                    std::vector<double> frame(predicted.begin() + frameOffset,
                                              predicted.begin() + frameOffset + frameSize);
                    filters::conv2d_separable_inplace(frame, opts.det_nz, opts.det_nx,
                                                      psfKernels->ky, psfKernels->kx, threads);
                    std::copy(frame.begin(), frame.end(), predicted.begin() + frameOffset);
                }
            }
        } else {
            resolvedTag = "approx-bare";
        }
    }

#pragma omp parallel for schedule(static) num_threads(threads)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(predicted.size()); ++i) {
        if (opts.w_proj) {
            predicted[static_cast<size_t>(i)] *= (*opts.w_proj)[static_cast<size_t>(i)];
        }
        if (opts.b_proj) {
            predicted[static_cast<size_t>(i)] += (*opts.b_proj)[static_cast<size_t>(i)];
        }
    }

    if (modelTag) {
        *modelTag = resolvedTag;
    }
    return predicted;
}

template <typename MatrixType>
inline MetricComputationResult computePoissonLogLikelihoodRuntime(
    const MatrixType& matrix,
    const std::vector<double>& image,
    const std::vector<double>& projections,
    const ReconOpts& opts,
    int Nx,
    int Ny,
    int Nz,
    const PSFKernels* psfKernels,
    int threads) {
    const size_t expectedProjectionCount =
        static_cast<size_t>(matrix.rows) * static_cast<size_t>(std::max(1, opts.bedCount));
    if (projections.size() != expectedProjectionCount) {
        throw std::runtime_error("Metric projection size mismatch.");
    }

    const char* modelTag = "runtime";
    const std::vector<double> predicted = computeExpectedProjectionRuntime(
        matrix, image, opts, Nx, Ny, Nz, psfKernels, threads, &modelTag);

    constexpr double eps = 1e-12;
    double logLikelihood = 0.0;
#pragma omp parallel for reduction(+ : logLikelihood) schedule(static) num_threads(threads)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(predicted.size()); ++i) {
        const double modeled = std::max(predicted[static_cast<size_t>(i)], eps);
        const double measured = projections[static_cast<size_t>(i)];
        logLikelihood += measured * std::log(modeled) - modeled;
    }

    return MetricComputationResult{logLikelihood, modelTag};
}

inline std::string formatMetricLine(double beta, double likelihood, double penalty,
                                    const char* modelTag = nullptr) {
    std::ostringstream os;
    os << "[METRIC] Beta=" << beta
       << " Likelihood=" << likelihood
       << " Penalty=" << penalty
       << " Objective=" << (likelihood - penalty);
    if (modelTag && *modelTag) {
        os << " Model=" << modelTag;
    }
    return os.str();
}

} // namespace app
} // namespace spect
