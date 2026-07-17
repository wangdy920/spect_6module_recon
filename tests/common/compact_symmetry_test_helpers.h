#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "io/CompactSymmetryMetadata.h"

namespace spect {
namespace io {
namespace test {

inline spect::io::CompactSymmetryMetadata makeToySym45ZMeta() {
    spect::io::CompactSymmetryMetadata meta;
    meta.Nx = 2;
    meta.Ny = 2;
    meta.Nz = 2;
    meta.totalVoxels = 8;
    meta.fullNumAngles = 16;
    meta.compactNumAngles = 3;
    meta.detNx = 2;
    meta.detNz = 2;
    meta.compactDetZ = 1;
    meta.compactRaysPerAngle = 2;
    meta.totalRows = 6;
    meta.useZSym = true;
    meta.use45Sym = true;
    meta.symCode = 45;
    meta.symmetryOrder = 8;
    meta.mode = spect::io::CompactSymmetryMode::Sym45Z;
    return meta;
}

struct ToyCompactCSR {
    std::vector<uint32_t> rowPtr;
    std::vector<uint32_t> colIdx;
    std::vector<float> values;
};

inline ToyCompactCSR makeToyCompactCSR() {
    ToyCompactCSR csr;
    csr.rowPtr = {0, 2, 3, 5, 6, 8, 10};
    csr.colIdx = {0, 3, 5, 1, 6, 2, 4, 7, 0, 6};
    csr.values = {1.0f, 0.5f, 0.75f, 0.2f, 1.1f,
                  0.9f, 0.4f, 1.2f, 0.3f, 0.8f};
    return csr;
}

inline std::vector<double> expandCompactToyToFullDense(
    const spect::io::CompactSymmetryMetadata& meta,
    const ToyCompactCSR& compact) {
    std::vector<double> full(static_cast<size_t>(meta.fullRows()) *
                                 static_cast<size_t>(meta.totalVoxels),
                             0.0);
    for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
        for (int z = 0; z < meta.zBranchCount(); ++z) {
            const bool zMirror = z != 0;
            for (int row = 0; row < meta.totalRows; ++row) {
                uint64_t fullRow = 0;
                if (!spect::io::fullProjectionIndexFromCompactRow(
                        meta, row, sym, zMirror, fullRow)) {
                    continue;
                }
                for (uint32_t t = compact.rowPtr[static_cast<size_t>(row)];
                     t < compact.rowPtr[static_cast<size_t>(row) + 1u]; ++t) {
                    const size_t compactCol = compact.colIdx[static_cast<size_t>(t)];
                    const size_t fullCol = spect::io::mapSym45ZVoxelIndex(
                        compactCol, meta.Nx, meta.Ny, meta.Nz, sym, zMirror);
                    full[static_cast<size_t>(fullRow) *
                             static_cast<size_t>(meta.totalVoxels) +
                         fullCol] += compact.values[static_cast<size_t>(t)];
                }
            }
        }
    }
    return full;
}

inline std::vector<double> denseDenom(const std::vector<double>& full, int rows,
                               int cols) {
    std::vector<double> denom(static_cast<size_t>(cols), 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            denom[static_cast<size_t>(col)] +=
                full[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                     static_cast<size_t>(col)];
        }
    }
    return denom;
}

inline std::vector<double> compactToyDenom(
    const spect::io::CompactSymmetryMetadata& meta,
    const ToyCompactCSR& compact) {
    std::vector<double> denom(static_cast<size_t>(meta.totalVoxels), 0.0);
    for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
        for (int z = 0; z < meta.zBranchCount(); ++z) {
            const bool zMirror = z != 0;
            for (int row = 0; row < meta.totalRows; ++row) {
                uint64_t fullRow = 0;
                if (!spect::io::fullProjectionIndexFromCompactRow(
                        meta, row, sym, zMirror, fullRow)) {
                    continue;
                }
                for (uint32_t t = compact.rowPtr[static_cast<size_t>(row)];
                     t < compact.rowPtr[static_cast<size_t>(row) + 1u]; ++t) {
                    const size_t fullCol = spect::io::mapSym45ZVoxelIndex(
                        compact.colIdx[static_cast<size_t>(t)], meta.Nx,
                        meta.Ny, meta.Nz, sym, zMirror);
                    denom[fullCol] += compact.values[static_cast<size_t>(t)];
                }
            }
        }
    }
    return denom;
}

inline std::vector<double> denseNumerator(const std::vector<double>& full,
                                   const std::vector<double>& projection,
                                   const std::vector<double>& image, int rows,
                                   int cols) {
    std::vector<double> yhat(static_cast<size_t>(rows), 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            yhat[static_cast<size_t>(row)] +=
                full[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                     static_cast<size_t>(col)] *
                image[static_cast<size_t>(col)];
        }
    }
    std::vector<double> numer(static_cast<size_t>(cols), 0.0);
    for (int row = 0; row < rows; ++row) {
        const double ratio = projection[static_cast<size_t>(row)] /
                             std::max(yhat[static_cast<size_t>(row)], 1e-10);
        for (int col = 0; col < cols; ++col) {
            numer[static_cast<size_t>(col)] +=
                full[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                     static_cast<size_t>(col)] *
                ratio;
        }
    }
    return numer;
}

inline std::vector<double> compactToyNumerator(
    const spect::io::CompactSymmetryMetadata& meta,
    const ToyCompactCSR& compact, const std::vector<double>& projection,
    const std::vector<double>& image) {
    std::vector<double> numer(static_cast<size_t>(meta.totalVoxels), 0.0);
    for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
        for (int z = 0; z < meta.zBranchCount(); ++z) {
            const bool zMirror = z != 0;
            std::vector<double> yhat(static_cast<size_t>(meta.totalRows), 0.0);
            for (int row = 0; row < meta.totalRows; ++row) {
                for (uint32_t t = compact.rowPtr[static_cast<size_t>(row)];
                     t < compact.rowPtr[static_cast<size_t>(row) + 1u]; ++t) {
                    const size_t compactCol = compact.colIdx[static_cast<size_t>(t)];
                    const size_t fullCol = spect::io::mapSym45ZVoxelIndex(
                        compactCol, meta.Nx, meta.Ny, meta.Nz, sym, zMirror);
                    yhat[static_cast<size_t>(row)] +=
                        compact.values[static_cast<size_t>(t)] * image[fullCol];
                }
            }
            for (int row = 0; row < meta.totalRows; ++row) {
                uint64_t fullRow = 0;
                if (!spect::io::fullProjectionIndexFromCompactRow(
                        meta, row, sym, zMirror, fullRow)) {
                    continue;
                }
                const double ratio = projection[static_cast<size_t>(fullRow)] /
                                     std::max(yhat[static_cast<size_t>(row)],
                                              1e-10);
                for (uint32_t t = compact.rowPtr[static_cast<size_t>(row)];
                     t < compact.rowPtr[static_cast<size_t>(row) + 1u]; ++t) {
                    const size_t fullCol = spect::io::mapSym45ZVoxelIndex(
                        compact.colIdx[static_cast<size_t>(t)], meta.Nx,
                        meta.Ny, meta.Nz, sym, zMirror);
                    numer[fullCol] +=
                        compact.values[static_cast<size_t>(t)] * ratio;
                }
            }
        }
    }
    return numer;
}

inline std::vector<double> transformedImage(const std::vector<double>& image, int nx,
                                     int ny, int nz, int sym,
                                     bool zMirror) {
    std::vector<double> out(image.size(), 0.0);
    for (size_t idx = 0; idx < image.size(); ++idx) {
        out[idx] = image[spect::io::mapSym45ZVoxelIndex(idx, nx, ny, nz, sym,
                                                        zMirror)];
    }
    return out;
}

inline void expectVectorsNear(const std::vector<double>& a,
                       const std::vector<double>& b, double tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "index=" << i;
    }
}

} // namespace test
} // namespace io
} // namespace spect
