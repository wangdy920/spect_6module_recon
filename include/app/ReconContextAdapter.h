/**
 * @file ReconContextAdapter.h
 * @brief 【T5.2】ReconContext到ReconOpts的适配器
 * 
 * 提供从新的ReconContext结构到旧ReconOpts的转换，
 * 支持渐进式迁移。
 */

#pragma once

#include "app/ReconContext.h"
#include "app/BedOffsets.h"
#include "app/ReconConfig.h"  // ReconOpts
#include "core/AttenuationConstants.h"
#include "sparse/SparseOpsBackend.h"
#include <vector>

namespace spect {
namespace app {

/**
 * @brief 从ReconContext创建ReconOpts
 * 
 * 用于在迁移期间桥接新旧API。
 */
inline ReconOpts createReconOpts(const ReconContext& ctx) {
    ReconOpts opts;
    
    // Basic settings
    opts.threads = ctx.numThreads;
    opts.nonneg = true;  // Default enforces non-negativity
    
    // Multi-bed settings
    opts.bedCount = ctx.bedCount;
    opts.bedOverlaps = ctx.bedOverlaps;
    
    // Calculate bed offset voxels
    opts.bedOffsetVoxels.resize(ctx.bedCount);
    if (!ctx.bedZStarts.empty()) {
        int sliceSize = ctx.Nx * ctx.Ny;
        for (int b = 0; b < ctx.bedCount; ++b) {
            opts.bedOffsetVoxels[b] = static_cast<long long>(ctx.bedZStarts[b]) * sliceSize;
        }
    } else {
        // Fallback: simple stacking
        int baseVoxels = ctx.Nx * ctx.Ny * ctx.baseNz;
        for (int b = 0; b < ctx.bedCount; ++b) {
            opts.bedOffsetVoxels[b] = static_cast<long long>(b) * baseVoxels;
        }
    }
    
    // Detector geometry for PSF
    opts.det_nz = ctx.detNz;
    opts.det_nx = ctx.detNx;
    opts.num_angles = ctx.numAngles;
    
    // PSF settings (basic)
    opts.psf.mode = PSFMode::None;  // Default off, caller can override

    SparseBackendChoice resolvedBackend = SparseBackendChoice::Auto;
    bool parsedBackend = false;
    opts.sparseOps =
        &selectSparseOps(ctx.args.sparseBackend, &resolvedBackend, &parsedBackend);

    return opts;
}

/**
 * @brief 从ReconArgs创建ReconOpts
 */
inline ReconOpts createReconOpts(const cli::ReconArgs& args) {
    ReconOpts opts;
    
    opts.threads = args.numThreads;
    opts.n_subsets = args.numSubsets;
    opts.nonneg = true;
    
    // Multi-bed
    opts.bedCount = args.getEffectiveBedCount();
    opts.bedOverlaps = args.bedOverlaps;
    
    // Pre-calculate offsets
    if (opts.bedCount > 1) {
        opts.bedOffsetVoxels.resize(opts.bedCount);
        const int baseNz = inferBaseNzFromCombinedNz(args.Nz, opts.bedCount, args.bedOverlaps);
        const std::vector<int> bedZStarts = computeBedZOffsets(opts.bedCount, baseNz, args.bedOverlaps);
        
        int sliceSize = args.Nx * args.Ny;
        for (int b = 0; b < opts.bedCount; ++b) {
            opts.bedOffsetVoxels[b] = static_cast<long long>(bedZStarts[b]) * sliceSize;
        }
    } else {
        opts.bedOffsetVoxels = {0};
    }
    
    // Detector geometry
    opts.det_nz = args.detNz > 0 ? args.detNz : args.detectorChannels;
    opts.det_nx = args.detNx > 0 ? args.detNx : args.detectorChannels;
    opts.num_angles = args.numAngles;

    SparseBackendChoice resolvedBackend = SparseBackendChoice::Auto;
    bool parsedBackend = false;
    opts.sparseOps =
        &selectSparseOps(args.sparseBackend, &resolvedBackend, &parsedBackend);

    return opts;
}

/**
 * @brief 将ReconContext的床位衰减因子转换为ReconOpts格式
 * 
 * ReconOpts使用uint16_t数组存储衰减因子 (0-65535 映射到 0.0-1.0)
 * 这个函数预留给需要使用衰减校正的情况
 */
inline void applyBedAttenFactors(ReconOpts& opts, const ReconContext& ctx,
                                  int numAngles, size_t voxelsPerBed) {
    if (ctx.bedInfos.empty()) return;
    
    opts.bedAttenFactors.resize(ctx.bedCount);
    
    for (int b = 0; b < ctx.bedCount; ++b) {
        double factor = ctx.bedInfos[b].attenFactor;
        if (std::abs(factor - 1.0) < 1e-9) {
            // No attenuation, leave empty
            continue;
        }
        
        // Convert to uint16_t format (placeholder - actual CTAC would need full maps)
        uint16_t attenVal = static_cast<uint16_t>(std::clamp(factor, 0.0, 1.0) * kAttenUint16Scale);
        size_t mapSize = static_cast<size_t>(numAngles) * voxelsPerBed;
        opts.bedAttenFactors[b].resize(mapSize, attenVal);
    }
}

/**
 * @brief 计算组合图像的总体素数
 */
inline size_t calculateTotalVoxels(const ReconContext& ctx) {
    return static_cast<size_t>(ctx.Nx) * ctx.Ny * ctx.Nz;
}

/**
 * @brief 计算每床位的射线数
 */
inline size_t calculateRaysPerBed(const ReconContext& ctx) {
    return static_cast<size_t>(ctx.detNz) * ctx.detNx * ctx.numAngles;
}

} // namespace app
} // namespace spect
