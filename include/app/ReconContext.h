// ReconContext.h - Reconstruction context and configuration
// 【T2.3】【T5.1】封装重建运行时所需的所有上下文
#pragma once

#include "cli/ReconArgs.h"
#include <array>
#include <string>
#include <vector>
#include <stdexcept>

namespace spect {
namespace app {

/**
 * @brief 多床位融合策略
 */
enum class BedFusionStrategy {
    Average,      ///< 重叠区域取平均
    Weighted,     ///< 加权平均（线性过渡）
    Max,          ///< 取最大值
    First,        ///< 取第一个床位值
    Last          ///< 取最后一个床位值
};

/**
 * @brief 床位信息结构
 */
struct BedInfo {
    int index = 0;              ///< 床位索引 (0-based)
    int zStart = 0;             ///< 在合成图像中的Z起始位置
    int zEnd = 0;               ///< 在合成图像中的Z结束位置
    int localNz = 0;            ///< 床位本地Z尺寸
    double attenFactor = 1.0;   ///< 衰减校正因子
    size_t rayOffset = 0;       ///< 投影数据偏移
    size_t rayCount = 0;        ///< 该床位的射线数
};

/**
 * @brief Reconstruction runtime context.
 * 
 * Holds all runtime state needed for reconstruction:
 * - Parsed arguments (ReconArgs)
 * - Loaded projection data
 * - System matrix paths/handles
 * - Multi-bed configuration
 */
struct ReconContext {
    // ========== From CLI ==========
    spect::cli::ReconArgs args;
    
    // ========== Projection Data ==========
    std::vector<double> projections;        ///< All bed projections concatenated
    std::vector<double> backgroundProj;     ///< Background/scatter term (optional)
    size_t raysPerBed = 0;                  ///< Rays per single bed
    size_t totalRays = 0;                   ///< Total rays across all beds
    
    // ========== Multi-Bed ==========
    int bedCount = 1;
    int baseNz = 0;                         ///< Original Nz per bed
    int combinedNz = 0;                     ///< Total Nz after overlap
    std::vector<int> bedOverlaps;           ///< Overlap pixels between adjacent beds
    std::vector<int> bedZStarts;            ///< Z start index for each bed
    std::vector<BedInfo> bedInfos;          ///< 【T5.1】详细床位信息
    BedFusionStrategy fusionStrategy = BedFusionStrategy::Weighted; ///< 【T5.1】融合策略
    
    // ========== Image Dimensions ==========
    int Nx = 0, Ny = 0, Nz = 0;
    std::array<double, 3> voxelSpacing = {1.0, 1.0, 1.0};
    size_t numVoxels = 0;
    
    // ========== Detector ==========
    int detNz = 0, detNx = 0;
    int numAngles = 0;
    size_t raysPerAngle = 0;
    
    // ========== Runtime State ==========
    int verbosity = 1;
    int numThreads = 1;
    std::string outputDir;
    std::string outputName;
    
    // ========== Helper Methods ==========
    
    /**
     * @brief Initialize context from parsed arguments.
     */
    void initFromArgs() {
        Nx = args.Nx;
        Ny = args.Ny;
        baseNz = args.Nz;
        Nz = args.Nz;
        voxelSpacing = args.getVoxelSpacing();
        numVoxels = static_cast<size_t>(Nx) * Ny * Nz;
        
        detNz = args.detNz > 0 ? args.detNz : args.detectorChannels;
        detNx = args.detNx > 0 ? args.detNx : args.detectorChannels;
        numAngles = args.numAngles;
        raysPerAngle = static_cast<size_t>(detNz) * detNx;
        
        verbosity = args.verbosity;
        numThreads = args.numThreads;
        outputDir = args.outputDir;
        outputName = args.outputName;
        
        bedCount = args.getEffectiveBedCount();
        bedOverlaps = args.bedOverlaps;
    }
    
    /**
     * @brief Calculate combined Nz for multi-bed.
     */
    void calculateCombinedNz() {
        if (bedCount <= 1) {
            combinedNz = baseNz;
            bedZStarts = {0};
            // 【T5.1】填充单床位信息
            bedInfos.resize(1);
            bedInfos[0] = BedInfo{0, 0, baseNz, baseNz, 1.0, 0, raysPerBed};
            return;
        }
        
        // Ensure we have enough overlaps
        if (bedOverlaps.size() < static_cast<size_t>(bedCount - 1)) {
            bedOverlaps.resize(bedCount - 1, 0);
        }
        
        // Calculate total Z: baseNz * bedCount - sum(overlaps)
        int totalOverlap = 0;
        for (int ov : bedOverlaps) totalOverlap += ov;
        combinedNz = baseNz * bedCount - totalOverlap;
        Nz = combinedNz;
        numVoxels = static_cast<size_t>(Nx) * Ny * Nz;
        
        // Calculate Z start for each bed
        bedZStarts.resize(bedCount);
        bedZStarts[0] = 0;
        for (int b = 1; b < bedCount; ++b) {
            int stride = baseNz - bedOverlaps[b - 1];
            bedZStarts[b] = bedZStarts[b - 1] + stride;
        }
        
        // 【T5.1】填充详细床位信息
        bedInfos.resize(bedCount);
        for (int b = 0; b < bedCount; ++b) {
            BedInfo& info = bedInfos[b];
            info.index = b;
            info.zStart = bedZStarts[b];
            info.zEnd = info.zStart + baseNz;
            if (info.zEnd > Nz) info.zEnd = Nz;
            info.localNz = baseNz;
            info.attenFactor = 1.0;  // 默认无衰减校正
            info.rayOffset = static_cast<size_t>(b) * raysPerBed;
            info.rayCount = raysPerBed;
        }
    }
    
    /**
     * @brief 【T5.1】获取床位信息
     * @param bedIndex 0-based床位索引
     * @return 床位信息引用
     */
    const BedInfo& getBedInfo(int bedIndex) const {
        if (bedIndex < 0 || bedIndex >= static_cast<int>(bedInfos.size())) {
            throw std::out_of_range("Bed index out of range");
        }
        return bedInfos[bedIndex];
    }
    
    /**
     * @brief 【T5.1】设置床位衰减因子
     */
    void setBedAttenFactor(int bedIndex, double factor) {
        if (bedIndex >= 0 && bedIndex < static_cast<int>(bedInfos.size())) {
            bedInfos[bedIndex].attenFactor = factor;
        }
    }
    
    /**
     * @brief 【T5.1】获取重叠区域的融合权重
     * @param z 全局Z索引
     * @param bedIndex 床位索引
     * @return 融合权重 (0.0-1.0)
     */
    double getOverlapWeight(int z, int bedIndex) const {
        if (!isMultiBed() || bedIndex < 0 || bedIndex >= bedCount) {
            return 1.0;
        }
        
        const BedInfo& info = bedInfos[bedIndex];
        
        // 检查是否在重叠区域
        if (bedIndex > 0) {
            // 与前一床位的重叠
            int overlapStart = bedZStarts[bedIndex];
            int overlapEnd = bedZStarts[bedIndex - 1] + baseNz;
            if (z >= overlapStart && z < overlapEnd) {
                // 线性权重过渡
                int overlapSize = overlapEnd - overlapStart;
                if (overlapSize > 0) {
                    double t = static_cast<double>(z - overlapStart) / overlapSize;
                    return t;  // 从0过渡到1
                }
            }
        }
        
        if (bedIndex < bedCount - 1) {
            // 与后一床位的重叠
            int overlapStart = bedZStarts[bedIndex + 1];
            int overlapEnd = info.zEnd;
            if (z >= overlapStart && z < overlapEnd) {
                int overlapSize = overlapEnd - overlapStart;
                if (overlapSize > 0) {
                    double t = static_cast<double>(z - overlapStart) / overlapSize;
                    return 1.0 - t;  // 从1过渡到0
                }
            }
        }
        
        return 1.0;  // 非重叠区域
    }
    
    /**
     * @brief Get Z range for a specific bed.
     * @param bedIndex 0-based bed index.
     * @return Pair of (zStart, zEnd) indices.
     */
    std::pair<int, int> getBedZRange(int bedIndex) const {
        if (bedIndex < 0 || bedIndex >= bedCount) {
            return {0, 0};
        }
        int zStart = bedZStarts[bedIndex];
        int zEnd = zStart + baseNz;
        if (zEnd > Nz) zEnd = Nz;
        return {zStart, zEnd};
    }
    
    /**
     * @brief Check if multi-bed mode is active.
     */
    bool isMultiBed() const { return bedCount > 1; }
};

} // namespace app
} // namespace spect
