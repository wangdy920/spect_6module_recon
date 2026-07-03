/**
 * @file CudaProjector.h
 * @brief 【T6.3】GPU投影器接口
 * 
 * 基于SparseOpsGPU的IProjector实现。
 */

#pragma once

#include "CudaConfig.h"
#include "SparseOpsGPU.h"
#include "../interfaces/IProjector.h"
#include "../core/CSRView.h"

#if CUDA_ENABLED

#include <memory>
#include <vector>

namespace spect {
namespace cuda {

/**
 * @brief GPU投影器
 * 
 * 封装SparseOpsGPU，实现IProjector接口。
 * 系统矩阵预上传到GPU，减少重复传输开销。
 */
class CudaProjector : public IProjector {
public:
    /**
     * @brief 构造函数
     * @param A 系统矩阵 (Rays × Voxels)
     * @param At 转置矩阵 (Voxels × Rays)
     * @param dims 图像维度
     */
    CudaProjector(const CSRView& A, const CSRView& At, Dims3D dims);
    
    ~CudaProjector() override = default;
    
    // IProjector接口
    void forward(const double* f, double* p, int threads = 1) const override;
    void backward(const double* p, double* f, int threads = 1) const override;
    
    Dims3D imageDims() const override { return dims_; }
    int64_t projectionSize() const override { return numRays_; }
    int64_t numVoxels() const override { return dims_.total(); }
    
    void computeSensitivity(double* sens, int threads = 1) const override;
    
    /**
     * @brief 预计算敏感度图像并缓存
     */
    void precomputeSensitivity();
    
    /**
     * @brief 获取缓存的敏感度
     */
    const std::vector<double>& cachedSensitivity() const { return sensitivity_; }
    
private:
    Dims3D dims_;
    int64_t numRays_;
    
    std::unique_ptr<SparseOpsGPU> opsA_;   // 前向投影
    std::unique_ptr<SparseOpsGPU> opsAt_;  // 反向投影
    
    std::vector<double> sensitivity_;
    bool sensitivityComputed_ = false;
};

/**
 * @brief GPU投影器工厂
 */
class CudaProjectorFactory {
public:
    /**
     * @brief 从CSRView创建GPU投影器
     */
    static std::unique_ptr<CudaProjector> create(
        const CSRView& A, const CSRView& At, Dims3D dims);
    
    /**
     * @brief 从文件加载并创建GPU投影器
     */
    static std::unique_ptr<CudaProjector> load(
        const std::string& matrixPath, Dims3D dims);
};

} // namespace cuda
} // namespace spect

#else // !CUDA_ENABLED

namespace spect {
namespace cuda {

class CudaProjector : public IProjector {
public:
    CudaProjector(const CSRView&, const CSRView&, Dims3D) {
        throw std::runtime_error("CUDA not available - CudaProjector cannot be used");
    }
    
    void forward(const double*, double*, int) const override {}
    void backward(const double*, double*, int) const override {}
    Dims3D imageDims() const override { return {}; }
    int64_t projectionSize() const override { return 0; }
    int64_t numVoxels() const override { return 0; }
    void computeSensitivity(double*, int) const override {}
};

} // namespace cuda
} // namespace spect

#endif // CUDA_ENABLED
