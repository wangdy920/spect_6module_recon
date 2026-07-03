/**
 * @file IProjector.h
 * @brief 投影/反投影算子统一接口
 * 
 * 设计约束:
 * - CSR-backed projector必须同时能访问A和A^T
 * - Backward投影必须基于A^T进行行式SpMV，严禁使用Atomic Scatter
 */

#pragma once

#include "../core/CSRView.h"
#include "../core/Types.h"

namespace spect {

/**
 * @brief 投影/反投影算子接口
 */
class IProjector {
public:
    virtual ~IProjector() = default;
    
    /**
     * @brief 前向投影 (p = A * f)
     * 
     * @param f 输入图像 (voxels)
     * @param p 输出投影数据 (detectors)
     * @param threads 线程数
     */
    virtual void forward(const double* f, double* p, int threads = 1) const = 0;
    
    /**
     * @brief 反向投影 (f = A^T * p)
     * 
     * @param p 输入投影数据 (detectors)
     * @param f 输出图像 (voxels)
     * @param threads 线程数
     */
    virtual void backward(const double* p, double* f, int threads = 1) const = 0;
    
    /**
     * @brief 获取图像维度
     */
    virtual Dims3D imageDims() const = 0;
    
    /**
     * @brief 获取投影维度 (探测器数 × 角度数)
     */
    virtual int64_t projectionSize() const = 0;
    
    /**
     * @brief 获取体素数
     */
    virtual int64_t numVoxels() const = 0;
    
    /**
     * @brief 获取敏感度图像 (A^T * 1)
     * 
     * @param sens 输出敏感度 (voxels)
     * @param threads 线程数
     */
    virtual void computeSensitivity(double* sens, int threads = 1) const = 0;
    
    /**
     * @brief 伴随测试 (<Af, p> vs <f, A^T p>)
     * 
     * 可选实现，用于验证投影器正确性。
     * 
     * @param rtol 相对容差
     * @return true 如果通过伴随测试
     */
    virtual bool adjointTest(double rtol = 1e-10) const {
        // 默认不实现
        return true;
    }
};

} // namespace spect
