/**
 * @file SparseOpsGPU.h
 * @brief 【T6.2】GPU稀疏矩阵运算接口
 * 
 * 基于cuSPARSE的ISparseOps实现。
 */

#pragma once

#include "CudaConfig.h"
#include "../interfaces/ISparseOps.h"
#include "../core/CSRView.h"

#if CUDA_ENABLED

#include <cusparse.h>
#include <memory>

namespace spect {
namespace cuda {

/**
 * @brief GPU端CSR矩阵
 * 
 * 管理设备内存中的CSR数据。
 */
class DeviceCSR {
public:
    DeviceCSR() = default;
    ~DeviceCSR();
    
    // 禁止拷贝
    DeviceCSR(const DeviceCSR&) = delete;
    DeviceCSR& operator=(const DeviceCSR&) = delete;
    
    // 允许移动
    DeviceCSR(DeviceCSR&&) noexcept;
    DeviceCSR& operator=(DeviceCSR&&) noexcept;
    
    /**
     * @brief 从主机CSRView上传
     */
    void upload(const CSRView& host);
    
    /**
     * @brief 获取行数
     */
    int64_t rows() const { return rows_; }
    
    /**
     * @brief 获取列数
     */
    int64_t cols() const { return cols_; }
    
    /**
     * @brief 获取非零元数
     */
    int64_t nnz() const { return nnz_; }
    
    // 设备指针访问
    const int64_t* rowPtr() const { return d_rowPtr_; }
    const int32_t* colIdx() const { return d_colIdx_; }
    const float* values() const { return d_values_; }
    
private:
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    int64_t nnz_ = 0;
    
    int64_t* d_rowPtr_ = nullptr;
    int32_t* d_colIdx_ = nullptr;
    float* d_values_ = nullptr;
};

/**
 * @brief GPU稀疏矩阵运算实现
 */
class SparseOpsGPU : public ISparseOps {
public:
    SparseOpsGPU();
    ~SparseOpsGPU() override;

    BackendType backend() const override { return BackendType::CUDA; }
    
    // ISparseOps接口
    void spmv(const CSRView& A, const double* x, double* y, int threads = 1) const override;
    void spmvT(const CSRView& A, const double* x, double* y, int threads = 1) const override;
    void computeNormAT(const CSRView& A, double* norm, int threads = 1) const override {
        std::vector<double> ones(static_cast<size_t>(A.rows), 1.0);
        std::fill(norm, norm + static_cast<size_t>(A.cols), 0.0);
        spmvT(A, ones.data(), norm, threads);
    }
    std::string name() const override { return "SparseOpsGPU"; }
    
    /**
     * @brief 预上传矩阵到GPU
     * 
     * 如果矩阵不变，可以预上传避免重复传输。
     */
    void uploadMatrix(const CSRView& A);
    
    /**
     * @brief 使用预上传的矩阵进行SpMV
     */
    void spmvCached(const double* x, double* y) const;
    
    /**
     * @brief 使用预上传的矩阵进行SpMV^T
     */
    void spmvTCached(const double* x, double* y) const;
    
private:
    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t matA_ = nullptr;
    
    std::unique_ptr<DeviceCSR> cachedMatrix_;
    
    // 临时设备缓冲区
    mutable double* d_x_ = nullptr;
    mutable double* d_y_ = nullptr;
    mutable size_t d_x_size_ = 0;
    mutable size_t d_y_size_ = 0;
    mutable void* d_buffer_ = nullptr;
    mutable size_t d_bufferSize_ = 0;
    
    void ensureBuffer(size_t xSize, size_t ySize) const;
    void freeBuffers();
};

} // namespace cuda
} // namespace spect

#else // !CUDA_ENABLED

namespace spect {
namespace cuda {

// Stub when CUDA not available
class SparseOpsGPU : public ISparseOps {
public:
    SparseOpsGPU() {
        throw std::runtime_error("CUDA not available - SparseOpsGPU cannot be used");
    }

    BackendType backend() const override { return BackendType::CUDA; }
    
    void spmv(const CSRView&, const double*, double*, int) const override {}
    void spmvT(const CSRView&, const double*, double*, int) const override {}
    void computeNormAT(const CSRView&, double*, int) const override {}
    std::string name() const override { return "SparseOpsGPU (unavailable)"; }
};


} // namespace cuda
} // namespace spect

#endif // CUDA_ENABLED
