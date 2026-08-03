/**
 * @file ISparseOps.h
 * @brief 稀疏矩阵运算统一接口
 * 
 * 此接口统一CUDA后端的SpMV实现，算法层只调用接口。
 * 
 * 设计约束:
 * - 虚调用粒度必须是"整次SpMV"，严禁元素级虚调用
 * - 不得在算法层分支AVX2/CUDA，由接口实现分发
 */

#pragma once

#include "../core/AttenuationConstants.h"
#include "../core/Types.h"
#include "../io/CSRTypes.h"
#include "detail/SparseOpsDefaultsDecl.h"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

namespace spect {

struct SparseSubsetContext {
    int bedCount = 1;
    const std::vector<long long>* bedOffsetVoxels = nullptr;
    const std::vector<std::vector<uint16_t>>* bedAttenFactors = nullptr;
    int numAngles = 0;
};

/**
 * @brief 稀疏矩阵运算接口
 * 
 * 提供SpMV及相关稀疏运算的统一入口（io::CSRView，CSRB v1 float 布局）。
 */
class ISparseOps {
public:
    virtual ~ISparseOps() = default;
    
    /**
     * @brief 获取后端类型
     */
    virtual BackendType backend() const = 0;
    
    /**
     * @brief 稀疏矩阵-向量乘法 (io::CSRView, y = A * x)
     *
     * 默认提供标量实现，后端可覆盖以获得更高性能。
     */
    virtual void spmv(const io::CSRView& A,
                      const double* x,
                      double* y,
                      int /*threads*/ = 1) const {
        for (int row = 0; row < A.rows; ++row) {
            double sum = 0.0;
            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
                sum += static_cast<double>(A.val[t]) * x[A.col_idx[t]];
            }
            y[row] = sum;
        }
    }

    /**
     * @brief 稀疏矩阵转置-向量乘法 (io::CSRView, y = A^T * x)
     */
    virtual void spmvT(const io::CSRView& A,
                       const double* x,
                       double* y,
                       int /*threads*/ = 1) const {
        for (int row = 0; row < A.rows; ++row) {
            const double xval = x[row];
            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
                y[A.col_idx[t]] += static_cast<double>(A.val[t]) * xval;
            }
        }
    }

    /**
     * @brief 计算归一化向量 (io::CSRView, A^T * 1)
     */
    virtual void computeNormAT(const io::CSRView& A,
                               double* norm,
                               int /*threads*/ = 1) const {
        std::memset(norm, 0, static_cast<size_t>(A.cols) * sizeof(double));
        for (int row = 0; row < A.rows; ++row) {
            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
                norm[A.col_idx[t]] += static_cast<double>(A.val[t]);
            }
        }
    }

    /**
     * @brief 带缩放的SpMV (io::CSRView, y = alpha * A * x + beta * y)
     */
    virtual void spmvScaled(const io::CSRView& A,
                            const double* x,
                            double* y,
                            double alpha,
                            double beta,
                            int threads = 1) const {
        if (beta == 0.0) {
            spmv(A, x, y, threads);
            if (alpha != 1.0) {
                for (int i = 0; i < A.rows; ++i) {
                    y[i] *= alpha;
                }
            }
        } else {
            std::vector<double> tmp(static_cast<size_t>(A.rows), 0.0);
            spmv(A, x, tmp.data(), threads);
            for (int i = 0; i < A.rows; ++i) {
                y[i] = alpha * tmp[static_cast<size_t>(i)] + beta * y[i];
            }
        }
    }

    /**
     * @brief 子集前向投影: y = A(subset) * x
     *
     * @param subset_rows 全局行索引列表（包含bed偏移后的行）
     * @param ctx 多床位/CTAC上下文
     * @param y 输出向量（长度 = subset_rows.size()）
     */
    virtual void spmvSubset(const io::CSRView& A,
                            const double* x,
                            const std::vector<int>& subset_rows,
                            const SparseSubsetContext& ctx,
                            double* y,
                            int threads = 1) const {
        sparse_detail::spmvSubsetImpl(A, x, subset_rows, ctx, y, threads);
    }

    /**
     * @brief 子集反投影: y = A(subset)^T * x
     *
     * @param x 输入向量（长度 = subset_rows.size()）
     * @param y 输出向量（长度 = 全局体素数，调用前应初始化为0）
     */
    virtual void spmvTSubset(const io::CSRView& A,
                             const double* x,
                             const std::vector<int>& subset_rows,
                             const SparseSubsetContext& ctx,
                             double* y,
                             int threads = 1) const {
        sparse_detail::spmvTSubsetImpl(A, x, subset_rows, ctx, y, threads);
    }

    /**
     * @brief CTAC 加权的归一化向量: norm = A^T * att
     */
    virtual void computeNormAT_Atten(const io::CSRView& A,
                                     const std::vector<uint16_t>& atten,
                                     int numAngles,
                                     double* norm,
                                     int threads = 1) const {
        sparse_detail::computeNormAT_AttenImpl(A, atten, numAngles, norm, threads);
    }

    /**
     * @brief 子集敏感度: S_sub = A(subset)^T * 1
     */
    virtual void computeSubsetSensitivity(const io::CSRView& A,
                                          const std::vector<int>& subset_rows,
                                          int N,
                                          const SparseSubsetContext& ctx,
                                          double* sensitivity,
                                          int threads = 1) const {
        sparse_detail::computeSubsetSensitivityImpl(
            A, subset_rows, N, ctx, sensitivity, threads);
    }

protected:
    static int sparseops_effective_threads(int threads);
    static bool sparseops_should_use_threadlocal_accum(size_t nvox, int threads);
};

} // namespace spect
