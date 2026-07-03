/**
 * @file SparseOpsRef.h
 * @brief 标量参考SpMV实现
 * 
 * 【T3.1】无SIMD、无OpenMP的纯标量实现，作为：
 * - 其他优化实现的正确性基准
 * - 小数据测试的简单后端
 * - bitwise一致性测试的参考
 */

#pragma once

#include "interfaces/ISparseOps.h"
#include "io/CSRTypes.h"
#include <algorithm>
#include <cstring>

namespace spect {

/**
 * @brief 标量参考SpMV实现
 * 
 * 特点：
 * - 纯C++标量循环，无SIMD指令
 * - 单线程，忽略threads参数
 * - 结果与任何正确实现应bitwise一致
 */
class SparseOpsRef : public ISparseOps {
public:
    BackendType backend() const override {
        return BackendType::Reference;
    }
    
    /**
     * @brief SpMV: y = A * x
     */
    void spmv(const CSRView& A, 
              const double* x, 
              double* y, 
              int /*threads*/) const override {
        for (int64_t row = 0; row < A.rows; ++row) {
            double sum = 0.0;
            const int64_t r0 = A.row_ptr[row];
            const int64_t r1 = A.row_ptr[row + 1];
            for (int64_t t = r0; t < r1; ++t) {
                sum += A.values[t] * x[A.col_idx[t]];
            }
            y[row] = sum;
        }
    }
    
    /**
     * @brief SpMV^T: y = A^T * x (scatter模式)
     * 
     * @note 调用前y必须初始化为0
     */
    void spmvT(const CSRView& A, 
               const double* x, 
               double* y, 
               int /*threads*/) const override {
        for (int64_t row = 0; row < A.rows; ++row) {
            const double xval = x[row];
            const int64_t r0 = A.row_ptr[row];
            const int64_t r1 = A.row_ptr[row + 1];
            for (int64_t t = r0; t < r1; ++t) {
                y[A.col_idx[t]] += A.values[t] * xval;
            }
        }
    }
    
    /**
     * @brief 计算归一化向量: norm = A^T * 1
     */
    void computeNormAT(const CSRView& A, 
                       double* norm, 
                       int /*threads*/) const override {
        // 初始化为0
        std::memset(norm, 0, static_cast<size_t>(A.cols) * sizeof(double));
        
        // 遍历所有非零元素，累加到对应列
        for (int64_t row = 0; row < A.rows; ++row) {
            const int64_t r0 = A.row_ptr[row];
            const int64_t r1 = A.row_ptr[row + 1];
            for (int64_t t = r0; t < r1; ++t) {
                norm[A.col_idx[t]] += A.values[t];
            }
        }
    }
    
    /**
     * @brief 带缩放的SpMV: y = alpha * A * x + beta * y
     */
    void spmvScaled(const CSRView& A, 
                    const double* x, 
                    double* y, 
                    double alpha, 
                    double beta,
                    int /*threads*/) const override {
        for (int64_t row = 0; row < A.rows; ++row) {
            double sum = 0.0;
            const int64_t r0 = A.row_ptr[row];
            const int64_t r1 = A.row_ptr[row + 1];
            for (int64_t t = r0; t < r1; ++t) {
                sum += A.values[t] * x[A.col_idx[t]];
            }
            y[row] = alpha * sum + beta * y[row];
        }
    }

    /**
     * @brief SpMV: y = A * x (io::CSRView, float values)
     */
    void spmv(const io::CSRView& A,
              const double* x,
              double* y,
              int /*threads*/) const override {
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
     * @brief SpMV^T: y = A^T * x (io::CSRView)
     */
    void spmvT(const io::CSRView& A,
               const double* x,
               double* y,
               int /*threads*/) const override {
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
     * @brief 计算归一化向量: norm = A^T * 1 (io::CSRView)
     */
    void computeNormAT(const io::CSRView& A,
                       double* norm,
                       int /*threads*/) const override {
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
     * @brief 带缩放的SpMV: y = alpha * A * x + beta * y (io::CSRView)
     */
    void spmvScaled(const io::CSRView& A,
                    const double* x,
                    double* y,
                    double alpha,
                    double beta,
                    int /*threads*/) const override {
        for (int row = 0; row < A.rows; ++row) {
            double sum = 0.0;
            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
                sum += static_cast<double>(A.val[t]) * x[A.col_idx[t]];
            }
            y[row] = alpha * sum + beta * y[row];
        }
    }
};

/**
 * @brief 获取标量参考SpMV实例（单例）
 */
inline const SparseOpsRef& getSparseOpsRefImpl() {
    static SparseOpsRef instance;
    return instance;
}

inline const ISparseOps& getSparseOpsRef() {
    return getSparseOpsRefImpl();
}

} // namespace spect
