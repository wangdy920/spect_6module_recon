/**
 * @file SparseOpsCPU.h
 * @brief AVX2+OpenMP优化的SpMV实现
 * 
 * 【T3.2】生产级CPU后端，支持：
 * - AVX2 SIMD向量化
 * - OpenMP多线程并行（使用schedule(static)保证确定性）
 * - 与SparseOpsRef保持结果一致性
 * - 符合Determinism Contract（Plan Section 0）
 */

#pragma once

#include "interfaces/ISparseOps.h"
#include "io/CSRTypes.h"
#include "sparse/SparseOpsRef.h"
#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// AVX2支持检测
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#define SPECT_HAS_AVX2 1
#include <immintrin.h>
#else
#define SPECT_HAS_AVX2 0
#endif

namespace spect {

inline int sparse_actual_threads(int threads) {
    if (threads <= 0)
        threads = 1;
#ifdef _OPENMP
    return std::min(threads, omp_get_max_threads());
#else
    return threads;
#endif
}

inline bool sparse_should_use_threadlocal_accum(size_t ncols, int threads) {
    const int actual_threads = sparse_actual_threads(threads);
    if (ncols == 0 || actual_threads <= 1)
        return false;
    constexpr size_t kMaxThreadLocalBytes = 512ull * 1024ull * 1024ull;
    constexpr size_t kMaxElems = kMaxThreadLocalBytes / sizeof(double);
    return ncols <= (kMaxElems / static_cast<size_t>(actual_threads));
}

/**
 * @brief AVX2+OpenMP优化的SpMV实现
 * 
 * 特点：
 * - OpenMP并行化行循环
 * - AVX2向量化行内累加（可选）
 * - 线程安全的spmvT scatter操作
 */
class SparseOpsCPU : public ISparseOps {
public:
    BackendType backend() const override {
        return BackendType::CPU;
    }
    
    /**
     * @brief SpMV: y = A * x (OpenMP并行)
     * 
     * @note 使用schedule(static)保证确定性
     */
    void spmv(const CSRView& A, 
              const double* x, 
              double* y, 
              int threads) const override {
        const int64_t nrows = A.rows;
        
#ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(static)
#else
        (void)threads;
#endif
        for (int64_t row = 0; row < nrows; ++row) {
            const int64_t r0 = A.row_ptr[row];
            const int64_t r1 = A.row_ptr[row + 1];
            
            double sum = 0.0;
            
#if SPECT_HAS_AVX2
            // AVX2: 一次处理4个double
            int64_t t = r0;
            const int64_t nnz_row = r1 - r0;
            
            if (nnz_row >= 4) {
                __m256d vsum = _mm256_setzero_pd();
                const int64_t simd_end = r0 + (nnz_row / 4) * 4;
                
                for (; t < simd_end; t += 4) {
                    // 加载4个矩阵值
                    __m256d vval = _mm256_loadu_pd(&A.values[t]);
                    
                    // 收集4个x值（散乱访问，无法向量化）
                    __m256d vx = _mm256_set_pd(
                        x[A.col_idx[t + 3]],
                        x[A.col_idx[t + 2]],
                        x[A.col_idx[t + 1]],
                        x[A.col_idx[t + 0]]
                    );
                    
                    vsum = _mm256_fmadd_pd(vval, vx, vsum);
                }
                
                // 水平求和
                double temp[4];
                _mm256_storeu_pd(temp, vsum);
                sum = temp[0] + temp[1] + temp[2] + temp[3];
            }
            
            // 处理剩余元素
            for (; t < r1; ++t) {
                sum += A.values[t] * x[A.col_idx[t]];
            }
#else
            // 标量回退
            for (int64_t t = r0; t < r1; ++t) {
                sum += A.values[t] * x[A.col_idx[t]];
            }
#endif
            y[row] = sum;
        }
    }
    
    /**
     * @brief SpMV^T: y = A^T * x (线程本地累加+确定性归约)
     * 
     * @note 调用前y必须初始化为0
     * @note 使用schedule(static)和固定tid顺序合并保证确定性
     */
    void spmvT(const CSRView& A, 
               const double* x, 
               double* y, 
               int threads) const override {
        const int64_t nrows = A.rows;
        const int64_t ncols = A.cols;
        
#ifdef _OPENMP
        const int actual_threads = sparse_actual_threads(threads);
        if (actual_threads <= 1) {
            for (int64_t row = 0; row < nrows; ++row) {
                const double xval = x[row];
                const int64_t r0 = A.row_ptr[row];
                const int64_t r1 = A.row_ptr[row + 1];
                for (int64_t t = r0; t < r1; ++t) {
                    y[A.col_idx[t]] += A.values[t] * xval;
                }
            }
        } else if (!sparse_should_use_threadlocal_accum(static_cast<size_t>(ncols),
                                                         actual_threads)) {
            #pragma omp parallel for num_threads(actual_threads) schedule(static)
            for (int64_t row = 0; row < nrows; ++row) {
                const double xval = x[row];
                const int64_t r0 = A.row_ptr[row];
                const int64_t r1 = A.row_ptr[row + 1];
                for (int64_t t = r0; t < r1; ++t) {
                    const int64_t col = A.col_idx[t];
                    #pragma omp atomic
                    y[col] += A.values[t] * xval;
                }
            }
        } else {
            std::vector<std::vector<double>> local_y(
                actual_threads, std::vector<double>(ncols, 0.0));
            #pragma omp parallel num_threads(actual_threads)
            {
                const int tid = omp_get_thread_num();
                auto& my_y = local_y[tid];
                #pragma omp for schedule(static)
                for (int64_t row = 0; row < nrows; ++row) {
                    const double xval = x[row];
                    const int64_t r0 = A.row_ptr[row];
                    const int64_t r1 = A.row_ptr[row + 1];
                    for (int64_t t = r0; t < r1; ++t) {
                        my_y[A.col_idx[t]] += A.values[t] * xval;
                    }
                }
            }
            for (int t = 0; t < actual_threads; ++t) {
                for (int64_t c = 0; c < ncols; ++c) {
                    y[c] += local_y[t][c];
                }
            }
        }
#else
        (void)threads;
        // 单线程回退
        for (int64_t row = 0; row < nrows; ++row) {
            const double xval = x[row];
            const int64_t r0 = A.row_ptr[row];
            const int64_t r1 = A.row_ptr[row + 1];
            for (int64_t t = r0; t < r1; ++t) {
                y[A.col_idx[t]] += A.values[t] * xval;
            }
        }
#endif
    }
    
    /**
     * @brief 计算归一化向量: norm = A^T * 1
     * 
     * @note 使用schedule(static)和线程本地累加保证确定性
     */
    void computeNormAT(const CSRView& A, 
                       double* norm, 
                       int threads) const override {
        const int64_t ncols = A.cols;
        const int64_t nrows = A.rows;
        
        // 初始化为0
        std::memset(norm, 0, static_cast<size_t>(ncols) * sizeof(double));
        
#ifdef _OPENMP
        const int actual_threads = sparse_actual_threads(threads);
        if (actual_threads <= 1) {
            for (int64_t row = 0; row < nrows; ++row) {
                const int64_t r0 = A.row_ptr[row];
                const int64_t r1 = A.row_ptr[row + 1];
                for (int64_t t = r0; t < r1; ++t) {
                    norm[A.col_idx[t]] += A.values[t];
                }
            }
        } else if (!sparse_should_use_threadlocal_accum(static_cast<size_t>(ncols),
                                                         actual_threads)) {
            #pragma omp parallel for num_threads(actual_threads) schedule(static)
            for (int64_t row = 0; row < nrows; ++row) {
                const int64_t r0 = A.row_ptr[row];
                const int64_t r1 = A.row_ptr[row + 1];
                for (int64_t t = r0; t < r1; ++t) {
                    const int64_t col = A.col_idx[t];
                    #pragma omp atomic
                    norm[col] += A.values[t];
                }
            }
        } else {
            std::vector<std::vector<double>> local_norm(
                actual_threads, std::vector<double>(ncols, 0.0));
            #pragma omp parallel num_threads(actual_threads)
            {
                const int tid = omp_get_thread_num();
                auto& my_norm = local_norm[tid];
                #pragma omp for schedule(static)
                for (int64_t row = 0; row < nrows; ++row) {
                    const int64_t r0 = A.row_ptr[row];
                    const int64_t r1 = A.row_ptr[row + 1];
                    for (int64_t t = r0; t < r1; ++t) {
                        my_norm[A.col_idx[t]] += A.values[t];
                    }
                }
            }
            for (int t = 0; t < actual_threads; ++t) {
                for (int64_t c = 0; c < ncols; ++c) {
                    norm[c] += local_norm[t][c];
                }
            }
        }
#else
        (void)threads;
        for (int64_t row = 0; row < nrows; ++row) {
            const int64_t r0 = A.row_ptr[row];
            const int64_t r1 = A.row_ptr[row + 1];
            for (int64_t t = r0; t < r1; ++t) {
                norm[A.col_idx[t]] += A.values[t];
            }
        }
#endif
    }
    
    /**
     * @brief 带缩放的SpMV: y = alpha * A * x + beta * y
     * 
     * @note 使用schedule(static)保证确定性
     */
    void spmvScaled(const CSRView& A, 
                    const double* x, 
                    double* y, 
                    double alpha, 
                    double beta,
                    int threads) const override {
        const int64_t nrows = A.rows;
        
#ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(static)
#else
        (void)threads;
#endif
        for (int64_t row = 0; row < nrows; ++row) {
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
              int threads) const override {
        const int nrows = A.rows;

#ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(static)
#else
        (void)threads;
#endif
        for (int row = 0; row < nrows; ++row) {
            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            double sum = 0.0;

#if SPECT_HAS_AVX2
            size_t t = r0;
            const size_t nnz_row = r1 - r0;
            if (nnz_row >= 4) {
                __m256d vsum = _mm256_setzero_pd();
                const size_t simd_end = r0 + (nnz_row / 4) * 4;
                for (; t < simd_end; t += 4) {
                    __m128 v_val_f = _mm_loadu_ps(&A.val[t]);
                    __m256d v_val_d = _mm256_cvtps_pd(v_val_f);
                    __m128i v_idx = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&A.col_idx[t]));
                    __m256d vx = _mm256_i32gather_pd(x, v_idx, 8);
                    vsum = _mm256_fmadd_pd(v_val_d, vx, vsum);
                }
                double temp[4];
                _mm256_storeu_pd(temp, vsum);
                sum = temp[0] + temp[1] + temp[2] + temp[3];
            }
            for (; t < r1; ++t) {
                sum += static_cast<double>(A.val[t]) * x[A.col_idx[t]];
            }
#else
            for (size_t t = r0; t < r1; ++t) {
                sum += static_cast<double>(A.val[t]) * x[A.col_idx[t]];
            }
#endif
            y[row] = sum;
        }
    }

    /**
     * @brief SpMV^T: y = A^T * x (io::CSRView)
     */
    void spmvT(const io::CSRView& A,
               const double* x,
               double* y,
               int threads) const override {
        const int nrows = A.rows;
        const int ncols = A.cols;

#ifdef _OPENMP
        const int actual_threads = sparse_actual_threads(threads);
        if (actual_threads <= 1) {
            for (int row = 0; row < nrows; ++row) {
                const double xval = x[row];
                const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                for (size_t t = r0; t < r1; ++t) {
                    y[A.col_idx[t]] += static_cast<double>(A.val[t]) * xval;
                }
            }
        } else if (!sparse_should_use_threadlocal_accum(
                       static_cast<size_t>(ncols), actual_threads)) {
            #pragma omp parallel for num_threads(actual_threads) schedule(static)
            for (int row = 0; row < nrows; ++row) {
                const double xval = x[row];
                const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                for (size_t t = r0; t < r1; ++t) {
                    const size_t col = static_cast<size_t>(A.col_idx[t]);
                    #pragma omp atomic
                    y[col] += static_cast<double>(A.val[t]) * xval;
                }
            }
        } else {
            std::vector<std::vector<double>> local_y(
                actual_threads, std::vector<double>(ncols, 0.0));
            #pragma omp parallel num_threads(actual_threads)
            {
                const int tid = omp_get_thread_num();
                auto& my_y = local_y[tid];
                #pragma omp for schedule(static)
                for (int row = 0; row < nrows; ++row) {
                    const double xval = x[row];
                    const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                    const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                    for (size_t t = r0; t < r1; ++t) {
                        my_y[A.col_idx[t]] +=
                            static_cast<double>(A.val[t]) * xval;
                    }
                }
            }
            for (int t = 0; t < actual_threads; ++t) {
                for (int c = 0; c < ncols; ++c) {
                    y[c] += local_y[t][static_cast<size_t>(c)];
                }
            }
        }
#else
        (void)threads;
        for (int row = 0; row < nrows; ++row) {
            const double xval = x[row];
            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
                y[A.col_idx[t]] += static_cast<double>(A.val[t]) * xval;
            }
        }
#endif
    }

    /**
     * @brief 计算归一化向量: norm = A^T * 1 (io::CSRView)
     */
    void computeNormAT(const io::CSRView& A,
                       double* norm,
                       int threads) const override {
        const int ncols = A.cols;
        const int nrows = A.rows;
        std::memset(norm, 0, static_cast<size_t>(ncols) * sizeof(double));

#ifdef _OPENMP
        const int actual_threads = sparse_actual_threads(threads);
        if (actual_threads <= 1) {
            for (int row = 0; row < nrows; ++row) {
                const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                for (size_t t = r0; t < r1; ++t) {
                    norm[A.col_idx[t]] += static_cast<double>(A.val[t]);
                }
            }
        } else if (!sparse_should_use_threadlocal_accum(
                       static_cast<size_t>(ncols), actual_threads)) {
            #pragma omp parallel for num_threads(actual_threads) schedule(static)
            for (int row = 0; row < nrows; ++row) {
                const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                for (size_t t = r0; t < r1; ++t) {
                    const size_t col = static_cast<size_t>(A.col_idx[t]);
                    #pragma omp atomic
                    norm[col] += static_cast<double>(A.val[t]);
                }
            }
        } else {
            std::vector<std::vector<double>> local_norm(
                actual_threads, std::vector<double>(ncols, 0.0));
            #pragma omp parallel num_threads(actual_threads)
            {
                const int tid = omp_get_thread_num();
                auto& my_norm = local_norm[tid];
                #pragma omp for schedule(static)
                for (int row = 0; row < nrows; ++row) {
                    const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                    const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                    for (size_t t = r0; t < r1; ++t) {
                        my_norm[A.col_idx[t]] +=
                            static_cast<double>(A.val[t]);
                    }
                }
            }
            for (int t = 0; t < actual_threads; ++t) {
                for (int c = 0; c < ncols; ++c) {
                    norm[c] += local_norm[t][static_cast<size_t>(c)];
                }
            }
        }
#else
        (void)threads;
        for (int row = 0; row < nrows; ++row) {
            const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
            const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
            for (size_t t = r0; t < r1; ++t) {
                norm[A.col_idx[t]] += static_cast<double>(A.val[t]);
            }
        }
#endif
    }

    /**
     * @brief 带缩放的SpMV: y = alpha * A * x + beta * y (io::CSRView)
     */
    void spmvScaled(const io::CSRView& A,
                    const double* x,
                    double* y,
                    double alpha,
                    double beta,
                    int threads) const override {
        const int nrows = A.rows;

#ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(static)
#else
        (void)threads;
#endif
        for (int row = 0; row < nrows; ++row) {
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
 * @brief 获取CPU优化SpMV实例（单例）
 */
inline const SparseOpsCPU& getSparseOpsCPUImpl() {
    static SparseOpsCPU instance;
    return instance;
}

inline const ISparseOps& getSparseOpsCPU() {
    return getSparseOpsCPUImpl();
}

/**
 * @brief 获取默认SpMV实现
 * 
 * 根据编译选项返回最优实现。
 */
inline const ISparseOps& getDefaultSparseOps() {
#ifdef _OPENMP
    return getSparseOpsCPU();
#else
    return getSparseOpsRef();
#endif
}

} // namespace spect
