/**
 * @file ISparseOps.h
 * @brief 稀疏矩阵运算统一接口
 * 
 * 此接口统一CPU/CUDA后端的SpMV实现，算法层只调用接口。
 * 
 * 设计约束:
 * - 虚调用粒度必须是"整次SpMV"，严禁元素级虚调用
 * - 不得在算法层分支AVX2/CUDA，由接口实现分发
 */

#pragma once

#include "../core/CSRView.h"
#include "../core/Types.h"
#include "../io/CSRTypes.h"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#define SPECT_ISPARSEOPS_HAS_AVX2 1
#include <immintrin.h>
#else
#define SPECT_ISPARSEOPS_HAS_AVX2 0
#endif

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
 * 提供SpMV及相关稀疏运算的统一入口。
 * CPU和CUDA后端分别实现此接口。
 */
class ISparseOps {
public:
    virtual ~ISparseOps() = default;
    
    /**
     * @brief 获取后端类型
     */
    virtual BackendType backend() const = 0;
    
    /**
     * @brief 稀疏矩阵-向量乘法 (y = A * x)
     * 
     * @param A CSR格式的稀疏矩阵视图
     * @param x 输入向量 (长度 = A.cols)
     * @param y 输出向量 (长度 = A.rows)
     * @param threads 线程数 (仅CPU后端使用)
     */
    virtual void spmv(const CSRView& A, 
                      const double* x, 
                      double* y, 
                      int threads = 1) const = 0;
    
    /**
     * @brief 稀疏矩阵转置-向量乘法 (y = A^T * x)
     * 
     * @param A CSR格式的稀疏矩阵视图
     * @param x 输入向量 (长度 = A.rows)
     * @param y 输出向量 (长度 = A.cols)，调用前应初始化为0
     * @param threads 线程数 (仅CPU后端使用)
     * 
     * @note 默认实现使用scatter模式，可能有性能问题。
     *       建议使用显式转置矩阵的spmv()替代。
     */
    virtual void spmvT(const CSRView& A, 
                       const double* x, 
                       double* y, 
                       int threads = 1) const = 0;
    
    /**
     * @brief 计算归一化向量 (A^T * 1)
     * 
     * 即每列的和，用于MLEM/OSEM的敏感度计算。
     * 
     * @param A CSR格式的稀疏矩阵视图
     * @param norm 输出归一化向量 (长度 = A.cols)
     * @param threads 线程数
     */
    virtual void computeNormAT(const CSRView& A, 
                               double* norm, 
                               int threads = 1) const = 0;
    
    /**
     * @brief 带缩放的SpMV (y = alpha * A * x + beta * y)
     * 
     * 可选实现，默认调用spmv后手动缩放。
     */
    virtual void spmvScaled(const CSRView& A, 
                            const double* x, 
                            double* y, 
                            double alpha, 
                            double beta,
                            int threads = 1) const {
        // 默认实现
        if (beta == 0.0) {
            spmv(A, x, y, threads);
            if (alpha != 1.0) {
                for (int64_t i = 0; i < A.rows; ++i) {
                    y[i] *= alpha;
                }
            }
        } else {
            // 需要临时缓冲区：先计算 tmp = A*x，再做 y = alpha*tmp + beta*y
            std::vector<double> tmp(static_cast<size_t>(A.rows), 0.0);
            spmv(A, x, tmp.data(), threads);
            for (int64_t i = 0; i < A.rows; ++i) {
                y[i] = alpha * tmp[static_cast<size_t>(i)] + beta * y[i];
            }
        }
    }

    /**
     * @brief 稀疏矩阵-向量乘法 (io::CSRView, float values)
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
     * @brief 稀疏矩阵转置-向量乘法 (io::CSRView)
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
     * @brief 计算归一化向量 (io::CSRView)
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
     * @brief 带缩放的SpMV (io::CSRView)
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
    virtual void spmvSubset(const CSRView& A,
                            const double* x,
                            const std::vector<int>& subset_rows,
                            const SparseSubsetContext& ctx,
                            double* y,
                            int threads = 1) const {
        spmvSubsetImpl(A, x, subset_rows, ctx, y, threads);
    }

    virtual void spmvSubset(const io::CSRView& A,
                            const double* x,
                            const std::vector<int>& subset_rows,
                            const SparseSubsetContext& ctx,
                            double* y,
                            int threads = 1) const {
        spmvSubsetImpl(A, x, subset_rows, ctx, y, threads);
    }

    /**
     * @brief 子集反投影: y = A(subset)^T * x
     *
     * @param x 输入向量（长度 = subset_rows.size()）
     * @param y 输出向量（长度 = 全局体素数，调用前应初始化为0）
     */
    virtual void spmvTSubset(const CSRView& A,
                             const double* x,
                             const std::vector<int>& subset_rows,
                             const SparseSubsetContext& ctx,
                             double* y,
                             int threads = 1) const {
        spmvTSubsetImpl(A, x, subset_rows, ctx, y, threads);
    }

    virtual void spmvTSubset(const io::CSRView& A,
                             const double* x,
                             const std::vector<int>& subset_rows,
                             const SparseSubsetContext& ctx,
                             double* y,
                             int threads = 1) const {
        spmvTSubsetImpl(A, x, subset_rows, ctx, y, threads);
    }

    /**
     * @brief CTAC 加权的归一化向量: norm = A^T * att
     */
    virtual void computeNormAT_Atten(const CSRView& A,
                                     const std::vector<uint16_t>& atten,
                                     int numAngles,
                                     double* norm,
                                     int threads = 1) const {
        computeNormAT_AttenImpl(A, atten, numAngles, norm, threads);
    }

    virtual void computeNormAT_Atten(const io::CSRView& A,
                                     const std::vector<uint16_t>& atten,
                                     int numAngles,
                                     double* norm,
                                     int threads = 1) const {
        computeNormAT_AttenImpl(A, atten, numAngles, norm, threads);
    }

    /**
     * @brief 子集敏感度: S_sub = A(subset)^T * 1
     */
    virtual void computeSubsetSensitivity(const CSRView& A,
                                          const std::vector<int>& subset_rows,
                                          int N,
                                          const SparseSubsetContext& ctx,
                                          double* sensitivity,
                                          int threads = 1) const {
        computeSubsetSensitivityImpl(A, subset_rows, N, ctx, sensitivity,
                                     threads);
    }

    virtual void computeSubsetSensitivity(const io::CSRView& A,
                                          const std::vector<int>& subset_rows,
                                          int N,
                                          const SparseSubsetContext& ctx,
                                          double* sensitivity,
                                          int threads = 1) const {
        computeSubsetSensitivityImpl(A, subset_rows, N, ctx, sensitivity,
                                     threads);
    }

protected:
    static int sparseops_effective_threads(int threads) {
        if (threads <= 0)
            threads = 1;
#ifdef SPECT_DETERMINISTIC
        return 1;
#else
        return threads;
#endif
    }

    static bool sparseops_should_use_threadlocal_accum(size_t nvox, int threads) {
        threads = sparseops_effective_threads(threads);
        if (threads <= 1 || nvox == 0)
            return false;
        constexpr size_t kMaxThreadLocalBytes = 512ull * 1024ull * 1024ull;
        constexpr size_t kMaxElems = kMaxThreadLocalBytes / sizeof(double);
        const size_t t = static_cast<size_t>(threads);
        return nvox <= (kMaxElems / t);
    }

private:
    template <typename MatrixType>
    static void spmvSubsetImpl(const MatrixType& A,
                               const double* x,
                               const std::vector<int>& subset_rows,
                               const SparseSubsetContext& ctx,
                               double* y,
                               int threads) {
        if (subset_rows.empty())
            return;

        const int threads_eff = sparseops_effective_threads(threads);
        const int baseRows = static_cast<int>(A.rows);
        const int baseCols = static_cast<int>(A.cols);
        const size_t NvoxLocal = static_cast<size_t>(baseCols);
        const size_t raysPerAngle =
            (ctx.numAngles > 0) ? (static_cast<size_t>(baseRows) / ctx.numAngles)
                                : 0;
        const double invScale = 1.0 / 65535.0;
        const int bedCount = std::max(1, ctx.bedCount);
        const bool hasOffsets =
            ctx.bedOffsetVoxels && !ctx.bedOffsetVoxels->empty();
        const bool hasAtten =
            ctx.bedAttenFactors && !ctx.bedAttenFactors->empty();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_eff)
#endif
        for (ptrdiff_t idx = 0;
             idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
            const int global_row = subset_rows[static_cast<size_t>(idx)];
            const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
            const int row = (bedCount > 1) ? (global_row % baseRows) : global_row;

            const uint16_t* attPtr = nullptr;
            if (hasAtten) {
                int angle = 0;
                if (raysPerAngle > 0)
                    angle = static_cast<int>(static_cast<size_t>(row) / raysPerAngle);
                if (angle >= ctx.numAngles)
                    angle = ctx.numAngles - 1;
                attPtr = (*ctx.bedAttenFactors)[bed].data() +
                         static_cast<size_t>(angle) * NvoxLocal;
            }

            const size_t col_offset = hasOffsets
                ? static_cast<size_t>((*ctx.bedOffsetVoxels)[bed])
                : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

            double sum = 0.0;

            if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
                const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);

#if SPECT_ISPARSEOPS_HAS_AVX2
                size_t t = r0;
                const size_t nnz_row = r1 - r0;
                if (nnz_row >= 4) {
                    __m256d v_sum = _mm256_setzero_pd();
                    const size_t simd_end = r0 + (nnz_row / 4) * 4;
                    __m128i v_col_offset =
                        _mm_set1_epi32(static_cast<int>(col_offset));

                    for (; t < simd_end; t += 4) {
                        __m128 v_val_f = _mm_loadu_ps(&A.val[t]);
                        if (attPtr) {
                            float a0 = static_cast<float>(attPtr[A.col_idx[t]]) *
                                       static_cast<float>(invScale);
                            float a1 =
                                static_cast<float>(attPtr[A.col_idx[t + 1]]) *
                                static_cast<float>(invScale);
                            float a2 =
                                static_cast<float>(attPtr[A.col_idx[t + 2]]) *
                                static_cast<float>(invScale);
                            float a3 =
                                static_cast<float>(attPtr[A.col_idx[t + 3]]) *
                                static_cast<float>(invScale);
                            __m128 v_att = _mm_set_ps(a3, a2, a1, a0);
                            v_val_f = _mm_mul_ps(v_val_f, v_att);
                        }

                        __m256d v_val_d = _mm256_cvtps_pd(v_val_f);
                        __m128i v_idx = _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(&A.col_idx[t]));
                        if (bedCount > 1)
                            v_idx = _mm_add_epi32(v_idx, v_col_offset);
                        __m256d v_f = _mm256_i32gather_pd(x, v_idx, 8);
                        v_sum = _mm256_fmadd_pd(v_val_d, v_f, v_sum);
                    }

                    double buf[4];
                    _mm256_storeu_pd(buf, v_sum);
                    sum = buf[0] + buf[1] + buf[2] + buf[3];
                }

                for (; t < r1; ++t) {
                    double val = static_cast<double>(A.val[t]);
                    if (attPtr)
                        val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
                    sum += val * x[col_offset + static_cast<size_t>(A.col_idx[t])];
                }
#else
                for (size_t t = r0; t < r1; ++t) {
                    double val = static_cast<double>(A.val[t]);
                    if (attPtr)
                        val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
                    sum += val * x[col_offset + static_cast<size_t>(A.col_idx[t])];
                }
#endif
            } else {
                const int64_t r0 = A.row_ptr[row];
                const int64_t r1 = A.row_ptr[row + 1];
                for (int64_t t = r0; t < r1; ++t) {
                    double val = A.values[t];
                    if (attPtr)
                        val *= static_cast<double>(attPtr[A.col_idx[t]]) * invScale;
                    sum += val * x[col_offset + static_cast<size_t>(A.col_idx[t])];
                }
            }

            y[static_cast<size_t>(idx)] = sum;
        }
    }

    template <typename MatrixType>
    static void spmvTSubsetImpl(const MatrixType& A,
                                const double* x,
                                const std::vector<int>& subset_rows,
                                const SparseSubsetContext& ctx,
                                double* y,
                                int threads) {
        if (subset_rows.empty())
            return;

        const int threads_eff = sparseops_effective_threads(threads);
        const int baseRows = static_cast<int>(A.rows);
        const int baseCols = static_cast<int>(A.cols);
        const size_t NvoxLocal = static_cast<size_t>(baseCols);
        const size_t raysPerAngle =
            (ctx.numAngles > 0) ? (static_cast<size_t>(baseRows) / ctx.numAngles)
                                : 0;
        const double invScale = 1.0 / 65535.0;
        const int bedCount = std::max(1, ctx.bedCount);
        const bool hasOffsets =
            ctx.bedOffsetVoxels && !ctx.bedOffsetVoxels->empty();
        const bool hasAtten =
            ctx.bedAttenFactors && !ctx.bedAttenFactors->empty();

        if (!sparseops_should_use_threadlocal_accum(NvoxLocal, threads_eff)) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(threads_eff)
#endif
            for (ptrdiff_t idx = 0;
                 idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
                const int global_row = subset_rows[static_cast<size_t>(idx)];
                const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
                const int row =
                    (bedCount > 1) ? (global_row % baseRows) : global_row;

                const uint16_t* attPtr = nullptr;
                if (hasAtten) {
                    int angle = 0;
                    if (raysPerAngle > 0)
                        angle = static_cast<int>(
                            static_cast<size_t>(row) / raysPerAngle);
                    if (angle >= ctx.numAngles)
                        angle = ctx.numAngles - 1;
                    attPtr = (*ctx.bedAttenFactors)[bed].data() +
                             static_cast<size_t>(angle) * NvoxLocal;
                }

                const size_t col_offset = hasOffsets
                    ? static_cast<size_t>((*ctx.bedOffsetVoxels)[bed])
                    : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

                const double ratio = x[static_cast<size_t>(idx)];

                if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
                    const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                    const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                    for (size_t t = r0; t < r1; ++t) {
                        double val = static_cast<double>(A.val[t]);
                        if (attPtr)
                            val *= static_cast<double>(attPtr[A.col_idx[t]]) *
                                   invScale;
                        const double inc = ratio * val;
#pragma omp atomic
                        y[col_offset + static_cast<size_t>(A.col_idx[t])] += inc;
                    }
                } else {
                    for (int64_t t = A.row_ptr[row]; t < A.row_ptr[row + 1];
                         ++t) {
                        double val = A.values[t];
                        if (attPtr)
                            val *= static_cast<double>(attPtr[A.col_idx[t]]) *
                                   invScale;
                        const double inc = ratio * val;
#pragma omp atomic
                        y[col_offset + static_cast<size_t>(A.col_idx[t])] += inc;
                    }
                }
            }
            return;
        }

        std::vector<std::vector<int>> bed_indices(bedCount);
        for (size_t i = 0; i < subset_rows.size(); ++i) {
            const int global_row = subset_rows[i];
            const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
            bed_indices[bed].push_back(static_cast<int>(i));
        }

        const size_t threadCount = static_cast<size_t>(threads_eff);
        std::vector<double> tls(threadCount * NvoxLocal, 0.0);

        for (int bed = 0; bed < bedCount; ++bed) {
            if (bed_indices[bed].empty())
                continue;

            const size_t col_offset = hasOffsets
                ? static_cast<size_t>((*ctx.bedOffsetVoxels)[bed])
                : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

#ifdef _OPENMP
#pragma omp parallel num_threads(threads_eff)
#endif
            {
#ifdef _OPENMP
                const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
                const size_t tid = 0;
#endif
                double* local = tls.data() + tid * NvoxLocal;
                std::fill(local, local + NvoxLocal, 0.0);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
                for (ptrdiff_t ii = 0;
                     ii < static_cast<ptrdiff_t>(bed_indices[bed].size());
                     ++ii) {
                    const int idx = bed_indices[bed][static_cast<size_t>(ii)];
                    const int global_row = subset_rows[static_cast<size_t>(idx)];
                    const int row =
                        (bedCount > 1) ? (global_row % baseRows) : global_row;

                    const double ratio = x[static_cast<size_t>(idx)];

                    const uint16_t* attPtr = nullptr;
                    if (hasAtten) {
                        int angle = 0;
                        if (raysPerAngle > 0)
                            angle = static_cast<int>(
                                static_cast<size_t>(row) / raysPerAngle);
                        if (angle >= ctx.numAngles)
                            angle = ctx.numAngles - 1;
                        attPtr = (*ctx.bedAttenFactors)[bed].data() +
                                 static_cast<size_t>(angle) * NvoxLocal;
                    }

                    if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
                        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                        for (size_t t = r0; t < r1; ++t) {
                            double val = static_cast<double>(A.val[t]);
                            if (attPtr)
                                val *=
                                    static_cast<double>(attPtr[A.col_idx[t]]) *
                                    invScale;
                            local[static_cast<size_t>(A.col_idx[t])] += ratio *
                                                                         val;
                        }
                    } else {
                        for (int64_t t = A.row_ptr[row]; t < A.row_ptr[row + 1];
                             ++t) {
                            double val = A.values[t];
                            if (attPtr)
                                val *=
                                    static_cast<double>(attPtr[A.col_idx[t]]) *
                                    invScale;
                            local[static_cast<size_t>(A.col_idx[t])] += ratio *
                                                                         val;
                        }
                    }
                }
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_eff)
#endif
            for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(NvoxLocal); ++j) {
                double sum = 0.0;
                const size_t jj = static_cast<size_t>(j);
                for (size_t tid = 0; tid < threadCount; ++tid)
                    sum += tls[tid * NvoxLocal + jj];
                y[col_offset + jj] += sum;
            }
        }
    }

    template <typename MatrixType>
    static void computeNormAT_AttenImpl(const MatrixType& A,
                                        const std::vector<uint16_t>& atten,
                                        int numAngles,
                                        double* norm,
                                        int threads) {
        const int threads_eff = sparseops_effective_threads(threads);
        const int baseRows = static_cast<int>(A.rows);
        const size_t Nvox = static_cast<size_t>(A.cols);
        const size_t raysPerAngle =
            (numAngles > 0) ? (static_cast<size_t>(baseRows) / numAngles) : 0;
        const double invScale = 1.0 / 65535.0;

        std::memset(norm, 0, Nvox * sizeof(double));

        if (!sparseops_should_use_threadlocal_accum(Nvox, threads_eff)) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(threads_eff)
#endif
            for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(baseRows);
                 ++row) {
                int angle = 0;
                if (raysPerAngle > 0)
                    angle = static_cast<int>(
                        static_cast<size_t>(row) / raysPerAngle);
                if (angle >= numAngles)
                    angle = numAngles - 1;

                const uint16_t* attPtr =
                    atten.data() + static_cast<size_t>(angle) * Nvox;

                if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
                    const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                    const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                    for (size_t t = r0; t < r1; ++t) {
                        double val = static_cast<double>(A.val[t]);
                        val *= static_cast<double>(attPtr[A.col_idx[t]]) *
                               invScale;
#pragma omp atomic
                        norm[static_cast<size_t>(A.col_idx[t])] += val;
                    }
                } else {
                    for (int64_t t = A.row_ptr[row]; t < A.row_ptr[row + 1];
                         ++t) {
                        double val = A.values[t];
                        val *= static_cast<double>(attPtr[A.col_idx[t]]) *
                               invScale;
#pragma omp atomic
                        norm[static_cast<size_t>(A.col_idx[t])] += val;
                    }
                }
            }
            return;
        }

        const size_t threadCount = static_cast<size_t>(threads_eff);
        std::vector<double> tls(threadCount * Nvox, 0.0);

#ifdef _OPENMP
#pragma omp parallel num_threads(threads_eff)
#endif
        {
#ifdef _OPENMP
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid = 0;
#endif
            double* local = tls.data() + tid * Nvox;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
            for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(baseRows);
                 ++row) {
                int angle = 0;
                if (raysPerAngle > 0)
                    angle = static_cast<int>(
                        static_cast<size_t>(row) / raysPerAngle);
                if (angle >= numAngles)
                    angle = numAngles - 1;

                const uint16_t* attPtr =
                    atten.data() + static_cast<size_t>(angle) * Nvox;

                if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
                    const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                    const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                    for (size_t t = r0; t < r1; ++t) {
                        double val = static_cast<double>(A.val[t]);
                        val *= static_cast<double>(attPtr[A.col_idx[t]]) *
                               invScale;
                        local[static_cast<size_t>(A.col_idx[t])] += val;
                    }
                } else {
                    for (int64_t t = A.row_ptr[row]; t < A.row_ptr[row + 1];
                         ++t) {
                        double val = A.values[t];
                        val *= static_cast<double>(attPtr[A.col_idx[t]]) *
                               invScale;
                        local[static_cast<size_t>(A.col_idx[t])] += val;
                    }
                }
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_eff)
#endif
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(Nvox); ++j) {
            double sum = 0.0;
            const size_t jj = static_cast<size_t>(j);
            for (size_t tid = 0; tid < threadCount; ++tid)
                sum += tls[tid * Nvox + jj];
            norm[jj] = sum;
        }
    }

    template <typename MatrixType>
    static void computeSubsetSensitivityImpl(const MatrixType& A,
                                             const std::vector<int>& subset_rows,
                                             int N,
                                             const SparseSubsetContext& ctx,
                                             double* sensitivity,
                                             int threads) {
        const int threads_eff = sparseops_effective_threads(threads);
        std::fill(sensitivity, sensitivity + static_cast<size_t>(N), 0.0);

        if (subset_rows.empty())
            return;

        const int baseRows = static_cast<int>(A.rows);
        const int baseCols = static_cast<int>(A.cols);
        const size_t NvoxLocal = static_cast<size_t>(baseCols);
        const size_t raysPerAngle =
            (ctx.numAngles > 0) ? (static_cast<size_t>(baseRows) / ctx.numAngles)
                                : 0;
        const double invScale = 1.0 / 65535.0;
        const int bedCount = std::max(1, ctx.bedCount);
        const bool hasOffsets =
            ctx.bedOffsetVoxels && !ctx.bedOffsetVoxels->empty();
        const bool hasAtten =
            ctx.bedAttenFactors && !ctx.bedAttenFactors->empty();

        if (!sparseops_should_use_threadlocal_accum(NvoxLocal, threads_eff)) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(threads_eff)
#endif
            for (ptrdiff_t idx = 0;
                 idx < static_cast<ptrdiff_t>(subset_rows.size()); ++idx) {
                const int global_row = subset_rows[static_cast<size_t>(idx)];
                const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
                const int row =
                    (bedCount > 1) ? (global_row % baseRows) : global_row;

                const uint16_t* attPtr = nullptr;
                if (hasAtten) {
                    int angle = 0;
                    if (raysPerAngle > 0)
                        angle = static_cast<int>(
                            static_cast<size_t>(row) / raysPerAngle);
                    if (angle >= ctx.numAngles)
                        angle = ctx.numAngles - 1;
                    attPtr = (*ctx.bedAttenFactors)[bed].data() +
                             static_cast<size_t>(angle) * NvoxLocal;
                }

                const size_t col_offset = hasOffsets
                    ? static_cast<size_t>((*ctx.bedOffsetVoxels)[bed])
                    : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

                if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
                    const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                    const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                    for (size_t t = r0; t < r1; ++t) {
                        double val = static_cast<double>(A.val[t]);
                        if (attPtr)
                            val *= static_cast<double>(attPtr[A.col_idx[t]]) *
                                   invScale;
#pragma omp atomic
                        sensitivity[col_offset + static_cast<size_t>(A.col_idx[t])] +=
                            val;
                    }
                } else {
                    for (int64_t t = A.row_ptr[row]; t < A.row_ptr[row + 1];
                         ++t) {
                        double val = A.values[t];
                        if (attPtr)
                            val *= static_cast<double>(attPtr[A.col_idx[t]]) *
                                   invScale;
#pragma omp atomic
                        sensitivity[col_offset + static_cast<size_t>(A.col_idx[t])] +=
                            val;
                    }
                }
            }
            return;
        }

        std::vector<std::vector<int>> bed_indices(bedCount);
        for (size_t i = 0; i < subset_rows.size(); ++i) {
            const int global_row = subset_rows[i];
            const int bed = (bedCount > 1) ? (global_row / baseRows) : 0;
            bed_indices[bed].push_back(static_cast<int>(i));
        }

        const size_t threadCount = static_cast<size_t>(threads_eff);
        std::vector<double> tls(threadCount * NvoxLocal, 0.0);

        for (int bed = 0; bed < bedCount; ++bed) {
            if (bed_indices[bed].empty())
                continue;

            const size_t col_offset = hasOffsets
                ? static_cast<size_t>((*ctx.bedOffsetVoxels)[bed])
                : static_cast<size_t>(bed) * static_cast<size_t>(baseCols);

#ifdef _OPENMP
#pragma omp parallel num_threads(threads_eff)
#endif
            {
#ifdef _OPENMP
                const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
                const size_t tid = 0;
#endif
                double* local = tls.data() + tid * NvoxLocal;
                std::fill(local, local + NvoxLocal, 0.0);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
                for (ptrdiff_t ii = 0;
                     ii < static_cast<ptrdiff_t>(bed_indices[bed].size());
                     ++ii) {
                    const int idx = bed_indices[bed][static_cast<size_t>(ii)];
                    const int global_row = subset_rows[static_cast<size_t>(idx)];
                    const int row =
                        (bedCount > 1) ? (global_row % baseRows) : global_row;

                    const uint16_t* attPtr = nullptr;
                    if (hasAtten) {
                        int angle = 0;
                        if (raysPerAngle > 0)
                            angle = static_cast<int>(
                                static_cast<size_t>(row) / raysPerAngle);
                        if (angle >= ctx.numAngles)
                            angle = ctx.numAngles - 1;
                        attPtr = (*ctx.bedAttenFactors)[bed].data() +
                                 static_cast<size_t>(angle) * NvoxLocal;
                    }

                    if constexpr (std::is_same_v<MatrixType, io::CSRView>) {
                        const size_t r0 = static_cast<size_t>(A.row_ptr[row]);
                        const size_t r1 = static_cast<size_t>(A.row_ptr[row + 1]);
                        for (size_t t = r0; t < r1; ++t) {
                            double val = static_cast<double>(A.val[t]);
                            if (attPtr)
                                val *=
                                    static_cast<double>(attPtr[A.col_idx[t]]) *
                                    invScale;
                            local[static_cast<size_t>(A.col_idx[t])] += val;
                        }
                    } else {
                        for (int64_t t = A.row_ptr[row]; t < A.row_ptr[row + 1];
                             ++t) {
                            double val = A.values[t];
                            if (attPtr)
                                val *=
                                    static_cast<double>(attPtr[A.col_idx[t]]) *
                                    invScale;
                            local[static_cast<size_t>(A.col_idx[t])] += val;
                        }
                    }
                }
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_eff)
#endif
            for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(NvoxLocal); ++j) {
                double sum = 0.0;
                const size_t jj = static_cast<size_t>(j);
                for (size_t tid = 0; tid < threadCount; ++tid)
                    sum += tls[tid * NvoxLocal + jj];
                sensitivity[col_offset + jj] += sum;
            }
        }
    }
};

} // namespace spect
