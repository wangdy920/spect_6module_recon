#pragma once

#include "cuda/CudaConfig.h"
#include "sparse/SparseOpsCUDA.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS) && \
    defined(SPECT_CUDA_SPARSEOPS_EXTERNAL_IMPL) && CUDA_ENABLED

#include <cusparse.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace spect {

inline void checkCusparse(cusparseStatus_t status, const char *expr,
                          const char *file, int line) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuSPARSE error at ") + file + ":" +
                             std::to_string(line) + " for " + expr +
                             " (status=" +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}

#define SPECT_CUSPARSE_CHECK(expr) \
  ::spect::checkCusparse((expr), #expr, __FILE__, __LINE__)

class CuSparseHandleGuard {
public:
  CuSparseHandleGuard() { SPECT_CUSPARSE_CHECK(cusparseCreate(&handle_)); }
  ~CuSparseHandleGuard() {
    if (handle_ != nullptr) {
      cusparseDestroy(handle_);
    }
  }

  CuSparseHandleGuard(const CuSparseHandleGuard &) = delete;
  CuSparseHandleGuard &operator=(const CuSparseHandleGuard &) = delete;

  cusparseHandle_t get() const { return handle_; }
  void setStream(cudaStream_t stream) const {
    SPECT_CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
  }

private:
  cusparseHandle_t handle_ = nullptr;
};

class CuSparseMatGuard {
public:
  CuSparseMatGuard() = default;
  ~CuSparseMatGuard() {
    reset();
  }

  CuSparseMatGuard(const CuSparseMatGuard &) = delete;
  CuSparseMatGuard &operator=(const CuSparseMatGuard &) = delete;

  void reset() {
    if (mat_ != nullptr) {
      cusparseDestroySpMat(mat_);
      mat_ = nullptr;
    }
  }

  void create(int64_t rows, int64_t cols, int64_t nnz, int32_t *rowPtr,
              int32_t *colIdx, float *values) {
    reset();
    SPECT_CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_, rows, cols, nnz, rowPtr, colIdx, values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  }

  cusparseSpMatDescr_t get() const { return mat_; }

private:
  cusparseSpMatDescr_t mat_ = nullptr;
};

class CuSparseVecGuard {
public:
  CuSparseVecGuard() = default;
  ~CuSparseVecGuard() {
    if (vec_ != nullptr) {
      cusparseDestroyDnVec(vec_);
    }
  }

  CuSparseVecGuard(const CuSparseVecGuard &) = delete;
  CuSparseVecGuard &operator=(const CuSparseVecGuard &) = delete;

  void create(int64_t size, float *data) {
    SPECT_CUSPARSE_CHECK(cusparseCreateDnVec(&vec_, size, data, CUDA_R_32F));
  }

  cusparseDnVecDescr_t get() const { return vec_; }

private:
  cusparseDnVecDescr_t vec_ = nullptr;
};

class CuSparseDnMatGuard {
public:
  CuSparseDnMatGuard() = default;
  ~CuSparseDnMatGuard() {
    if (mat_ != nullptr) {
      cusparseDestroyDnMat(mat_);
    }
  }

  CuSparseDnMatGuard(const CuSparseDnMatGuard &) = delete;
  CuSparseDnMatGuard &operator=(const CuSparseDnMatGuard &) = delete;

  void create(int64_t rows, int64_t cols, int64_t ld, float *data,
              cudaDataType type, cusparseOrder_t order) {
    if (mat_ != nullptr) {
      cusparseDestroyDnMat(mat_);
      mat_ = nullptr;
    }
    SPECT_CUSPARSE_CHECK(
        cusparseCreateDnMat(&mat_, rows, cols, ld, data, type, order));
  }

  cusparseDnMatDescr_t get() const { return mat_; }

private:
  cusparseDnMatDescr_t mat_ = nullptr;
};

struct HostCSRData {
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t nnz = 0;
  std::vector<int32_t> rowPtr;
  std::vector<int32_t> colIdx;
  std::vector<float> values;
};

int getHostParallelThreads(int requestedThreads, uint64_t workItems);

HostCSRData convertCSR(const io::CSRView &matrix);

void buildContiguousSubsetCSR(const io::CSRView &matrix, int rowBegin,
                              int rowEnd, int threads, HostCSRData &host);
void buildSubsetCSR(const io::CSRView &matrix,
                    const std::vector<int> &subset_rows, int threads,
                    HostCSRData &host);

} // namespace spect

#endif