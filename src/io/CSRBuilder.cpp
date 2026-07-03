#include "io/CSRBuilder.h"

#include <cmath>
#include <iostream>
#include <limits>

namespace spect {
namespace io {

CSR buildCSR_from_mat(const MatArray &W_ind, const MatArray &W_dat,
                      size_t Nvox, double prune_eps,
                      const std::vector<uint8_t> *mask) {
  if (W_ind.cls != MAT_C_INT32)
    throw std::runtime_error("W_ind must be int32");
  if (!(W_dat.cls == MAT_C_DOUBLE || W_dat.cls == MAT_C_SINGLE))
    throw std::runtime_error("W_dat must be double/single");
  if (W_ind.dims.size() != 2 || W_dat.dims.size() != 2)
    throw std::runtime_error("W_ind/W_dat must be 2D");

  const size_t M = W_ind.dims[0];
  const size_t K = W_ind.dims[1];
  if (W_dat.dims[0] != M || W_dat.dims[1] != K)
    throw std::runtime_error("W_ind/W_dat dims mismatch");

  const int32_t *indices = W_ind.i32.data();
  const double *weights_d = (W_dat.cls == MAT_C_DOUBLE) ? W_dat.d.data() : nullptr;
  const float *weights_f = (W_dat.cls == MAT_C_SINGLE) ? W_dat.f.data() : nullptr;
  auto get_weight = [&](size_t r, size_t c) {
    const size_t linear = matlab2dElem(r, c, M);
    return weights_d ? weights_d[linear] : static_cast<double>(weights_f[linear]);
  };

  CSR A;
  A.rows = static_cast<int>(M);
  A.cols = static_cast<int>(Nvox);
  A.row_ptr.assign(M + 1, 0);

  uint64_t nnzAccum = 0;
  for (size_t r = 0; r < M; ++r) {
    uint64_t count = 0;
    for (size_t c = 0; c < K; ++c) {
      const int32_t j1 = indices[matlab2dElem(r, c, M)];
      if (j1 > 0 && static_cast<size_t>(j1) <= Nvox) {
        const double w = get_weight(r, c);
        if (std::abs(w) > prune_eps && (!mask || (*mask)[static_cast<size_t>(j1 - 1)]))
          ++count;
      }
    }
    nnzAccum += count;
    if (nnzAccum > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("CSR nnz exceeds CSRB v1 uint32 row_ptr limit: " +
                               std::to_string(nnzAccum));
    }
    A.row_ptr[r + 1] = static_cast<uint32_t>(nnzAccum);
  }

  const size_t nnz = A.row_ptr[M];
  A.col_idx.resize(nnz);
  A.val.resize(nnz);
  for (size_t r = 0; r < M; ++r) {
    size_t pos = A.row_ptr[r];
    for (size_t c = 0; c < K; ++c) {
      const int32_t j1 = indices[matlab2dElem(r, c, M)];
      if (j1 > 0 && static_cast<size_t>(j1) <= Nvox) {
        const double w = get_weight(r, c);
        if (std::abs(w) > prune_eps && (!mask || (*mask)[static_cast<size_t>(j1 - 1)])) {
          A.col_idx[pos] = static_cast<uint32_t>(j1 - 1);
          A.val[pos] = static_cast<float>(w);
          ++pos;
        }
      }
    }
  }

  validateCSR(A, kDefaultCSRValidationLevel);
  return A;
}

CSR buildCSR_from_mat_stream(const std::string &idxPath, const std::string &idxVar,
                             const std::string &wgtPath, const std::string &wgtVar,
                             size_t Nvox, double prune_eps,
                             const std::vector<uint8_t> *mask) {
  mat_t *mIdx = Mat_Open(idxPath.c_str(), MAT_ACC_RDONLY);
  if (!mIdx)
    throw std::runtime_error("Cannot open MAT file: " + idxPath);
  matvar_t *vIdx = idxVar.empty() ? Mat_VarReadNext(mIdx) : Mat_VarRead(mIdx, idxVar.c_str());
  if (!vIdx) {
    Mat_Close(mIdx);
    throw std::runtime_error("Variable not found in " + idxPath + " (" + idxVar + ")");
  }

  mat_t *mWgt = Mat_Open(wgtPath.c_str(), MAT_ACC_RDONLY);
  if (!mWgt) {
    Mat_VarFree(vIdx);
    Mat_Close(mIdx);
    throw std::runtime_error("Cannot open MAT file: " + wgtPath);
  }
  matvar_t *vWgt = wgtVar.empty() ? Mat_VarReadNext(mWgt) : Mat_VarRead(mWgt, wgtVar.c_str());
  if (!vWgt) {
    Mat_VarFree(vIdx);
    Mat_Close(mIdx);
    Mat_Close(mWgt);
    throw std::runtime_error("Variable not found in " + wgtPath + " (" + wgtVar + ")");
  }

  if (vIdx->rank != 2 || vWgt->rank != 2) {
    Mat_VarFree(vIdx);
    Mat_VarFree(vWgt);
    Mat_Close(mIdx);
    Mat_Close(mWgt);
    throw std::runtime_error("W_ind/W_dat must be 2D");
  }

  const size_t M = vIdx->dims[0];
  const size_t K = vIdx->dims[1];
  if (vWgt->dims[0] != M || vWgt->dims[1] != K) {
    Mat_VarFree(vIdx);
    Mat_VarFree(vWgt);
    Mat_Close(mIdx);
    Mat_Close(mWgt);
    throw std::runtime_error("W_ind/W_dat dims mismatch");
  }
  if (vIdx->class_type != MAT_C_INT32) {
    Mat_VarFree(vIdx);
    Mat_VarFree(vWgt);
    Mat_Close(mIdx);
    Mat_Close(mWgt);
    throw std::runtime_error("W_ind must be int32");
  }
  if (!(vWgt->class_type == MAT_C_SINGLE || vWgt->class_type == MAT_C_DOUBLE)) {
    Mat_VarFree(vIdx);
    Mat_VarFree(vWgt);
    Mat_Close(mIdx);
    Mat_Close(mWgt);
    throw std::runtime_error("W_dat must be single/double");
  }

  CSR A;
  A.rows = static_cast<int>(M);
  A.cols = static_cast<int>(Nvox);
  A.row_ptr.assign(M + 1, 0);

  std::vector<int32_t> bufIdx(M);
  std::vector<double> bufWgtD;
  std::vector<float> bufWgtF;
  const bool weights_are_double = (vWgt->class_type == MAT_C_DOUBLE);
  if (weights_are_double)
    bufWgtD.resize(M);
  else
    bufWgtF.resize(M);

  const std::string idxLabel = idxVar.empty() && vIdx->name ? vIdx->name : idxVar;
  const std::string wgtLabel = wgtVar.empty() && vWgt->name ? vWgt->name : wgtVar;
  auto cleanupAndThrowReadFailure = [&]() {
    Mat_VarFree(vIdx);
    Mat_VarFree(vWgt);
    Mat_Close(mIdx);
    Mat_Close(mWgt);
  };

  std::cout << "[CSR-stream] Rows=" << M << ", Cols=" << K << ", Nvox=" << Nvox << "\n";
  std::cout << "[CSR-stream] Pass 1/2 (counting nonzeros) ...\n";
  const size_t progressStep = std::max<size_t>(1, K / 20);
  std::vector<uint64_t> rowPtr64(M + 1, 0);

  for (size_t c = 0; c < K; ++c) {
    const size_t start = c * M;
    if (Mat_VarReadDataLinear(mIdx, vIdx, bufIdx.data(), start, 1, M) != 0) {
      cleanupAndThrowReadFailure();
      throw std::runtime_error("Mat_VarReadDataLinear failed for variable '" + idxLabel +
                               "' in " + idxPath);
    }
    if (weights_are_double) {
      if (Mat_VarReadDataLinear(mWgt, vWgt, bufWgtD.data(), start, 1, M) != 0) {
        cleanupAndThrowReadFailure();
        throw std::runtime_error("Mat_VarReadDataLinear failed for variable '" + wgtLabel +
                                 "' in " + wgtPath);
      }
    } else {
      if (Mat_VarReadDataLinear(mWgt, vWgt, bufWgtF.data(), start, 1, M) != 0) {
        cleanupAndThrowReadFailure();
        throw std::runtime_error("Mat_VarReadDataLinear failed for variable '" + wgtLabel +
                                 "' in " + wgtPath);
      }
    }

    for (size_t r = 0; r < M; ++r) {
      const int32_t j1 = bufIdx[r];
      if (j1 <= 0 || static_cast<size_t>(j1) > Nvox)
        continue;
      const double w = weights_are_double ? bufWgtD[r] : static_cast<double>(bufWgtF[r]);
      if (std::abs(w) <= prune_eps)
        continue;
      if (mask && !(*mask)[static_cast<size_t>(j1 - 1)])
        continue;
      rowPtr64[r + 1]++;
    }

    if ((c % progressStep) == 0 || c + 1 == K) {
      std::cout << "[CSR-stream]   Pass1 processed " << (c + 1) << "/" << K << " columns\n";
    }
  }

  for (size_t r = 0; r < M; ++r) {
    rowPtr64[r + 1] += rowPtr64[r];
    if (rowPtr64[r + 1] > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      cleanupAndThrowReadFailure();
      throw std::runtime_error("CSR nnz exceeds CSRB v1 uint32 row_ptr limit: " +
                               std::to_string(rowPtr64[r + 1]));
    }
    A.row_ptr[r + 1] = static_cast<uint32_t>(rowPtr64[r + 1]);
  }

  const size_t nnz = A.row_ptr[M];
  A.col_idx.resize(nnz);
  A.val.resize(nnz);
  std::cout << "[CSR-stream] Pass1 done, nnz=" << nnz << "\n";

  std::cout << "[CSR-stream] Pass 2/2 (filling values) ...\n";
  std::vector<uint32_t> cursor(A.row_ptr.begin(), A.row_ptr.begin() + static_cast<ptrdiff_t>(M));
  for (size_t c = 0; c < K; ++c) {
    const size_t start = c * M;
    if (Mat_VarReadDataLinear(mIdx, vIdx, bufIdx.data(), start, 1, M) != 0) {
      cleanupAndThrowReadFailure();
      throw std::runtime_error("Mat_VarReadDataLinear failed for variable '" + idxLabel +
                               "' in " + idxPath);
    }
    if (weights_are_double) {
      if (Mat_VarReadDataLinear(mWgt, vWgt, bufWgtD.data(), start, 1, M) != 0) {
        cleanupAndThrowReadFailure();
        throw std::runtime_error("Mat_VarReadDataLinear failed for variable '" + wgtLabel +
                                 "' in " + wgtPath);
      }
    } else {
      if (Mat_VarReadDataLinear(mWgt, vWgt, bufWgtF.data(), start, 1, M) != 0) {
        cleanupAndThrowReadFailure();
        throw std::runtime_error("Mat_VarReadDataLinear failed for variable '" + wgtLabel +
                                 "' in " + wgtPath);
      }
    }

    for (size_t r = 0; r < M; ++r) {
      const int32_t j1 = bufIdx[r];
      if (j1 <= 0 || static_cast<size_t>(j1) > Nvox)
        continue;
      const double w = weights_are_double ? bufWgtD[r] : static_cast<double>(bufWgtF[r]);
      if (std::abs(w) <= prune_eps)
        continue;
      if (mask && !(*mask)[static_cast<size_t>(j1 - 1)])
        continue;
      const uint32_t pos = cursor[r]++;
      A.col_idx[pos] = static_cast<uint32_t>(j1 - 1);
      A.val[pos] = static_cast<float>(w);
    }

    if ((c % progressStep) == 0 || c + 1 == K) {
      std::cout << "[CSR-stream]   Pass2 processed " << (c + 1) << "/" << K << " columns\n";
    }
  }
  std::cout << "[CSR-stream] Pass2 done (nnz=" << nnz << ").\n";

  Mat_VarFree(vIdx);
  Mat_VarFree(vWgt);
  Mat_Close(mIdx);
  Mat_Close(mWgt);

  validateCSR(A, kDefaultCSRValidationLevel);
  return A;
}

} // namespace io
} // namespace spect