#include "io/MatFileReader.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace spect {
namespace io {

MatArray readMat(const std::string &path, const std::string &wantVar) {
  mat_t *mat = Mat_Open(path.c_str(), MAT_ACC_RDONLY);
  if (!mat)
    throw std::runtime_error("Cannot open MAT file: " + path);

  matvar_t *var = wantVar.empty() ? Mat_VarReadNext(mat) : Mat_VarRead(mat, wantVar.c_str());
  if (!var) {
    Mat_Close(mat);
    throw std::runtime_error("Variable not found in " + path + " (var=" + wantVar + ")");
  }

  MatArray out;
  out.cls = var->class_type;
  out.dtype = var->data_type;
  out.dims.assign(var->dims, var->dims + var->rank);
  const size_t n = out.numel();

  if (!var->data && n != 0) {
    Mat_VarFree(var);
    Mat_Close(mat);
    throw std::runtime_error("matvar->data is null");
  }

  switch (var->class_type) {
  case MAT_C_DOUBLE:
    out.d.resize(n);
    if (n)
      std::memcpy(out.d.data(), var->data, n * sizeof(double));
    break;
  case MAT_C_SINGLE:
    out.f.resize(n);
    if (n)
      std::memcpy(out.f.data(), var->data, n * sizeof(float));
    break;
  case MAT_C_INT32:
    out.i32.resize(n);
    if (n)
      std::memcpy(out.i32.data(), var->data, n * sizeof(int32_t));
    break;
  default:
    Mat_VarFree(var);
    Mat_Close(mat);
    throw std::runtime_error("Unsupported MATLAB class (expect double/single/int32)");
  }

  Mat_VarFree(var);
  Mat_Close(mat);
  return out;
}

void printMatInfo(const MatArray &arr, const std::string &name) {
  std::cout << "=== " << name << " ===\n";
  std::cout << "Dim: [";
  for (size_t i = 0; i < arr.dims.size(); ++i) {
    std::cout << arr.dims[i];
    if (i + 1 < arr.dims.size())
      std::cout << " x ";
  }
  std::cout << "]  Class: " << static_cast<int>(arr.cls)
            << ", Numel: " << arr.numel() << "\n";
}

} // namespace io
} // namespace spect