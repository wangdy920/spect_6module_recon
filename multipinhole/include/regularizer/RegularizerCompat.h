#pragma once

#include "regularizer/RegConfig.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace spect::regularizer {

inline size_t IDX(int x, int y, int z, int nx, int ny) {
  return static_cast<size_t>(z) * ny * nx + static_cast<size_t>(y) * nx + x;
}

inline bool inb(int x, int y, int z, int nx, int ny, int nz) {
  return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
}

void compute_reg_gradient(const std::vector<double> &f, int nx, int ny, int nz,
                          const RegConf &reg,
                          const std::vector<uint8_t> *mask,
                          std::vector<double> &grad, int threads);

void osl_update_inplace(std::vector<double> &f, const std::vector<double> &q,
                        const std::vector<double> &d,
                        const std::vector<double> &grad, double beta,
                        const std::vector<uint8_t> *mask, bool nonneg,
                        int threads);

double compute_reg_energy(const std::vector<double> &f, int nx, int ny, int nz,
                          const RegConf &reg,
                          const std::vector<uint8_t> *mask);

} // namespace spect::regularizer
