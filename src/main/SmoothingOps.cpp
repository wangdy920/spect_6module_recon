#include "main/SmoothingOps.h"

#include <algorithm>
#include <vector>

namespace spect::main {

void applyJosephSmoothing(std::vector<double> &buffer, int nx, int ny, int nz,
                          int passes, int threads) {
  if (passes <= 0 || buffer.empty())
    return;

  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> tmp(buffer.size(), 0.0);
  auto idx = [=](int x, int y, int z) {
    return static_cast<size_t>(z) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
  };

  for (int pass = 0; pass < passes; ++pass) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (ptrdiff_t z = 0; z < nz; ++z) {
      for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
          double sum = buffer[idx(x, y, static_cast<int>(z))];
          double weight = 1.0;
          const int dx[6] = {-1, 1, 0, 0, 0, 0};
          const int dy[6] = {0, 0, -1, 1, 0, 0};
          const int dz[6] = {0, 0, 0, 0, -1, 1};
          for (int k = 0; k < 6; ++k) {
            const int xx = x + dx[k];
            const int yy = y + dy[k];
            const int zz = static_cast<int>(z) + dz[k];
            if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz)
              continue;
            sum += buffer[idx(xx, yy, zz)];
            weight += 1.0;
          }
          tmp[idx(x, y, static_cast<int>(z))] = sum / weight;
        }
      }
    }
    buffer.swap(tmp);
  }
}

void clampNonNegativeInPlace(std::vector<double> &buffer, int threads) {
#pragma omp parallel for schedule(static) num_threads(threads)
  for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(buffer.size()); ++i) {
    if (buffer[static_cast<size_t>(i)] < 0.0)
      buffer[static_cast<size_t>(i)] = 0.0;
  }
}

} // namespace spect::main