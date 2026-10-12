#include "ctac/AutoMuMap.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace spect::main {

namespace {

struct BallOffsets {
  std::vector<int> dx, dy, dz;
};

BallOffsets makeBallOffsets(int radius) {
  BallOffsets b;
  const int r2 = radius * radius;
  for (int z = -radius; z <= radius; ++z)
    for (int y = -radius; y <= radius; ++y)
      for (int x = -radius; x <= radius; ++x)
        if (x * x + y * y + z * z <= r2) {
          b.dx.push_back(x);
          b.dy.push_back(y);
          b.dz.push_back(z);
        }
  return b;
}

/// dilate=true: out = max over ball; dilate=false: out = min over ball.
std::vector<uint8_t> morph(const std::vector<uint8_t> &in, int nx, int ny,
                           int nz, const BallOffsets &ball, bool dilate) {
  std::vector<uint8_t> out(in.size(), dilate ? 0 : 1);
  const size_t plane = static_cast<size_t>(nx) * ny;
#pragma omp parallel for schedule(static)
  for (ptrdiff_t iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        uint8_t v = dilate ? 0 : 1;
        for (size_t k = 0; k < ball.dx.size(); ++k) {
          const int x = ix + ball.dx[k];
          const int y = iy + ball.dy[k];
          const int z = static_cast<int>(iz) + ball.dz[k];
          uint8_t s;
          if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) {
            // Neutral element outside the volume (0 for dilation, 1 for
            // erosion): closing must not erode bodies that touch the volume
            // border, since the preceding dilation could not grow past it.
            s = dilate ? 0 : 1;
          } else {
            s = in[static_cast<size_t>(z) * plane +
                   static_cast<size_t>(y) * nx + static_cast<size_t>(x)];
          }
          if (dilate) {
            if (s) { v = 1; break; }
          } else {
            if (!s) { v = 0; break; }
          }
        }
        out[static_cast<size_t>(iz) * plane + static_cast<size_t>(iy) * nx +
            static_cast<size_t>(ix)] = v;
      }
    }
  }
  return out;
}

/// 6-connected flood fill assigning `label` to `labels` from `seed`, over
/// voxels where mask[i]==value and labels[i]==0. Returns component size.
size_t floodFill(const std::vector<uint8_t> &mask, uint8_t value,
                 std::vector<int> &labels, int label, size_t seed, int nx,
                 int ny, int nz) {
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::queue<size_t> q;
  q.push(seed);
  labels[seed] = label;
  size_t count = 0;
  while (!q.empty()) {
    const size_t idx = q.front();
    q.pop();
    ++count;
    const int iz = static_cast<int>(idx / plane);
    const int iy = static_cast<int>((idx % plane) / nx);
    const int ix = static_cast<int>(idx % nx);
    const int nbr[6][3] = {{ix - 1, iy, iz}, {ix + 1, iy, iz},
                           {ix, iy - 1, iz}, {ix, iy + 1, iz},
                           {ix, iy, iz - 1}, {ix, iy, iz + 1}};
    for (const auto &n : nbr) {
      if (n[0] < 0 || n[1] < 0 || n[2] < 0 || n[0] >= nx || n[1] >= ny ||
          n[2] >= nz)
        continue;
      const size_t j = static_cast<size_t>(n[2]) * plane +
                       static_cast<size_t>(n[1]) * nx +
                       static_cast<size_t>(n[0]);
      if (labels[j] == 0 && mask[j] == value) {
        labels[j] = label;
        q.push(j);
      }
    }
  }
  return count;
}

} // namespace

AutoMuMapResult buildAutoMuMapFromActivity(
    const std::vector<double> &activity, int nx, int ny, int nz,
    double thresholdFrac, double waterMuPerMM,
    const std::vector<uint8_t> *fovMask, int closingRadiusVoxels) {
  const size_t voxels =
      static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  if (nx <= 0 || ny <= 0 || nz <= 0 || activity.size() != voxels)
    throw std::runtime_error("buildAutoMuMapFromActivity: volume size mismatch");
  if (fovMask && fovMask->size() != voxels)
    throw std::runtime_error("buildAutoMuMapFromActivity: FOV mask size mismatch");
  if (!(thresholdFrac > 0.0) || thresholdFrac >= 1.0)
    throw std::runtime_error(
        "buildAutoMuMapFromActivity: threshold fraction must be in (0, 1)");
  if (!std::isfinite(waterMuPerMM) || !(waterMuPerMM > 0.0))
    throw std::runtime_error(
        "buildAutoMuMapFromActivity: water mu must be positive");

  AutoMuMapResult result;
  result.muMap.assign(voxels, 0.0);

  // 1. FOV subtraction + collect positive values for the robust maximum.
  std::vector<double> masked(voxels, 0.0);
  std::vector<double> positives;
  positives.reserve(voxels / 4);
  for (size_t i = 0; i < voxels; ++i) {
    if (fovMask && (*fovMask)[i] == 0)
      continue;
    const double v = activity[i];
    if (v > 0.0) {
      masked[i] = v;
      positives.push_back(v);
    }
  }
  if (positives.empty())
    return result; // empty activity -> empty mu map (caller warns)

  // 2. Robust maximum (P99.9) so a few hot voxels do not inflate the
  //    threshold, then absolute threshold.
  const size_t rank = static_cast<size_t>(
      std::min<double>(static_cast<double>(positives.size()) - 1.0,
                       0.999 * static_cast<double>(positives.size())));
  std::nth_element(positives.begin(), positives.begin() + rank, positives.end());
  result.robustMax = positives[rank];
  if (!(result.robustMax > 0.0))
    return result;
  result.thresholdUsed = thresholdFrac * result.robustMax;

  std::vector<uint8_t> body(voxels, 0);
  for (size_t i = 0; i < voxels; ++i)
    body[i] = (masked[i] >= result.thresholdUsed) ? 1 : 0;

  // 3. Morphological closing bridges small gaps (cold rods, noise dropouts).
  if (closingRadiusVoxels > 0) {
    const BallOffsets ball = makeBallOffsets(closingRadiusVoxels);
    body = morph(body, nx, ny, nz, ball, /*dilate=*/true);
    body = morph(body, nx, ny, nz, ball, /*dilate=*/false);
  }

  // 4. Keep the largest 6-connected body component.
  {
    std::vector<int> labels(voxels, 0);
    int nextLabel = 0;
    int bestLabel = 0;
    size_t bestCount = 0;
    for (size_t i = 0; i < voxels; ++i) {
      if (body[i] && labels[i] == 0) {
        ++nextLabel;
        const size_t count = floodFill(body, 1, labels, nextLabel, i, nx, ny, nz);
        if (count > bestCount) {
          bestCount = count;
          bestLabel = nextLabel;
        }
      }
    }
    if (bestLabel == 0)
      return result;
    for (size_t i = 0; i < voxels; ++i)
      body[i] = (labels[i] == bestLabel) ? 1 : 0;
  }

  // 5. Fill internal holes: background voxels not reachable from the volume
  //    border belong to the body (cold cavities still attenuate).
  {
    std::vector<int> reach(voxels, 0);
    const size_t plane = static_cast<size_t>(nx) * ny;
    for (int iz = 0; iz < nz; ++iz)
      for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
          const bool border = (ix == 0 || iy == 0 || iz == 0 || ix == nx - 1 ||
                               iy == ny - 1 || iz == nz - 1);
          if (!border)
            continue;
          const size_t idx = static_cast<size_t>(iz) * plane +
                             static_cast<size_t>(iy) * nx +
                             static_cast<size_t>(ix);
          if (!body[idx] && reach[idx] == 0)
            floodFill(body, 0, reach, 1, idx, nx, ny, nz);
        }
    for (size_t i = 0; i < voxels; ++i)
      if (!body[i] && reach[i] == 0)
        body[i] = 1;
  }

  // 6. Water fill.
  for (size_t i = 0; i < voxels; ++i) {
    if (body[i]) {
      result.muMap[i] = waterMuPerMM;
      ++result.bodyVoxels;
    }
  }
  return result;
}

} // namespace spect::main
