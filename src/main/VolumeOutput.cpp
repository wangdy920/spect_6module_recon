#include "main/VolumeOutput.h"

#include "io/MHDWriter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace spect {
namespace main {

CropSettings sanitizeCropSettings(int cropStart, int cropEnd) {
  CropSettings crop;
  crop.start = std::max(0, cropStart);
  crop.end = std::max(0, cropEnd);
  return crop;
}

bool isCropSettingsValid(int nz, const CropSettings &crop) {
  return crop.start >= 0 && crop.end >= 0 && crop.start < nz && crop.end < nz && (crop.start + crop.end) < nz;
}

namespace {

bool shouldFlipZPerBed(const OutputTransform *transform) {
  return transform && transform->flipZPerBed && !transform->jointVirtualZFlip &&
         transform->baseNz > 0 &&
         !transform->bedZStarts.empty();
}

bool shouldFlipZWholeOutput(const OutputTransform *transform) {
  return transform && transform->flipZWholeOutput;
}

bool shouldFlipXWholeOutput(const OutputTransform *transform) {
  return transform && transform->flipXWholeOutput;
}

bool shouldHardCutOutputOverlap(const OutputTransform *transform) {
  return transform && !transform->bedOverlapInRecon &&
         transform->baseNz > 0 && transform->bedZStarts.size() > 1;
}

void validateBedZFlipTransform(const OutputTransform &transform, int nz) {
  for (int zStart : transform.bedZStarts) {
    if (zStart < 0 || zStart + transform.baseNz > nz) {
      throw std::invalid_argument(
          "prepareOutputVolume: invalid per-bed Z flip transform");
    }
  }
}

std::vector<double> flipDoubleZPerBed(const std::vector<double> &source, int nx,
                                      int ny, int nz,
                                      const OutputTransform &transform) {
  validateBedZFlipTransform(transform, nz);
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> out(source.size(), 0.0);
  std::vector<int> hits(source.size(), 0);
  for (int zStart : transform.bedZStarts) {
    for (int z = 0; z < transform.baseNz; ++z) {
      const int dstZ = zStart + z;
      const int srcZ = zStart + (transform.baseNz - 1 - z);
      const size_t dstOffset = static_cast<size_t>(dstZ) * plane;
      const size_t srcOffset = static_cast<size_t>(srcZ) * plane;
      for (size_t xy = 0; xy < plane; ++xy) {
        out[dstOffset + xy] += source[srcOffset + xy];
        ++hits[dstOffset + xy];
      }
    }
  }
  for (size_t i = 0; i < out.size(); ++i) {
    if (hits[i] > 1) {
      out[i] /= static_cast<double>(hits[i]);
    } else if (hits[i] == 0) {
      out[i] = source[i];
    }
  }
  return out;
}

template <typename T>
std::vector<T> hardCutOutputOverlap(const std::vector<T> &source, int nx,
                                    int ny, int nz,
                                    const OutputTransform &transform) {
  validateBedZFlipTransform(transform, nz);
  const size_t beds = transform.bedZStarts.size();
  if (transform.outputBedOverlaps.size() + 1 != beds)
    throw std::invalid_argument(
        "prepareOutputVolume: output overlap count does not match beds");
  for (size_t bed = 0; bed < beds; ++bed) {
    const int expectedStart = static_cast<int>(bed) * transform.baseNz;
    if (transform.bedZStarts[bed] != expectedStart)
      throw std::invalid_argument(
          "prepareOutputVolume: output hard-cut requires non-overlapping reconstruction beds");
  }

  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  size_t outputSlices = 0;
  for (size_t bed = 0; bed < beds; ++bed) {
    const int trimBefore = bed == 0
                               ? 0
                               : transform.outputBedOverlaps[bed - 1] -
                                     transform.outputBedOverlaps[bed - 1] / 2;
    const int trimAfter = bed + 1 == beds
                              ? 0
                              : transform.outputBedOverlaps[bed] / 2;
    if (trimBefore < 0 || trimAfter < 0 ||
        trimBefore + trimAfter >= transform.baseNz)
      throw std::invalid_argument(
          "prepareOutputVolume: invalid output hard-cut overlap");
    outputSlices += static_cast<size_t>(transform.baseNz - trimBefore - trimAfter);
  }

  std::vector<T> out;
  out.reserve(outputSlices * plane);
  for (size_t bed = 0; bed < beds; ++bed) {
    const int trimBefore = bed == 0
                               ? 0
                               : transform.outputBedOverlaps[bed - 1] -
                                     transform.outputBedOverlaps[bed - 1] / 2;
    const int trimAfter = bed + 1 == beds
                              ? 0
                              : transform.outputBedOverlaps[bed] / 2;
    const int zBegin = transform.bedZStarts[bed] + trimBefore;
    const int zEnd = transform.bedZStarts[bed] + transform.baseNz - trimAfter;
    const auto first = source.begin() + static_cast<std::ptrdiff_t>(zBegin) *
                                            static_cast<std::ptrdiff_t>(plane);
    const auto last = source.begin() + static_cast<std::ptrdiff_t>(zEnd) *
                                           static_cast<std::ptrdiff_t>(plane);
    out.insert(out.end(), first, last);
  }
  return out;
}

std::vector<int> outputBedStarts(const OutputTransform &transform) {
  const size_t beds = transform.bedZStarts.size();
  if (transform.outputBedOverlaps.size() + 1 != beds)
    throw std::invalid_argument(
        "prepareOutputVolume: output overlap count does not match beds");
  std::vector<int> starts(beds, 0);
  for (size_t bed = 0; bed < beds; ++bed) {
    const int expectedStart = static_cast<int>(bed) * transform.baseNz;
    if (transform.bedZStarts[bed] != expectedStart)
      throw std::invalid_argument(
          "prepareOutputVolume: output stitching requires non-overlapping reconstruction beds");
    if (bed > 0) {
      const int overlap = transform.outputBedOverlaps[bed - 1];
      if (overlap < 0 || overlap >= transform.baseNz)
        throw std::invalid_argument(
            "prepareOutputVolume: invalid output overlap");
      starts[bed] = starts[bed - 1] + transform.baseNz - overlap;
    }
  }
  return starts;
}

double sigmoidRamp(double t) {
  t = std::clamp(t, 0.0, 1.0);
  return 1.0 / (1.0 + std::exp(-12.0 * (t - 0.5)));
}

std::vector<double> sigmoidBlendOutputOverlap(
    const std::vector<double> &source, int nx, int ny, int nz,
    const OutputTransform &transform) {
  validateBedZFlipTransform(transform, nz);
  const std::vector<int> starts = outputBedStarts(transform);
  const int outputNz = starts.back() + transform.baseNz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> out(static_cast<size_t>(outputNz) * plane, 0.0);
  std::vector<double> weights(static_cast<size_t>(outputNz), 0.0);
  for (size_t bed = 0; bed < starts.size(); ++bed) {
    const int before = bed == 0 ? 0 : transform.outputBedOverlaps[bed - 1];
    const int after = bed + 1 == starts.size()
                          ? 0
                          : transform.outputBedOverlaps[bed];
    for (int localZ = 0; localZ < transform.baseNz; ++localZ) {
      double weight = 1.0;
      if (before > 0 && localZ < before)
        weight *= sigmoidRamp(static_cast<double>(localZ) / before);
      if (after > 0 && localZ >= transform.baseNz - after) {
        const double phase = static_cast<double>(
            localZ - (transform.baseNz - after)) / after;
        weight *= 1.0 - sigmoidRamp(phase);
      }
      const int srcZ = transform.bedZStarts[bed] + localZ;
      const int dstZ = starts[bed] + localZ;
      const size_t srcOffset = static_cast<size_t>(srcZ) * plane;
      const size_t dstOffset = static_cast<size_t>(dstZ) * plane;
      for (size_t xy = 0; xy < plane; ++xy)
        out[dstOffset + xy] += weight * source[srcOffset + xy];
      weights[static_cast<size_t>(dstZ)] += weight;
    }
  }
  for (int z = 0; z < outputNz; ++z) {
    const double weight = weights[static_cast<size_t>(z)];
    if (weight <= 0.0)
      continue;
    const size_t offset = static_cast<size_t>(z) * plane;
    for (size_t xy = 0; xy < plane; ++xy)
      out[offset + xy] /= weight;
  }
  return out;
}

double percentile(std::vector<double> values, double q) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const double position = std::clamp(q, 0.0, 100.0) *
                          static_cast<double>(values.size() - 1) / 100.0;
  const size_t lo = static_cast<size_t>(std::floor(position));
  const size_t hi = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lo);
  return values[lo] * (1.0 - fraction) + values[hi] * fraction;
}

std::vector<int> componentSizes2D(const std::vector<uint8_t> &mask,
                                  int height, int width) {
  std::vector<uint8_t> visited(mask.size(), 0);
  std::vector<int> sizes;
  const int dy[4] = {-1, 1, 0, 0};
  const int dx[4] = {0, 0, -1, 1};
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t seed = static_cast<size_t>(y) * width + x;
      if (!mask[seed] || visited[seed])
        continue;
      int size = 0;
      std::deque<std::pair<int, int>> queue;
      queue.emplace_back(y, x);
      visited[seed] = 1;
      while (!queue.empty()) {
        const auto [cy, cx] = queue.front();
        queue.pop_front();
        ++size;
        for (int direction = 0; direction < 4; ++direction) {
          const int ny = cy + dy[direction];
          const int nx = cx + dx[direction];
          if (ny < 0 || ny >= height || nx < 0 || nx >= width)
            continue;
          const size_t index = static_cast<size_t>(ny) * width + nx;
          if (mask[index] && !visited[index]) {
            visited[index] = 1;
            queue.emplace_back(ny, nx);
          }
        }
      }
      if (size >= 8)
        sizes.push_back(size);
    }
  }
  return sizes;
}

std::string classifyOutputObject(const std::vector<double> &volume, int nx,
                                 int ny, int nz) {
  std::vector<double> positive;
  positive.reserve(volume.size() / 8);
  for (double value : volume) {
    if (value > 0.0)
      positive.push_back(value);
  }
  if (positive.empty())
    return "unknown";
  const double p995 = percentile(positive, 99.5);
  const double threshold = 0.10 * p995;
  const double highThreshold = 0.50 * p995;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> componentCounts;
  componentCounts.reserve(static_cast<size_t>(nz));
  size_t maskCount = 0;
  size_t highCount = 0;
  for (int z = 0; z < nz; ++z) {
    std::vector<uint8_t> mask(plane, 0);
    const size_t offset = static_cast<size_t>(z) * plane;
    for (size_t xy = 0; xy < plane; ++xy) {
      const double value = volume[offset + xy];
      if (value >= threshold) {
        mask[xy] = 1;
        ++maskCount;
        if (value >= highThreshold)
          ++highCount;
      }
    }
    componentCounts.push_back(
        static_cast<double>(componentSizes2D(mask, ny, nx).size()));
  }

  std::vector<uint8_t> coronal(static_cast<size_t>(nz) * nx, 0);
  for (int z = 0; z < nz; ++z) {
    const size_t offset = static_cast<size_t>(z) * plane;
    for (int x = 0; x < nx; ++x) {
      double maximum = 0.0;
      for (int y = 0; y < ny; ++y)
        maximum = std::max(maximum,
                           volume[offset + static_cast<size_t>(y) * nx + x]);
      coronal[static_cast<size_t>(z) * nx + x] = maximum >= threshold;
    }
  }
  const int projectionComponents =
      static_cast<int>(componentSizes2D(coronal, nz, nx).size());
  const double highFraction = maskCount > 0
                                  ? static_cast<double>(highCount) / maskCount
                                  : 0.0;
  const double p90Components = percentile(componentCounts, 90.0);

  // ponytail: provisional gate from capillary, tube, mouse, and zimu; unknown
  // remains a no-op and the explicit profile override is the safety valve.
  if (projectionComponents >= 12 && highFraction <= 0.25 &&
      p90Components >= 4.0)
    return "mouse";
  if (projectionComponents >= 4 && projectionComponents <= 12 &&
      highFraction >= 0.30 && p90Components <= 3.0)
    return "zimu";
  if (projectionComponents <= 3 && highFraction >= 0.30 &&
      p90Components <= 3.0)
    return "phantom";
  return "unknown";
}

struct SeamComponent {
  std::vector<uint8_t> mask;
  std::vector<uint8_t> softMask;
  double centerY = 0.0;
  double centerX = 0.0;
};

std::vector<uint8_t> dilateMask(std::vector<uint8_t> mask, int ny, int nx,
                                int iterations) {
  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::vector<uint8_t> next = mask;
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const size_t index = static_cast<size_t>(y) * nx + x;
        if (!mask[index])
          continue;
        if (y > 0)
          next[index - nx] = 1;
        if (y + 1 < ny)
          next[index + nx] = 1;
        if (x > 0)
          next[index - 1] = 1;
        if (x + 1 < nx)
          next[index + 1] = 1;
      }
    }
    mask.swap(next);
  }
  return mask;
}

std::vector<SeamComponent>
seamComponents(const std::vector<double> &plane, int ny, int nx,
               double threshold) {
  const size_t pixels = static_cast<size_t>(ny) * nx;
  std::vector<uint8_t> visited(pixels, 0);
  std::vector<SeamComponent> components;
  constexpr int dy[4] = {-1, 1, 0, 0};
  constexpr int dx[4] = {0, 0, -1, 1};
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const size_t seed = static_cast<size_t>(y) * nx + x;
      if (visited[seed] || plane[seed] < threshold)
        continue;
      std::deque<std::pair<int, int>> queue{{y, x}};
      visited[seed] = 1;
      std::vector<size_t> indices;
      while (!queue.empty()) {
        const auto [cy, cx] = queue.front();
        queue.pop_front();
        indices.push_back(static_cast<size_t>(cy) * nx + cx);
        for (int direction = 0; direction < 4; ++direction) {
          const int yy = cy + dy[direction];
          const int xx = cx + dx[direction];
          if (yy < 0 || yy >= ny || xx < 0 || xx >= nx)
            continue;
          const size_t index = static_cast<size_t>(yy) * nx + xx;
          if (!visited[index] && plane[index] >= threshold) {
            visited[index] = 1;
            queue.emplace_back(yy, xx);
          }
        }
      }
      if (indices.size() < 8)
        continue;
      SeamComponent component;
      component.mask.assign(pixels, 0);
      double sum = 0.0;
      for (size_t index : indices) {
        component.mask[index] = 1;
        const double value = plane[index];
        sum += value;
        component.centerY += value * static_cast<double>(index / nx);
        component.centerX += value * static_cast<double>(index % nx);
      }
      if (sum > 0.0) {
        component.centerY /= sum;
        component.centerX /= sum;
      }
      component.softMask = dilateMask(component.mask, ny, nx, 3);
      components.push_back(std::move(component));
    }
  }
  return components;
}

std::vector<double> squaredDistanceTransform1D(const std::vector<double> &input) {
  const int n = static_cast<int>(input.size());
  std::vector<int> sites(static_cast<size_t>(n));
  std::vector<double> boundaries(static_cast<size_t>(n + 1));
  std::vector<double> output(static_cast<size_t>(n));
  int k = 0;
  sites[0] = 0;
  boundaries[0] = -std::numeric_limits<double>::infinity();
  boundaries[1] = std::numeric_limits<double>::infinity();
  for (int q = 1; q < n; ++q) {
    double boundary = 0.0;
    do {
      const int site = sites[static_cast<size_t>(k)];
      boundary = ((input[static_cast<size_t>(q)] + q * q) -
                  (input[static_cast<size_t>(site)] + site * site)) /
                 (2.0 * (q - site));
      if (boundary <= boundaries[static_cast<size_t>(k)])
        --k;
      else
        break;
    } while (k >= 0);
    ++k;
    sites[static_cast<size_t>(k)] = q;
    boundaries[static_cast<size_t>(k)] = boundary;
    boundaries[static_cast<size_t>(k + 1)] =
        std::numeric_limits<double>::infinity();
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (boundaries[static_cast<size_t>(k + 1)] < q)
      ++k;
    const int site = sites[static_cast<size_t>(k)];
    const double delta = static_cast<double>(q - site);
    output[static_cast<size_t>(q)] =
        delta * delta + input[static_cast<size_t>(site)];
  }
  return output;
}

std::vector<double> euclideanDistanceTo(const std::vector<uint8_t> &mask,
                                        int ny, int nx, uint8_t target) {
  constexpr double infinity = 1e20;
  const size_t pixels = static_cast<size_t>(ny) * nx;
  std::vector<double> horizontal(pixels);
  for (int y = 0; y < ny; ++y) {
    std::vector<double> row(static_cast<size_t>(nx));
    for (int x = 0; x < nx; ++x)
      row[static_cast<size_t>(x)] =
          mask[static_cast<size_t>(y) * nx + x] == target ? 0.0 : infinity;
    const auto distance = squaredDistanceTransform1D(row);
    std::copy(distance.begin(), distance.end(),
              horizontal.begin() + static_cast<size_t>(y) * nx);
  }

  std::vector<double> output(pixels);
  for (int x = 0; x < nx; ++x) {
    std::vector<double> column(static_cast<size_t>(ny));
    for (int y = 0; y < ny; ++y)
      column[static_cast<size_t>(y)] =
          horizontal[static_cast<size_t>(y) * nx + x];
    const auto distance = squaredDistanceTransform1D(column);
    for (int y = 0; y < ny; ++y)
      output[static_cast<size_t>(y) * nx + x] =
          std::sqrt(distance[static_cast<size_t>(y)]);
  }
  return output;
}

std::vector<double> signedEuclideanDistance(const std::vector<uint8_t> &mask,
                                            int ny, int nx) {
  const auto toInside = euclideanDistanceTo(mask, ny, nx, 1);
  const auto toOutside = euclideanDistanceTo(mask, ny, nx, 0);
  std::vector<double> result(mask.size());
  for (size_t index = 0; index < mask.size(); ++index)
    result[index] = mask[index] ? toOutside[index] : -toInside[index];
  return result;
}

double bilinearSample(const std::vector<double> &image, int ny, int nx,
                      double y, double x, double outside = 0.0) {
  const int y0 = static_cast<int>(std::floor(y));
  const int x0 = static_cast<int>(std::floor(x));
  const double fy = y - y0;
  const double fx = x - x0;
  const auto at = [&](int yy, int xx) {
    if (yy < 0 || yy >= ny || xx < 0 || xx >= nx)
      return outside;
    return image[static_cast<size_t>(yy) * nx + xx];
  };
  return (1.0 - fy) * ((1.0 - fx) * at(y0, x0) + fx * at(y0, x0 + 1)) +
         fy * ((1.0 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1));
}

void applyPhantomContourInterpolation(std::vector<double> &volume, int nx,
                                      int ny, int nz, double threshold,
                                      const OutputTransform &transform) {
  constexpr int anchorWidth = 3;
  constexpr double contourMargin = 1.0;
  const size_t planeSize = static_cast<size_t>(nx) * ny;
  const std::vector<double> input = volume;
  const std::vector<int> starts = outputBedStarts(transform);
  for (size_t bed = 1; bed < starts.size(); ++bed) {
    const int seam = starts[bed] + transform.outputBedOverlaps[bed - 1] / 2;
    const int z0 = seam - transform.seamHalfWidth;
    const int z1 = seam + transform.seamHalfWidth + 1;
    if (z0 - anchorWidth < 0 || z1 + anchorWidth > nz)
      continue;
    std::vector<double> left(planeSize, 0.0), right(planeSize, 0.0);
    for (int dz = 0; dz < anchorWidth; ++dz) {
      const size_t leftOffset = static_cast<size_t>(z0 - anchorWidth + dz) * planeSize;
      const size_t rightOffset = static_cast<size_t>(z1 + dz) * planeSize;
      for (size_t xy = 0; xy < planeSize; ++xy) {
        left[xy] += input[leftOffset + xy] / anchorWidth;
        right[xy] += input[rightOffset + xy] / anchorWidth;
      }
    }
    auto leftComponents = seamComponents(left, ny, nx, threshold);
    auto rightComponents = seamComponents(right, ny, nx, threshold);
    if (leftComponents.empty() || leftComponents.size() != rightComponents.size() ||
        leftComponents.size() > 3)
      continue;

    std::vector<int> permutation(rightComponents.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<int> best = permutation;
    double bestCost = std::numeric_limits<double>::infinity();
    do {
      double cost = 0.0;
      for (size_t i = 0; i < leftComponents.size(); ++i) {
        const double dy = leftComponents[i].centerY -
                          rightComponents[static_cast<size_t>(permutation[i])].centerY;
        const double dx = leftComponents[i].centerX -
                          rightComponents[static_cast<size_t>(permutation[i])].centerX;
        cost += std::hypot(dy, dx);
      }
      if (cost < bestCost) {
        bestCost = cost;
        best = permutation;
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    struct PairData {
      const SeamComponent *left = nullptr;
      const SeamComponent *right = nullptr;
      std::vector<double> leftImage;
      std::vector<double> rightImage;
      std::vector<double> leftDistance;
      std::vector<double> rightDistance;
    };
    std::vector<PairData> pairs;
    for (size_t i = 0; i < leftComponents.size(); ++i) {
      PairData pair;
      pair.left = &leftComponents[i];
      pair.right = &rightComponents[static_cast<size_t>(best[i])];
      pair.leftImage.resize(planeSize);
      pair.rightImage.resize(planeSize);
      for (size_t xy = 0; xy < planeSize; ++xy) {
        pair.leftImage[xy] = pair.left->softMask[xy] ? left[xy] : 0.0;
        pair.rightImage[xy] = pair.right->softMask[xy] ? right[xy] : 0.0;
      }
      pair.leftDistance = signedEuclideanDistance(pair.left->mask, ny, nx);
      pair.rightDistance = signedEuclideanDistance(pair.right->mask, ny, nx);
      pairs.push_back(std::move(pair));
    }

    std::vector<double> candidate(static_cast<size_t>(z1 - z0) * planeSize, 0.0);
    for (int index = 0; index < z1 - z0; ++index) {
      const double fraction = static_cast<double>(index + 1) /
                              static_cast<double>(z1 - z0 + 1);
      const size_t offset = static_cast<size_t>(index) * planeSize;
      for (const PairData &pair : pairs) {
        const double targetY = (1.0 - fraction) * pair.left->centerY +
                               fraction * pair.right->centerY;
        const double targetX = (1.0 - fraction) * pair.left->centerX +
                               fraction * pair.right->centerX;
        const double leftShiftY = targetY - pair.left->centerY;
        const double leftShiftX = targetX - pair.left->centerX;
        const double rightShiftY = targetY - pair.right->centerY;
        const double rightShiftX = targetX - pair.right->centerX;
        for (int y = 0; y < ny; ++y) {
          for (int x = 0; x < nx; ++x) {
            const size_t xy = static_cast<size_t>(y) * nx + x;
            const double imageLeft = bilinearSample(
                pair.leftImage, ny, nx, y - leftShiftY, x - leftShiftX);
            const double imageRight = bilinearSample(
                pair.rightImage, ny, nx, y - rightShiftY, x - rightShiftX);
            const double distanceLeft = bilinearSample(
                pair.leftDistance, ny, nx, y - leftShiftY, x - leftShiftX, -100.0);
            const double distanceRight = bilinearSample(
                pair.rightDistance, ny, nx, y - rightShiftY, x - rightShiftX, -100.0);
            const double gate = std::clamp(
                (1.0 - fraction) * distanceLeft + fraction * distanceRight +
                    contourMargin,
                0.0, 1.0);
            candidate[offset + xy] +=
                ((1.0 - fraction) * imageLeft + fraction * imageRight) * gate;
          }
        }
      }
    }

    bool topologyValid = true;
    for (int index = 0; index < z1 - z0; ++index) {
      std::vector<uint8_t> mask(planeSize, 0);
      const size_t offset = static_cast<size_t>(index) * planeSize;
      for (size_t xy = 0; xy < planeSize; ++xy)
        mask[xy] = candidate[offset + xy] >= threshold;
      if (componentSizes2D(mask, ny, nx).size() != pairs.size()) {
        topologyValid = false;
        break;
      }
    }
    if (!topologyValid)
      continue;

    for (int index = 0; index < z1 - z0; ++index) {
      const int z = z0 + index;
      const size_t inputOffset = static_cast<size_t>(z) * planeSize;
      const size_t candidateOffset = static_cast<size_t>(index) * planeSize;
      std::vector<uint8_t> support(planeSize, 0);
      for (size_t xy = 0; xy < planeSize; ++xy)
        support[xy] = input[inputOffset + xy] >= 0.5 * threshold;
      support = dilateMask(std::move(support), ny, nx, 3);
      for (size_t xy = 0; xy < planeSize; ++xy) {
        if (!support[xy] && candidate[candidateOffset + xy] <= 0.0)
          continue;
        const double interpolated = candidate[candidateOffset + xy];
        volume[inputOffset + xy] =
            input[inputOffset + xy] + transform.phantomCloseStrength *
                                          (interpolated - input[inputOffset + xy]);
      }
    }
  }
}

void applyZimuTopologyLockedSmooth(std::vector<double> &volume, int nx, int ny,
                                   int nz, double threshold,
                                   const OutputTransform &transform) {
  const size_t planeSize = static_cast<size_t>(nx) * ny;
  const std::vector<double> input = volume;
  // scipy.ndimage.gaussian_filter1d(sigma=1, truncate=4) uses radius 4.
  constexpr int radius = 4;
  std::vector<double> weights(static_cast<size_t>(2 * radius + 1));
  double weightSum = 0.0;
  for (int dz = -radius; dz <= radius; ++dz) {
    const double weight = std::exp(-0.5 * dz * dz);
    weights[static_cast<size_t>(dz + radius)] = weight;
    weightSum += weight;
  }
  for (double &weight : weights)
    weight /= weightSum;
  const std::vector<int> starts = outputBedStarts(transform);
  const int halfWidth = std::min(transform.seamHalfWidth, 4);
  for (size_t bed = 1; bed < starts.size(); ++bed) {
    const int seam = starts[bed] + transform.outputBedOverlaps[bed - 1] / 2;
    const int z0 = std::max(0, seam - halfWidth);
    const int z1 = std::min(nz, seam + halfWidth + 1);
    std::vector<double> candidate(static_cast<size_t>(z1 - z0) * planeSize);
    double beforeSum = 0.0;
    double afterSum = 0.0;
    for (int z = z0; z < z1; ++z) {
      const size_t inputOffset = static_cast<size_t>(z) * planeSize;
      const size_t candidateOffset = static_cast<size_t>(z - z0) * planeSize;
      for (size_t xy = 0; xy < planeSize; ++xy) {
        const double original = input[inputOffset + xy];
        candidate[candidateOffset + xy] = original;
        if (original < threshold)
          continue;
        double smooth = 0.0;
        for (int dz = -radius; dz <= radius; ++dz) {
          const int zz = std::clamp(z + dz, 0, nz - 1);
          smooth += weights[static_cast<size_t>(dz + radius)] *
                    input[static_cast<size_t>(zz) * planeSize + xy];
        }
        candidate[candidateOffset + xy] =
            (1.0 - transform.zimuCloseStrength) * original +
            transform.zimuCloseStrength * smooth;
        beforeSum += original;
        afterSum += candidate[candidateOffset + xy];
      }
    }
    const double scale = beforeSum > 0.0 && afterSum > 0.0
                             ? beforeSum / afterSum
                             : 1.0;
    for (int z = z0; z < z1; ++z) {
      const size_t inputOffset = static_cast<size_t>(z) * planeSize;
      const size_t candidateOffset = static_cast<size_t>(z - z0) * planeSize;
      for (size_t xy = 0; xy < planeSize; ++xy) {
        const double original = input[inputOffset + xy];
        if (original >= threshold)
          volume[inputOffset + xy] =
              std::max(threshold, candidate[candidateOffset + xy] * scale);
        else
          volume[inputOffset + xy] =
              std::min(original, std::nextafter(threshold, 0.0));
      }
    }
  }
}

void applyObjectAwareSeamPostprocessImpl(std::vector<double> &volume, int nx,
                                         int ny, int nz,
                                         const OutputTransform &transform) {
  const std::string profile = transform.objectProfile == "auto"
                                  ? classifyOutputObject(volume, nx, ny, nz)
                                  : transform.objectProfile;
  if ((profile == "phantom" && transform.phantomCloseStrength == 0.0) ||
      (profile == "zimu" && transform.zimuCloseStrength == 0.0) ||
      (profile != "phantom" && profile != "zimu"))
    return;
  for (int overlap : transform.outputBedOverlaps) {
    if (overlap != 16)
      throw std::invalid_argument(
          "prepareOutputVolume: seam postprocess requires overlap=16");
  }

  std::vector<double> positive;
  positive.reserve(volume.size() / 8);
  for (double value : volume) {
    if (value > 0.0)
      positive.push_back(value);
  }
  if (positive.empty())
    return;
  const double threshold = 0.10 * percentile(positive, 99.5);
  if (profile == "phantom")
    applyPhantomContourInterpolation(volume, nx, ny, nz, threshold, transform);
  else
    applyZimuTopologyLockedSmooth(volume, nx, ny, nz, threshold, transform);
}

std::vector<uint8_t> blendOutputMask(const std::vector<uint8_t> &source,
                                     int nx, int ny, int nz,
                                     const OutputTransform &transform) {
  validateBedZFlipTransform(transform, nz);
  const std::vector<int> starts = outputBedStarts(transform);
  const int outputNz = starts.back() + transform.baseNz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<uint8_t> out(static_cast<size_t>(outputNz) * plane, 0);
  for (size_t bed = 0; bed < starts.size(); ++bed) {
    for (int localZ = 0; localZ < transform.baseNz; ++localZ) {
      const size_t srcOffset = static_cast<size_t>(
          transform.bedZStarts[bed] + localZ) * plane;
      const size_t dstOffset = static_cast<size_t>(starts[bed] + localZ) * plane;
      for (size_t xy = 0; xy < plane; ++xy)
        out[dstOffset + xy] =
            std::max(out[dstOffset + xy], source[srcOffset + xy]);
    }
  }
  return out;
}

std::vector<uint8_t> flipMaskZPerBed(const std::vector<uint8_t> &source, int nx,
                                     int ny, int nz,
                                     const OutputTransform &transform) {
  validateBedZFlipTransform(transform, nz);
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<uint8_t> out(source.size(), 0);
  std::vector<int> hits(source.size(), 0);
  for (int zStart : transform.bedZStarts) {
    for (int z = 0; z < transform.baseNz; ++z) {
      const int dstZ = zStart + z;
      const int srcZ = zStart + (transform.baseNz - 1 - z);
      const size_t dstOffset = static_cast<size_t>(dstZ) * plane;
      const size_t srcOffset = static_cast<size_t>(srcZ) * plane;
      for (size_t xy = 0; xy < plane; ++xy) {
        out[dstOffset + xy] =
            std::max(out[dstOffset + xy], source[srcOffset + xy]);
        ++hits[dstOffset + xy];
      }
    }
  }
  for (size_t i = 0; i < out.size(); ++i) {
    if (hits[i] == 0) {
      out[i] = source[i];
    }
  }
  return out;
}

std::vector<double> flipDoubleZWholeOutput(const std::vector<double> &source,
                                           int nx, int ny, int nz) {
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> out(source.size(), 0.0);
  for (int z = 0; z < nz; ++z) {
    const int srcZ = nz - 1 - z;
    const size_t dstOffset = static_cast<size_t>(z) * plane;
    const size_t srcOffset = static_cast<size_t>(srcZ) * plane;
    std::copy(source.begin() + static_cast<std::ptrdiff_t>(srcOffset),
              source.begin() + static_cast<std::ptrdiff_t>(srcOffset + plane),
              out.begin() + static_cast<std::ptrdiff_t>(dstOffset));
  }
  return out;
}

std::vector<uint8_t> flipMaskZWholeOutput(const std::vector<uint8_t> &source,
                                          int nx, int ny, int nz) {
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<uint8_t> out(source.size(), 0);
  for (int z = 0; z < nz; ++z) {
    const int srcZ = nz - 1 - z;
    const size_t dstOffset = static_cast<size_t>(z) * plane;
    const size_t srcOffset = static_cast<size_t>(srcZ) * plane;
    std::copy(source.begin() + static_cast<std::ptrdiff_t>(srcOffset),
              source.begin() + static_cast<std::ptrdiff_t>(srcOffset + plane),
              out.begin() + static_cast<std::ptrdiff_t>(dstOffset));
  }
  return out;
}

template <typename T>
std::vector<T> flipXWholeOutput(const std::vector<T> &source, int nx, int ny,
                                int nz) {
  const size_t row = static_cast<size_t>(nx);
  const size_t plane = row * static_cast<size_t>(ny);
  std::vector<T> out(source.size());
  for (int z = 0; z < nz; ++z) {
    const size_t zOffset = static_cast<size_t>(z) * plane;
    for (int y = 0; y < ny; ++y) {
      const size_t yOffset = zOffset + static_cast<size_t>(y) * row;
      for (int x = 0; x < nx; ++x) {
        out[yOffset + static_cast<size_t>(x)] =
            source[yOffset + static_cast<size_t>(nx - 1 - x)];
      }
    }
  }
  return out;
}

} // namespace

void postprocessStitchedOutput(std::vector<double> &volume, int nx, int ny,
                               int nz, const OutputTransform &transform) {
  const size_t expected = static_cast<size_t>(nx) * ny * nz;
  if (nx <= 0 || ny <= 0 || nz <= 0 || volume.size() != expected)
    throw std::invalid_argument(
        "postprocessStitchedOutput: invalid volume dimensions");
  applyObjectAwareSeamPostprocessImpl(volume, nx, ny, nz, transform);
}

std::vector<double> prepareOutputVolume(const std::vector<double> &source, int nx,
                                        int ny, int nz,
                                        const std::vector<uint8_t> *mask,
                                        const CropSettings &crop, int *outNz,
                                        const OutputTransform *transform,
                                        double voxelSizeXmm,
                                        double voxelSizeYmm) {
  const size_t expected = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  if (source.size() != expected)
    throw std::invalid_argument("prepareOutputVolume: source size mismatch");
  if (mask && mask->size() != expected)
    throw std::invalid_argument("prepareOutputVolume: mask size mismatch");

  std::vector<double> transformedSource;
  std::vector<uint8_t> transformedMask;
  const std::vector<double> *sourceForOutput = &source;
  const std::vector<uint8_t> *maskForOutput = mask;
  int transformedNz = nz;
  if (shouldFlipZPerBed(transform)) {
    transformedSource = flipDoubleZPerBed(source, nx, ny, nz, *transform);
    sourceForOutput = &transformedSource;
    if (mask) {
      transformedMask = flipMaskZPerBed(*mask, nx, ny, nz, *transform);
      maskForOutput = &transformedMask;
    }
  }
  if (shouldHardCutOutputOverlap(transform)) {
    if (transform->sigmoidBlendOutput) {
      transformedSource = sigmoidBlendOutputOverlap(
          *sourceForOutput, nx, ny, nz, *transform);
    } else {
      transformedSource = hardCutOutputOverlap(*sourceForOutput, nx, ny, nz,
                                               *transform);
    }
    sourceForOutput = &transformedSource;
    transformedNz = static_cast<int>(transformedSource.size() /
                                     (static_cast<size_t>(nx) * ny));
    if (maskForOutput) {
      transformedMask = transform->sigmoidBlendOutput
                            ? blendOutputMask(*maskForOutput, nx, ny, nz,
                                              *transform)
                            : hardCutOutputOverlap(*maskForOutput, nx, ny, nz,
                                                   *transform);
      maskForOutput = &transformedMask;
    }
  }
  if (transform && transform->objectAwareSeamPostprocess) {
    if (!transform->sigmoidBlendOutput || transform->bedOverlapInRecon)
      throw std::invalid_argument(
          "prepareOutputVolume: seam postprocess requires output sigmoid stitching");
    postprocessStitchedOutput(transformedSource, nx, ny, transformedNz,
                              *transform);
    sourceForOutput = &transformedSource;
  }
  if (shouldFlipZWholeOutput(transform)) {
    transformedSource =
        flipDoubleZWholeOutput(*sourceForOutput, nx, ny, transformedNz);
    sourceForOutput = &transformedSource;
    if (maskForOutput) {
      transformedMask =
          flipMaskZWholeOutput(*maskForOutput, nx, ny, transformedNz);
      maskForOutput = &transformedMask;
    }
  }
  if (shouldFlipXWholeOutput(transform)) {
    transformedSource = flipXWholeOutput(*sourceForOutput, nx, ny, transformedNz);
    sourceForOutput = &transformedSource;
    if (maskForOutput) {
      transformedMask = flipXWholeOutput(*maskForOutput, nx, ny, transformedNz);
      maskForOutput = &transformedMask;
    }
  }

  if (!isCropSettingsValid(transformedNz, crop) &&
      !(crop.start == 0 && crop.end == 0))
    throw std::invalid_argument("prepareOutputVolume: invalid crop range");
  const int croppedNz = transformedNz - crop.start - crop.end;
  if (outNz)
    *outNz = croppedNz;

  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> fovWeights;
  if (transform && transform->fovRadiusMm > 0.0) {
    if (voxelSizeXmm <= 0.0 || voxelSizeYmm <= 0.0)
      throw std::invalid_argument("prepareOutputVolume: voxel size must be positive for XY FOV");
    fovWeights.resize(plane);
    const double centerX = 0.5 * (nx - 1);
    const double centerY = 0.5 * (ny - 1);
    const double taperMm = 4.0 * std::min(voxelSizeXmm, voxelSizeYmm);
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const double dx = (x - centerX) * voxelSizeXmm;
        const double dy = (y - centerY) * voxelSizeYmm;
        fovWeights[static_cast<size_t>(y) * nx + x] = std::clamp(
            (transform->fovRadiusMm + taperMm - std::hypot(dx, dy)) /
                taperMm,
            0.0, 1.0);
      }
    }
  }
  std::vector<double> out(static_cast<size_t>(croppedNz) * plane, 0.0);
  for (int z = 0; z < croppedNz; ++z) {
    const int srcZ = z + crop.start;
    for (size_t xy = 0; xy < plane; ++xy) {
      const size_t srcIdx = static_cast<size_t>(srcZ) * plane + xy;
      const double value = (maskForOutput && !(*maskForOutput)[srcIdx])
                               ? 0.0
                               : (*sourceForOutput)[srcIdx];
      out[static_cast<size_t>(z) * plane + xy] =
          fovWeights.empty() ? value : value * fovWeights[xy];
    }
  }
  return out;
}

void saveVolumeOutput(const std::vector<double> &source,
                      const OutputGeometry &geometry,
                      const std::string &outputDir,
                      const std::string &outputPrefix, int iteration,
                      const std::vector<uint8_t> *mask,
                      const CropSettings &crop, bool littleEndian,
                      const OutputTransform *transform) {
  int outNz = 0;
  const std::vector<double> prepared = prepareOutputVolume(
      source, geometry.nx, geometry.ny, geometry.nz, mask, crop, &outNz,
      transform, geometry.vx, geometry.vy);
  io::saveMHDRAW(prepared, geometry.nx, geometry.ny, outNz, geometry.vx,
                 geometry.vy, geometry.vz, outputDir, outputPrefix, iteration,
                 littleEndian);
}

} // namespace main
} // namespace spect
