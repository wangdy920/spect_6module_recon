#include "postprocess/SeamPostprocess.h"

#include "pipeline/RunLogger.h"
#include "pipeline/VolumeOutput.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace spect {
namespace main {

namespace {

std::vector<int> seamOutputBedStarts(const OutputTransform &transform) {
  const size_t beds = transform.bedZStarts.size();
  if (transform.outputBedOverlaps.size() + 1 != beds)
    throw std::invalid_argument(
        "postprocessStitchedOutput: output overlap count does not match beds");
  std::vector<int> starts(beds, 0);
  for (size_t bed = 0; bed < beds; ++bed) {
    const int expectedStart = static_cast<int>(bed) * transform.baseNz;
    if (transform.bedZStarts[bed] != expectedStart)
      throw std::invalid_argument(
          "postprocessStitchedOutput: requires non-overlapping reconstruction beds");
    if (bed > 0) {
      const int overlap = transform.outputBedOverlaps[bed - 1];
      if (overlap < 0 || overlap >= transform.baseNz)
        throw std::invalid_argument(
            "postprocessStitchedOutput: invalid output overlap");
      starts[bed] = starts[bed - 1] + transform.baseNz - overlap;
    }
  }
  return starts;
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

struct ObjectClassification {
  std::string profile = "unknown";
  int projectionComponents = 0;
  double highFraction = 0.0;
  double p90Components = 0.0;
  double medianComponents = 0.0;
  double mainFraction = 0.0;
};

ObjectClassification classifyOutputObject(const std::vector<double> &volume,
                                           int nx, int ny, int nz) {
  ObjectClassification result;
  std::vector<double> positive;
  positive.reserve(volume.size() / 8);
  for (double value : volume) {
    if (value > 0.0)
      positive.push_back(value);
  }
  if (positive.empty())
    return result;
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
  const auto coronalSizes = componentSizes2D(coronal, nz, nx);
  result.projectionComponents = static_cast<int>(coronalSizes.size());
  result.highFraction = maskCount > 0
                            ? static_cast<double>(highCount) / maskCount
                            : 0.0;
  result.p90Components = percentile(componentCounts, 90.0);
  result.medianComponents = percentile(componentCounts, 50.0);
  if (!coronalSizes.empty()) {
    const double total = std::accumulate(coronalSizes.begin(),
                                         coronalSizes.end(), 0.0);
    const double largest = static_cast<double>(
        *std::max_element(coronalSizes.begin(), coronalSizes.end()));
    result.mainFraction = total > 0.0 ? largest / total : 0.0;
  }

  // ponytail: provisional gate from capillary, tube/ruanguan, EPguan, mouse,
  // and zimu. Bed-seam intensity drops can fragment an otherwise compact
  // coronal support (EPguan), so dominant-component phantoms are accepted
  // even when raw projection_components is inflated. Unknown remains a
  // no-op; explicit object_profile is the safety valve.
  const bool dominantCompact =
      result.highFraction >= 0.20 && result.mainFraction >= 0.95 &&
      result.medianComponents <= 2.0 && result.p90Components <= 5.0 &&
      result.projectionComponents >= 4;
  const bool diffuseIrregularMouse =
      result.highFraction < 0.20 && result.projectionComponents >= 2 &&
      result.p90Components >= 3.0;
  if (dominantCompact)
    result.profile = "epguan";
  else if (diffuseIrregularMouse ||
           (result.projectionComponents >= 12 &&
            result.highFraction <= 0.25 && result.p90Components >= 4.0))
    result.profile = "mouse";
  else if (result.projectionComponents >= 4 &&
           result.projectionComponents <= 12 && result.highFraction >= 0.30 &&
           result.p90Components <= 3.0)
    result.profile = "zimu";
  else if (result.projectionComponents <= 3 && result.highFraction >= 0.30 &&
           result.p90Components <= 3.0)
    result.profile = "phantom";
  return result;
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
               double threshold, size_t minSize = 8) {
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
      if (indices.size() < minSize)
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
      component.softMask = dilateMask(component.mask, ny, nx, 5);
      components.push_back(std::move(component));
    }
  }
  return components;
}

// Drop FOV-edge / grain speckles that are tiny versus the dominant support.
// Real multi-tube phantoms keep peers at similar area (tube/ruanguan ~1:1).
std::vector<SeamComponent>
filterSignificantSeamComponents(std::vector<SeamComponent> components) {
  if (components.size() <= 1)
    return components;
  std::vector<size_t> sizes(components.size(), 0);
  size_t largest = 0;
  for (size_t i = 0; i < components.size(); ++i) {
    sizes[i] = static_cast<size_t>(std::count(components[i].mask.begin(),
                                              components[i].mask.end(),
                                              static_cast<uint8_t>(1)));
    largest = std::max(largest, sizes[i]);
  }
  if (largest == 0)
    return components;
  const size_t minSize = std::max<size_t>(8, (largest + 9) / 10);
  std::vector<SeamComponent> kept;
  kept.reserve(components.size());
  for (size_t i = 0; i < components.size(); ++i) {
    if (sizes[i] >= minSize)
      kept.push_back(std::move(components[i]));
  }
  return kept;
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

std::vector<double> gaussianBlur2D(const std::vector<double> &input, int ny,
                                   int nx, double sigmaY, double sigmaX) {
  const auto makeKernel = [](double sigma) {
    if (sigma <= 0.0)
      return std::vector<double>{1.0};
    const int radius = std::max(1, static_cast<int>(std::ceil(4.0 * sigma)));
    std::vector<double> kernel(static_cast<size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
      const double value = std::exp(-0.5 * i * i / (sigma * sigma));
      kernel[static_cast<size_t>(i + radius)] = value;
      sum += value;
    }
    for (double &value : kernel)
      value /= sum;
    return kernel;
  };
  const auto ky = makeKernel(sigmaY);
  const auto kx = makeKernel(sigmaX);
  const int ry = static_cast<int>(ky.size() / 2);
  const int rx = static_cast<int>(kx.size() / 2);
  std::vector<double> horizontal(input.size(), 0.0);
  std::vector<double> output(input.size(), 0.0);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      double sum = 0.0;
      for (int dx = -rx; dx <= rx; ++dx) {
        const int xx = std::clamp(x + dx, 0, nx - 1);
        sum += kx[static_cast<size_t>(dx + rx)] *
               input[static_cast<size_t>(y) * nx + xx];
      }
      horizontal[static_cast<size_t>(y) * nx + x] = sum;
    }
  }
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      double sum = 0.0;
      for (int dy = -ry; dy <= ry; ++dy) {
        const int yy = std::clamp(y + dy, 0, ny - 1);
        sum += ky[static_cast<size_t>(dy + ry)] *
               horizontal[static_cast<size_t>(yy) * nx + x];
      }
      output[static_cast<size_t>(y) * nx + x] = sum;
    }
  }
  return output;
}

std::vector<uint8_t> binaryClose3(const std::vector<uint8_t> &input) {
  std::vector<uint8_t> dilated(input.size(), 0);
  std::vector<uint8_t> output(input.size(), 0);
  for (int i = 0; i < static_cast<int>(input.size()); ++i) {
    for (int delta = -1; delta <= 1; ++delta) {
      const int sample = i + delta;
      if (sample >= 0 && sample < static_cast<int>(input.size()) &&
          input[static_cast<size_t>(sample)]) {
        dilated[static_cast<size_t>(i)] = 1;
        break;
      }
    }
  }
  for (int i = 0; i < static_cast<int>(input.size()); ++i) {
    bool keep = true;
    for (int delta = -1; delta <= 1; ++delta) {
      const int sample = i + delta;
      if (sample < 0 || sample >= static_cast<int>(input.size()) ||
          !dilated[static_cast<size_t>(sample)]) {
        keep = false;
        break;
      }
    }
    output[static_cast<size_t>(i)] = keep ? 1 : 0;
  }
  return output;
}

std::pair<int, int> longestTrueRun(const std::vector<uint8_t> &mask) {
  int bestStart = -1;
  int bestStop = -1;
  int currentStart = -1;
  for (int i = 0; i <= static_cast<int>(mask.size()); ++i) {
    const bool enabled = i < static_cast<int>(mask.size()) &&
                         mask[static_cast<size_t>(i)] != 0;
    if (enabled && currentStart < 0)
      currentStart = i;
    if (!enabled && currentStart >= 0) {
      if (bestStart < 0 || i - currentStart > bestStop - bestStart) {
        bestStart = currentStart;
        bestStop = i;
      }
      currentStart = -1;
    }
  }
  return {bestStart, bestStop};
}

std::vector<double> rankFilter1D(const std::vector<double> &input, int length,
                                 bool takeMaximum) {
  if (length <= 1 || input.empty())
    return input;
  const int radius = length / 2;
  std::vector<double> output(input.size(), 0.0);
  for (int i = 0; i < static_cast<int>(input.size()); ++i) {
    double value = input[static_cast<size_t>(i)];
    for (int delta = -radius; delta <= radius; ++delta) {
      const int sample = std::clamp(i + delta, 0,
                                    static_cast<int>(input.size()) - 1);
      value = takeMaximum
                  ? std::max(value, input[static_cast<size_t>(sample)])
                  : std::min(value, input[static_cast<size_t>(sample)]);
    }
    output[static_cast<size_t>(i)] = value;
  }
  return output;
}

std::vector<double> greyCloseOpen1D(const std::vector<double> &input,
                                    int length) {
  auto output = rankFilter1D(input, length, true);
  output = rankFilter1D(output, length, false);
  output = rankFilter1D(output, length, false);
  return rankFilter1D(output, length, true);
}

std::vector<double> gaussianSmooth1D(const std::vector<double> &input,
                                     double sigma) {
  if (sigma <= 0.0 || input.empty())
    return input;
  const int radius = std::max(1, static_cast<int>(std::ceil(4.0 * sigma)));
  std::vector<double> kernel(static_cast<size_t>(2 * radius + 1));
  double kernelSum = 0.0;
  for (int delta = -radius; delta <= radius; ++delta) {
    const double value = std::exp(-0.5 * delta * delta / (sigma * sigma));
    kernel[static_cast<size_t>(delta + radius)] = value;
    kernelSum += value;
  }
  for (double &value : kernel)
    value /= kernelSum;
  std::vector<double> output(input.size(), 0.0);
  for (int i = 0; i < static_cast<int>(input.size()); ++i) {
    for (int delta = -radius; delta <= radius; ++delta) {
      const int sample = std::clamp(i + delta, 0,
                                    static_cast<int>(input.size()) - 1);
      output[static_cast<size_t>(i)] +=
          kernel[static_cast<size_t>(delta + radius)] *
          input[static_cast<size_t>(sample)];
    }
  }
  return output;
}

void applyEpguanBackgroundFilter(std::vector<double> &volume,
                                 const std::vector<double> &analysisVolume,
                                 int nx, int ny, int nz, double voxelSizeXmm,
                                 double voxelSizeYmm,
                                 const OutputTransform &transform) {
  if (!transform.epguanBackgroundFilter)
    return;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> positive;
  positive.reserve(volume.size() / 8);
  for (double value : analysisVolume) {
    if (value > 0.0)
      positive.push_back(value);
  }
  if (positive.empty())
    return;
  const double clip = percentile(positive, 99.5);
  std::vector<double> projection(plane, 0.0);
  std::vector<double> column(static_cast<size_t>(nz));
  for (size_t xy = 0; xy < plane; ++xy) {
    for (int z = 0; z < nz; ++z)
      column[static_cast<size_t>(z)] =
          std::min(analysisVolume[static_cast<size_t>(z) * plane + xy], clip);
    projection[xy] = percentile(column, 85.0);
  }
  constexpr double smoothMm = 2.0;
  const auto smooth = gaussianBlur2D(projection, ny, nx,
                                     smoothMm / voxelSizeYmm,
                                     smoothMm / voxelSizeXmm);
  const double reference = percentile(smooth, 99.5);
  if (reference <= 0.0)
    return;
  auto components = filterSignificantSeamComponents(seamComponents(
      smooth, ny, nx, transform.epguanSupportThreshold * reference));
  if (components.empty()) {
    logInfo("EPguan background filter: skipped reason=no_support");
    return;
  }
  std::vector<uint8_t> support(plane, 0);
  for (const auto &component : components) {
    for (size_t xy = 0; xy < plane; ++xy)
      support[xy] = support[xy] || component.mask[xy];
  }
  const auto distancePixels = euclideanDistanceTo(support, ny, nx, 1);
  const double pixelMm = std::min(voxelSizeXmm, voxelSizeYmm);
  const double marginMm = transform.epguanSupportMarginMm;
  size_t changedVoxels = 0;
  for (int z = 0; z < nz; ++z) {
    const size_t offset = static_cast<size_t>(z) * plane;
    for (size_t xy = 0; xy < plane; ++xy) {
      const double distanceMm = distancePixels[xy] * pixelMm;
      double weight = 1.0;
      if (!support[xy] && distanceMm > marginMm) {
        const double feather = marginMm > 0.0
                                   ? std::clamp(2.0 - distanceMm / marginMm,
                                                0.0, 1.0)
                                   : 0.0;
        weight = transform.epguanBackgroundRetain +
                 (1.0 - transform.epguanBackgroundRetain) * feather;
      }
      if (weight < 1.0 && volume[offset + xy] != 0.0) {
        volume[offset + xy] *= weight;
        ++changedVoxels;
      }
    }
  }
  logInfo("EPguan background filter: components=" +
          std::to_string(components.size()) + ", changed_voxels=" +
          std::to_string(changedVoxels));
}

void applyPhantomContourInterpolation(std::vector<double> &volume,
                                      const std::vector<double> &analysisVolume,
                                      int nx, int ny, int nz, double threshold,
                                      const OutputTransform &transform) {
  constexpr int anchorWidth = 3;
  constexpr double contourMargin = 3.0;
  const size_t planeSize = static_cast<size_t>(nx) * ny;
  const std::vector<double> input = volume;
  const std::vector<double> &analysisInput = analysisVolume;
  const std::vector<int> starts = seamOutputBedStarts(transform);
  for (size_t bed = 1; bed < starts.size(); ++bed) {
    const int overlap = transform.outputBedOverlaps[bed - 1];
    if (overlap <= 0)
      continue;
    const int seam = starts[bed] + overlap / 2;
    const int halfWidth = std::min(transform.seamHalfWidth, overlap / 2);
    const int z0 = seam - halfWidth;
    const int z1 = seam + halfWidth + 1;
    if (z0 - anchorWidth < 0 || z1 + anchorWidth > nz) {
      logInfo("Seam postprocess: seam_z=" + std::to_string(seam) +
              " skipped reason=missing_endpoint");
      continue;
    }
    std::vector<double> left(planeSize, 0.0), right(planeSize, 0.0);
    for (int dz = 0; dz < anchorWidth; ++dz) {
      const size_t leftOffset = static_cast<size_t>(z0 - anchorWidth + dz) * planeSize;
      const size_t rightOffset = static_cast<size_t>(z1 + dz) * planeSize;
      for (size_t xy = 0; xy < planeSize; ++xy) {
        left[xy] += analysisInput[leftOffset + xy] / anchorWidth;
        right[xy] += analysisInput[rightOffset + xy] / anchorWidth;
      }
    }
    auto leftComponents = filterSignificantSeamComponents(
        seamComponents(left, ny, nx, threshold));
    auto rightComponents = filterSignificantSeamComponents(
        seamComponents(right, ny, nx, threshold));
    if (leftComponents.empty() || leftComponents.size() != rightComponents.size() ||
        leftComponents.size() > 3) {
      logInfo("Seam postprocess: seam_z=" + std::to_string(seam) +
              " skipped reason=component_gate left_components=" +
              std::to_string(leftComponents.size()) + " right_components=" +
              std::to_string(rightComponents.size()));
      continue;
    }

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
      pair.leftDistance = signedEuclideanDistance(pair.left->softMask, ny, nx);
      pair.rightDistance = signedEuclideanDistance(pair.right->softMask, ny, nx);
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
            candidate[offset + xy] = std::max(
                candidate[offset + xy],
                ((1.0 - fraction) * imageLeft + fraction * imageRight) * gate);
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
    if (!topologyValid) {
      logInfo("Seam postprocess: seam_z=" + std::to_string(seam) +
              " skipped reason=topology_gate components=" +
              std::to_string(pairs.size()));
      continue;
    }

    // C4: near-wall never digs below original (kills residual outer notches).
    constexpr double wallInnerPx = 3.0;
    constexpr double wallOuterPx = 2.0;
    size_t changedVoxels = 0;
    for (int index = 0; index < z1 - z0; ++index) {
      const int z = z0 + index;
      double axialBlend = 1.0;
      if (transform.objectProfile == "epguan" &&
          transform.epguanSeamFeatherSlices > 0) {
        const int edgeDistance =
            std::min(index, z1 - z0 - 1 - index);
        const double phase = std::clamp(
            static_cast<double>(edgeDistance) /
                transform.epguanSeamFeatherSlices,
            0.0, 1.0);
        axialBlend = transform.epguanSeamEdgeStrength +
                     (1.0 - transform.epguanSeamEdgeStrength) * phase;
      }
      const size_t inputOffset = static_cast<size_t>(z) * planeSize;
      const size_t candidateOffset = static_cast<size_t>(index) * planeSize;
      std::vector<uint8_t> support(planeSize, 0);
      for (size_t xy = 0; xy < planeSize; ++xy)
        support[xy] = candidate[candidateOffset + xy] > 0.05 * threshold;
      const auto distIn = euclideanDistanceTo(support, ny, nx, 0);
      const auto distOut = euclideanDistanceTo(support, ny, nx, 1);
      for (size_t xy = 0; xy < planeSize; ++xy) {
        // Preserve exterior halo/scatter: never blend toward candidate zeros
        // over positive original voxels (avoids rectangular vacuum bands at
        // seams outside the tube). Only adopt positive candidate samples.
        const double interpolated = candidate[candidateOffset + xy];
        if (interpolated <= 0.0)
          continue;
        double updated =
            input[inputOffset + xy] +
            transform.phantomCloseStrength * axialBlend *
                (interpolated - input[inputOffset + xy]);
        const bool wall =
            support[xy] != 0 && distIn[xy] <= wallInnerPx;
        const bool outer = support[xy] == 0 && distOut[xy] <= wallOuterPx &&
                           input[inputOffset + xy] > 0.0;
        if ((wall || outer) &&
            (transform.objectProfile != "epguan" ||
             transform.epguanWallMaxPreserve))
          updated = std::max(updated, input[inputOffset + xy]);
        changedVoxels += updated != input[inputOffset + xy];
        volume[inputOffset + xy] = updated;
      }
    }
    logInfo("Seam postprocess: seam_z=" + std::to_string(seam) +
            " processed components=" + std::to_string(pairs.size()) +
            " changed_voxels=" + std::to_string(changedVoxels));
  }
}

void applyZimuTopologyLockedSmooth(std::vector<double> &volume,
                                   const std::vector<double> &analysisVolume,
                                   int nx, int ny, int nz, double threshold,
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
  const std::vector<int> starts = seamOutputBedStarts(transform);
  for (size_t bed = 1; bed < starts.size(); ++bed) {
    const int overlap = transform.outputBedOverlaps[bed - 1];
    if (overlap <= 0)
      continue;
    const int seam = starts[bed] + overlap / 2;
    const int halfWidth =
        std::min({transform.seamHalfWidth, 4, overlap / 2});
    const int z0 = std::max(0, seam - halfWidth);
    const int z1 = std::min(nz, seam + halfWidth + 1);
    std::vector<double> candidate(static_cast<size_t>(z1 - z0) * planeSize);
    double beforeSum = 0.0;
    double afterSum = 0.0;
    size_t changedVoxels = 0;
    for (int z = z0; z < z1; ++z) {
      const size_t inputOffset = static_cast<size_t>(z) * planeSize;
      const size_t candidateOffset = static_cast<size_t>(z - z0) * planeSize;
      for (size_t xy = 0; xy < planeSize; ++xy) {
        const double original = input[inputOffset + xy];
        candidate[candidateOffset + xy] = original;
        if (analysisVolume[inputOffset + xy] < threshold)
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
        const double updated = analysisVolume[inputOffset + xy] >= threshold
                                   ? std::max(threshold,
                                              candidate[candidateOffset + xy] * scale)
                                   : original;
        changedVoxels += updated != original;
        volume[inputOffset + xy] = updated;
      }
    }
    logInfo("Seam postprocess: seam_z=" + std::to_string(seam) +
            " processed mode=topology_locked changed_voxels=" +
            std::to_string(changedVoxels));
  }
}

void applyObjectAwareSeamPostprocessImpl(std::vector<double> &volume, int nx,
                                         int ny, int nz,
                                         const OutputTransform &transform,
                                         double voxelSizeXmm,
                                         double voxelSizeYmm,
                                         const std::vector<double> &analysisVolume) {
  ObjectClassification classification;
  if (transform.objectProfile == "auto")
    classification = classifyOutputObject(analysisVolume, nx, ny, nz);
  else
    classification.profile = transform.objectProfile;
  const std::string &profile = classification.profile;
  std::ostringstream classifierLog;
  classifierLog << "Seam classifier: selected_profile=" << profile;
  if (transform.objectProfile == "auto")
    classifierLog << ", projection_components="
                  << classification.projectionComponents
                  << ", high_fraction=" << classification.highFraction
                  << ", p90_components=" << classification.p90Components
                  << ", median_components=" << classification.medianComponents
                  << ", main_fraction=" << classification.mainFraction;
  else
    classifierLog << ", source=explicit";
  logInfo(classifierLog.str());
  const bool epguanNoop =
      profile == "epguan" && transform.epguanCloseStrength == 0.0 &&
      !transform.epguanBackgroundFilter;
  if ((profile == "phantom" && transform.phantomCloseStrength == 0.0) ||
      epguanNoop ||
      (profile == "zimu" && transform.zimuCloseStrength == 0.0) ||
      (profile != "phantom" && profile != "epguan" && profile != "zimu")) {
    logInfo("Seam postprocess: skipped reason=profile_noop");
    return;
  }
  std::vector<double> positive;
  positive.reserve(volume.size() / 8);
  for (double value : analysisVolume) {
    if (value > 0.0)
      positive.push_back(value);
  }
  if (positive.empty())
    return;
  const double threshold = 0.10 * percentile(positive, 99.5);
  const std::vector<double> before = volume;
  if (profile == "phantom") {
    applyPhantomContourInterpolation(volume, analysisVolume, nx, ny, nz,
                                     threshold, transform);
  } else if (profile == "epguan") {
    if (transform.epguanCloseStrength > 0.0) {
      OutputTransform epguanTransform = transform;
      epguanTransform.objectProfile = "epguan";
      epguanTransform.phantomCloseStrength = transform.epguanCloseStrength;
      applyPhantomContourInterpolation(volume, analysisVolume, nx, ny, nz, threshold,
                                       epguanTransform);
    }
    applyEpguanBackgroundFilter(volume, analysisVolume, nx, ny, nz, voxelSizeXmm,
                                voxelSizeYmm, transform);
  } else {
    applyZimuTopologyLockedSmooth(volume, analysisVolume, nx, ny, nz,
                                  threshold, transform);
  }
  size_t changedVoxels = 0;
  size_t changedSlices = 0;
  double maxAbs = 0.0;
  const size_t planeSize = static_cast<size_t>(nx) * ny;
  for (int z = 0; z < nz; ++z) {
    bool sliceChanged = false;
    for (size_t xy = 0; xy < planeSize; ++xy) {
      const size_t index = static_cast<size_t>(z) * planeSize + xy;
      const double difference = std::abs(volume[index] - before[index]);
      if (difference > 0.0) {
        ++changedVoxels;
        sliceChanged = true;
        maxAbs = std::max(maxAbs, difference);
      }
    }
    changedSlices += sliceChanged;
  }
  std::ostringstream summary;
  summary << "Seam postprocess summary: total_changed_slices=" << changedSlices
          << ", changed_voxels=" << changedVoxels
          << ", input_output_max_abs=" << maxAbs;
  logInfo(summary.str());
}


} // namespace

std::string classifyOutputObjectProfile(const std::vector<double> &volume,
                                        int nx, int ny, int nz) {
  const size_t expected = static_cast<size_t>(nx) * ny * nz;
  if (nx <= 0 || ny <= 0 || nz <= 0 || volume.size() != expected)
    throw std::invalid_argument(
        "classifyOutputObjectProfile: invalid volume dimensions");
  return classifyOutputObject(volume, nx, ny, nz).profile;
}

void postprocessDerenzoOutput(std::vector<double> &volume, int nx, int ny,
                             int nz, double voxelSizeXmm, double voxelSizeYmm,
                             double voxelSizeZmm, double axialStrength) {
  const size_t plane = static_cast<size_t>(nx) * ny;
  const size_t expected = plane * static_cast<size_t>(nz);
  if (nx <= 0 || ny <= 0 || nz <= 0 || volume.size() != expected)
    throw std::invalid_argument(
        "postprocessDerenzoOutput: invalid volume dimensions");
  if (voxelSizeXmm <= 0.0 || voxelSizeYmm <= 0.0 || voxelSizeZmm <= 0.0)
    throw std::invalid_argument(
        "postprocessDerenzoOutput: voxel sizes must be positive");
  if (!std::isfinite(axialStrength) || axialStrength < 0.0 ||
      axialStrength > 1.0)
    throw std::invalid_argument(
        "postprocessDerenzoOutput: axial strength must be in [0,1]");
  if (axialStrength == 0.0) {
    logInfo("Derenzo axial postprocess: skipped reason=strength_zero");
    return;
  }

  std::vector<double> positive;
  positive.reserve(volume.size() / 8);
  for (double value : volume) {
    if (value > 0.0)
      positive.push_back(value);
  }
  if (positive.empty()) {
    logInfo("Derenzo axial postprocess: skipped reason=no_positive_signal");
    return;
  }

  const double clipValue = percentile(positive, 99.5);
  std::vector<double> projection(plane, 0.0);
  std::vector<double> column(static_cast<size_t>(nz));
  for (size_t xy = 0; xy < plane; ++xy) {
    for (int z = 0; z < nz; ++z)
      column[static_cast<size_t>(z)] = std::min(
          volume[static_cast<size_t>(z) * plane + xy], clipValue);
    projection[xy] = percentile(column, 80.0);
  }
  const auto lowpass = gaussianBlur2D(
      projection, ny, nx, 2.0 / voxelSizeYmm, 2.0 / voxelSizeXmm);
  std::vector<double> highpass(plane, 0.0);
  for (size_t xy = 0; xy < plane; ++xy)
    highpass[xy] = std::max(0.0, projection[xy] - lowpass[xy]);
  const auto highpassSmooth = gaussianBlur2D(
      highpass, ny, nx, 0.25 / voxelSizeYmm, 0.25 / voxelSizeXmm);
  std::vector<double> highpassPositive;
  highpassPositive.reserve(plane / 4);
  for (double value : highpassSmooth) {
    if (value > 0.0)
      highpassPositive.push_back(value);
  }
  if (highpassPositive.empty()) {
    logInfo("Derenzo axial postprocess: skipped reason=no_highpass_signal");
    return;
  }

  constexpr double rodThresholdFraction = 0.40;
  const double highpassReference = percentile(highpassPositive, 99.0);
  const auto rods = seamComponents(highpassSmooth, ny, nx,
                                   rodThresholdFraction * highpassReference,
                                   4);
  int morphLength =
      std::max(1, static_cast<int>(std::round(1.50 / voxelSizeZmm)));
  if (morphLength % 2 == 0)
    ++morphLength;
  constexpr double continuitySigmaMm = 0.25;
  constexpr double bodyThresholdFraction = 0.15;
  constexpr double coreThresholdFraction = 0.60;
  constexpr double minCoreToBodyRatio = 0.50;

  size_t processedRods = 0;
  size_t changedVoxels = 0;
  double addedSignal = 0.0;
  double maxAbs = 0.0;
  int shortestSpan = nz + 1;
  int longestSpan = 0;
  for (const auto &rod : rods) {
    double weightSum = 0.0;
    for (size_t xy = 0; xy < plane; ++xy) {
      if (rod.mask[xy])
        weightSum += highpassSmooth[xy];
    }
    if (weightSum <= 0.0)
      continue;
    std::vector<double> profile(static_cast<size_t>(nz), 0.0);
    for (int z = 0; z < nz; ++z) {
      const size_t offset = static_cast<size_t>(z) * plane;
      double weighted = 0.0;
      for (size_t xy = 0; xy < plane; ++xy) {
        if (rod.mask[xy])
          weighted += volume[offset + xy] * highpassSmooth[xy];
      }
      profile[static_cast<size_t>(z)] = weighted / weightSum;
    }
    const double baseline = percentile(profile, 20.0);
    std::vector<double> signal(profile.size(), 0.0);
    double peak = 0.0;
    for (size_t z = 0; z < profile.size(); ++z) {
      signal[z] = std::max(0.0, profile[z] - baseline);
      peak = std::max(peak, signal[z]);
    }
    if (peak <= 0.0)
      continue;
    std::vector<uint8_t> body(static_cast<size_t>(nz), 0);
    std::vector<uint8_t> core(static_cast<size_t>(nz), 0);
    for (int z = 0; z < nz; ++z) {
      const double normalized = signal[static_cast<size_t>(z)] / peak;
      body[static_cast<size_t>(z)] =
          normalized >= bodyThresholdFraction ? 1 : 0;
      core[static_cast<size_t>(z)] =
          normalized >= coreThresholdFraction ? 1 : 0;
    }
    const auto bodyRun = longestTrueRun(binaryClose3(body));
    const auto coreRun = longestTrueRun(binaryClose3(core));
    if (bodyRun.first < 0)
      continue;
    auto trustedRun = bodyRun;
    if (coreRun.first >= 0) {
      const int bodyLength = std::max(1, bodyRun.second - bodyRun.first);
      const int coreLength = coreRun.second - coreRun.first;
      if (static_cast<double>(coreLength) / bodyLength >=
          minCoreToBodyRatio)
        trustedRun = coreRun;
    }
    const int start = std::clamp(trustedRun.first, 0, nz);
    const int stop = std::clamp(trustedRun.second, start, nz);
    if (stop - start < 2)
      continue;
    ++processedRods;
    shortestSpan = std::min(shortestSpan, stop - start);
    longestSpan = std::max(longestSpan, stop - start);

    for (size_t xy = 0; xy < plane; ++xy) {
      if (!rod.mask[xy])
        continue;
      std::vector<double> source(static_cast<size_t>(stop - start));
      for (int z = start; z < stop; ++z)
        source[static_cast<size_t>(z - start)] =
            volume[static_cast<size_t>(z) * plane + xy];
      auto target = greyCloseOpen1D(source, morphLength);
      target = gaussianSmooth1D(target,
                                continuitySigmaMm / voxelSizeZmm);
      for (int z = start; z < stop; ++z) {
        const size_t local = static_cast<size_t>(z - start);
        const double proposed = source[local] +
                                axialStrength *
                                    (target[local] - source[local]);
        const double updated = std::max(source[local], proposed);
        const double difference = updated - source[local];
        if (difference <= 0.0)
          continue;
        volume[static_cast<size_t>(z) * plane + xy] = updated;
        ++changedVoxels;
        addedSignal += difference;
        maxAbs = std::max(maxAbs, difference);
      }
    }
  }

  std::ostringstream summary;
  summary << "Derenzo axial postprocess: detected_rods=" << rods.size()
          << ", processed_rods=" << processedRods
          << ", changed_voxels=" << changedVoxels
          << ", added_signal=" << addedSignal << ", max_abs=" << maxAbs
          << ", support_extension_voxels=0";
  if (processedRods > 0)
    summary << ", trusted_span_voxels=" << shortestSpan << ".."
            << longestSpan;
  logInfo(summary.str());
}

void postprocessStitchedOutput(std::vector<double> &volume, int nx, int ny,
                               int nz, const OutputTransform &transform,
                               double voxelSizeXmm, double voxelSizeYmm,
                               const std::vector<double> *analysisVolume) {
  const size_t expected = static_cast<size_t>(nx) * ny * nz;
  if (nx <= 0 || ny <= 0 || nz <= 0 || volume.size() != expected)
    throw std::invalid_argument(
        "postprocessStitchedOutput: invalid volume dimensions");
  if (voxelSizeXmm <= 0.0 || voxelSizeYmm <= 0.0)
    throw std::invalid_argument(
        "postprocessStitchedOutput: voxel sizes must be positive");
  if (analysisVolume && analysisVolume->size() != expected)
    throw std::invalid_argument(
        "postprocessStitchedOutput: analysis volume size mismatch");
  const std::vector<double> &analysis =
      analysisVolume ? *analysisVolume : volume;
  applyObjectAwareSeamPostprocessImpl(volume, nx, ny, nz, transform,
                                      voxelSizeXmm, voxelSizeYmm, analysis);
}

} // namespace main
} // namespace spect
