#include "pipeline/VolumeOutput.h"

#include "io/MHDWriter.h"
#include "pipeline/RunLogger.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
  return transform && transform->flipZPerBed && transform->baseNz > 0 &&
         !transform->bedZStarts.empty();
}

bool shouldFlipZWholeOutput(const OutputTransform *transform) {
  return transform && transform->flipZWholeOutput;
}

bool shouldFlipXWholeOutput(const OutputTransform *transform) {
  return transform && transform->flipXWholeOutput;
}

bool shouldHardCutOutputOverlap(const OutputTransform *transform) {
  return transform && transform->baseNz > 0 &&
         transform->bedZStarts.size() > 1;
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

double center66CosineFadeIn(int overlapIndex) {
  constexpr double pi = 3.14159265358979323846;
  const double phase = static_cast<double>(overlapIndex) /
                       static_cast<double>(kCenter66OverlapLayers - 1);
  const double sine = std::sin(0.5 * pi * phase);
  return sine * sine;
}

void validateCenter66CosineTransform(const OutputTransform &transform, int nz) {
  validateBedZFlipTransform(transform, nz);
  if (transform.baseNz < kCenter66CropLayers)
    throw std::invalid_argument(
        "prepareOutputVolume: center66_cosine10 requires baseNz >= 66");
  const std::vector<int> starts = outputBedStarts(transform);
  for (size_t bed = 1; bed < starts.size(); ++bed) {
    if (starts[bed] - starts[bed - 1] != kCenter66BedStepLayers)
      throw std::invalid_argument(
          "prepareOutputVolume: center66_cosine10 requires a 56-layer bed step");
  }
}

std::vector<double> center66CosineBlendOutputOverlap(
    const std::vector<double> &source, int nx, int ny, int nz,
    const OutputTransform &transform) {
  validateCenter66CosineTransform(transform, nz);
  const std::vector<int> starts = outputBedStarts(transform);
  const int cropStart = (transform.baseNz - kCenter66CropLayers) / 2;
  const int outputNz = starts.back() + kCenter66CropLayers;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> out(static_cast<size_t>(outputNz) * plane, 0.0);
  std::vector<double> weights(static_cast<size_t>(outputNz), 0.0);
  for (size_t bed = 0; bed < starts.size(); ++bed) {
    for (int localZ = 0; localZ < kCenter66CropLayers; ++localZ) {
      double weight = 1.0;
      if (bed > 0 && localZ < kCenter66OverlapLayers)
        weight *= center66CosineFadeIn(localZ);
      if (bed + 1 < starts.size() &&
          localZ >= kCenter66BedStepLayers)
        weight *= 1.0 -
                  center66CosineFadeIn(localZ - kCenter66BedStepLayers);
      const int srcZ = transform.bedZStarts[bed] + cropStart + localZ;
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

std::vector<uint8_t> center66BlendOutputMask(
    const std::vector<uint8_t> &source, int nx, int ny, int nz,
    const OutputTransform &transform) {
  validateCenter66CosineTransform(transform, nz);
  const std::vector<int> starts = outputBedStarts(transform);
  const int cropStart = (transform.baseNz - kCenter66CropLayers) / 2;
  const int outputNz = starts.back() + kCenter66CropLayers;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<uint8_t> out(static_cast<size_t>(outputNz) * plane, 0);
  for (size_t bed = 0; bed < starts.size(); ++bed) {
    for (int localZ = 0; localZ < kCenter66CropLayers; ++localZ) {
      const size_t srcOffset = static_cast<size_t>(
          transform.bedZStarts[bed] + cropStart + localZ) * plane;
      const size_t dstOffset =
          static_cast<size_t>(starts[bed] + localZ) * plane;
      for (size_t xy = 0; xy < plane; ++xy)
        out[dstOffset + xy] =
            std::max(out[dstOffset + xy], source[srcOffset + xy]);
    }
  }
  return out;
}

OutputTransform center66PostprocessTransform(
    const OutputTransform &transform) {
  OutputTransform adjusted = transform;
  adjusted.baseNz = kCenter66CropLayers;
  for (size_t bed = 0; bed < adjusted.bedZStarts.size(); ++bed)
    adjusted.bedZStarts[bed] =
        static_cast<int>(bed) * kCenter66CropLayers;
  adjusted.outputBedOverlaps.assign(
      adjusted.bedZStarts.size() - 1, kCenter66OverlapLayers);
  return adjusted;
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

std::vector<double> applyCircularFov(const std::vector<double> &source, int nx,
                                     int ny, int nz, double radiusMm,
                                     double voxelSizeXmm,
                                     double voxelSizeYmm) {
  if (voxelSizeXmm <= 0.0 || voxelSizeYmm <= 0.0)
    throw std::invalid_argument(
        "prepareOutputVolume: voxel size must be positive for XY FOV");
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> weights(plane);
  const double centerX = 0.5 * (nx - 1);
  const double centerY = 0.5 * (ny - 1);
  const double taperMm = 4.0 * std::min(voxelSizeXmm, voxelSizeYmm);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const double dx = (x - centerX) * voxelSizeXmm;
      const double dy = (y - centerY) * voxelSizeYmm;
      weights[static_cast<size_t>(y) * nx + x] = std::clamp(
          (radiusMm + taperMm - std::hypot(dx, dy)) / taperMm, 0.0, 1.0);
    }
  }
  std::vector<double> output(source);
  for (int z = 0; z < nz; ++z) {
    const size_t offset = static_cast<size_t>(z) * plane;
    for (size_t xy = 0; xy < plane; ++xy)
      output[offset + xy] *= weights[xy];
  }
  return output;
}

int reflect101(int index, int size) {
  if (size <= 1)
    return 0;
  while (index < 0 || index >= size) {
    if (index < 0)
      index = -index;
    else
      index = 2 * size - index - 2;
  }
  return index;
}

void gaussianBlurSigma1(const double *input, double *output, int ny, int nx) {
  constexpr int radius = 3;
  double kernel[2 * radius + 1];
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(-0.5 * i * i);
    sum += kernel[i + radius];
  }
  for (double &weight : kernel)
    weight /= sum;
  std::vector<double> horizontal(static_cast<size_t>(ny) * nx, 0.0);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      double value = 0.0;
      for (int dx = -radius; dx <= radius; ++dx)
        value += kernel[dx + radius] *
                 input[static_cast<size_t>(y) * nx +
                       reflect101(x + dx, nx)];
      horizontal[static_cast<size_t>(y) * nx + x] = value;
    }
  }
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      double value = 0.0;
      for (int dy = -radius; dy <= radius; ++dy)
        value += kernel[dy + radius] *
                 horizontal[static_cast<size_t>(reflect101(y + dy, ny)) * nx +
                            x];
      output[static_cast<size_t>(y) * nx + x] = value;
    }
  }
}

void keepLargest6ConnectedComponent(std::vector<uint8_t> &support,
                                    int nz, int ny, int nx) {
  const size_t plane = static_cast<size_t>(ny) * nx;
  std::vector<uint8_t> visited(support.size(), 0);
  std::vector<size_t> queue;
  std::vector<size_t> largest;
  for (size_t seed = 0; seed < support.size(); ++seed) {
    if (!support[seed] || visited[seed])
      continue;
    queue.clear();
    queue.push_back(seed);
    visited[seed] = 1;
    for (size_t head = 0; head < queue.size(); ++head) {
      const size_t index = queue[head];
      const int z = static_cast<int>(index / plane);
      const size_t xy = index % plane;
      const int y = static_cast<int>(xy / nx);
      const int x = static_cast<int>(xy % nx);
      const int dz[6] = {-1, 1, 0, 0, 0, 0};
      const int dy[6] = {0, 0, -1, 1, 0, 0};
      const int dx[6] = {0, 0, 0, 0, -1, 1};
      for (int direction = 0; direction < 6; ++direction) {
        const int zz = z + dz[direction];
        const int yy = y + dy[direction];
        const int xx = x + dx[direction];
        if (zz < 0 || zz >= nz || yy < 0 || yy >= ny || xx < 0 || xx >= nx)
          continue;
        const size_t next = static_cast<size_t>(zz) * plane +
                            static_cast<size_t>(yy) * nx + xx;
        if (support[next] && !visited[next]) {
          visited[next] = 1;
          queue.push_back(next);
        }
      }
    }
    if (queue.size() > largest.size())
      largest = queue;
  }
  std::fill(support.begin(), support.end(), 0);
  for (size_t index : largest)
    support[index] = 1;
}

double positivePercentile995(const std::vector<double> &volume) {
  std::vector<double> positive;
  positive.reserve(volume.size() / 8);
  for (double value : volume)
    if (value > 0.0)
      positive.push_back(value);
  if (positive.empty())
    return 0.0;
  std::sort(positive.begin(), positive.end());
  const double position = 0.995 * static_cast<double>(positive.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(position));
  const size_t hi = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lo);
  return positive[lo] * (1.0 - fraction) + positive[hi] * fraction;
}

std::vector<double> mouseConfidenceBlendOutputOverlap(
    const std::vector<double> &source, int nx, int ny, int nz,
    const OutputTransform &transform, double analysisRadiusMm,
    double voxelSizeXmm, double voxelSizeYmm) {
  std::vector<double> output =
      sigmoidBlendOutputOverlap(source, nx, ny, nz, transform);
  if (transform.mouseSeamStrength <= 0.0)
    return output;
  constexpr double lowFraction = 0.02;
  constexpr double ratioGate = 3.0;
  constexpr double boost = 2.0;
  const std::vector<double> supportSource =
      analysisRadiusMm > 0.0
          ? applyCircularFov(source, nx, ny, nz, analysisRadiusMm,
                             voxelSizeXmm, voxelSizeYmm)
          : source;
  const double scale = positivePercentile995(supportSource);
  if (scale <= 0.0)
    return output;
  const double low = lowFraction * scale;
  const double epsilon = std::max(low * 0.25, 1e-12);
  const size_t plane = static_cast<size_t>(nx) * ny;
  const std::vector<int> starts = outputBedStarts(transform);
  for (size_t bed = 0; bed + 1 < starts.size(); ++bed) {
    const int overlap = transform.outputBedOverlaps[bed];
    if (overlap <= 0)
      continue;
    const int stride = transform.baseNz - overlap;
    const int outputStart = starts[bed + 1];
    std::vector<double> blurredA(static_cast<size_t>(overlap) * plane);
    std::vector<double> blurredB(static_cast<size_t>(overlap) * plane);
    std::vector<uint8_t> support(static_cast<size_t>(overlap) * plane, 0);
    for (int z = 0; z < overlap; ++z) {
      const size_t sourceA = static_cast<size_t>(
          transform.bedZStarts[bed] + stride + z) * plane;
      const size_t sourceB = static_cast<size_t>(
          transform.bedZStarts[bed + 1] + z) * plane;
      const size_t local = static_cast<size_t>(z) * plane;
      gaussianBlurSigma1(supportSource.data() + sourceA,
                         blurredA.data() + local, ny, nx);
      gaussianBlurSigma1(supportSource.data() + sourceB,
                         blurredB.data() + local, ny, nx);
      for (size_t xy = 0; xy < plane; ++xy)
        support[local + xy] = blurredA[local + xy] >= low &&
                              blurredB[local + xy] >= low;
    }
    keepLargest6ConnectedComponent(support, overlap, ny, nx);
    size_t changed = 0;
    for (int z = 0; z < overlap; ++z) {
      const double weightB = sigmoidRamp(static_cast<double>(z) / overlap);
      const double weightA = 1.0 - weightB;
      const size_t sourceA = static_cast<size_t>(
          transform.bedZStarts[bed] + stride + z) * plane;
      const size_t sourceB = static_cast<size_t>(
          transform.bedZStarts[bed + 1] + z) * plane;
      const size_t local = static_cast<size_t>(z) * plane;
      const size_t target = static_cast<size_t>(outputStart + z) * plane;
      for (size_t xy = 0; xy < plane; ++xy) {
        if (!support[local + xy])
          continue;
        const double ratio = (blurredA[local + xy] + epsilon) /
                             (blurredB[local + xy] + epsilon);
        if (ratio < ratioGate && ratio > 1.0 / ratioGate)
          continue;
        const double adjustedA = weightA * (ratio >= ratioGate ? boost : 1.0);
        const double adjustedB =
            weightB * (ratio <= 1.0 / ratioGate ? boost : 1.0);
        const double adaptive =
            (adjustedA * source[sourceA + xy] +
             adjustedB * source[sourceB + xy]) /
            (adjustedA + adjustedB);
        const double baseline = output[target + xy];
        output[target + xy] =
            baseline + transform.mouseSeamStrength * (adaptive - baseline);
        ++changed;
      }
    }
    logInfo("Mouse seam fusion: seam_z=" +
            std::to_string(outputStart + overlap / 2) +
            ", changed_voxels=" + std::to_string(changed));
  }
  return output;
}

} // namespace


std::vector<double> prepareOutputVolume(const std::vector<double> &source, int nx,
                                        int ny, int nz,
                                        const std::vector<uint8_t> *mask,
                                        const CropSettings &crop, int *outNz,
                                        const OutputTransform *transform,
                                        double voxelSizeXmm,
                                        double voxelSizeYmm,
                                        double voxelSizeZmm) {
  const size_t expected = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
  if (source.size() != expected)
    throw std::invalid_argument("prepareOutputVolume: source size mismatch");
  if (mask && mask->size() != expected)
    throw std::invalid_argument("prepareOutputVolume: mask size mismatch");
  if (transform && transform->center66CosineOutput &&
      transform->mouseSeamStrength > 0.0)
    throw std::invalid_argument(
        "prepareOutputVolume: mouse_seam_strength is only supported by sigmoid stitching");

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
  std::vector<double> preStitchSource;
  if (shouldHardCutOutputOverlap(transform) && transform->sigmoidBlendOutput &&
      transform->objectAwareSeamPostprocess &&
      transform->mouseSeamStrength > 0.0) {
    preStitchSource = *sourceForOutput;
  }
  if (shouldHardCutOutputOverlap(transform)) {
    if (transform->center66CosineOutput) {
      transformedSource = center66CosineBlendOutputOverlap(
          *sourceForOutput, nx, ny, nz, *transform);
    } else if (transform->sigmoidBlendOutput) {
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
      if (transform->center66CosineOutput) {
        transformedMask = center66BlendOutputMask(
            *maskForOutput, nx, ny, nz, *transform);
      } else if (transform->sigmoidBlendOutput) {
        transformedMask = blendOutputMask(*maskForOutput, nx, ny, nz,
                                          *transform);
      } else {
        transformedMask = hardCutOutputOverlap(*maskForOutput, nx, ny, nz,
                                               *transform);
      }
      maskForOutput = &transformedMask;
    }
  }
  std::vector<double> analysisSource;
  const std::vector<double> *analysisForPostprocess = nullptr;
  OutputTransform frozenTransform;
  const OutputTransform *postprocessTransform = transform;
  if (transform && transform->center66CosineOutput) {
    frozenTransform = center66PostprocessTransform(*transform);
    postprocessTransform = &frozenTransform;
  }
  double analysisRadiusMm = 0.0;
  if (transform && transform->objectAwareSeamPostprocess) {
    analysisRadiusMm = transform->analysisFovRadiusMm > 0.0
                           ? transform->analysisFovRadiusMm
                           : transform->fovRadiusMm;
    if (analysisRadiusMm > 0.0 &&
        (analysisRadiusMm != transform->fovRadiusMm ||
         transform->mouseSeamStrength > 0.0)) {
      analysisSource = applyCircularFov(*sourceForOutput, nx, ny, transformedNz,
                                        analysisRadiusMm, voxelSizeXmm,
                                        voxelSizeYmm);
      analysisForPostprocess = &analysisSource;
    }
    if (transform->mouseSeamStrength > 0.0) {
      const std::vector<double> &analysis =
          analysisForPostprocess ? *analysisForPostprocess : *sourceForOutput;
      frozenTransform = *transform;
      if (frozenTransform.objectProfile == "auto")
        frozenTransform.objectProfile =
            classifyOutputObjectProfile(analysis, nx, ny, transformedNz);
      postprocessTransform = &frozenTransform;
      if (frozenTransform.objectProfile == "mouse") {
        transformedSource = mouseConfidenceBlendOutputOverlap(
            preStitchSource, nx, ny, nz, frozenTransform, analysisRadiusMm,
            voxelSizeXmm, voxelSizeYmm);
        sourceForOutput = &transformedSource;
      }
    }
  }
  if (transform && transform->fovRadiusMm > 0.0) {
    transformedSource = applyCircularFov(
        *sourceForOutput, nx, ny, transformedNz, transform->fovRadiusMm,
        voxelSizeXmm, voxelSizeYmm);
    sourceForOutput = &transformedSource;
  }
  if (transform && transform->derenzoAxialPostprocess) {
    if (sourceForOutput != &transformedSource)
      transformedSource = *sourceForOutput;
    postprocessDerenzoOutput(
        transformedSource, nx, ny, transformedNz, voxelSizeXmm,
        voxelSizeYmm, voxelSizeZmm, transform->derenzoAxialStrength);
    sourceForOutput = &transformedSource;
  }
  if (transform && transform->objectAwareSeamPostprocess) {
    if (!transform->sigmoidBlendOutput && !transform->center66CosineOutput)
      throw std::invalid_argument(
          "prepareOutputVolume: seam postprocess requires weighted output stitching");
    postprocessStitchedOutput(transformedSource, nx, ny, transformedNz,
                              *postprocessTransform, voxelSizeXmm, voxelSizeYmm,
                              analysisForPostprocess);
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
  if (outNz)
    *outNz = transformedNz;

  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> out(static_cast<size_t>(transformedNz) * plane, 0.0);
  for (int z = 0; z < transformedNz; ++z) {
    if (z < crop.start || z >= transformedNz - crop.end)
      continue;
    for (size_t xy = 0; xy < plane; ++xy) {
      const size_t srcIdx = static_cast<size_t>(z) * plane + xy;
      const double value = (maskForOutput && !(*maskForOutput)[srcIdx])
                               ? 0.0
                               : (*sourceForOutput)[srcIdx];
      out[static_cast<size_t>(z) * plane + xy] = value;
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
      transform, geometry.vx, geometry.vy, geometry.vz);
  io::saveMHDRAW(prepared, geometry.nx, geometry.ny, outNz, geometry.vx,
                 geometry.vy, geometry.vz, outputDir, outputPrefix, iteration,
                 littleEndian);
}

} // namespace main
} // namespace spect
