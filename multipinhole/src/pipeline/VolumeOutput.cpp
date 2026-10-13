#include "pipeline/VolumeOutput.h"

#include "filters/ButterworthFilter.h"
#include "filters/Convolution.h"
#include "filters/cuda/VolumeOpsCUDA.h"
#include "io/MHDWriter.h"
#include "pipeline/RunLogger.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

CanonicalOutputGeometry computeCanonicalOutputGeometry(
    int baseNz, const std::vector<int>& outputBedOverlaps,
    double voxelSizeZmm, double bedStepMm) {
  if (bedStepMm <= 0.0)
    bedStepMm = spect::core::sixmodule::kMultiBedStepMm;
  constexpr int kBlendMargin =
      spect::core::sixmodule::kDefaultBlendMarginLayers;
  const double kTolMm = 1e-4;

  if (outputBedOverlaps.empty())
    throw std::invalid_argument(
        "computeCanonicalOutputGeometry: outputBedOverlaps is empty");
  if (baseNz <= 0)
    throw std::invalid_argument(
        "computeCanonicalOutputGeometry: baseNz must be positive");
  if (voxelSizeZmm <= 0.0)
    throw std::invalid_argument(
        "computeCanonicalOutputGeometry: voxelSizeZmm must be positive");

  // Expected step in layers: round the physical scanner step to the nearest
  // integer number of voxels (prototype uses llround(bed_step_mm/voxel_mm)).
  const int expectedStepLayers =
      static_cast<int>(std::llround(bedStepMm / voxelSizeZmm));
  if (expectedStepLayers <= 0)
    throw std::invalid_argument(
        "computeCanonicalOutputGeometry: bed step " +
        std::to_string(bedStepMm) + " mm is smaller than one voxel (" +
        std::to_string(voxelSizeZmm) + " mm).");

  const int stepLayers = baseNz - outputBedOverlaps[0];
  if (stepLayers <= 0)
    throw std::invalid_argument(
        "computeCanonicalOutputGeometry: bed_overlap must be less than baseNz; "
        "got step=" + std::to_string(stepLayers) + " from baseNz=" +
        std::to_string(baseNz) + " and overlap=" +
        std::to_string(outputBedOverlaps[0]));
  if (stepLayers >= baseNz)
    throw std::invalid_argument(
        "computeCanonicalOutputGeometry: bed_overlap must be positive; "
        "effective step=" + std::to_string(stepLayers) +
        " is not < baseNz=" + std::to_string(baseNz));

  for (size_t i = 1; i < outputBedOverlaps.size(); ++i) {
    if (baseNz - outputBedOverlaps[i] != stepLayers)
      throw std::invalid_argument(
          "computeCanonicalOutputGeometry: every bed_overlap must produce "
          "the same bed step. Joint 0 gives step=" +
          std::to_string(stepLayers) + ", joint " + std::to_string(i) +
          " gives step=" +
          std::to_string(baseNz - outputBedOverlaps[i]) + ".");
  }

  // The overlap-derived step must match the configured scanner step within
  // half-layer quantization (a non-integer layer ratio rounds to the nearest
  // layer) plus a small absolute tolerance.
  const double stepMm = stepLayers * voxelSizeZmm;
  const double layerQuantTol = 0.5 * voxelSizeZmm + kTolMm;
  if (stepLayers != expectedStepLayers ||
      std::abs(stepMm - bedStepMm) > layerQuantTol)
    throw std::invalid_argument(
        "computeCanonicalOutputGeometry: effective bed step " +
        std::to_string(stepMm) + " mm (" + std::to_string(stepLayers) +
        " layers at voxel_z=" + std::to_string(voxelSizeZmm) +
        " mm) deviates from the scanner bed step " +
        std::to_string(bedStepMm) + " mm. Set -bedStepMm / [MultiBed] "
        "bed_step_mm to the actual scanner step, or pass bed_overlap = "
        "baseNz - round(bed_step_mm / voxel_z).");

  const int cropLayers =
      std::min(stepLayers + kBlendMargin, baseNz);

  CanonicalOutputGeometry geom;
  geom.stepLayers = stepLayers;
  geom.cropLayers = cropLayers;
  geom.blendOverlap = cropLayers - stepLayers;
  return geom;
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

bool shouldStitchCanonicalOutput(const OutputTransform *transform) {
  // Matrix-grid beds are reconstructed jointly; the volume is already global.
  return transform && !transform->matrixGridOutput && transform->baseNz > 0 &&
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

template <typename T>
std::vector<T> centerCropVolume(const std::vector<T> &source,
                                 int &nx, int &ny, int &nz,
                                 int requestedNx, int requestedNy,
                                 int requestedNz) {
  const int targetNx = requestedNx > 0 ? requestedNx : nx;
  const int targetNy = requestedNy > 0 ? requestedNy : ny;
  const int targetNz = requestedNz > 0 ? requestedNz : nz;
  if (targetNx > nx || targetNy > ny || targetNz > nz || targetNx <= 0 ||
      targetNy <= 0 || targetNz <= 0) {
    throw std::invalid_argument(
        "Fixed 6module output mask exceeds reconstructed geometry.");
  }
  if (targetNx == nx && targetNy == ny && targetNz == nz)
    return source;

  const int x0 = (nx - targetNx) / 2;
  const int y0 = (ny - targetNy) / 2;
  const int z0 = (nz - targetNz) / 2;
  const size_t sourcePlane = static_cast<size_t>(nx) * ny;
  const size_t targetPlane = static_cast<size_t>(targetNx) * targetNy;
  std::vector<T> cropped(
      targetPlane * static_cast<size_t>(targetNz), T{});
  for (int z = 0; z < targetNz; ++z) {
    for (int y = 0; y < targetNy; ++y) {
      const size_t sourceOffset =
          static_cast<size_t>(z + z0) * sourcePlane +
          static_cast<size_t>(y + y0) * nx + x0;
      const size_t targetOffset =
          static_cast<size_t>(z) * targetPlane +
          static_cast<size_t>(y) * targetNx;
      std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(sourceOffset),
                  targetNx,
                  cropped.begin() + static_cast<std::ptrdiff_t>(targetOffset));
    }
  }
  nx = targetNx;
  ny = targetNy;
  nz = targetNz;
  return cropped;
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

double center66CosineFadeIn(int overlapIndex, int blendOverlap) {
  if (blendOverlap == 1) return 0.5;
  constexpr double pi = 3.14159265358979323846;
  const double phase =
      static_cast<double>(overlapIndex) / static_cast<double>(blendOverlap - 1);
  const double sine = std::sin(0.5 * pi * phase);
  return sine * sine;
}

void validateCenter66CosineTransform(const OutputTransform &transform, int nz) {
  validateBedZFlipTransform(transform, nz);
  const auto& geom = transform.canonicalGeom;
  if (transform.baseNz < geom.cropLayers)
    throw std::invalid_argument(
        "prepareOutputVolume: canonical stitch requires baseNz >= " +
        std::to_string(geom.cropLayers));
  const std::vector<int> starts = outputBedStarts(transform);
  for (size_t bed = 1; bed < starts.size(); ++bed) {
    if (starts[bed] - starts[bed - 1] != geom.stepLayers)
      throw std::invalid_argument(
          "prepareOutputVolume: canonical stitch requires a " +
          std::to_string(geom.stepLayers) + "-layer bed step");
  }
}

double percentile99(std::vector<double> values) {
  if (values.empty())
    return 0.0;
  const size_t index = static_cast<size_t>(
      0.99 * static_cast<double>(values.size() - 1));
  std::nth_element(values.begin(),
                   values.begin() + static_cast<std::ptrdiff_t>(index),
                   values.end());
  return values[index];
}

double median(std::vector<double> values) {
  if (values.empty())
    throw std::runtime_error("center66 gain fit received an empty sample");
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(),
                   values.begin() + static_cast<std::ptrdiff_t>(middle),
                   values.end());
  double result = values[middle];
  if ((values.size() & 1u) == 0u) {
    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(middle - 1),
        values.end());
    result = 0.5 * (result + values[middle - 1]);
  }
  return result;
}

double center66PairLogRatio(const std::vector<double> &source, int nx, int ny,
                            const OutputTransform &transform, int cropStart,
                            size_t lowBed, size_t highBed,
                            double voxelSizeXmm, double voxelSizeYmm,
                            size_t &sampleCount) {
  const auto& geom = transform.canonicalGeom;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const double radius =
      0.5 * spect::core::sixmodule::kMultiBedGainRoiDiameterMm;
  const double radiusSquared = radius * radius;
  const int lowStart = transform.bedZStarts[lowBed] + cropStart +
      (transform.reverseLocalZInReconStorage ? 0 : geom.stepLayers);
  const int highStart = transform.bedZStarts[highBed] + cropStart +
      (transform.reverseLocalZInReconStorage ? geom.stepLayers : 0);
  const int cropNx = transform.centerCropNx > 0 ? transform.centerCropNx : nx;
  const int cropNy = transform.centerCropNy > 0 ? transform.centerCropNy : ny;
  if (cropNx > nx || cropNy > ny)
    throw std::invalid_argument(
        "center66 gain fit XY crop exceeds reconstruction geometry");
  const int cropX0 = (nx - cropNx) / 2;
  const int cropY0 = (ny - cropNy) / 2;

  std::vector<double> joint;
  joint.reserve(plane * static_cast<size_t>(geom.blendOverlap));
  for (int z = 0; z < geom.blendOverlap; ++z) {
    for (int y = 0; y < ny; ++y) {
      const double yy =
          (static_cast<double>(y) - 0.5 * static_cast<double>(ny - 1)) *
          voxelSizeYmm;
      for (int x = 0; x < nx; ++x) {
        const double xx =
            (static_cast<double>(x) - 0.5 * static_cast<double>(nx - 1)) *
            voxelSizeXmm;
        if (xx * xx + yy * yy > radiusSquared)
          continue;
        const size_t xy = static_cast<size_t>(y) * nx + x;
        const double low =
            source[static_cast<size_t>(lowStart + z) * plane + xy];
        const double high =
            source[static_cast<size_t>(highStart + z) * plane + xy];
        if (std::isfinite(low) && std::isfinite(high) &&
            std::max(low, high) > 0.0) {
          joint.push_back(std::max(low, high));
        }
      }
    }
  }
  if (joint.empty())
    throw std::runtime_error(
        "center66 gain fit found no positive signal in an adjacent overlap");

  const double threshold = 0.10 * percentile99(joint);
  std::vector<double> ratios;
  ratios.reserve(joint.size());
  for (int z = 0; z < geom.blendOverlap; ++z) {
    for (int y = 0; y < ny; ++y) {
      const double yy =
          (static_cast<double>(y) - 0.5 * static_cast<double>(ny - 1)) *
          voxelSizeYmm;
      for (int x = 0; x < nx; ++x) {
        const double xx =
            (static_cast<double>(x) - 0.5 * static_cast<double>(nx - 1)) *
            voxelSizeXmm;
        if (xx * xx + yy * yy > radiusSquared)
          continue;
        const size_t xy = static_cast<size_t>(y) * nx + x;
        const double low =
            source[static_cast<size_t>(lowStart + z) * plane + xy];
        const double high =
            source[static_cast<size_t>(highStart + z) * plane + xy];
        if (low > threshold && high > threshold && std::isfinite(low) &&
            std::isfinite(high)) {
          ratios.push_back(std::log(low) - std::log(high));
        }
      }
    }
  }

  if (ratios.size() < 100) {
    ratios.clear();
    for (int z = 0; z < geom.blendOverlap; ++z) {
      for (int y = cropY0; y < cropY0 + cropNy; ++y) {
        for (int x = cropX0; x < cropX0 + cropNx; ++x) {
          const size_t xy = static_cast<size_t>(y) * nx + x;
          const double low =
              source[static_cast<size_t>(lowStart + z) * plane + xy];
          const double high =
              source[static_cast<size_t>(highStart + z) * plane + xy];
          if (low > 0.0 && high > 0.0 && std::isfinite(low) &&
              std::isfinite(high)) {
            ratios.push_back(std::log(low) - std::log(high));
          }
        }
      }
    }
  }
  if (ratios.empty())
    throw std::runtime_error(
        "center66 gain fit found no common positive overlap voxels");

  double center = median(ratios);
  std::vector<double> deviations(ratios.size());
  for (size_t i = 0; i < ratios.size(); ++i)
    deviations[i] = std::abs(ratios[i] - center);
  const double mad = median(deviations);
  if (mad > 0.0) {
    const double limit = 3.5 * 1.4826 * mad;
    std::vector<double> filtered;
    filtered.reserve(ratios.size());
    for (double ratio : ratios) {
      if (std::abs(ratio - center) <= limit)
        filtered.push_back(ratio);
    }
    if (!filtered.empty())
      center = median(filtered);
    sampleCount = filtered.size();
  } else {
    sampleCount = ratios.size();
  }
  return center;
}

struct SeamXyShift {
  int dx = 0;
  int dy = 0;
  double ncc = 0.0;
  int samples = 0;
};

double overlapPlaneNcc(const std::vector<double> &low,
                       const std::vector<double> &high, int nx, int ny,
                       int dx, int dy, double radiusSquared,
                       double voxelSizeXmm, double voxelSizeYmm,
                       int &samples) {
  double sumL = 0.0;
  double sumH = 0.0;
  double sumLL = 0.0;
  double sumHH = 0.0;
  double sumLH = 0.0;
  samples = 0;
  for (int y = 0; y < ny; ++y) {
    const double yy =
        (static_cast<double>(y) - 0.5 * static_cast<double>(ny - 1)) *
        voxelSizeYmm;
    const int sy = y + dy;
    if (sy < 0 || sy >= ny)
      continue;
    for (int x = 0; x < nx; ++x) {
      const double xx =
          (static_cast<double>(x) - 0.5 * static_cast<double>(nx - 1)) *
          voxelSizeXmm;
      if (xx * xx + yy * yy > radiusSquared)
        continue;
      const int sx = x + dx;
      if (sx < 0 || sx >= nx)
        continue;
      const double lv = low[static_cast<size_t>(y) * nx + x];
      const double hv = high[static_cast<size_t>(sy) * nx + sx];
      if (!(lv > 0.0 && hv > 0.0) || !std::isfinite(lv) || !std::isfinite(hv))
        continue;
      sumL += lv;
      sumH += hv;
      sumLL += lv * lv;
      sumHH += hv * hv;
      sumLH += lv * hv;
      ++samples;
    }
  }
  if (samples < 32)
    return -1.0;
  const double n = static_cast<double>(samples);
  const double cov = sumLH - sumL * sumH / n;
  const double varL = sumLL - sumL * sumL / n;
  const double varH = sumHH - sumH * sumH / n;
  if (varL <= 0.0 || varH <= 0.0)
    return -1.0;
  return cov / std::sqrt(varL * varH);
}

SeamXyShift estimateOverlapXyShift(const std::vector<double> &source, int nx,
                                   int ny, int lowStart, int highStart,
                                   int blendOverlap, double voxelSizeXmm,
                                   double voxelSizeYmm) {
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> low(plane, 0.0);
  std::vector<double> high(plane, 0.0);
  for (int z = 0; z < blendOverlap; ++z) {
    const size_t lowOff = static_cast<size_t>(lowStart + z) * plane;
    const size_t highOff = static_cast<size_t>(highStart + z) * plane;
    for (size_t xy = 0; xy < plane; ++xy) {
      low[xy] += source[lowOff + xy];
      high[xy] += source[highOff + xy];
    }
  }
  const double inv = 1.0 / static_cast<double>(std::max(blendOverlap, 1));
  for (size_t xy = 0; xy < plane; ++xy) {
    low[xy] *= inv;
    high[xy] *= inv;
  }
  const double radius =
      0.5 * spect::core::sixmodule::kMultiBedGainRoiDiameterMm;
  const double radiusSquared = radius * radius;
  constexpr int kSearch = 8;
  constexpr double kMinNcc = 0.25;
  SeamXyShift best;
  best.ncc = -1.0;
  for (int dy = -kSearch; dy <= kSearch; ++dy) {
    for (int dx = -kSearch; dx <= kSearch; ++dx) {
      int samples = 0;
      const double ncc = overlapPlaneNcc(
          low, high, nx, ny, dx, dy, radiusSquared, voxelSizeXmm,
          voxelSizeYmm, samples);
      if (ncc > best.ncc) {
        best.dx = dx;
        best.dy = dy;
        best.ncc = ncc;
        best.samples = samples;
      }
    }
  }
  if (best.ncc < kMinNcc) {
    best.dx = 0;
    best.dy = 0;
  }
  return best;
}

std::vector<double> center66CosineBlendOutputOverlap(
    const std::vector<double> &source, int nx, int ny, int nz,
    const OutputTransform &transform, double voxelSizeXmm,
    double voxelSizeYmm) {
  validateCenter66CosineTransform(transform, nz);
  const auto& geom = transform.canonicalGeom;
  const std::vector<int> starts = outputBedStarts(transform);
  const int cropStart = (transform.baseNz - geom.cropLayers) / 2;
  const std::vector<double> gains = fitCenter66BedGains(
      source, nx, ny, transform, cropStart, voxelSizeXmm, voxelSizeYmm);
  const size_t bedCount = starts.size();
  std::vector<int> sampleDx(bedCount, 0);
  std::vector<int> sampleDy(bedCount, 0);
  if (transform.seamXyAlign && bedCount > 1) {
    for (size_t highBed = 0; highBed + 1 < bedCount; ++highBed) {
      const size_t lowBed = highBed + 1;
      const int lowStart = transform.bedZStarts[lowBed] + cropStart +
          (transform.reverseLocalZInReconStorage ? 0 : geom.stepLayers);
      const int highStart = transform.bedZStarts[highBed] + cropStart +
          (transform.reverseLocalZInReconStorage ? geom.stepLayers : 0);
      const SeamXyShift shift = estimateOverlapXyShift(
          source, nx, ny, lowStart, highStart, geom.blendOverlap,
          voxelSizeXmm, voxelSizeYmm);
      const bool apply = shift.ncc >= 0.25;
      // high[y+dy, x+dx] matches low[y,x]; sample the higher-index bed
      // (caudal, lowBed) with the inverse so both land in Bed1's frame.
      if (apply) {
        sampleDx[lowBed] = sampleDx[highBed] - shift.dx;
        sampleDy[lowBed] = sampleDy[highBed] - shift.dy;
      } else {
        sampleDx[lowBed] = sampleDx[highBed];
        sampleDy[lowBed] = sampleDy[highBed];
      }
      logDebug("center66 seam XY Bed" + std::to_string(lowBed + 1) + "/Bed" +
              std::to_string(highBed + 1) + ": dx=" +
              std::to_string(shift.dx) + ", dy=" + std::to_string(shift.dy) +
              ", ncc=" + std::to_string(shift.ncc) + ", samples=" +
              std::to_string(shift.samples) +
              (apply ? "" : " (below gate, no shift)"));
    }
  }
  const int outputNz = starts.back() + geom.cropLayers;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<double> out(static_cast<size_t>(outputNz) * plane, 0.0);
  std::vector<double> weights(static_cast<size_t>(outputNz), 0.0);
  for (size_t bed = 0; bed < starts.size(); ++bed) {
    const int dx = sampleDx[bed];
    const int dy = sampleDy[bed];
    for (int localZ = 0; localZ < geom.cropLayers; ++localZ) {
      double weight = 1.0;
      // Fade along output Z, not storage Z. reverseLocalZInReconStorage
      // only selects which overlap voxels are compared for gain; inverting
      // these weights produced a paired bed-switch at each 10-layer joint.
      if (bed > 0 && localZ < geom.blendOverlap) {
        const double fade = center66CosineFadeIn(localZ, geom.blendOverlap);
        weight *= fade;
      }
      if (bed + 1 < starts.size() && localZ >= geom.stepLayers) {
        const double fade =
            center66CosineFadeIn(localZ - geom.stepLayers, geom.blendOverlap);
        weight *= 1.0 - fade;
      }
      const int srcZ = transform.bedZStarts[bed] + cropStart + localZ;
      const int dstZ = starts[bed] + localZ;
      const size_t srcOffset = static_cast<size_t>(srcZ) * plane;
      const size_t dstOffset = static_cast<size_t>(dstZ) * plane;
      const double scaled = weight * gains[bed];
      if (dx == 0 && dy == 0) {
        for (size_t xy = 0; xy < plane; ++xy)
          out[dstOffset + xy] += scaled * source[srcOffset + xy];
      } else {
        for (int y = 0; y < ny; ++y) {
          const int sy = y + dy;
          if (sy < 0 || sy >= ny)
            continue;
          for (int x = 0; x < nx; ++x) {
            const int sx = x + dx;
            if (sx < 0 || sx >= nx)
              continue;
            out[dstOffset + static_cast<size_t>(y) * nx + x] +=
                scaled *
                source[srcOffset + static_cast<size_t>(sy) * nx + sx];
          }
        }
      }
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
  const auto& geom = transform.canonicalGeom;
  const std::vector<int> starts = outputBedStarts(transform);
  const int cropStart = (transform.baseNz - geom.cropLayers) / 2;
  const int outputNz = starts.back() + geom.cropLayers;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<uint8_t> out(static_cast<size_t>(outputNz) * plane, 0);
  for (size_t bed = 0; bed < starts.size(); ++bed) {
    for (int localZ = 0; localZ < geom.cropLayers; ++localZ) {
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
  const auto& geom = transform.canonicalGeom;
  OutputTransform adjusted = transform;
  adjusted.baseNz = geom.cropLayers;
  for (size_t bed = 0; bed < adjusted.bedZStarts.size(); ++bed)
    adjusted.bedZStarts[bed] =
        static_cast<int>(bed) * geom.cropLayers;
  adjusted.outputBedOverlaps.assign(
      adjusted.bedZStarts.size() - 1, geom.blendOverlap);
  return adjusted;
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

template <typename T>
std::vector<T> flipZWholeOutput(const std::vector<T> &source,
                                 int nx, int ny, int nz) {
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  std::vector<T> out(source.size(), T{});
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

void clampNonNegativeInPlace(std::vector<double> &vol) {
  for (double &v : vol)
    v = std::max(0.0, v);
}

std::vector<double> rotateOutputXY(const std::vector<double> &source,
                                   int nx, int ny, int nz,
                                   double vx, double vy, double degrees) {
  if (!std::isfinite(degrees) || !std::isfinite(vx) || !std::isfinite(vy) ||
      vx <= 0.0 || vy <= 0.0)
    throw std::invalid_argument("Output XY rotation requires a finite angle and positive finite XY spacing");
  degrees = std::remainder(degrees, 360.0);
  if (degrees == 0.0)
    return source;
  const double radians = degrees * (std::acos(-1.0) / 180.0);
  double c = std::cos(radians), s = std::sin(radians);
  // Exact cardinal angles avoid resampling noise on aligned voxel grids.
  if (std::remainder(degrees, 90.0) == 0.0) {
    c = std::round(c);
    s = std::round(s);
  }
  const double cx = (nx - 1) * 0.5, cy = (ny - 1) * 0.5;
  const size_t plane = static_cast<size_t>(nx) * ny;
  std::vector<double> out(source.size(), 0.0);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      // Inverse mapping in millimetres preserves anisotropic voxel geometry.
      const double px = (x - cx) * vx, py = (y - cy) * vy;
      const double sx = (c * px + s * py) / vx + cx;
      const double sy = (-s * px + c * py) / vy + cy;
      if (sx <= -1.0 || sx >= nx || sy <= -1.0 || sy >= ny)
        continue;
      const int x0 = static_cast<int>(std::floor(sx));
      const int y0 = static_cast<int>(std::floor(sy));
      const double fx = sx - x0, fy = sy - y0;
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
          if (x0 + i < 0 || x0 + i >= nx || y0 + j < 0 || y0 + j >= ny)
            continue;
          const double weight = (i ? fx : 1.0 - fx) * (j ? fy : 1.0 - fy);
          if (weight == 0.0)
            continue;
          for (int z = 0; z < nz; ++z)
            out[z * plane + y * nx + x] += weight *
                source[z * plane + (y0 + j) * nx + x0 + i];
        }
      }
    }
  }
  return out;
}

} // namespace

std::vector<double> fitCenter66BedGains(
    const std::vector<double> &source, int nx, int ny,
    const OutputTransform &transform, int cropStart, double voxelSizeXmm,
    double voxelSizeYmm) {
  const size_t bedCount = transform.bedZStarts.size();
  if (bedCount < 2)
    return std::vector<double>(bedCount, 1.0);
  std::vector<double> logGains(bedCount, 0.0);
  for (size_t lowBed = bedCount - 1; lowBed > 0; --lowBed) {
    const size_t highBed = lowBed - 1;
    size_t samples = 0;
    const double ratio = center66PairLogRatio(
        source, nx, ny, transform, cropStart, lowBed, highBed, voxelSizeXmm,
        voxelSizeYmm, samples);
    logGains[highBed] = logGains[lowBed] + ratio;
    logDebug("center66 gain pair Bed" + std::to_string(lowBed + 1) + "/Bed" +
            std::to_string(highBed + 1) + ": log_ratio=" +
            std::to_string(ratio) + ", samples=" +
            std::to_string(samples));
  }
  // Prototype behavior: always apply the full log-ratio chain anchored at
  // Bed1 (no reliability gate).
  std::vector<double> gains(bedCount, 1.0);
  if (bedCount > 0) {
    const double anchor = logGains[0];
    for (size_t i = 0; i < bedCount; ++i)
      gains[i] = std::exp(logGains[i] - anchor);
    for (size_t i = 0; i < bedCount; ++i) {
      logDebug("center66 gain Bed" + std::to_string(i + 1) + "=" +
              std::to_string(gains[i]));
    }
  }
  return gains;
}

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

  const std::vector<double> *sourceForOutput = &source;
  const std::vector<uint8_t> *maskForOutput = mask;
  std::vector<double> perBedFlipped;
  std::vector<double> transformedSource;
  std::vector<uint8_t> transformedMask;
  int transformedNz = nz;

  if (shouldFlipZPerBed(transform)) {
    perBedFlipped = flipDoubleZPerBed(*sourceForOutput, nx, ny, nz, *transform);
    sourceForOutput = &perBedFlipped;
    if (maskForOutput) {
      transformedMask = flipMaskZPerBed(*maskForOutput, nx, ny, nz, *transform);
      maskForOutput = &transformedMask;
    }
  }

  std::vector<double> wholeZFlipped;
  std::vector<uint8_t> wholeZMaskFlipped;
  if (shouldFlipZWholeOutput(transform)) {
    wholeZFlipped = flipZWholeOutput(*sourceForOutput, nx, ny, transformedNz);
    sourceForOutput = &wholeZFlipped;
    if (maskForOutput) {
      wholeZMaskFlipped = flipZWholeOutput(*maskForOutput, nx, ny, transformedNz);
      maskForOutput = &wholeZMaskFlipped;
    }
  }

  std::vector<double> xFlipped;
  std::vector<uint8_t> xMaskFlipped;
  if (shouldFlipXWholeOutput(transform)) {
    xFlipped = flipXWholeOutput(*sourceForOutput, nx, ny, transformedNz);
    sourceForOutput = &xFlipped;
    if (maskForOutput) {
      xMaskFlipped = flipXWholeOutput(*maskForOutput, nx, ny, transformedNz);
      maskForOutput = &xMaskFlipped;
    }
  }

  if (shouldStitchCanonicalOutput(transform)) {
    transformedSource = center66CosineBlendOutputOverlap(
        *sourceForOutput, nx, ny, transformedNz, *transform,
        voxelSizeXmm, voxelSizeYmm);
    sourceForOutput = &transformedSource;
    const int stitchOutputNz = static_cast<int>(transformedSource.size() /
                                     (static_cast<size_t>(nx) * ny));
    if (maskForOutput) {
      transformedMask = center66BlendOutputMask(
          *maskForOutput, nx, ny, transformedNz, *transform);
      maskForOutput = &transformedMask;
    }
    transformedNz = stitchOutputNz;
  }

  std::vector<double> centerCropped;
  std::vector<uint8_t> centerCropMask;
  int croppedNx = nx, croppedNy = ny, croppedNz = transformedNz;
  // Volume and mask must each be cropped from the *original* dimensions:
  // centerCropVolume updates nx/ny/nz by reference, so reusing the already
  // updated values for the mask would skip the mask crop and misalign it.
  if (transform && transform->centerCropNx > 0 && transform->centerCropNy > 0) {
    int volNx = nx, volNy = ny, volNz = transformedNz;
    centerCropped = centerCropVolume(
        *sourceForOutput, volNx, volNy, volNz,
        transform->centerCropNx, transform->centerCropNy, -1);
    sourceForOutput = &centerCropped;
    if (maskForOutput) {
      int maskNx = nx, maskNy = ny, maskNz = transformedNz;
      centerCropMask = centerCropVolume(
          *maskForOutput, maskNx, maskNy, maskNz,
          transform->centerCropNx, transform->centerCropNy, -1);
      maskForOutput = &centerCropMask;
    }
    croppedNx = volNx;
    croppedNy = volNy;
    croppedNz = volNz;
  }
  if (transform && transform->centerCropNz > 0) {
    int volNx = croppedNx, volNy = croppedNy, volNz = croppedNz;
    centerCropped = centerCropVolume(
        *sourceForOutput, volNx, volNy, volNz,
        -1, -1, transform->centerCropNz);
    sourceForOutput = &centerCropped;
    if (maskForOutput) {
      int maskNx = croppedNx, maskNy = croppedNy, maskNz = croppedNz;
      centerCropMask = centerCropVolume(
          *maskForOutput, maskNx, maskNy, maskNz,
          -1, -1, transform->centerCropNz);
      maskForOutput = &centerCropMask;
    }
    croppedNx = volNx;
    croppedNy = volNy;
    croppedNz = volNz;
  }

  std::vector<double> out = *sourceForOutput;
  if (crop.start > 0 || crop.end > 0) {
    const int zTotal = croppedNz;
    const size_t plane = static_cast<size_t>(croppedNx) * croppedNy;
    for (int z = 0; z < crop.start && z < zTotal; ++z)
      std::fill_n(out.begin() + static_cast<std::ptrdiff_t>(z) * plane, plane, 0.0);
    for (int z = std::max(0, zTotal - crop.end); z < zTotal; ++z)
      std::fill_n(out.begin() + static_cast<std::ptrdiff_t>(z) * plane, plane, 0.0);
  }

  if (maskForOutput) {
    for (size_t i = 0; i < out.size(); ++i) {
      if (!(*maskForOutput)[i])
        out[i] = 0.0;
    }
  }

  if (transform && transform->xyRotationDeg != 0.0)
    out = rotateOutputXY(out, croppedNx, croppedNy, croppedNz,
                         voxelSizeXmm, voxelSizeYmm, transform->xyRotationDeg);

  if (outNz)
    *outNz = croppedNz;
  return out;
}

void saveVolumeOutput(const std::vector<double> &source,
                      const OutputGeometry &geometry,
                      const std::string &outputDir,
                      const std::string &outputPrefix, int iteration,
                      const std::vector<uint8_t> *mask,
                      const CropSettings &crop, bool littleEndian,
                      const OutputTransform *transform,
                      const OutputFilterSettings *outputFilters) {
  int outNz = 0;
  const std::vector<double> prepared = prepareOutputVolume(
      source, geometry.nx, geometry.ny, geometry.nz, mask, crop, &outNz,
      transform, geometry.vx, geometry.vy, geometry.vz);
  int outNx = geometry.nx;
  int outNy = geometry.ny;
  if (transform && transform->centerCropNx > 0) outNx = transform->centerCropNx;
  if (transform && transform->centerCropNy > 0) outNy = transform->centerCropNy;

  const auto iterStem = std::filesystem::path(outputDir) /
      (outputPrefix + (iteration >= 0 ? "_it" + std::to_string(iteration) : ""));
  const bool autoMuCompatible = littleEndian && crop.start == 0 && crop.end == 0 &&
      outNx == geometry.nx && outNy == geometry.ny &&
      outNz == geometry.nz && (!transform ||
      (!transform->flipZPerBed && !transform->flipZWholeOutput &&
       !transform->flipXWholeOutput && !transform->seamXyAlign &&
       transform->bedZStarts.size() <= 1 &&
       std::fmod(transform->xyRotationDeg, 360.0) == 0.0));
  const auto markCoordinates = [&](const std::string &suffix) {
    const auto header = iterStem.string() + suffix + ".mhd";
    std::ofstream out(header, std::ios::app);
    out << "SPECTAutoMuCompatible = " << (autoMuCompatible ? 1 : 0) << '\n';
    out.flush();
    if (!out) throw std::runtime_error("Failed writing volume coordinate marker: " + header);
  };
  const bool clamp = outputFilters == nullptr || outputFilters->nonnegativeClamp;
  const bool filtersActive = outputFilters != nullptr && iteration >= 0;
  const bool overwritePrimary =
      filtersActive && outputFilters->overwritePrimary;
  std::vector<double> primary = prepared;
  auto savePrimary = [&](std::vector<double> &volume) {
    if (clamp)
      clampNonNegativeInPlace(volume);
    io::saveMHDRAW(volume, outNx, outNy, outNz, geometry.vx,
                   geometry.vy, geometry.vz, outputDir, outputPrefix,
                   iteration, littleEndian);
    markCoordinates("");
  };
  if (!overwritePrimary)
    savePrimary(primary);
  if (!filtersActive)
    return;

  auto applySmooth = [&](std::vector<double> &volume) {
    const auto kx = filters::makeCastorErf1D(
        outputFilters->smoothFwhmMm, geometry.vx,
        outputFilters->smoothSigmas);
    const auto ky = filters::makeCastorErf1D(
        outputFilters->smoothFwhmMm, geometry.vy,
        outputFilters->smoothSigmas);
    const auto kz = filters::makeCastorErf1D(
        outputFilters->smoothFwhmMm, geometry.vz,
        outputFilters->smoothSigmas);
    if (!cuda::applyCastorConv3DInplaceCUDAForcedFloat32Staging(
            volume, outNx, outNy, outNz, kx, ky, kz)) {
      throw std::runtime_error(
          "Output Gaussian requires the CUDA CASToR convolution backend.");
    }
  };
  auto applyButter = [&](std::vector<double> &volume) {
    filters::apply_butterworth_output_cufft(
        volume, outNx, outNy, outNz,
        filters::Spacing3{geometry.vx, geometry.vy, geometry.vz},
        outputFilters->butterCutoffCpcm, outputFilters->butterOrder);
  };

  std::vector<double> smoothed;
  if (filtersActive && outputFilters->smoothEnabled) {
    smoothed = prepared;
    applySmooth(smoothed);
  }
  std::vector<double> buttered;
  if (filtersActive && outputFilters->butterEnabled) {
    buttered = prepared;
    applyButter(buttered);
  }

  if (overwritePrimary) {
    if (outputFilters->smoothEnabled)
      primary = smoothed;
    if (outputFilters->butterEnabled) {
      if (outputFilters->smoothEnabled)
        applyButter(primary);
      else
        primary = buttered;
    }
    savePrimary(primary);
  }

  if (outputFilters->smoothEnabled) {
    if (clamp)
      clampNonNegativeInPlace(smoothed);
    io::writeMHD((iterStem.string() + "_smooth"), smoothed, outNx, outNy,
                 outNz, static_cast<float>(geometry.vx),
                 static_cast<float>(geometry.vy),
                 static_cast<float>(geometry.vz), littleEndian);
    markCoordinates("_smooth");
  }
  if (outputFilters->butterEnabled) {
    if (clamp)
      clampNonNegativeInPlace(buttered);
    io::writeMHD((iterStem.string() + "_butter"), buttered, outNx, outNy,
                 outNz, static_cast<float>(geometry.vx),
                 static_cast<float>(geometry.vy),
                 static_cast<float>(geometry.vz), littleEndian);
    markCoordinates("_butter");
  }
}

} // namespace main
} // namespace spect
