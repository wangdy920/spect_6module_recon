#include "io/ProjectionLoader.h"

#include "io/MatFileReader.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace spect {
namespace io {
namespace {

std::string lowerExtension(const std::string &path) {
  std::string ext = std::filesystem::path(path).extension().string();
  for (char &ch : ext)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return ext;
}

uint64_t checkedMul(uint64_t a, uint64_t b, const std::string &what) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
    throw std::runtime_error(what + " overflows uint64");
  return a * b;
}

uint64_t checkedAdd(uint64_t a, uint64_t b, const std::string &what) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::runtime_error(what + " overflows uint64");
  return a + b;
}

void readExact(std::istream &in, void *dst, size_t bytes,
               const std::string &context) {
  if (bytes == 0)
    return;
  in.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
  if (!in)
    throw std::runtime_error("Failed to read " + context);
}

ProjectionData loadProjectionBinInternal(const std::string &path,
                                         int requestedAngles,
                                         ProjectionReorderStats *stats,
                                         MatArray *outMat) {
  if (outMat)
    *outMat = MatArray{};
  if (stats)
    *stats = ProjectionReorderStats{};

  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open projection BIN file: " + path);

  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < static_cast<std::streamoff>(2 * sizeof(int32_t)))
    throw std::runtime_error("Projection BIN file is too small: " + path);
  const uint64_t fileSize = static_cast<uint64_t>(end);
  f.seekg(0, std::ios::beg);

  int32_t totalAngles = 0;
  int32_t raysPerAngle = 0;
  readExact(f, &totalAngles, sizeof(totalAngles), "projection BIN num_angles");
  readExact(f, &raysPerAngle, sizeof(raysPerAngle), "projection BIN rays_per_angle");
  if (totalAngles <= 0 || raysPerAngle <= 0)
    throw std::runtime_error("Projection BIN header has invalid dimensions: " +
                             path);

  const uint64_t totalCount =
      checkedMul(static_cast<uint64_t>(totalAngles),
                 static_cast<uint64_t>(raysPerAngle),
                 "projection BIN element count");
  const uint64_t expectedSize =
      checkedAdd(2 * sizeof(int32_t), checkedMul(totalCount, sizeof(double),
                                                 "projection BIN byte count"),
                 "projection BIN file size");
  if (fileSize != expectedSize) {
    std::ostringstream oss;
    oss << "Projection BIN file size mismatch: " << path
        << " (expected " << expectedSize << " bytes, got " << fileSize << ")";
    throw std::runtime_error(oss.str());
  }

  int selectedAngles = (requestedAngles > 0) ? requestedAngles : totalAngles;
  selectedAngles = std::max(1, std::min(selectedAngles, totalAngles));
  const uint64_t selectedCount =
      checkedMul(static_cast<uint64_t>(selectedAngles),
                 static_cast<uint64_t>(raysPerAngle),
                 "selected projection BIN element count");
  if (selectedCount >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("Projection BIN selected data is too large for this build");
  }

  ProjectionData out;
  out.data.resize(static_cast<size_t>(selectedCount));
  readExact(f, out.data.data(), out.data.size() * sizeof(double),
            "projection BIN payload");
  out.numAngles = selectedAngles;
  out.detNz = 1;
  out.detNx = raysPerAngle;
  out.raysPerAngle = static_cast<size_t>(raysPerAngle);
  out.detectorShapeKnown = false;
  out.binaryFormat = true;

  if (stats) {
    stats->isEventList = false;
    stats->selectedAngles = selectedAngles;
    stats->detectorsPerAngle = out.raysPerAngle;
    stats->recordCount = out.data.size();
  }
  return out;
}

} // namespace

bool isProjectionBinPath(const std::string &path) {
  return lowerExtension(path) == ".bin";
}

std::vector<double> reorderProjection(const MatArray &mat, int requestedAngles) {
  return reorderProjection(mat, requestedAngles, nullptr);
}

std::vector<double> reorderProjection(const MatArray &mat, int requestedAngles,
                                      ProjectionReorderStats *stats) {
  const double *srcD = mat.d.empty() ? nullptr : mat.d.data();
  const float *srcF = mat.f.empty() ? nullptr : mat.f.data();
  if (!srcD && !srcF)
    throw std::runtime_error("Projection MAT must contain double or single data");

  if (stats)
    *stats = ProjectionReorderStats{};

  if (mat.dims.size() == 3) {
    const int detZ = static_cast<int>(mat.dims[0]);
    const int detX = static_cast<int>(mat.dims[1]);
    const int totalAngles = static_cast<int>(mat.dims[2]);
    if (detZ <= 0 || detX <= 0 || totalAngles <= 0)
      throw std::runtime_error("Projection MAT has invalid dimensions");

    int numAngles = (requestedAngles > 0) ? requestedAngles : totalAngles;
    numAngles = std::max(1, std::min(numAngles, totalAngles));
    const size_t raysPerAngle = static_cast<size_t>(detZ) * static_cast<size_t>(detX);

    std::vector<double> out(raysPerAngle * static_cast<size_t>(numAngles), 0.0);
    size_t dst = 0;
    for (int a = 0; a < numAngles; ++a) {
      for (int z = 0; z < detZ; ++z) {
        for (int x = 0; x < detX; ++x) {
          const size_t srcIdx = static_cast<size_t>(z) +
                                static_cast<size_t>(detZ) *
                                    (static_cast<size_t>(x) + static_cast<size_t>(detX) * static_cast<size_t>(a));
          out[dst++] = srcD ? srcD[srcIdx] : static_cast<double>(srcF[srcIdx]);
        }
      }
    }

    if (stats) {
      stats->isEventList = false;
      stats->selectedAngles = numAngles;
      stats->detectorsPerAngle = raysPerAngle;
      stats->recordCount = out.size();
    }
    return out;
  }

  if (mat.dims.size() == 2) {
    const size_t rows = mat.dims[0];
    const size_t cols = mat.dims[1];
    if (!(cols == 3 || rows == 3))
      throw std::runtime_error("Event-list MAT must be Nx3 or 3xN");

    const bool rowTriples = (cols == 3);
    const size_t recordCount = rowTriples ? rows : cols;
    auto valueAt = [&](size_t r, size_t c) -> double {
      const size_t idx = r + rows * c;
      return srcD ? srcD[idx] : static_cast<double>(srcF[idx]);
    };

    struct ListEntry {
      int angle;
      int detector;
      double value;
    };

    std::vector<ListEntry> entries;
    entries.reserve(recordCount);
    int minDet = std::numeric_limits<int>::max();
    int maxDet = std::numeric_limits<int>::min();
    std::set<int> uniqueAngles;

    for (size_t rec = 0; rec < recordCount; ++rec) {
      const double rawAngle = rowTriples ? valueAt(rec, 0) : valueAt(0, rec);
      const double rawDet = rowTriples ? valueAt(rec, 1) : valueAt(1, rec);
      const double rawVal = rowTriples ? valueAt(rec, 2) : valueAt(2, rec);
      if (!std::isfinite(rawAngle) || !std::isfinite(rawDet) || !std::isfinite(rawVal))
        continue;

      const int angleIdx = static_cast<int>(std::llround(rawAngle));
      const int detIdx = static_cast<int>(std::llround(rawDet));
      entries.push_back({angleIdx, detIdx, rawVal});
      uniqueAngles.insert(angleIdx);
      minDet = std::min(minDet, detIdx);
      maxDet = std::max(maxDet, detIdx);
    }

    if (entries.empty())
      return {};

    std::vector<int> angleList(uniqueAngles.begin(), uniqueAngles.end());
    size_t selectedAngles = angleList.size();
    if (requestedAngles > 0 && static_cast<size_t>(requestedAngles) < selectedAngles)
      selectedAngles = static_cast<size_t>(requestedAngles);
    if (selectedAngles == 0)
      return {};

    std::unordered_map<int, int> angleToIndex;
    angleToIndex.reserve(selectedAngles);
    for (size_t i = 0; i < selectedAngles; ++i)
      angleToIndex.emplace(angleList[i], static_cast<int>(i));

    const size_t detPerAngle = static_cast<size_t>(maxDet - minDet + 1);
    if (detPerAngle == 0)
      throw std::runtime_error("Detector indexing in event list is invalid");

    std::vector<double> out(detPerAngle * selectedAngles, 0.0);
    for (const auto &entry : entries) {
      const auto it = angleToIndex.find(entry.angle);
      if (it == angleToIndex.end())
        continue;
      const size_t detIndex = static_cast<size_t>(entry.detector - minDet);
      if (detIndex >= detPerAngle)
        continue;
      out[static_cast<size_t>(it->second) * detPerAngle + detIndex] += entry.value;
    }

    if (stats) {
      stats->isEventList = true;
      stats->selectedAngles = static_cast<int>(selectedAngles);
      stats->detectorsPerAngle = detPerAngle;
      stats->recordCount = entries.size();
    }
    return out;
  }

  throw std::runtime_error("Unsupported projection MAT dimensionality");
}

ProjectionData loadProjection(const std::string &path, const std::string &varName,
                              int requestedAngles, ProjectionReorderStats *stats,
                              MatArray *outMat) {
  if (isProjectionBinPath(path))
    return loadProjectionBinInternal(path, requestedAngles, stats, outMat);

  MatArray localMat;
  MatArray &mat = outMat ? *outMat : localMat;
  mat = readMat(path, varName);
  ProjectionReorderStats localStats;
  ProjectionReorderStats *effectiveStats = stats ? stats : &localStats;

  ProjectionData out;
  out.data = reorderProjection(mat, requestedAngles, effectiveStats);

  if (mat.dims.size() == 3) {
    out.detNz = static_cast<int>(mat.dims[0]);
    out.detNx = static_cast<int>(mat.dims[1]);
    const int totalAngles = static_cast<int>(mat.dims[2]);
    out.numAngles = (requestedAngles > 0) ? std::min(requestedAngles, totalAngles) : totalAngles;
    out.raysPerAngle = static_cast<size_t>(out.detNz) * static_cast<size_t>(out.detNx);
  } else if (stats != nullptr || !out.data.empty()) {
    out.numAngles = effectiveStats->selectedAngles;
    out.detNz = 1;
    out.detNx = static_cast<int>(effectiveStats->detectorsPerAngle);
    out.raysPerAngle = effectiveStats->detectorsPerAngle;
  }

  return out;
}

} // namespace io
} // namespace spect
