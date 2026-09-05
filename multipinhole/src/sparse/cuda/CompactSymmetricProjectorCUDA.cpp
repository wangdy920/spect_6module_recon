#include "sparse/CompactSymmetricProjector.h"

#if defined(SPECT_ENABLE_CUDA_SPARSEOPS)

#include "core/ReconLimits.h"
#include "cuda/CudaConfig.h"
#include "cuda/CudaLaunchConfig.h"
#include "sparse/SparseOpsCUDA.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spect::cuda {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint64_t kCompactChunkResidentReserveBytes =
    kCompactChunkResidentDeviceReserveBytes;

struct CompactChunkResidentCacheLease {
  uint64_t cacheBudget = 0;
  bool residentWarmupComplete = false;
  bool localityLogged = false;
  bool sortedHint = false;
  std::future<void> hostPreload;
  // Retain the actual allocations until uploads finish and registrations are
  // removed. A raw-pointer registration in the device cache outlives a chunk
  // when the host cache is too small, leaving a stale CUDA registration.
  std::vector<std::shared_ptr<const spect::io::CompactChunkCSRData>> pinnedChunks;
  std::vector<void *> pinnedArrays;

  ~CompactChunkResidentCacheLease() {
    if (pinnedArrays.empty())
      return;
    const auto sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess)
      std::cerr << "[ERROR] compact host-pin release synchronization: "
                << cudaGetErrorString(sync) << '\n';
    for (void *ptr : pinnedArrays) {
      const auto status = cudaHostUnregister(ptr);
      if (status != cudaSuccess)
        std::cerr << "[ERROR] compact host-pin release: "
                  << cudaGetErrorString(status) << '\n';
    }
  }

  void pin(const std::shared_ptr<const spect::io::CompactChunkCSRData> &chunk) {
    // Reserve bookkeeping before registering so allocation failure cannot
    // leave a registration untracked. No second copy of the payload is made.
    pinnedArrays.reserve(pinnedArrays.size() + 2);
    pinnedChunks.push_back(chunk);
    if (chunk->info.nnz == 0)
      return;
    const size_t bytes = static_cast<size_t>(chunk->info.nnz) * sizeof(uint32_t);
    for (void *ptr : {static_cast<void *>(const_cast<uint32_t *>(chunk->colIdxData())),
                      static_cast<void *>(const_cast<float *>(chunk->valuesData()))}) {
      CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable));
      pinnedArrays.push_back(ptr);
    }
  }
};

struct CompactCsrLocalitySample {
  bool sampled = false;
  int sampledRows = 0;
  int sortedRows = 0;
  bool sortedHint = false;
  bool fullFootprint = false;
  bool compactSpmmCandidate = false;
  double medianSpan = 0.0;
  double medianZfill = 0.0;
  double p75Span = 0.0;
  double p75Zfill = 0.0;
};

double localityPercentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double idx = p * static_cast<double>(values.size() - 1);
  return values[static_cast<size_t>(idx)];
}

void accumulateLocalityRows(const spect::io::CSRView &matrix, int nx, int ny,
                            int nz, int slotBegin, int slotCount,
                            int slotTotal, std::vector<double> *spans,
                            std::vector<double> *zfills, int *sortedRows) {
  const int64_t plane =
      static_cast<int64_t>(nx) * static_cast<int64_t>(ny);
  const int64_t nCols = matrix.cols > 0
                            ? static_cast<int64_t>(matrix.cols)
                            : plane * static_cast<int64_t>(nz);
  if (nCols <= 0 || matrix.row_ptr == nullptr || matrix.col_idx == nullptr) {
    return;
  }
  std::vector<uint8_t> zSeen(
      nz > 0 ? static_cast<size_t>(nz) : static_cast<size_t>(1), 0);
  for (int slot = slotBegin; slot < slotBegin + slotCount; ++slot) {
    int row0 = static_cast<int>(static_cast<int64_t>(slot) * matrix.rows /
                                slotTotal);
    int row1 = static_cast<int>(static_cast<int64_t>(slot + 1) * matrix.rows /
                                slotTotal);
    if (row1 <= row0) {
      row1 = row0 + 1;
    }
    if (row1 > matrix.rows) {
      row1 = matrix.rows;
    }
    int row = row0;
    while (row < row1 &&
           matrix.row_ptr[static_cast<size_t>(row)] ==
               matrix.row_ptr[static_cast<size_t>(row) + 1u]) {
      ++row;
    }
    if (row >= row1) {
      continue;
    }
    const uint32_t start = matrix.row_ptr[static_cast<size_t>(row)];
    const uint32_t end = matrix.row_ptr[static_cast<size_t>(row) + 1u];
    uint32_t minCol = std::numeric_limits<uint32_t>::max();
    uint32_t maxCol = 0;
    bool sorted = true;
    uint32_t prev = 0;
    std::fill(zSeen.begin(), zSeen.end(), 0);
    int zCount = 0;
    for (uint32_t t = start; t < end; ++t) {
      const uint32_t col = matrix.col_idx[t];
      if (col < minCol) {
        minCol = col;
      }
      if (col > maxCol) {
        maxCol = col;
      }
      if (t > start && col <= prev) {
        sorted = false;
      }
      prev = col;
      if (plane > 0 && nz > 0) {
        const int z = static_cast<int>(static_cast<uint64_t>(col) /
                                       static_cast<uint64_t>(plane));
        if (z >= 0 && z < nz && zSeen[static_cast<size_t>(z)] == 0) {
          zSeen[static_cast<size_t>(z)] = 1;
          ++zCount;
        }
      }
    }
    spans->push_back(static_cast<double>(maxCol - minCol + 1u) /
                     static_cast<double>(nCols));
    zfills->push_back(nz > 0 ? static_cast<double>(zCount) /
                                   static_cast<double>(nz)
                             : 0.0);
    if (sorted) {
      ++(*sortedRows);
    }
  }
}

CompactCsrLocalitySample sampleCompactCsrLocality(
    const spect::io::CSRView &matrix, int nx, int ny, int nz) {
  CompactCsrLocalitySample sample;
  if (!matrix.valid() || matrix.rows <= 0 || nx <= 0 || ny <= 0 || nz <= 0) {
    return sample;
  }
  std::vector<double> spans;
  std::vector<double> zfills;
  spans.reserve(32);
  zfills.reserve(32);
  accumulateLocalityRows(matrix, nx, ny, nz, 0, 16, 16, &spans, &zfills,
                         &sample.sortedRows);
  auto classify = [&]() {
    sample.sampled = !spans.empty();
    sample.sampledRows = static_cast<int>(spans.size());
    if (!sample.sampled) {
      return;
    }
    sample.medianSpan = localityPercentile(spans, 0.50);
    sample.medianZfill = localityPercentile(zfills, 0.50);
    sample.p75Span = localityPercentile(spans, 0.75);
    sample.p75Zfill = localityPercentile(zfills, 0.75);
    sample.fullFootprint =
        sample.medianSpan >= 0.80 || sample.medianZfill >= 0.80;
    sample.compactSpmmCandidate =
        sample.p75Span <= 0.35 && sample.p75Zfill <= 0.35;
    sample.sortedHint =
        static_cast<double>(sample.sortedRows) /
            static_cast<double>(sample.sampledRows) >=
        0.95;
  };
  classify();
  if (sample.sampled && !sample.fullFootprint &&
      !sample.compactSpmmCandidate) {
    accumulateLocalityRows(matrix, nx, ny, nz, 16, 16, 32, &spans, &zfills,
                           &sample.sortedRows);
    classify();
  }
  return sample;
}

void applyCompactCsrLocalityHint(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta) {
  auto lease = std::static_pointer_cast<CompactChunkResidentCacheLease>(
      chunks.acceleratorLease());
  if (lease == nullptr) {
    noteCompactCsrLocalityCUDA(false);
    return;
  }
  if (!lease->localityLogged) {
    CompactCsrLocalitySample sample;
    if (!chunks.chunks().empty()) {
      const auto chunk = chunks.loadChunk(0);
      sample = sampleCompactCsrLocality(chunk->view(), meta.Nx, meta.Ny,
                                        meta.Nz);
    }
    lease->sortedHint = sample.sortedHint;
    const char *cls = "unknown";
    if (sample.fullFootprint) {
      cls = "full_footprint";
    } else if (sample.compactSpmmCandidate) {
      cls = "compact_candidate";
    }
    std::cerr << "[NOTE] compact CSR locality: class=" << cls
              << " sorted_hint=" << (sample.sortedHint ? 1 : 0)
              << " sampled=" << sample.sampledRows
              << " span_med=" << sample.medianSpan
              << " zfill_med=" << sample.medianZfill
              << " span_p75=" << sample.p75Span
              << " zfill_p75=" << sample.p75Zfill
              << "; wide SpMM is not auto-enabled\n";
    lease->localityLogged = true;
  }
  noteCompactCsrLocalityCUDA(lease->sortedHint);
}

int effectiveThreads(int threads) { return threads > 0 ? threads : 1; }

double elapsedSeconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void addTiming(double &target, Clock::time_point start) {
  target += elapsedSeconds(start);
}


bool compactBranchBatchEnabled() {
  const char *value = std::getenv("SPECT_COMPACT_BRANCH_BATCH");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  return value[0] != '0' && value[0] != 'n' && value[0] != 'N' &&
         value[0] != 'f' && value[0] != 'F';
}

CompactDeviceRatioParams makeCompactDeviceRatioParams(
    const spect::io::CompactSymmetryMetadata &meta, int compactRowBase,
    int symId, bool zMirror) {
  CompactDeviceRatioParams params;
  params.symmetryMode = 90;
  params.compactRowBase = compactRowBase;
  params.compactRaysPerAngle = meta.compactRaysPerAngle;
  params.compactNumAngles = meta.compactNumAngles;
  params.detNx = meta.detNx;
  params.detNz = meta.detNz;
  params.compactDetZ = meta.compactDetZ;
  params.fullNumAngles = meta.fullNumAngles;
  params.fullRaysPerAngle = meta.fullRaysPerAngle();
  params.symId = symId;
  params.zMirror = zMirror;
  return params;
}

CompactDeviceTransformParams makeCompactDeviceTransformParams(
    const spect::io::CompactSymmetryMetadata &meta, int symId, bool zMirror) {
  CompactDeviceTransformParams params;
  params.symmetryMode = 90;
  params.nx = meta.Nx;
  params.ny = meta.Ny;
  params.nz = meta.Nz;
  params.symId = symId;
  params.zMirror = zMirror;
  return params;
}

size_t compactAttenTableCount(const spect::io::CompactSymmetryMetadata &meta) {
  return static_cast<size_t>(meta.fullNumAngles) *
         static_cast<size_t>(meta.totalVoxels);
}


uint64_t estimatedCompactChunkResidentBytes(
    const spect::io::CompactChunkCSRDirectory &chunks) {
  uint64_t totalBytes = 0;
  for (const auto &info : chunks.chunks()) {
    if (std::numeric_limits<uint64_t>::max() - totalBytes < info.fileSizeBytes) {
      return std::numeric_limits<uint64_t>::max();
    }
    totalBytes += info.fileSizeBytes;
  }
  return totalBytes;
}

void warmCompactChunkResidentDeviceCache(
    const spect::io::CompactChunkCSRDirectory &chunks) {
  const uint64_t cacheBudget = chunks.cacheBudgetBytes();
  const uint64_t matrixBytes = estimatedCompactChunkResidentBytes(chunks);
  const bool hostCacheCoversMatrix =
      cacheBudget == std::numeric_limits<uint64_t>::max() ||
      cacheBudget >= matrixBytes;

  static std::mutex warmupMutex;
  std::lock_guard<std::mutex> lock(warmupMutex);
  auto lease = std::static_pointer_cast<CompactChunkResidentCacheLease>(
      chunks.acceleratorLease());
  if (lease && lease->cacheBudget == cacheBudget &&
      lease->residentWarmupComplete) {
    return;
  }

  constexpr uint64_t kMiB = 1024ull * 1024ull;
  if (cacheBudget != std::numeric_limits<uint64_t>::max() &&
      cacheBudget < matrixBytes) {
    uint64_t cachedPrefixBytes = 0;
    size_t cachedPrefixCount = 0;
    for (const auto &info : chunks.chunks()) {
      if (info.fileSizeBytes > cacheBudget - cachedPrefixBytes)
        break;
      cachedPrefixBytes += info.fileSizeBytes;
      ++cachedPrefixCount;
    }
    std::cerr << "[NOTE] compact host cache " << (cacheBudget / kMiB)
              << " MiB is below matrix size " << (matrixBytes / kMiB)
              << " MiB; estimated persistent prefix " << cachedPrefixCount
              << " / " << chunks.chunks().size() << ", up to "
              << ((matrixBytes - cachedPrefixBytes) / kMiB)
              << " MiB reread per matrix pass; GPU resident warmup remains "
                 "enabled.\n";
  }

  chunks.setAcceleratorLease({});
  lease = std::make_shared<CompactChunkResidentCacheLease>();
  lease->cacheBudget = cacheBudget;

  size_t residentPrefixCount = 1;
  try {
    const auto memory = cuda::getMemoryInfo();
    const uint64_t available =
        memory.free > kCompactChunkResidentReserveBytes
            ? static_cast<uint64_t>(memory.free) -
                  kCompactChunkResidentReserveBytes
            : 0;
    uint64_t bytes = 0;
    residentPrefixCount = 0;
    for (const auto &info : chunks.chunks()) {
      if (bytes > available || info.fileSizeBytes > available - bytes)
        break;
      bytes += info.fileSizeBytes;
      ++residentPrefixCount;
    }
    residentPrefixCount = std::max<size_t>(1, residentPrefixCount);
  } catch (const std::exception &ex) {
    std::cerr << "[NOTE] CUDA memory query failed before resident warmup: "
              << ex.what() << "; trying one matrix.\n";
    residentPrefixCount = 1;
  } catch (...) {
    std::cerr << "[NOTE] CUDA memory query failed before resident warmup; "
                 "trying one matrix.\n";
    residentPrefixCount = 1;
  }
  residentPrefixCount =
      std::min(residentPrefixCount, chunks.chunks().size());
  std::vector<std::shared_ptr<const spect::io::CompactChunkCSRData>> loaded;
  std::vector<spect::io::CSRView> views;
  loaded.reserve(residentPrefixCount);
  views.reserve(residentPrefixCount);
  for (size_t chunkIndex = 0; chunkIndex < residentPrefixCount; ++chunkIndex) {
    auto chunk = chunks.loadChunk(chunkIndex);
    views.push_back(chunk->view());
    loaded.push_back(std::move(chunk));
  }

  if (residentPrefixCount < chunks.chunks().size() && hostCacheCoversMatrix) {
    const auto *chunkDirectory = &chunks;
    lease->hostPreload = std::async(
        std::launch::async, [chunkDirectory, residentPrefixCount]() {
          try {
            for (size_t chunkIndex = residentPrefixCount;
                 chunkIndex < chunkDirectory->chunks().size(); ++chunkIndex) {
              (void)chunkDirectory->loadChunk(chunkIndex);
            }
          } catch (const std::exception &ex) {
            std::cerr << "[NOTE] compact host preload stopped: " << ex.what()
                      << "; foreground loading will retry.\n";
          }
        });
  }

  lease->residentWarmupComplete =
      views.empty() ||
      warmupResidentMatrixSetCUDA(views, kCompactChunkResidentReserveBytes);
  if (!lease->residentWarmupComplete) {
    std::cerr << "[NOTE] GPU resident matrix warmup was unavailable; "
                 "foreground matrix transfers remain active.\n";
  }
  // Join after the 2-chunk GPU upload so host load of the remaining chunks
  // overlaps that H2D. Leaving this future running pushed disk I/O into
  // iteration_core_s on the first numerator (Sampling 3-iter: 18 s vs 5 s).
  if (lease->hostPreload.valid()) {
    lease->hostPreload.get();
  }
  // Only register a complete host cache. Transient chunks in a partial cache
  // keep the existing pageable upload path and cannot acquire stale pins.
  if (hostCacheCoversMatrix) {
    for (size_t chunkIndex = 0; chunkIndex < chunks.chunks().size(); ++chunkIndex)
      lease->pin(chunks.loadChunk(chunkIndex));
    std::cerr << "[NOTE] compact host cache page-locked: "
              << lease->pinnedChunks.size() << " chunks, "
              << (matrixBytes / kMiB) << " MiB; registration owns chunk storage.\n";
  }
  chunks.setAcceleratorLease(lease);
}

// CSRQ (CompactChunkFormat::CsrqRunU16): the whole matrix is handed to the
// device once per chunk directory into matrix slot matrixIndex (resident, or
// streamed from pinned host memory when it does not fit).
struct CsrqResidentLease {
  bool complete = false;
  int matrixIndex = 0;
};

std::atomic<bool> g_csrqForceStreaming{false};
std::atomic<int> g_csrqTestFitSkips{0};
std::atomic<int> g_csrqTestFitFailures{0};

bool isCsrqDirectory(const spect::io::CompactChunkCSRDirectory &chunks) {
  return chunks.format() == spect::io::CompactChunkFormat::CsrqRunU16;
}

// Device bytes of one CSRQ chunk (row pointers, run pointers, row max, run
// words, u16 values); mirrors the state's estimate.
uint64_t csrqChunkDeviceBytes(const spect::io::CompactChunkCSRInfo &info) {
  const uint64_t rows = static_cast<uint64_t>(info.rows);
  return 2ull * sizeof(uint32_t) * (rows + 1ull) + sizeof(float) * rows +
         sizeof(uint32_t) * info.runCount + sizeof(uint16_t) * info.nnz;
}

void warmCsrqResident(const spect::io::CompactChunkCSRDirectory &chunks,
                      int matrixIndex, bool forceStreamed) {
  static std::mutex warmupMutex;
  std::lock_guard<std::mutex> lock(warmupMutex);
  auto lease = std::static_pointer_cast<CsrqResidentLease>(
      chunks.acceleratorLease());
  if (lease && lease->complete && lease->matrixIndex == matrixIndex &&
      csrqResidentCUDA(matrixIndex)) {
    return;
  }
  // Chunks are validated contiguous in angle order by the directory loader,
  // so the compact row base is the running row count. Shapes come from the
  // headers; chunk payloads are loaded one at a time and dropped right after
  // the state took them (uploaded or pinned), so the host holds one chunk.
  std::vector<CsrqChunkShape> shapes;
  shapes.reserve(chunks.chunks().size());
  int compactRowBase = 0;
  for (const auto &info : chunks.chunks()) {
    CsrqChunkShape shape;
    shape.rows = info.rows;
    shape.nnz = info.nnz;
    shape.runCount = info.runCount;
    shape.compactRowBase = compactRowBase;
    compactRowBase += info.rows;
    shapes.push_back(shape);
  }
  bool streamed = false;
  const bool forced = forceStreamed || g_csrqForceStreaming.load();
  if (!forced && g_csrqTestFitFailures.load() > 0) {
    if (g_csrqTestFitSkips.load() > 0) {
      g_csrqTestFitSkips.fetch_sub(1);
    } else {
      g_csrqTestFitFailures.fetch_sub(1);
      throw CsrqDeviceMemoryError(
          "CSRQ matrix does not fit on the device even streamed (test hook).");
    }
  }
  if (!csrqWarmupBeginCUDA(matrixIndex, chunks.cols(), shapes,
                           kCsrqResidentDeviceReserveBytes, forced,
                           &streamed)) {
    throw std::runtime_error(
        "CSRQ compact matrix requires the CUDA sparse backend.");
  }
  for (size_t chunkIndex = 0; chunkIndex < chunks.chunks().size(); ++chunkIndex) {
    const auto chunk = chunks.loadChunk(chunkIndex);
    if (!csrqWarmupChunkCUDA(matrixIndex, chunkIndex, chunk->runView())) {
      throw std::runtime_error(
          "CSRQ compact matrix requires the CUDA sparse backend.");
    }
  }
  if (!csrqWarmupEndCUDA(matrixIndex)) {
    throw std::runtime_error(
        "CSRQ compact matrix requires the CUDA sparse backend.");
  }
  lease = std::make_shared<CsrqResidentLease>();
  lease->complete = true;
  lease->matrixIndex = matrixIndex;
  chunks.setAcceleratorLease(lease);
}

int csrqAssignedSlot(const spect::io::CompactChunkCSRDirectory &chunks) {
  auto lease = std::static_pointer_cast<CsrqResidentLease>(
      chunks.acceleratorLease());
  return (lease && lease->complete) ? lease->matrixIndex : -1;
}

bool sameDirectory(const spect::io::CompactChunkCSRDirectory &a,
                   const spect::io::CompactChunkCSRDirectory &b) {
  if (&a == &b) {
    return true;
  }
  std::error_code ec;
  return std::filesystem::equivalent(a.directory(), b.directory(), ec);
}

// First device slot that is neither claimed by a channel in this set nor
// still holding another matrix. A slot is never reassigned while occupied:
// a second window silently replacing window 0's matrix would corrupt the
// joint likelihood, so exhaustion is a hard error.
int csrqFreeSlot(const std::vector<int> &claimed) {
  for (int slot = 0; slot < kCsrqMaxMatrices; ++slot) {
    const bool taken =
        std::find(claimed.begin(), claimed.end(), slot) != claimed.end() ||
        csrqResidentCUDA(slot);
    if (!taken) {
      return slot;
    }
  }
  throw std::runtime_error(
      "no free CSRQ device matrix slot: at most " +
      std::to_string(kCsrqMaxMatrices) +
      " distinct matrices can be resident (release the previous "
      "reconstruction first).");
}

// Device matrix slot of a CSRQ directory used outside a channel set (forward
// projection, single-window callers): keeps an existing lease, otherwise
// warms the matrix into a free slot (resident when it fits, else streamed).
int csrqMatrixIndexOf(const spect::io::CompactChunkCSRDirectory &chunks) {
  int slot = csrqAssignedSlot(chunks);
  if (slot < 0 || !csrqResidentCUDA(slot)) {
    slot = slot >= 0 ? slot : csrqFreeSlot({});
    warmCsrqResident(chunks, slot, false);
  }
  return slot;
}

// Residency plan for a channel set that is not loaded yet: the matrices that
// still have to be loaded are placed largest first, each kept resident only
// if the remaining ones can still get their two streaming slots; whatever
// does not fit is streamed (PCIe cost grows with the streamed bytes, so the
// small matrix streams first). Aliased channels share one decision.
std::vector<bool> planCsrqStreaming(
    const std::vector<CompactMatrixChannel> &channels,
    const std::vector<int> &slots) {
  std::vector<bool> streamed(channels.size(), false);
  struct Pending {
    size_t channel = 0;
    uint64_t bytes = 0;
    uint64_t twoSlots = 0;
  };
  std::vector<Pending> pending;
  uint64_t nvox = 0;
  uint64_t maxRows = 0;
  for (size_t index = 0; index < channels.size(); ++index) {
    const auto &chunks = *channels[index].chunks;
    if (!isCsrqDirectory(chunks) || csrqResidentCUDA(slots[index])) {
      continue;
    }
    bool seen = false;
    for (const auto &p : pending) {
      seen = seen || slots[p.channel] == slots[index];
    }
    if (seen) {
      continue;
    }
    Pending p;
    p.channel = index;
    uint64_t largest = 0;
    for (const auto &info : chunks.chunks()) {
      const uint64_t bytes = csrqChunkDeviceBytes(info);
      p.bytes += bytes;
      largest = std::max(largest, bytes);
    }
    p.twoSlots = static_cast<uint64_t>(kCsrqStreamSlots) * largest;
    pending.push_back(p);
    nvox = std::max<uint64_t>(nvox, static_cast<uint64_t>(chunks.cols()));
    maxRows = std::max<uint64_t>(maxRows, chunks.totalRows());
  }
  if (pending.size() < 2) {
    return streamed; // the state decides alone for a single matrix
  }
  std::sort(pending.begin(), pending.end(),
            [](const Pending &a, const Pending &b) { return a.bytes > b.bytes; });
  uint64_t freeBytes = 0;
  try {
    freeBytes = static_cast<uint64_t>(cuda::getMemoryInfo().free);
  } catch (...) {
    return streamed;
  }
  const uint64_t scratch = 2ull * nvox * kCsrqBranchCount * sizeof(float) +
                           maxRows * kCsrqBranchCount * sizeof(float);
  const uint64_t fixed = kCsrqResidentDeviceReserveBytes + scratch;
  uint64_t budget = freeBytes > fixed ? freeBytes - fixed : 0ull;
  constexpr uint64_t kMiB = 1024ull * 1024ull;
  std::cerr << "[NOTE] CSRQ residency plan for " << pending.size()
            << " matrices: free " << (freeBytes / kMiB) << " MiB, reserve+scratch "
            << (fixed / kMiB) << " MiB;";
  for (size_t i = 0; i < pending.size(); ++i) {
    uint64_t othersMinimum = 0;
    for (size_t j = i + 1; j < pending.size(); ++j) {
      othersMinimum += pending[j].twoSlots;
    }
    const bool resident = pending[i].bytes + othersMinimum <= budget;
    budget -= std::min(budget, resident ? pending[i].bytes : pending[i].twoSlots);
    std::cerr << " matrix " << slots[pending[i].channel] << " ("
              << (pending[i].bytes / kMiB) << " MiB) "
              << (resident ? "resident" : "streamed") << ';';
    if (!resident) {
      for (size_t index = 0; index < channels.size(); ++index) {
        if (slots[index] == slots[pending[i].channel]) {
          streamed[index] = true;
        }
      }
    }
  }
  std::cerr << '\n';
  return streamed;
}

bool isCsrqChannelSet(const std::vector<CompactMatrixChannel> &channels) {
  return !channels.empty() && isCsrqDirectory(*channels.front().chunks);
}

// Shape/format rules shared by every channel-aware entry point.
void validateChannels(const std::vector<CompactMatrixChannel> &channels,
                      const spect::io::CompactSymmetryMetadata &meta,
                      const char *what) {
  if (channels.empty() || channels.size() > static_cast<size_t>(kCsrqMaxMatrices)) {
    throw std::runtime_error(std::string("compact ") + what + " needs 1.." +
                             std::to_string(kCsrqMaxMatrices) +
                             " matrix channels.");
  }
  for (const auto &channel : channels) {
    if (channel.chunks == nullptr) {
      throw std::runtime_error(std::string("compact ") + what +
                               " received a null matrix channel.");
    }
    if (channel.chunks->totalRows() !=
            static_cast<uint64_t>(meta.compactRows()) ||
        channel.chunks->cols() != meta.totalVoxels) {
      throw std::runtime_error(std::string("compact ") + what +
                               " matrix/metadata mismatch.");
    }
    if (!(channel.weight > 0.0)) {
      throw std::runtime_error(std::string("compact ") + what +
                               " channel weight must be positive.");
    }
    if (channels.size() > 1 && !isCsrqDirectory(*channel.chunks)) {
      throw std::runtime_error(
          "joint multi-window reconstruction requires CSRQ matrices for "
          "every channel; convert the CSRB directories first.");
    }
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error("compact chunk CUDA helper requires Nx == Ny.");
  }
}

// Device matrix slot per channel: an existing lease wins, channels naming
// the same directory as an earlier channel share its copy (one matrix
// modelling several windows with different weights), and every other
// directory takes a free slot in channel order.
std::vector<int> csrqChannelMatrixIndices(
    const std::vector<CompactMatrixChannel> &channels) {
  std::vector<int> indices(channels.size(), -1);
  std::vector<int> claimed;
  for (size_t index = 0; index < channels.size(); ++index) {
    const int assigned = csrqAssignedSlot(*channels[index].chunks);
    if (assigned >= 0) {
      indices[index] = assigned;
      claimed.push_back(assigned);
    }
  }
  for (size_t index = 0; index < channels.size(); ++index) {
    if (indices[index] >= 0) {
      continue;
    }
    for (size_t earlier = 0; earlier < index; ++earlier) {
      if (indices[earlier] >= 0 &&
          sameDirectory(*channels[earlier].chunks, *channels[index].chunks)) {
        indices[index] = indices[earlier];
        break;
      }
    }
    if (indices[index] < 0) {
      indices[index] = csrqFreeSlot(claimed);
      claimed.push_back(indices[index]);
    }
  }
  return indices;
}

// Loads every channel's matrix. The residency plan is advisory: under WDDM
// cudaMemGetInfo can overstate the free memory (another process's
// allocations are not always visible), so a matrix planned resident may leave
// too little room for the next one. When a later matrix does not fit even
// streamed, the matrices this call already made resident are demoted to
// streamed (re-read from their directories into pinned memory) and the
// failed one is retried with the freed memory, instead of aborting the run.
void warmCompactChunks(const std::vector<CompactMatrixChannel> &channels) {
  const std::vector<int> slots = csrqChannelMatrixIndices(channels);
  std::vector<bool> streamed = planCsrqStreaming(channels, slots);
  std::vector<size_t> warmedHere;
  for (size_t index = 0; index < channels.size(); ++index) {
    const auto &chunks = *channels[index].chunks;
    if (!isCsrqDirectory(chunks)) {
      warmCompactChunkResidentDeviceCache(chunks);
      continue;
    }
    bool aliased = false;
    for (size_t earlier = 0; earlier < index && !aliased; ++earlier) {
      aliased = slots[earlier] == slots[index];
    }
    if (aliased) {
      // Same directory as an earlier channel: reuse its device copy.
      if (csrqAssignedSlot(chunks) != slots[index]) {
        auto lease = std::make_shared<CsrqResidentLease>();
        lease->complete = true;
        lease->matrixIndex = slots[index];
        chunks.setAcceleratorLease(lease);
      }
      continue;
    }
    if (csrqResidentCUDA(slots[index])) {
      continue;
    }
    try {
      warmCsrqResident(chunks, slots[index], streamed[index]);
    } catch (const CsrqDeviceMemoryError &ex) {
      std::vector<size_t> residentEarlier;
      for (size_t earlier : warmedHere) {
        if (!streamed[earlier]) {
          residentEarlier.push_back(earlier);
        }
      }
      if (residentEarlier.empty()) {
        throw;
      }
      std::cerr << "[NOTE] " << ex.what()
                << "\n[NOTE] CSRQ residency: demoting "
                << residentEarlier.size()
                << " resident matrix(es) loaded by this call to streamed and "
                   "retrying (the free-memory estimate under WDDM can be "
                   "optimistic).\n";
      for (size_t earlier : residentEarlier) {
        streamed[earlier] = true;
        // Drop the lease so the matrix is reloaded (the warmup begin frees
        // its resident chunks) instead of being reported as already warm.
        channels[earlier].chunks->setAcceleratorLease(nullptr);
        warmCsrqResident(*channels[earlier].chunks, slots[earlier], true);
      }
      warmCsrqResident(chunks, slots[index], false);
    }
    warmedHere.push_back(index);
  }
}

void warmCompactChunks(const spect::io::CompactChunkCSRDirectory &chunks) {
  if (isCsrqDirectory(chunks)) {
    (void)csrqMatrixIndexOf(chunks);
  } else {
    warmCompactChunkResidentDeviceCache(chunks);
  }
}

std::vector<CompactMatrixChannel>
singleChannel(const spect::io::CompactChunkCSRDirectory &chunks) {
  CompactMatrixChannel channel;
  channel.chunks = &chunks;
  channel.weight = 1.0;
  return {channel};
}

int csrqProjectionSlot(int channel, int bed) {
  return channel * spect::core::kMaxBeds + bed;
}

struct CompactBranchDescriptor {
  int symId = 0;
  bool zMirror = false;
};

void prefetchNextCompactChunk(
    const spect::io::CompactChunkCSRDirectory &chunks, size_t chunkIndex,
    size_t chunkCount, bool reverseChunkOrder,
    const spect::io::CSRView &current) {
  if (chunkCount == 0) {
    return;
  }
  size_t nextIndex = 0;
  if (reverseChunkOrder) {
    if (chunkIndex == 0) {
      return;
    }
    nextIndex = chunkIndex - 1;
  } else {
    if (chunkIndex + 1 >= chunkCount) {
      return;
    }
    nextIndex = chunkIndex + 1;
  }
  const auto next = chunks.loadChunk(nextIndex);
  prefetchCompactFloatMatrixCUDA(next->view(), current);
}

std::vector<CompactBranchDescriptor>
enumerateCompactBranches(const spect::io::CompactSymmetryMetadata &meta) {
  std::vector<CompactBranchDescriptor> branches;
  branches.reserve(static_cast<size_t>(meta.xySymmetryCount() *
                                       meta.zBranchCount()));
  for (int sym = 0; sym < meta.xySymmetryCount(); ++sym) {
    for (int z = 0; z < meta.zBranchCount(); ++z) {
      branches.push_back(CompactBranchDescriptor{sym, z != 0});
    }
  }
  return branches;
}

void fillBranchSymmetryTables(const std::vector<CompactBranchDescriptor> &branches,
                              std::vector<int> &symIds,
                              std::vector<int> &zMirrors) {
  symIds.resize(branches.size());
  zMirrors.resize(branches.size());
  for (size_t b = 0; b < branches.size(); ++b) {
    symIds[b] = branches[b].symId;
    zMirrors[b] = branches[b].zMirror ? 1 : 0;
  }
}



void transformForwardInto(const double *input, double *output,
                          const spect::io::CompactSymmetryMetadata &meta,
                          int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(nvox); ++idx) {
    const size_t uidx = static_cast<size_t>(idx);
    const int iz = static_cast<int>(uidx / plane);
    const int rem = static_cast<int>(uidx - static_cast<size_t>(iz) * plane);
    const int iy = rem / nx;
    const int ix = rem - iy * nx;
    const spect::io::CompactVoxelCoord mapped =
        spect::io::mapCompactVoxelCoord(ix, iy, iz, nx, nz, symId, zMirror);
    output[uidx] =
        input[static_cast<size_t>(mapped.z) * plane +
              static_cast<size_t>(mapped.y) * static_cast<size_t>(nx) +
              static_cast<size_t>(mapped.x)];
  }
}

void transformForwardIntoFloat(const double *input, float *output,
                               const spect::io::CompactSymmetryMetadata &meta,
                               int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(nvox); ++idx) {
    const size_t uidx = static_cast<size_t>(idx);
    const int iz = static_cast<int>(uidx / plane);
    const int rem = static_cast<int>(uidx - static_cast<size_t>(iz) * plane);
    const int iy = rem / nx;
    const int ix = rem - iy * nx;
    const spect::io::CompactVoxelCoord mapped =
        spect::io::mapCompactVoxelCoord(ix, iy, iz, nx, nz, symId, zMirror);
    output[uidx] = static_cast<float>(
        input[static_cast<size_t>(mapped.z) * plane +
              static_cast<size_t>(mapped.y) * static_cast<size_t>(nx) +
              static_cast<size_t>(mapped.x)]);
  }
}

void transformForward(const double *input, std::vector<double> &output,
                      const spect::io::CompactSymmetryMetadata &meta,
                      int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);
  output.resize(nvox);
  transformForwardInto(input, output.data(), meta, symId, zMirror, threads);
}

void transformBackwardAdd(const float *rotated, std::vector<double> &accum,
                          const spect::io::CompactSymmetryMetadata &meta,
                          int symId, bool zMirror, int threads) {
  const int nx = meta.Nx;
  const int ny = meta.Ny;
  const int nz = meta.Nz;
  const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  const size_t nvox = plane * static_cast<size_t>(nz);
  if (accum.size() != nvox) {
    accum.assign(nvox, 0.0);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(nvox); ++idx) {
    const size_t uidx = static_cast<size_t>(idx);
    const int iz = static_cast<int>(uidx / plane);
    const int rem = static_cast<int>(uidx - static_cast<size_t>(iz) * plane);
    const int iy = rem / nx;
    const int ix = rem - iy * nx;
    const spect::io::CompactVoxelCoord mapped =
        spect::io::mapCompactVoxelCoord(ix, iy, iz, nx, nz, symId, zMirror);
    accum[static_cast<size_t>(mapped.z) * plane +
          static_cast<size_t>(mapped.y) * static_cast<size_t>(nx) +
          static_cast<size_t>(mapped.x)] +=
        static_cast<double>(rotated[uidx]);
  }
}

// Host sensitivity input for one branch: rowWeights[fullIndex] (1 when null)
// where the row maps to a full projection row, else 0.
void makeUnitRatioChunkFloat(
    const spect::io::CompactSymmetryMetadata &meta, int compactRowBase,
    int rows, int symId, bool zMirror, const double *rowWeights,
    std::vector<float> &ratio, int threads) {
  ratio.resize(static_cast<size_t>(rows));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(effectiveThreads(threads))
#endif
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(rows); ++row) {
    uint64_t fullIndex = 0;
    float value = 0.0f;
    if (spect::io::fullProjectionIndexFromCompactRow(
            meta, compactRowBase + static_cast<int>(row), symId, zMirror,
            fullIndex)) {
      value = rowWeights == nullptr
                  ? 1.0f
                  : static_cast<float>(rowWeights[fullIndex]);
    }
    ratio[static_cast<size_t>(row)] = value;
  }
}

template <typename ValueType>
void scatterCompactForwardProjection(
    const spect::io::CompactSymmetryMetadata &meta, int compactRowBase,
    const std::vector<ValueType> &compactProjection, int symId, bool zMirror,
    double *fullExpected, std::vector<uint8_t> &coverage) {
  for (size_t row = 0; row < compactProjection.size(); ++row) {
    uint64_t fullIndex = 0;
    if (!spect::io::fullProjectionIndexFromCompactRow(
            meta, compactRowBase + static_cast<int>(row), symId, zMirror,
            fullIndex)) {
      continue;
    }
    const size_t index = static_cast<size_t>(fullIndex);
    if (index >= coverage.size() || coverage[index] != 0) {
      throw std::runtime_error(
          "compact forward projection produced duplicate/out-of-range full row.");
    }
    const double value = static_cast<double>(compactProjection[row]);
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "compact forward projection produced a non-finite value.");
    }
    fullExpected[index] = value;
    coverage[index] = 1;
  }
}

void requireCompleteForwardProjectionCoverage(
    const std::vector<uint8_t> &coverage) {
  if (std::find(coverage.begin(), coverage.end(), uint8_t{0}) !=
      coverage.end()) {
    throw std::runtime_error(
        "compact forward projection did not cover every full projection row.");
  }
}

// CSRQ numerator for 1..kMaxBeds beds and 1..kCsrqMaxMatrices windows:
// every window of every bed accumulates weight_k * H_k^T(p_k / mean_k) into
// the bed's device accumulator. fullProjections / fullBackgrounds are
// indexed [channel * bedCount + bed]; the CSRQ projection cache slot is
// channel * kMaxBeds + bed. attFactors (null or one full-angle table per bed)
// turns A into H = A att_bed on the device, per compact angle. Channels are
// already validated and warm.
bool runCsrqMlemNumerMultiBed(
    const std::vector<CompactMatrixChannel> &channels,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *const *fullProjections,
    const double *const *fullBackgrounds,
    const double *const *smoothedImages, int bedCount, int threads,
    double *const *numers, CompactSymmetricStageTiming *timing,
    const uint16_t *const *attFactors, bool refreshProjectionInputs,
    bool branchBatchInputPrepared, bool downloadBackwardOutputs) {
  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const auto branches = enumerateCompactBranches(meta);
  const size_t branchCount = branches.size();
  const size_t channelCount = channels.size();
  const size_t fullProjectionCount = static_cast<size_t>(meta.fullRows());

  auto projectionAt = [&](size_t channel, int bed) {
    return fullProjections[channel * static_cast<size_t>(bedCount) +
                           static_cast<size_t>(bed)];
  };
  auto backgroundAt = [&](size_t channel, int bed) -> const double * {
    return fullBackgrounds == nullptr
               ? nullptr
               : fullBackgrounds[channel * static_cast<size_t>(bedCount) +
                                 static_cast<size_t>(bed)];
  };

  if (refreshProjectionInputs) {
    for (size_t channel = 0; channel < channelCount; ++channel) {
      for (int bed = 0; bed < bedCount; ++bed) {
        if (!refreshCompactProjectionCUDA(
                projectionAt(channel, bed), fullProjectionCount,
                csrqProjectionSlot(static_cast<int>(channel), bed),
                backgroundAt(channel, bed))) {
          return false;
        }
      }
    }
  }
  if (!branchBatchInputPrepared) {
    std::vector<float> rotatedImages(
        static_cast<size_t>(bedCount) * branchCount * nvox, 0.0f);
    for (int bed = 0; bed < bedCount; ++bed) {
      for (size_t branchIndex = 0; branchIndex < branchCount; ++branchIndex) {
        const auto &branch = branches[branchIndex];
        float *imgRot =
            rotatedImages.data() +
            (static_cast<size_t>(bed) * branchCount + branchIndex) * nvox;
        const auto stageStart = Clock::now();
        transformForwardIntoFloat(smoothedImages[bed], imgRot, meta,
                                  branch.symId, branch.zMirror, threads);
        if (timing)
          addTiming(timing->transformForwardSeconds, stageStart);
      }
    }
    if (!refreshCompactBranchBatchInputCUDA(rotatedImages.data(),
                                            rotatedImages.size())) {
      return false;
    }
  }
  std::vector<int> symIds;
  std::vector<int> zMirrors;
  fillBranchSymmetryTables(branches, symIds, zMirrors);
  const CompactDeviceRatioParams commonRatio =
      makeCompactDeviceRatioParams(meta, 0, 0, false);
  const CompactDeviceTransformParams commonTransform =
      makeCompactDeviceTransformParams(meta, 0, false);
  for (int bed = 0; bed < bedCount; ++bed) {
    if (downloadBackwardOutputs)
      std::fill(numers[bed], numers[bed] + nvox, 0.0);
    if (!compactDeviceBackwardBeginCUDA(nvox, bed)) {
      return false;
    }
  }
  // One chunk-outer pass per window over all beds (a streamed matrix then
  // crosses PCIe once per iteration); windows accumulate into the same
  // per-bed accumulators.
  for (size_t channel = 0; channel < channelCount; ++channel) {
    const double *const *channelProjections =
        fullProjections + channel * static_cast<size_t>(bedCount);
    const double *const *channelBackgrounds =
        fullBackgrounds == nullptr
            ? nullptr
            : fullBackgrounds + channel * static_cast<size_t>(bedCount);
    CompactDeviceRatioTiming deviceTiming;
    CompactDeviceBackwardTiming backwardTiming;
    if (!csrqRunNumerCUDA(csrqMatrixIndexOf(*channels[channel].chunks),
                          bedCount, static_cast<int>(branchCount), commonRatio,
                          commonTransform, symIds.data(), zMirrors.data(),
                          channelProjections, channelBackgrounds,
                          fullProjectionCount, channels[channel].weight, 0,
                          csrqProjectionSlot(static_cast<int>(channel), 0),
                          &deviceTiming, &backwardTiming, attFactors,
                          attFactors == nullptr ? 0
                                                : compactAttenTableCount(meta),
                          channels[channel].attenPower)) {
      return false;
    }
    if (timing) {
      timing->forwardSeconds += deviceTiming.forwardSeconds;
      timing->ratioSeconds += deviceTiming.ratioSeconds;
      timing->backwardSeconds += deviceTiming.backwardSeconds;
      timing->transformBackwardSeconds +=
          backwardTiming.transformBackwardSeconds;
    }
  }
  if (downloadBackwardOutputs) {
    for (int bed = 0; bed < bedCount; ++bed) {
      if (!compactDeviceBackwardDownloadCUDA(numers[bed], nvox, bed)) {
        return false;
      }
    }
  }
  if (timing) {
    timing->numeratorSeconds += elapsedSeconds(totalStart);
    ++timing->numeratorCalls;
  }
  return true;
}

} // namespace

bool computePoissonSensitivityCUDA(
    const std::vector<CompactMatrixChannel> &channels,
    const spect::io::CompactSymmetryMetadata &meta, int threads, double *denom,
    CompactSymmetricStageTiming *timing, const uint16_t *attFactors,
    const double *rowWeights) {
  validateChannels(channels, meta, "sensitivity");
  warmCompactChunks(channels);

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullProjectionCount = static_cast<size_t>(meta.fullRows());
  const auto branches = enumerateCompactBranches(meta);
  std::fill(denom, denom + nvox, 0.0);

  if (isCsrqChannelSet(channels)) {
    std::vector<int> symIds;
    std::vector<int> zMirrors;
    fillBranchSymmetryTables(branches, symIds, zMirrors);
    if (!compactDeviceBackwardBeginCUDA(nvox, 0)) {
      return false;
    }
    for (size_t channel = 0; channel < channels.size(); ++channel) {
      CompactDeviceBackwardTiming backwardTiming;
      if (!csrqRunSensitivityCUDA(
              csrqMatrixIndexOf(*channels[channel].chunks),
              static_cast<int>(branches.size()), channels[channel].weight,
              makeCompactDeviceRatioParams(meta, 0, 0, false),
              makeCompactDeviceTransformParams(meta, 0, false), symIds.data(),
              zMirrors.data(), 0, fullProjectionCount, rowWeights,
              &backwardTiming, attFactors,
              attFactors == nullptr ? 0 : compactAttenTableCount(meta),
              channels[channel].attenPower)) {
        return false;
      }
      if (timing) {
        timing->backwardSeconds += backwardTiming.backwardSeconds;
        timing->transformBackwardSeconds +=
            backwardTiming.transformBackwardSeconds;
      }
    }
    if (!compactDeviceBackwardDownloadCUDA(denom, nvox, 0)) {
      return false;
    }
    if (timing) {
      timing->sensitivitySeconds += elapsedSeconds(totalStart);
      ++timing->sensitivityCalls;
    }
    return true;
  }

  const spect::io::CompactChunkCSRDirectory &chunks = *channels.front().chunks;
  const size_t chunkCount = chunks.chunks().size();
  const bool useBranchBatch =
      compactBranchBatchEnabled() && branches.size() > 1;
  if (attFactors != nullptr && !useBranchBatch) {
    throw std::runtime_error(
        "compact CTAC sensitivity needs the branch-batch device path "
        "(SPECT_COMPACT_BRANCH_BATCH must not be disabled).");
  }
  if (!compactDeviceBackwardBeginCUDA(nvox, 0)) {
    return false;
  }
  std::vector<int> symIds;
  std::vector<int> zMirrors;
  fillBranchSymmetryTables(branches, symIds, zMirrors);
  const CompactDeviceTransformParams commonTransform =
      makeCompactDeviceTransformParams(meta, 0, false);
  const double attenPower = channels.front().attenPower;

  std::vector<float> ratio;
  for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
    const auto chunk = chunks.loadChunk(chunkIndex);
    const spect::io::CSRView view = chunk->view();
    const int compactRowBase = chunk->info.angleOffset * meta.compactRaysPerAngle;
    if (useBranchBatch) {
      CompactDeviceBackwardTiming backwardTiming;
      const CompactDeviceRatioParams commonRatio =
          makeCompactDeviceRatioParams(meta, compactRowBase, 0, false);
      const bool ok =
          attFactors != nullptr
              ? compactSensitivityBranchBatchAttenSpmvTAccumulateFloatCUDA(
                    view, commonRatio, commonTransform, symIds.data(),
                    zMirrors.data(), static_cast<int>(branches.size()),
                    attFactors, compactAttenTableCount(meta), attenPower,
                    rowWeights, fullProjectionCount, &backwardTiming, 0)
              : compactSensitivityBranchBatchSpmvTAccumulateFloatCUDA(
                    view, commonTransform, symIds.data(), zMirrors.data(),
                    static_cast<int>(branches.size()), &backwardTiming, 0, 0,
                    &commonRatio, rowWeights,
                    rowWeights == nullptr ? 0 : fullProjectionCount);
      if (!ok) {
        return false;
      }
      if (timing) {
        timing->backwardSeconds += backwardTiming.backwardSeconds;
        timing->transformBackwardSeconds +=
            backwardTiming.transformBackwardSeconds;
      }
      continue;
    }
    for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
      const auto &branch = branches[branchIndex];
      auto stageStart = Clock::now();
      makeUnitRatioChunkFloat(meta, compactRowBase, chunk->info.rows,
                              branch.symId, branch.zMirror, rowWeights, ratio,
                              threads);
      if (timing)
        addTiming(timing->ratioSeconds, stageStart);
      CompactDeviceBackwardTiming deviceTiming;
      const CompactDeviceTransformParams transformParams =
          makeCompactDeviceTransformParams(meta, branch.symId, branch.zMirror);
      if (!compactDeviceBackwardSpmvTAccumulateFloatCUDA(
              view, ratio.data(), transformParams, &deviceTiming, 0)) {
        return false;
      }
      if (timing) {
        timing->backwardSeconds += deviceTiming.backwardSeconds;
        timing->transformBackwardSeconds +=
            deviceTiming.transformBackwardSeconds;
      }
    }
  }

  if (!compactDeviceBackwardDownloadCUDA(denom, nvox, 0)) {
    return false;
  }
  if (timing) {
    timing->sensitivitySeconds += elapsedSeconds(totalStart);
    ++timing->sensitivityCalls;
  }
  return true;
}

bool computeCompactChunkSensitivityCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta, int threads, double *denom,
    CompactSymmetricStageTiming *timing, const uint16_t *attFactors) {
  return computePoissonSensitivityCUDA(singleChannel(chunks), meta, threads,
                                       denom, timing, attFactors);
}

bool prepareChannelExecutionCUDA(
    const std::vector<CompactMatrixChannel> &channels) {
  if (channels.empty()) {
    throw std::runtime_error("compact execution prepare needs a matrix.");
  }
  warmCompactChunks(channels);
  return true;
}

void setCsrqForceStreamingCUDA(bool force) { g_csrqForceStreaming.store(force); }

void setCsrqWarmupFitFailuresForTestsCUDA(int skip, int count) {
  g_csrqTestFitSkips.store(std::max(0, skip));
  g_csrqTestFitFailures.store(std::max(0, count));
}

bool csrqForceStreamingCUDA() { return g_csrqForceStreaming.load(); }

bool prepareCompactChunkExecutionCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks) {
  warmCompactChunks(chunks);
  return true;
}

bool prepareCompactBranchBatchInputFromGlobalCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    const double *globalImage, size_t globalImageCount, int globalNz,
    const int *bedZStarts, int bedCount, bool flipLocalZ,
    const spect::filters::PSFKernels *castorPsf, bool uploadGlobalImage) {
  if (bedCount < 1 || !compactBranchBatchEnabled())
    return false;
  const CompactDeviceTransformParams commonTransform =
      makeCompactDeviceTransformParams(meta, 0, false);
  return spect::cuda::prepareCompactBranchBatchInputFromGlobalCUDA(
      globalImage, globalImageCount, globalNz, bedZStarts, bedCount,
      flipLocalZ, uploadGlobalImage, commonTransform,
      meta.xySymmetryCount() * meta.zBranchCount(), meta.zBranchCount(),
      castorPsf == nullptr ? nullptr : &castorPsf->kx,
      castorPsf == nullptr ? nullptr : &castorPsf->ky,
      castorPsf == nullptr ? nullptr : &castorPsf->kz);
}

bool finishCompactResidentMlemIterationCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ, const double *denom, const uint8_t *mask,
    const spect::filters::PSFKernels *castorPsf,
    bool requirePositiveNumerator, bool refreshStaticInputs,
    bool *imageAllZero) {
  const CompactDeviceTransformParams commonTransform =
      makeCompactDeviceTransformParams(meta, 0, false);
  return spect::cuda::finishCompactResidentMlemIterationCUDA(
      globalImageCount, globalNz, bedZStarts, bedCount, flipLocalZ,
      commonTransform, denom, mask,
      castorPsf == nullptr ? nullptr : &castorPsf->kx,
      castorPsf == nullptr ? nullptr : &castorPsf->ky,
      castorPsf == nullptr ? nullptr : &castorPsf->kz,
      requirePositiveNumerator, refreshStaticInputs, imageAllZero);
}

bool finishCompactResidentBsremIterationCUDA(
    const spect::io::CompactSymmetryMetadata &meta,
    size_t globalImageCount, int globalNz, const int *bedZStarts,
    int bedCount, bool flipLocalZ, const double *denom, const uint8_t *mask,
    const spect::filters::PSFKernels *castorPsf,
    const CompactBsremUpdateParams &params, bool requirePositiveNumerator,
    bool refreshStaticInputs, bool *imageAllZero) {
  const CompactDeviceTransformParams commonTransform =
      makeCompactDeviceTransformParams(meta, 0, false);
  return spect::cuda::finishCompactResidentBsremIterationCUDA(
      globalImageCount, globalNz, bedZStarts, bedCount, flipLocalZ,
      commonTransform, denom, mask,
      castorPsf == nullptr ? nullptr : &castorPsf->kx,
      castorPsf == nullptr ? nullptr : &castorPsf->ky,
      castorPsf == nullptr ? nullptr : &castorPsf->kz, params,
      requirePositiveNumerator, refreshStaticInputs, imageAllZero);
}

bool runCompactChunkMlemNumerCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *fullProjection, const double *fullBackground,
    const double *smoothedImage, int threads, double *numer,
    bool reverseChunkOrder, CompactSymmetricStageTiming *timing,
    const uint16_t *attFactors, bool refreshProjectionInput) {
  // One-bed form of the channel path; both formats share it.
  const double *const projections[] = {fullProjection};
  const double *const backgrounds[] = {fullBackground};
  const double *const images[] = {smoothedImage};
  double *const numers[] = {numer};
  const uint16_t *const attFactorList[] = {attFactors};
  return accumulatePoissonBackprojectedRatioCUDA(
      singleChannel(chunks), meta, projections,
      fullBackground == nullptr ? nullptr : backgrounds, images, 1, threads,
      numers, reverseChunkOrder, timing,
      attFactors == nullptr ? nullptr : attFactorList, refreshProjectionInput,
      false, true);
}

bool computeCompactChunkForwardProjectionCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *smoothedImage, int threads, double *fullExpected,
    const uint16_t *attFactors, double attenPower) {
  if (chunks.totalRows() != static_cast<uint64_t>(meta.compactRows()) ||
      chunks.cols() != meta.totalVoxels) {
    throw std::runtime_error(
        "compact chunk forward projection matrix/metadata mismatch.");
  }
  if (meta.Nx != meta.Ny) {
    throw std::runtime_error(
        "compact chunk forward projection requires Nx == Ny.");
  }
  if (smoothedImage == nullptr || fullExpected == nullptr) {
    throw std::runtime_error(
        "compact chunk forward projection received null image/output data.");
  }
  warmCompactChunks(chunks);

  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const size_t fullRows = static_cast<size_t>(meta.fullRows());
  std::fill(fullExpected, fullExpected + fullRows, 0.0);
  std::vector<uint8_t> coverage(fullRows, 0);

  if (isCsrqDirectory(chunks)) {
    const int matrixIndex = csrqMatrixIndexOf(chunks);
    const auto branches = enumerateCompactBranches(meta);
    std::vector<float> rotatedImages(branches.size() * nvox, 0.0f);
    for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
      transformForwardIntoFloat(smoothedImage,
                                rotatedImages.data() + branchIndex * nvox, meta,
                                branches[branchIndex].symId,
                                branches[branchIndex].zMirror, threads);
    }
    if (!refreshCompactBranchBatchInputCUDA(rotatedImages.data(),
                                            rotatedImages.size())) {
      return false;
    }
    const size_t compactRows = static_cast<size_t>(chunks.totalRows());
    std::vector<float> yhatPacked(compactRows * branches.size(), 0.0f);
    std::vector<int> symIds;
    std::vector<int> zMirrors;
    fillBranchSymmetryTables(branches, symIds, zMirrors);
    const CompactDeviceRatioParams commonRatio =
        makeCompactDeviceRatioParams(meta, 0, 0, false);
    const CompactDeviceTransformParams commonTransform =
        makeCompactDeviceTransformParams(meta, 0, false);
    if (!csrqRunForwardCUDA(matrixIndex, static_cast<int>(branches.size()),
                            yhatPacked.data(), yhatPacked.size(), &commonRatio,
                            &commonTransform, symIds.data(), zMirrors.data(),
                            attFactors,
                            attFactors == nullptr ? 0
                                                  : compactAttenTableCount(meta),
                            attenPower)) {
      return false;
    }
    // Packed [compactRow][branch] -> one full projection row per valid pair,
    // same coverage contract as the CSRB path below.
    for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
      const auto &branch = branches[branchIndex];
      for (size_t row = 0; row < compactRows; ++row) {
        uint64_t fullIndex = 0;
        if (!spect::io::fullProjectionIndexFromCompactRow(
                meta, static_cast<int>(row), branch.symId, branch.zMirror,
                fullIndex)) {
          continue;
        }
        const size_t index = static_cast<size_t>(fullIndex);
        if (index >= coverage.size() || coverage[index] != 0) {
          throw std::runtime_error(
              "compact forward projection produced duplicate/out-of-range "
              "full row.");
        }
        const double value = static_cast<double>(
            yhatPacked[row * branches.size() + branchIndex]);
        if (!std::isfinite(value)) {
          throw std::runtime_error(
              "compact forward projection produced a non-finite value.");
        }
        fullExpected[index] = value;
        coverage[index] = 1;
      }
    }
    requireCompleteForwardProjectionCoverage(coverage);
    return true;
  }

  if (attFactors != nullptr) {
    // CSRB CTAC: the eight rotated images as one branch batch, then per
    // chunk the attenuated per-angle forward ([branch][chunkRow] output).
    const auto branches = enumerateCompactBranches(meta);
    std::vector<float> rotatedImages(branches.size() * nvox, 0.0f);
    for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
      transformForwardIntoFloat(smoothedImage,
                                rotatedImages.data() + branchIndex * nvox, meta,
                                branches[branchIndex].symId,
                                branches[branchIndex].zMirror, threads);
    }
    if (!refreshCompactBranchBatchInputCUDA(rotatedImages.data(),
                                            rotatedImages.size())) {
      return false;
    }
    std::vector<int> symIds;
    std::vector<int> zMirrors;
    fillBranchSymmetryTables(branches, symIds, zMirrors);
    const CompactDeviceTransformParams commonTransform =
        makeCompactDeviceTransformParams(meta, 0, false);
    std::vector<float> yBatch;
    std::vector<float> y;
    for (size_t chunkIndex = 0; chunkIndex < chunks.chunks().size();
         ++chunkIndex) {
      const auto chunk = chunks.loadChunk(chunkIndex);
      const spect::io::CSRView view = chunk->view();
      const int compactRowBase =
          chunk->info.angleOffset * meta.compactRaysPerAngle;
      const size_t rows = static_cast<size_t>(view.rows);
      yBatch.assign(rows * branches.size(), 0.0f);
      if (!compactBranchBatchAttenForwardCUDA(
              view, makeCompactDeviceRatioParams(meta, compactRowBase, 0, false),
              commonTransform, symIds.data(), zMirrors.data(),
              static_cast<int>(branches.size()), attFactors,
              compactAttenTableCount(meta), attenPower, yBatch.data(),
              yBatch.size())) {
        return false;
      }
      for (size_t branchIndex = 0; branchIndex < branches.size(); ++branchIndex) {
        y.assign(yBatch.begin() + static_cast<ptrdiff_t>(branchIndex * rows),
                 yBatch.begin() + static_cast<ptrdiff_t>((branchIndex + 1) * rows));
        scatterCompactForwardProjection(meta, compactRowBase, y,
                                        branches[branchIndex].symId,
                                        branches[branchIndex].zMirror,
                                        fullExpected, coverage);
      }
    }
    requireCompleteForwardProjectionCoverage(coverage);
    return true;
  }

  std::vector<float> imgRot(nvox, 0.0f);
  std::vector<float> y;
  for (const auto &branch : enumerateCompactBranches(meta)) {
    transformForwardIntoFloat(smoothedImage, imgRot.data(), meta, branch.symId,
                              branch.zMirror, threads);
    for (size_t chunkIndex = 0; chunkIndex < chunks.chunks().size();
         ++chunkIndex) {
      const auto chunk = chunks.loadChunk(chunkIndex);
      const spect::io::CSRView view = chunk->view();
      const int compactRowBase =
          chunk->info.angleOffset * meta.compactRaysPerAngle;
      y.assign(static_cast<size_t>(view.rows), 0.0f);
      if (!spmvFloatCUDA(view, imgRot.data(), y.data())) {
        return false;
      }
      scatterCompactForwardProjection(meta, compactRowBase, y, branch.symId,
                                      branch.zMirror, fullExpected, coverage);
    }
  }
  requireCompleteForwardProjectionCoverage(coverage);
  return true;
}

bool runCompactChunkMlemNumerMultiBedCUDA(
    const spect::io::CompactChunkCSRDirectory &chunks,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *const *fullProjections,
    const double *const *fullBackgrounds,
    const double *const *smoothedImages, int bedCount, int threads,
    double *const *numers, bool reverseChunkOrder,
    CompactSymmetricStageTiming *timing,
    const uint16_t *const *attFactors, bool refreshProjectionInputs,
    bool branchBatchInputPrepared, bool downloadBackwardOutputs) {
  return accumulatePoissonBackprojectedRatioCUDA(
      singleChannel(chunks), meta, fullProjections, fullBackgrounds,
      smoothedImages, bedCount, threads, numers, reverseChunkOrder, timing,
      attFactors, refreshProjectionInputs, branchBatchInputPrepared,
      downloadBackwardOutputs);
}

bool accumulatePoissonBackprojectedRatioCUDA(
    const std::vector<CompactMatrixChannel> &channels,
    const spect::io::CompactSymmetryMetadata &meta,
    const double *const *fullProjections,
    const double *const *fullBackgrounds,
    const double *const *smoothedImages, int bedCount, int threads,
    double *const *numers, bool reverseChunkOrder,
    CompactSymmetricStageTiming *timing,
    const uint16_t *const *attFactors, bool refreshProjectionInputs,
    bool branchBatchInputPrepared, bool downloadBackwardOutputs) {
  if (bedCount < 1 || bedCount > spect::core::kMaxBeds) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper supports 1.." +
        std::to_string(spect::core::kMaxBeds) + " beds.");
  }
  validateChannels(channels, meta, "numerator");
  const size_t channelCount = channels.size();
  if (fullProjections == nullptr ||
      (smoothedImages == nullptr && !branchBatchInputPrepared) ||
      (numers == nullptr && downloadBackwardOutputs)) {
    throw std::runtime_error(
        "compact chunk multi-bed CUDA helper received null bed pointer arrays.");
  }
  for (size_t channel = 0; channel < channelCount; ++channel) {
    for (int bed = 0; bed < bedCount; ++bed) {
      if (fullProjections[channel * static_cast<size_t>(bedCount) + bed] ==
          nullptr) {
        throw std::runtime_error(
            "compact chunk multi-bed CUDA helper received null projections.");
      }
    }
  }
  for (int bed = 0; bed < bedCount; ++bed) {
    if ((!branchBatchInputPrepared && smoothedImages[bed] == nullptr) ||
        (downloadBackwardOutputs && numers[bed] == nullptr)) {
      throw std::runtime_error(
          "compact chunk multi-bed CUDA helper received null bed data.");
    }
  }

  bool hasBackground = false;
  if (fullBackgrounds != nullptr) {
    for (size_t index = 0; index < channelCount * static_cast<size_t>(bedCount);
         ++index) {
      hasBackground = hasBackground || fullBackgrounds[index] != nullptr;
    }
  }
  bool hasAtten = false;
  if (attFactors != nullptr) {
    for (int bed = 0; bed < bedCount; ++bed) {
      hasAtten = hasAtten || attFactors[bed] != nullptr;
    }
  }

  warmCompactChunks(channels);
  const bool csrq = isCsrqChannelSet(channels);
  if (csrq) {
    return runCsrqMlemNumerMultiBed(
        channels, meta, fullProjections, hasBackground ? fullBackgrounds : nullptr,
        smoothedImages, bedCount, threads, numers, timing,
        hasAtten ? attFactors : nullptr, refreshProjectionInputs,
        branchBatchInputPrepared, downloadBackwardOutputs);
  }

  // CSRB: one device path for any bed count. The ratio p/(A x + b) and the
  // backward accumulation always stay on the device; CTAC selects the
  // attenuated branch-batch pass (factor per compact angle in the image
  // domain) instead of the plain one.
  const spect::io::CompactChunkCSRDirectory &chunks = *channels.front().chunks;
  applyCompactCsrLocalityHint(chunks, meta);

  const auto totalStart = Clock::now();
  const size_t nvox = static_cast<size_t>(meta.totalVoxels);
  const auto branches = enumerateCompactBranches(meta);
  const size_t branchCount = branches.size();
  const size_t chunkCount = chunks.chunks().size();

  const bool useBranchBatch = compactBranchBatchEnabled() && branchCount > 1;
  if (branchBatchInputPrepared && !useBranchBatch) {
    throw std::runtime_error(
        "compact resident branch-batch input requires the batch CUDA path.");
  }
  if (hasAtten && !useBranchBatch) {
    throw std::runtime_error(
        "compact CTAC needs the branch-batch device path "
        "(SPECT_COMPACT_BRANCH_BATCH must not be disabled).");
  }
  const double attenPower = channels.front().attenPower;
  const size_t fullProjectionCount = static_cast<size_t>(meta.fullRows());
  if (refreshProjectionInputs) {
    for (int bed = 0; bed < bedCount; ++bed) {
      if (!refreshCompactProjectionCUDA(
              fullProjections[bed], fullProjectionCount, bed,
              fullBackgrounds == nullptr ? nullptr : fullBackgrounds[bed])) {
        return false;
      }
    }
  }

  std::vector<float> rotatedImages;
  if (!branchBatchInputPrepared) {
    rotatedImages.assign(static_cast<size_t>(bedCount) * branchCount * nvox,
                         0.0f);
  }

  for (int bed = 0; bed < bedCount; ++bed) {
    if (downloadBackwardOutputs)
      std::fill(numers[bed], numers[bed] + nvox, 0.0);
    if (!compactDeviceBackwardBeginCUDA(nvox, bed)) {
      return false;
    }
    if (branchBatchInputPrepared)
      continue;
    for (size_t branchIndex = 0; branchIndex < branchCount; ++branchIndex) {
      const auto &branch = branches[branchIndex];
      float *imgRot = rotatedImages.data() +
                      (static_cast<size_t>(bed) * branchCount + branchIndex) *
                          nvox;
      const auto stageStart = Clock::now();
      transformForwardIntoFloat(smoothedImages[bed], imgRot, meta, branch.symId,
                                branch.zMirror, threads);
      if (timing)
        addTiming(timing->transformForwardSeconds, stageStart);
    }
  }
  if (useBranchBatch && !branchBatchInputPrepared &&
      !refreshCompactBranchBatchInputCUDA(rotatedImages.data(),
                                          rotatedImages.size())) {
    return false;
  }

  std::vector<int> branchSymIds(branchCount);
  std::vector<int> branchZMirrors(branchCount);
  for (size_t branchIndex = 0; branchIndex < branchCount; ++branchIndex) {
    branchSymIds[branchIndex] = branches[branchIndex].symId;
    branchZMirrors[branchIndex] = branches[branchIndex].zMirror ? 1 : 0;
  }

  const auto processChunk = [&](size_t chunkIndex) -> bool {
    const auto chunk = chunks.loadChunk(chunkIndex);
    const spect::io::CSRView view = chunk->view();
    prefetchNextCompactChunk(chunks, chunkIndex, chunkCount, reverseChunkOrder,
                             view);
    const int compactRowBase = chunk->info.angleOffset * meta.compactRaysPerAngle;
    if (useBranchBatch) {
      CompactDeviceRatioTiming deviceTiming;
      CompactDeviceBackwardTiming backwardTiming;
      const CompactDeviceRatioParams commonRatio =
          makeCompactDeviceRatioParams(meta, compactRowBase, 0, false);
      const CompactDeviceTransformParams commonTransform =
          makeCompactDeviceTransformParams(meta, 0, false);
      const bool ok =
          hasAtten
              ? compactDeviceRatioBedBranchBatchAttenSpmvTAccumulateFloatCUDA(
                    view, fullProjections, fullProjectionCount, commonRatio,
                    commonTransform, branchSymIds.data(), branchZMirrors.data(),
                    static_cast<int>(branchCount), bedCount, attFactors,
                    compactAttenTableCount(meta), attenPower, &deviceTiming,
                    &backwardTiming)
              : compactDeviceRatioBedBranchBatchSpmvTAccumulateFloatCUDA(
                    view, fullProjections, fullProjectionCount, commonRatio,
                    commonTransform, branchSymIds.data(), branchZMirrors.data(),
                    static_cast<int>(branchCount), bedCount, &deviceTiming,
                    &backwardTiming);
      if (!ok) {
        return false;
      }
      if (timing) {
        timing->forwardSeconds += deviceTiming.forwardSeconds;
        timing->ratioSeconds += deviceTiming.ratioSeconds;
        timing->backwardSeconds += deviceTiming.backwardSeconds;
        timing->transformBackwardSeconds +=
            backwardTiming.transformBackwardSeconds;
      }
      return true;
    }
    for (int bed = 0; bed < bedCount; ++bed) {
      const double *fullProjection = fullProjections[bed];
      for (size_t branchIndex = 0; branchIndex < branchCount; ++branchIndex) {
        const auto &branch = branches[branchIndex];
        const float *imgRot =
            rotatedImages.data() +
            (static_cast<size_t>(bed) * branchCount + branchIndex) * nvox;
        CompactDeviceRatioTiming deviceTiming;
        CompactDeviceBackwardTiming backwardTiming;
        const CompactDeviceRatioParams params = makeCompactDeviceRatioParams(
            meta, compactRowBase, branch.symId, branch.zMirror);
        const CompactDeviceTransformParams transformParams =
            makeCompactDeviceTransformParams(meta, branch.symId,
                                             branch.zMirror);
        const bool ok = compactDeviceRatioSpmvTAccumulateFloatCUDA(
            view, imgRot, fullProjection, fullProjectionCount, params,
            transformParams, &deviceTiming, &backwardTiming, bed, bed);
        if (!ok) {
          return false;
        }
        if (timing) {
          timing->forwardSeconds += deviceTiming.forwardSeconds;
          timing->ratioSeconds += deviceTiming.ratioSeconds;
          timing->backwardSeconds += deviceTiming.backwardSeconds;
          timing->transformBackwardSeconds +=
              backwardTiming.transformBackwardSeconds;
        }
      }
    }
    return true;
  };

  if (reverseChunkOrder) {
    for (size_t chunkIndex = chunkCount; chunkIndex-- > 0;) {
      if (!processChunk(chunkIndex)) {
        return false;
      }
    }
  } else {
    for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
      if (!processChunk(chunkIndex)) {
        return false;
      }
    }
  }

  if (downloadBackwardOutputs) {
    for (int bed = 0; bed < bedCount; ++bed) {
      if (!compactDeviceBackwardDownloadCUDA(numers[bed], nvox, bed)) {
        return false;
      }
    }
  }
  if (timing) {
    timing->numeratorSeconds += elapsedSeconds(totalStart);
    ++timing->numeratorCalls;
  }
  return true;
}

} // namespace spect::cuda

#endif
