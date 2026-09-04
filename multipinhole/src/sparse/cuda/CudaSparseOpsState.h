#pragma once

// NOTE: this header is included inside namespace spect by SparseOpsCUDA.cpp;
// standard includes it needs (<atomic>, <unordered_map>) live in that TU.

constexpr int kCompactDeviceBackwardMaxAccumulators =
  cuda::kCompactDeviceBackwardMaxAccumulators;
// Projection/background cache slots: one per (energy window, bed). CSRB uses
// slots [0, kMaxBeds); CSRQ window m uses [m * kMaxBeds, (m + 1) * kMaxBeds).
constexpr int kCsrqMaxMatrices = cuda::kCsrqMaxMatrices;
constexpr int kCsrqStreamSlots = cuda::kCsrqStreamSlots;
constexpr int kCompactProjectionSlots =
  spect::core::kMaxBeds * kCsrqMaxMatrices;

// Phase-B transpose experiment switch (default off; A/B measurement only).
// The production compact transpose path stays per-RHS cusparseSpMV CSR_ALG2
// when SPECT_COMPACT_T_SPMM is unset. At convergence this switch is either
// removed (variant rejected) or the winning variant becomes the only path.
//   unset/""/"off"/"0" -> per-RHS cusparseSpMV CSR_ALG2 (protected baseline)
//   "alg1:K"    -> K-grouped cusparseSpMM CSR_ALG1 on COL-major batch slices
//   "alg1pre:K" -> alg1 plus cusparseSpMM_preprocess once per chunk visit
//   "default:K" -> K-grouped cusparseSpMM ALG_DEFAULT
//   "spmvpre"   -> per-RHS CSR_ALG2 plus cusparseSpMV_preprocess per visit
//   "cscn"      -> control: same device buffers reinterpreted as CSC(A^T),
//                  per-RHS SpMV op=N ALG_DEFAULT (no data movement)
//   "csrt"      -> R1 route: true CSR(A^T) halves, built on host at first
//                  visit (pinned), uploaded JIT on a side stream, backward
//                  runs op=N cusparseSpMM per half (2nd half beta=1).
//                  Forward/A path untouched.
enum class CompactTransposeMode {
  KeepSpmvAlg2,
  SpmmCsrAlg1,
  SpmmCsrAlg1Preprocess,
  SpmmDefault,
  SpmvAlg2Preprocess,
  CscSpmvN,
  TrueCsrTranspose,
};

struct CompactTransposeConfig {
  CompactTransposeMode mode = CompactTransposeMode::KeepSpmvAlg2;
  int groupK = 0;
};

inline const CompactTransposeConfig &compactTransposeConfig() {
  static const CompactTransposeConfig config = [] {
    CompactTransposeConfig parsed;
    const char *raw = std::getenv("SPECT_COMPACT_T_SPMM");
    if (raw == nullptr || raw[0] == '\0') {
      return parsed;
    }
    const std::string text(raw);
    if (text == "off" || text == "0") {
      return parsed;
    }
    if (text == "spmvpre") {
      parsed.mode = CompactTransposeMode::SpmvAlg2Preprocess;
      return parsed;
    }
    if (text == "cscn") {
      parsed.mode = CompactTransposeMode::CscSpmvN;
      return parsed;
    }
    if (text == "csrt") {
      parsed.mode = CompactTransposeMode::TrueCsrTranspose;
      return parsed;
    }
    std::string name = text;
    int k = 0;
    const size_t colon = text.find(':');
    if (colon != std::string::npos) {
      name = text.substr(0, colon);
      k = std::atoi(text.c_str() + colon + 1);
    }
    if (name == "alg1") {
      parsed.mode = CompactTransposeMode::SpmmCsrAlg1;
    } else if (name == "alg1pre") {
      parsed.mode = CompactTransposeMode::SpmmCsrAlg1Preprocess;
    } else if (name == "default") {
      parsed.mode = CompactTransposeMode::SpmmDefault;
    } else {
      throw std::runtime_error(
          "SPECT_COMPACT_T_SPMM must be off|alg1:K|alg1pre:K|default:K|"
          "spmvpre|cscn|csrt, got '" +
          text + "'");
    }
    if (k < 2) {
      throw std::runtime_error(
          "SPECT_COMPACT_T_SPMM SpMM variants require ':K' with K >= 2, "
          "got '" +
          text + "'");
    }
    parsed.groupK = k;
    return parsed;
  }();
  return config;
}

inline uint64_t estimatePreparedChunkHostBytes(const io::CSRView &matrix);

class CudaStreamGuard {
public:
  CudaStreamGuard() = default;
  ~CudaStreamGuard() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  CudaStreamGuard(const CudaStreamGuard &) = delete;
  CudaStreamGuard &operator=(const CudaStreamGuard &) = delete;

  cudaStream_t get() {
    if (stream_ == nullptr) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
    return stream_;
  }

private:
  cudaStream_t stream_ = nullptr;
};

class CudaEventGuard {
public:
  CudaEventGuard() = default;
  ~CudaEventGuard() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  CudaEventGuard(const CudaEventGuard &) = delete;
  CudaEventGuard &operator=(const CudaEventGuard &) = delete;

  cudaEvent_t get() {
    if (event_ == nullptr) {
      CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDefault));
    }
    return event_;
  }

  void record(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(get(), stream)); }
  void wait(cudaStream_t stream) { CUDA_CHECK(cudaStreamWaitEvent(stream, get(), 0)); }

private:
  cudaEvent_t event_ = nullptr;
};

class TimedCudaStagePair {
public:
  TimedCudaStagePair() {
    CUDA_CHECK(cudaEventCreateWithFlags(&start_, cudaEventDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&stop_, cudaEventDefault));
  }

  ~TimedCudaStagePair() {
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
  }

  TimedCudaStagePair(const TimedCudaStagePair &) = delete;
  TimedCudaStagePair &operator=(const TimedCudaStagePair &) = delete;

  TimedCudaStagePair(TimedCudaStagePair &&other) noexcept
      : start_(other.start_), stop_(other.stop_) {
    other.start_ = nullptr;
    other.stop_ = nullptr;
  }

  TimedCudaStagePair &operator=(TimedCudaStagePair &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
    start_ = other.start_;
    stop_ = other.stop_;
    other.start_ = nullptr;
    other.stop_ = nullptr;
    return *this;
  }

  void recordStart(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
  void recordStop(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(stop_, stream)); }
  void synchronizeStop() const { CUDA_CHECK(cudaEventSynchronize(stop_)); }

  double elapsedSeconds() const {
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start_, stop_));
    return static_cast<double>(elapsedMs) / 1000.0;
  }

private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

double accumulateTimedCudaPairs(const std::vector<TimedCudaStagePair> &pairs,
                                size_t count) {
  const size_t n = std::min(count, pairs.size());
  double seconds = 0.0;
  for (size_t i = 0; i < n; ++i) {
    seconds += pairs[i].elapsedSeconds();
  }
  return seconds;
}

double accumulateTimedCudaPairs(const std::vector<TimedCudaStagePair> &pairs) {
  return accumulateTimedCudaPairs(pairs, pairs.size());
}

template <typename T> class DeviceArray {
public:
  DeviceArray() = default;

  explicit DeviceArray(size_t count) { allocate(count); }

  ~DeviceArray() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;

  DeviceArray(DeviceArray &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  DeviceArray &operator=(DeviceArray &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cudaFree(ptr_);
      }
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  T *get() const { return ptr_; }
  size_t size() const { return size_; }

  void allocate(size_t count) {
    if (count == 0) {
      return;
    }
    if (ptr_ != nullptr && count <= size_) {
      return;
    }
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T)));
    size_ = count;
  }

  void copyFromHost(const T *host, size_t count) {
    allocate(count);
    if (count == 0) {
      return;
    }
    CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copyFromHostAsync(const T *host, size_t count, cudaStream_t stream) {
    allocate(count);
    if (count == 0) {
      return;
    }
    CUDA_CHECK(cudaMemcpyAsync(ptr_, host, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  void copyToHost(T *host, size_t count) const {
    if (count == 0) {
      return;
    }
    if (count > size_) {
      throw std::runtime_error("CUDA device buffer is smaller than requested host copy.");
    }
    CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void copyToHostAsync(T *host, size_t count, cudaStream_t stream) const {
    if (count == 0) {
      return;
    }
    if (count > size_) {
      throw std::runtime_error("CUDA device buffer is smaller than requested host copy.");
    }
    CUDA_CHECK(cudaMemcpyAsync(host, ptr_, count * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

  void reset() {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }
  }

private:
  T *ptr_ = nullptr;
  size_t size_ = 0;
};

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void ensure(size_t bytes) {
    if (bytes <= size_) {
      return;
    }
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }
    if (bytes == 0) {
      return;
    }
    CUDA_CHECK(cudaMalloc(&ptr_, bytes));
    size_ = bytes;
  }

  void *get() const { return ptr_; }

  void reset() {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }
  }

private:
  void *ptr_ = nullptr;
  size_t size_ = 0;
};

inline void ensureCudaRuntimeAvailable() {
  if (!cuda::isCudaAvailable()) {
    throw std::runtime_error(
        "CUDA sparse backend was selected, but no CUDA device is available at runtime.");
  }
}

class SparseOpsCUDAState {
private:
  // CSRQ resident matrices: every chunk of every energy window lives on the
  // device for the whole reconstruction (declared first so the method
  // signatures below can name them).
  struct CsrqDeviceChunk {
    int rows = 0;
    int compactRowBase = 0;
    uint64_t nnz = 0;
    uint64_t runCount = 0;
    DeviceArray<uint32_t> dRowNnzPtr;
    DeviceArray<uint32_t> dRowRunPtr;
    DeviceArray<float> dRowMax;
    DeviceArray<uint32_t> dRuns;
    DeviceArray<uint16_t> dValuesQ;

    CsrqDeviceChunkPtrs ptrs() const {
      CsrqDeviceChunkPtrs p;
      p.rowNnzPtr = dRowNnzPtr.get();
      p.rowRunPtr = dRowRunPtr.get();
      p.rowMax = dRowMax.get();
      p.runs = dRuns.get();
      p.valuesQ = dValuesQ.get();
      p.rows = rows;
      return p;
    }
  };
  // Pinned host mirror of one chunk of a streamed matrix.
  struct CsrqHostChunk {
    PinnedHostArray<uint32_t> rowNnzPtr;
    PinnedHostArray<uint32_t> rowRunPtr;
    PinnedHostArray<float> rowMax;
    PinnedHostArray<uint32_t> runs;
    PinnedHostArray<uint16_t> valuesQ;
  };
  struct CsrqChunkDims {
    size_t rows = 0;
    size_t nnz = 0;
    size_t runCount = 0;
  };
  // CTAC plan of one matrix (see csrqEnsureAttenPlanLocked): per chunk the
  // compact angle count, the offset of its [angle][8] slice-id table inside
  // dSliceIndex and the distinct full angles it reaches, in slice order.
  struct CsrqAttenPlan {
    bool valid = false;
    int symmetryMode = 0;
    int compactRaysPerAngle = 0;
    int compactNumAngles = 0;
    int fullNumAngles = 0;
    std::vector<int> symIds;
    std::vector<int> chunkAngles;
    std::vector<size_t> chunkSliceIndexOffset;
    std::vector<std::vector<int>> chunkFullAngles;
    size_t maxSlices = 0;
    DeviceArray<int> dSliceIndex;
  };
  // One energy window's matrix. Resident: chunks[i] holds chunk i. Streamed:
  // chunks holds kCsrqStreamSlots device slots sized to the largest chunk and
  // host[i] the pinned copy of chunk i.
  struct CsrqResidentMatrix {
    std::vector<std::unique_ptr<CsrqDeviceChunk>> chunks;
    std::vector<CsrqHostChunk> host;
    std::vector<cuda::CsrqChunkShape> shapes;
    std::vector<bool> loaded;
    bool streamed = false;
    bool ready = false;
    int cols = 0;
    size_t totalRows = 0;
    uint64_t bytes = 0;
    CsrqAttenPlan attenPlan;
  };
  // A caller-owned CTAC table page-locked for asynchronous slice copies.
  struct CsrqPinnedHostTable {
    const void *ptr = nullptr;
    size_t bytes = 0;
    bool pinned = false;
  };
  static constexpr int kCsrqAttSlots = 2;

public:
  void executeSpMVFloat(const io::CSRView &matrix, const float *x, float *y,
                        cusparseOperation_t operation) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;
    const bool backward = operation == CUSPARSE_OPERATION_TRANSPOSE;

    const auto stageStart = std::chrono::steady_clock::now();
    stageInputH2D(x, static_cast<size_t>(inputSize));
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      stageStart)
            .count());
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    TimedCudaStagePair deviceTiming;
    deviceTiming.recordStart(nullptr);
    executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f);
    deviceTiming.recordStop(nullptr);
    const auto outputStart = std::chrono::steady_clock::now();
    stageOutputD2H(dOutput_, y, static_cast<size_t>(outputSize));
    const double outputSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      outputStart)
            .count();
    noteCudaCompactOutputD2H(outputSeconds,
                             static_cast<size_t>(outputSize) * sizeof(float),
                             backward);
    const double deviceSeconds = deviceTiming.elapsedSeconds();
    if (backward) {
      noteCudaCusparseBackward(deviceSeconds, 1);
    } else {
      noteCudaCusparseForward(deviceSeconds, 1);
    }
  }

  // ---- CSRB CTAC (factorised, per compact angle) ---------------------------
  // Same algebra as the CSRQ passes: H = A att^power, the factor applied to
  // the branch batch before the forward (compact_branch_batch_atten_scale)
  // and to the scattered branch outputs after the backward
  // (compact_transform_backward_add_batch_atten), one compact angle at a
  // time. Each angle's rows are addressed as a sub-CSR (zero-based row
  // pointers built once per chunk, colIdx / values offset by the angle's
  // first nnz) so cuSPARSE runs unchanged on them. Only the factor slices a
  // chunk reaches are staged (shared double-buffered slots with CSRQ).

  // Numerator: bed b accumulates H_b^T(p_b / (H_b x_b + bg_b)) into
  // accumulator accumulatorBaseIndex + b from the refreshed branch batch.
  void executeCompactBedBranchBatchAttenDeviceRatioSpmvTAccumulateFloat(
      const io::CSRView &matrix, const double *const *fullProjections,
      size_t fullProjectionCount,
      const cuda::CompactDeviceRatioParams &commonRatioParams,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const int *symIds, const int *zMirrors, int branchCount, int bedCount,
      const uint16_t *const *bedAttFactors, size_t attCount, double attPower,
      cuda::CompactDeviceRatioTiming *ratioTimingOut,
      cuda::CompactDeviceBackwardTiming *backwardTimingOut,
      int accumulatorBaseIndex, int cacheSlotBase) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (branchCount <= 0 || bedCount <= 0 || fullProjections == nullptr ||
        bedAttFactors == nullptr) {
      throw std::runtime_error(
          "compact attenuated bed-branch batch requires positive branch/bed "
          "counts, projections and attenuation tables.");
    }
    if (accumulatorBaseIndex < 0 || cacheSlotBase < 0 ||
        accumulatorBaseIndex + bedCount > spect::core::kMaxBeds ||
        cacheSlotBase + bedCount > spect::core::kMaxBeds) {
      throw std::runtime_error("compact bed-branch batch slots out of range.");
    }
    if (commonRatioParams.symmetryMode != 45 &&
        commonRatioParams.symmetryMode != 90) {
      throw std::runtime_error(
          "compact bed-branch batch requires symmetryMode 45 or 90.");
    }
    const float power = csrqCheckedAttPower(attPower);
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t cols = cached.cols;
    for (int bed = 0; bed < bedCount; ++bed) {
      const size_t accumSlot =
          compactBackwardAccumulatorSlot(accumulatorBaseIndex + bed);
      if (compactBackwardAccumSize_[accumSlot] != static_cast<size_t>(cols) ||
          dCompactBackwardAccum_[accumSlot].get() == nullptr) {
        throw std::runtime_error(
            "compact bed-branch accumulator does not match matrix columns.");
      }
      ensureCompactFullProjectionUploaded(fullProjections[bed],
                                          fullProjectionCount,
                                          cacheSlotBase + bed);
    }
    const size_t expectedInputCount = static_cast<size_t>(cols) *
                                      static_cast<size_t>(branchCount) *
                                      static_cast<size_t>(bedCount);
    if (compactBatchInputCount_ != expectedInputCount ||
        dBatchInput_.get() == nullptr) {
      throw std::runtime_error(
          "compact branch-batch input was not refreshed for this pass.");
    }
    ensureBranchMetadataLocked(symIds, zMirrors, branchCount);
    ensureProjDomainKernelsLoaded();
    const CsrbAttenChunk &chunk = csrbEnsureAttenChunkLocked(
        matrix, cached, commonRatioParams.compactRowBase, commonRatioParams);
    const int rpa = commonRatioParams.compactRaysPerAngle;
    const size_t nvox = static_cast<size_t>(cols);
    dCsrbAttenBatch_.allocate(nvox * static_cast<size_t>(branchCount));
    dBatchOutput_.allocate(static_cast<size_t>(rpa) *
                           static_cast<size_t>(branchCount));
    dFullBatchOutput_.allocate(nvox * static_cast<size_t>(branchCount));
    CsrbAngleMatrices angleMats = csrbCreateAngleMatricesLocked(cached, chunk);

    const size_t stages =
        static_cast<size_t>(bedCount) * static_cast<size_t>(chunk.angles);
    csrbEnsureAttenTimersLocked(stages);
    size_t stage = 0;
    for (int bed = 0; bed < bedCount; ++bed) {
      const int slot = stageAttenSlicesLocked(
          chunk.fullAngles, chunk.fullAngles.size(), chunk.fullNumAngles,
          bedAttFactors[bed], attCount, nvox);
      const uint16_t *dSlices = dCsrqAttSlices_[slot].get();
      const float *bedInput =
          dBatchInput_.get() +
          static_cast<size_t>(bed) * static_cast<size_t>(branchCount) * nvox;
      const double *dProjection =
          dCompactProjectionInput_[static_cast<size_t>(cacheSlotBase + bed)]
              .get();
      const double *dBackground = compactBackgroundDeviceLocked(cacheSlotBase + bed);
      double *accum =
          dCompactBackwardAccum_[compactBackwardAccumulatorSlot(
                                     accumulatorBaseIndex + bed)]
              .get();
      for (int i = 0; i < chunk.angles; ++i, ++stage) {
        const int *dSlice = chunk.dSliceIndex.get() + i * branchCount;
        cuda::CompactDeviceRatioParams angleParams = commonRatioParams;
        angleParams.compactRowBase = commonRatioParams.compactRowBase + i * rpa;

        csrbAttenForwardTimers_[stage].recordStart(nullptr);
        launchCompactBranchBatchAttenScaleKernel(
            bedInput, dSlices, dSlice, dCsrbAttenBatch_.get(), nvox,
            branchCount, commonTransformParams, dBatchSymIds_.get(),
            dBatchZMirrors_.get(), power);
        csrbAngleSpMVLocked(angleMats, cached, chunk, i,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            dCsrbAttenBatch_.get(), nvox, dBatchOutput_.get(),
                            static_cast<size_t>(rpa), static_cast<size_t>(rpa),
                            branchCount, dFullBatchOutput_.get());
        launchCompactRatioBatchKernel(
            dBatchOutput_.get(), dProjection, dBackground, fullProjectionCount,
            angleParams, dBatchSymIds_.get(), dBatchZMirrors_.get(),
            branchCount, dBatchOutput_.get(), static_cast<size_t>(rpa), 1e-10);
        csrbAttenForwardTimers_[stage].recordStop(nullptr);

        csrbAttenBackwardTimers_[stage].recordStart(nullptr);
        csrbAngleSpMVLocked(angleMats, cached, chunk, i,
                            CUSPARSE_OPERATION_TRANSPOSE, dBatchOutput_.get(),
                            static_cast<size_t>(rpa), dFullBatchOutput_.get(),
                            nvox, nvox, branchCount, nullptr);
        csrbAttenBackwardTimers_[stage].recordStop(nullptr);

        csrbAttenTransformTimers_[stage].recordStart(nullptr);
        launchCompactTransformBackwardAddBatchAttenKernel(
            dFullBatchOutput_.get(), accum, nvox, branchCount,
            commonTransformParams, dBatchSymIds_.get(), dBatchZMirrors_.get(),
            dSlices, dSlice, power);
        csrbAttenTransformTimers_[stage].recordStop(nullptr);
      }
      csrqReleaseAttenSlotLocked(slot);
    }
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const size_t calls = stages * static_cast<size_t>(branchCount);
    const double forwardSeconds =
        accumulateTimedCudaPairs(csrbAttenForwardTimers_, stages);
    const double backwardSeconds =
        accumulateTimedCudaPairs(csrbAttenBackwardTimers_, stages);
    const double transformSeconds =
        accumulateTimedCudaPairs(csrbAttenTransformTimers_, stages);
    noteCudaCusparseForward(forwardSeconds, calls);
    noteCudaCusparseBackward(backwardSeconds, calls);
    noteCudaTransformBackward(transformSeconds, calls);
    if (ratioTimingOut != nullptr) {
      ratioTimingOut->forwardSeconds += forwardSeconds;
      ratioTimingOut->backwardSeconds += backwardSeconds;
    }
    if (backwardTimingOut != nullptr) {
      backwardTimingOut->backwardSeconds += backwardSeconds;
      backwardTimingOut->transformBackwardSeconds += transformSeconds;
    }
  }

  // Sensitivity: H^T W 1 of one chunk for one bed's table, added to
  // accumulatorIndex (rowWeights may be null for W = 1).
  void executeCompactSensitivityBranchBatchAttenSpmvTAccumulateFloat(
      const io::CSRView &matrix,
      const cuda::CompactDeviceRatioParams &commonRatioParams,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const int *symIds, const int *zMirrors, int branchCount,
      const uint16_t *attFactors, size_t attCount, double attPower,
      const double *rowWeights, size_t fullProjectionCount,
      cuda::CompactDeviceBackwardTiming *backwardTimingOut,
      int accumulatorIndex) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (branchCount <= 0 || attFactors == nullptr || fullProjectionCount == 0) {
      throw std::runtime_error(
          "compact attenuated sensitivity needs branches, a table and the full "
          "projection row count.");
    }
    if (commonRatioParams.symmetryMode != 45 &&
        commonRatioParams.symmetryMode != 90) {
      throw std::runtime_error(
          "compact attenuated sensitivity requires symmetryMode 45 or 90.");
    }
    const float power = csrqCheckedAttPower(attPower);
    const size_t accumSlot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[accumSlot];
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t cols = cached.cols;
    if (compactBackwardAccumSize_[accumSlot] != static_cast<size_t>(cols) ||
        accum.get() == nullptr) {
      throw std::runtime_error(
          "compact device accumulator does not match matrix columns.");
    }
    ensureBranchMetadataLocked(symIds, zMirrors, branchCount);
    ensureProjDomainKernelsLoaded();
    const double *dRowWeights = ensureCompactRowWeightsUploaded(
        rowWeights, rowWeights == nullptr ? 0 : fullProjectionCount);
    const CsrbAttenChunk &chunk = csrbEnsureAttenChunkLocked(
        matrix, cached, commonRatioParams.compactRowBase, commonRatioParams);
    const int rpa = commonRatioParams.compactRaysPerAngle;
    const size_t nvox = static_cast<size_t>(cols);
    dBatchOnes_.allocate(static_cast<size_t>(rpa) *
                         static_cast<size_t>(branchCount));
    dFullBatchOutput_.allocate(nvox * static_cast<size_t>(branchCount));
    CsrbAngleMatrices angleMats = csrbCreateAngleMatricesLocked(cached, chunk);
    const int slot = stageAttenSlicesLocked(
        chunk.fullAngles, chunk.fullAngles.size(), chunk.fullNumAngles,
        attFactors, attCount, nvox);
    const uint16_t *dSlices = dCsrqAttSlices_[slot].get();

    const size_t stages = static_cast<size_t>(chunk.angles);
    csrbEnsureAttenTimersLocked(stages);
    for (int i = 0; i < chunk.angles; ++i) {
      const int *dSlice = chunk.dSliceIndex.get() + i * branchCount;
      cuda::CompactDeviceRatioParams angleParams = commonRatioParams;
      angleParams.compactRowBase = commonRatioParams.compactRowBase + i * rpa;
      // W per (row, branch) for rows with a full projection counterpart
      // (1 without weights), the same contract as the CSRQ unit ratio.
      launchCompactUnitRatioBatchKernel(
          dRowWeights, fullProjectionCount, angleParams, dBatchSymIds_.get(),
          dBatchZMirrors_.get(), branchCount, dBatchOnes_.get(),
          static_cast<size_t>(rpa));
      csrbAttenBackwardTimers_[static_cast<size_t>(i)].recordStart(nullptr);
      csrbAngleSpMVLocked(angleMats, cached, chunk, i,
                          CUSPARSE_OPERATION_TRANSPOSE, dBatchOnes_.get(),
                          static_cast<size_t>(rpa), dFullBatchOutput_.get(),
                          nvox, nvox, branchCount, nullptr);
      csrbAttenBackwardTimers_[static_cast<size_t>(i)].recordStop(nullptr);
      csrbAttenTransformTimers_[static_cast<size_t>(i)].recordStart(nullptr);
      launchCompactTransformBackwardAddBatchAttenKernel(
          dFullBatchOutput_.get(), accum.get(), nvox, branchCount,
          commonTransformParams, dBatchSymIds_.get(), dBatchZMirrors_.get(),
          dSlices, dSlice, power);
      csrbAttenTransformTimers_[static_cast<size_t>(i)].recordStop(nullptr);
    }
    csrqReleaseAttenSlotLocked(slot);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const double backwardSeconds =
        accumulateTimedCudaPairs(csrbAttenBackwardTimers_, stages);
    const double transformSeconds =
        accumulateTimedCudaPairs(csrbAttenTransformTimers_, stages);
    noteCudaCusparseBackward(backwardSeconds,
                             stages * static_cast<size_t>(branchCount));
    if (backwardTimingOut != nullptr) {
      backwardTimingOut->backwardSeconds += backwardSeconds;
      backwardTimingOut->transformBackwardSeconds += transformSeconds;
    }
  }

  // Forward projection H x of one chunk over the refreshed single-bed branch
  // batch; hostYhatBatch receives [branch][chunkRow] floats.
  void executeCompactBranchBatchAttenForward(
      const io::CSRView &matrix,
      const cuda::CompactDeviceRatioParams &commonRatioParams,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const int *symIds, const int *zMirrors, int branchCount,
      const uint16_t *attFactors, size_t attCount, double attPower,
      float *hostYhatBatch, size_t count) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (branchCount <= 0 || attFactors == nullptr || hostYhatBatch == nullptr) {
      throw std::runtime_error(
          "compact attenuated forward needs branches, a table and an output.");
    }
    const float power = csrqCheckedAttPower(attPower);
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t cols = cached.cols;
    const size_t nvox = static_cast<size_t>(cols);
    if (count != static_cast<size_t>(cached.rows) *
                     static_cast<size_t>(branchCount)) {
      throw std::runtime_error(
          "compact attenuated forward output must hold rows * branches floats.");
    }
    if (compactBatchInputCount_ != nvox * static_cast<size_t>(branchCount) ||
        dBatchInput_.get() == nullptr) {
      throw std::runtime_error(
          "compact attenuated forward requires a refreshed single-bed branch "
          "batch input.");
    }
    ensureBranchMetadataLocked(symIds, zMirrors, branchCount);
    ensureProjDomainKernelsLoaded();
    const CsrbAttenChunk &chunk = csrbEnsureAttenChunkLocked(
        matrix, cached, commonRatioParams.compactRowBase, commonRatioParams);
    const int rpa = commonRatioParams.compactRaysPerAngle;
    dCsrbAttenBatch_.allocate(nvox * static_cast<size_t>(branchCount));
    dBatchOutput_.allocate(count);
    dFullBatchOutput_.allocate(nvox * static_cast<size_t>(branchCount));
    CsrbAngleMatrices angleMats = csrbCreateAngleMatricesLocked(cached, chunk);
    const int slot = stageAttenSlicesLocked(
        chunk.fullAngles, chunk.fullAngles.size(), chunk.fullNumAngles,
        attFactors, attCount, nvox);
    const uint16_t *dSlices = dCsrqAttSlices_[slot].get();
    const size_t stages = static_cast<size_t>(chunk.angles);
    csrbEnsureAttenTimersLocked(stages);
    for (int i = 0; i < chunk.angles; ++i) {
      const int *dSlice = chunk.dSliceIndex.get() + i * branchCount;
      csrbAttenForwardTimers_[static_cast<size_t>(i)].recordStart(nullptr);
      launchCompactBranchBatchAttenScaleKernel(
          dBatchInput_.get(), dSlices, dSlice, dCsrbAttenBatch_.get(), nvox,
          branchCount, commonTransformParams, dBatchSymIds_.get(),
          dBatchZMirrors_.get(), power);
      // Output is [branch][chunkRow]: column b starts at b * rows, this
      // angle's rows at + i * rpa.
      csrbAngleSpMVLocked(
          angleMats, cached, chunk, i, CUSPARSE_OPERATION_NON_TRANSPOSE,
          dCsrbAttenBatch_.get(), nvox,
          dBatchOutput_.get() + static_cast<size_t>(i) * static_cast<size_t>(rpa),
          static_cast<size_t>(rpa), static_cast<size_t>(cached.rows), branchCount,
          dFullBatchOutput_.get());
      csrbAttenForwardTimers_[static_cast<size_t>(i)].recordStop(nullptr);
    }
    csrqReleaseAttenSlotLocked(slot);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    noteCudaCusparseForward(
        accumulateTimedCudaPairs(csrbAttenForwardTimers_, stages),
        stages * static_cast<size_t>(branchCount));
    const auto outputStart = std::chrono::steady_clock::now();
    dBatchOutput_.copyToHost(hostYhatBatch, count);
    noteCudaCompactOutputD2H(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      outputStart)
            .count(),
        count * sizeof(float), false);
  }

  void beginCompactDeviceBackward(size_t voxelCount, int accumulatorIndex) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    accum.allocate(voxelCount);
    if (voxelCount > 0) {
      CUDA_CHECK(cudaMemset(accum.get(), 0,
                            voxelCount * sizeof(double)));
    }
    compactBackwardAccumSize_[slot] = voxelCount;
  }

  void executeCompactDeviceBackwardSpmvTAccumulateFloat(
      const io::CSRView &matrix, const float *x,
      const cuda::CompactDeviceTransformParams &params,
      cuda::CompactDeviceBackwardTiming *timing, int accumulatorIndex) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    const size_t accumSize = compactBackwardAccumSize_[slot];
    if (accumSize == 0 || accum.get() == nullptr) {
      throw std::runtime_error(
          "compact device-backward accumulator was not initialized.");
    }
    if (params.symmetryMode != 45 && params.symmetryMode != 90) {
      throw std::runtime_error(
          "compact device-backward CUDA helper requires symmetryMode 45 or 90.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;
    if (static_cast<size_t>(cols) != accumSize) {
      throw std::runtime_error(
          "compact device-backward accumulator size does not match matrix columns.");
    }

    const auto inputStart = std::chrono::steady_clock::now();
    dInput_.copyFromHost(x, static_cast<size_t>(rows));
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      inputStart)
            .count());
    dFullOutput_.allocate(static_cast<size_t>(cols));
    ensureProjDomainKernelsLoaded();

    CuSparseVecGuard vecXBackward;
    CuSparseVecGuard vecYBackward;
    vecXBackward.create(rows, dInput_.get());
    vecYBackward.create(cols, dFullOutput_.get());
    TimedCudaStagePair backwardTiming;
    backwardTiming.recordStart(nullptr);
    executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                      vecXBackward.get(), vecYBackward.get(), 0.0f);
    backwardTiming.recordStop(nullptr);

    TimedCudaStagePair transformTiming;
    transformTiming.recordStart(nullptr);
    launchCompactTransformBackwardAddKernel(
        dFullOutput_.get(), accum.get(), accumSize, params);
    transformTiming.recordStop(nullptr);

    backwardTiming.synchronizeStop();
    transformTiming.synchronizeStop();
    const double backwardSeconds = backwardTiming.elapsedSeconds();
    const double transformSeconds = transformTiming.elapsedSeconds();
    noteCudaCusparseBackward(backwardSeconds, 1);
    if (timing != nullptr) {
      timing->backwardSeconds += backwardSeconds;
      timing->transformBackwardSeconds += transformSeconds;
    }
  }

  void executeCompactDeviceRatioSpmvTAccumulateFloat(
      const io::CSRView &matrix, const float *x, const double *fullProjection,
      size_t fullProjectionCount,
      const cuda::CompactDeviceRatioParams &ratioParams,
      const cuda::CompactDeviceTransformParams &transformParams,
      cuda::CompactDeviceRatioTiming *ratioTimingOut,
      cuda::CompactDeviceBackwardTiming *backwardTimingOut,
      int accumulatorIndex, int cacheSlot) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    if (cacheSlot < 0 || cacheSlot >= spect::core::kMaxBeds) {
      throw std::runtime_error(
          "compact device-ratio accumulator cache slot out of range.");
    }
    const size_t projectionSlot = static_cast<size_t>(cacheSlot);
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    const size_t accumSize = compactBackwardAccumSize_[slot];
    if (accumSize == 0 || accum.get() == nullptr) {
      throw std::runtime_error(
          "compact device accumulator was not initialized.");
    }
    if (ratioParams.symmetryMode != 45 && ratioParams.symmetryMode != 90) {
      throw std::runtime_error(
          "compact device-ratio accumulator helper requires symmetryMode 45 or 90.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;
    if (static_cast<size_t>(cols) != accumSize) {
      throw std::runtime_error(
          "compact device accumulator size does not match matrix columns.");
    }

    const auto inputStart = std::chrono::steady_clock::now();
    dInput_.copyFromHost(x, static_cast<size_t>(cols));
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      inputStart)
            .count());
    ensureCompactFullProjectionUploaded(fullProjection, fullProjectionCount,
                                        cacheSlot);
    ensureProjDomainKernelsLoaded();

    dOutput_.allocate(static_cast<size_t>(rows));
    dChunkInput_.allocate(static_cast<size_t>(rows));
    dFullOutput_.allocate(static_cast<size_t>(cols));

    CuSparseVecGuard vecXForward;
    CuSparseVecGuard vecYForward;
    vecXForward.create(cols, dInput_.get());
    vecYForward.create(rows, dOutput_.get());
    TimedCudaStagePair forwardTiming;
    forwardTiming.recordStart(nullptr);
    executeSpMVLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      vecXForward.get(), vecYForward.get(), 0.0f);
    forwardTiming.recordStop(nullptr);

    TimedCudaStagePair ratioTiming;
    ratioTiming.recordStart(nullptr);
    launchCompactRatioKernel(dOutput_.get(),
                             dCompactProjectionInput_[projectionSlot].get(),
                             compactBackgroundDeviceLocked(cacheSlot),
                             fullProjectionCount, ratioParams,
                             dChunkInput_.get(), static_cast<size_t>(rows),
                             1e-10);
    ratioTiming.recordStop(nullptr);

    CuSparseVecGuard vecXBackward;
    CuSparseVecGuard vecYBackward;
    vecXBackward.create(rows, dChunkInput_.get());
    vecYBackward.create(cols, dFullOutput_.get());
    TimedCudaStagePair backwardTiming;
    backwardTiming.recordStart(nullptr);
    executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                      vecXBackward.get(), vecYBackward.get(), 0.0f);
    backwardTiming.recordStop(nullptr);

    TimedCudaStagePair transformTiming;
    transformTiming.recordStart(nullptr);
    launchCompactTransformBackwardAddKernel(
        dFullOutput_.get(), accum.get(), accumSize, transformParams);
    transformTiming.recordStop(nullptr);

    transformTiming.synchronizeStop();
    const double forwardSeconds = forwardTiming.elapsedSeconds();
    const double ratioSeconds = ratioTiming.elapsedSeconds();
    const double backwardSeconds = backwardTiming.elapsedSeconds();
    const double transformSeconds = transformTiming.elapsedSeconds();
    noteCudaCusparseForward(forwardSeconds, 1);
    noteCudaRatio(ratioSeconds, 1);
    noteCudaCusparseBackward(backwardSeconds, 1);
    if (ratioTimingOut != nullptr) {
      ratioTimingOut->forwardSeconds += forwardSeconds;
      ratioTimingOut->ratioSeconds += ratioSeconds;
      ratioTimingOut->backwardSeconds += backwardSeconds;
    }
    if (backwardTimingOut != nullptr) {
      backwardTimingOut->backwardSeconds += backwardSeconds;
      backwardTimingOut->transformBackwardSeconds += transformSeconds;
    }
  }

  void executeCompactBedBranchBatchDeviceRatioSpmvTAccumulateFloat(
      const io::CSRView &matrix, const double *const *fullProjections,
      size_t fullProjectionCount,
      const cuda::CompactDeviceRatioParams &commonRatioParams,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const int *symIds, const int *zMirrors, int branchCount, int bedCount,
      cuda::CompactDeviceRatioTiming *ratioTimingOut,
      cuda::CompactDeviceBackwardTiming *backwardTimingOut,
      int accumulatorBaseIndex = 0, int cacheSlotBase = 0) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (branchCount <= 0 || bedCount <= 0 || fullProjections == nullptr) {
      throw std::runtime_error(
          "compact bed-branch batch requires positive branch/bed counts and "
          "projections.");
    }
    if (accumulatorBaseIndex < 0 || cacheSlotBase < 0 ||
        accumulatorBaseIndex + bedCount > spect::core::kMaxBeds ||
        cacheSlotBase + bedCount > spect::core::kMaxBeds) {
      throw std::runtime_error("compact bed-branch batch slots out of range.");
    }
    if (commonRatioParams.symmetryMode != 45 &&
        commonRatioParams.symmetryMode != 90) {
      throw std::runtime_error(
          "compact bed-branch batch requires symmetryMode 45 or 90.");
    }

    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    if (compactTransposeConfig().mode ==
        CompactTransposeMode::TrueCsrTranspose) {
      // Queue A^T half uploads now so they hide under forward + ratio.
      queueCompactTrueTransposeUploadLocked(matrix);
    }
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;
    for (int bed = 0; bed < bedCount; ++bed) {
      const size_t accumSlot = compactBackwardAccumulatorSlot(
          accumulatorBaseIndex + bed);
      if (compactBackwardAccumSize_[accumSlot] != static_cast<size_t>(cols) ||
          dCompactBackwardAccum_[accumSlot].get() == nullptr) {
        throw std::runtime_error(
            "compact bed-branch accumulator does not match matrix columns.");
      }
      ensureCompactFullProjectionUploaded(fullProjections[bed],
                                          fullProjectionCount,
                                          cacheSlotBase + bed);
    }

    const int batchSize = branchCount * bedCount;
    const size_t expectedInputCount =
        static_cast<size_t>(cols) * static_cast<size_t>(batchSize);
    if (compactBatchInputCount_ != expectedInputCount ||
        dBatchInput_.get() == nullptr) {
      throw std::runtime_error(
          "compact branch-batch input was not refreshed for this pass.");
    }
    const bool branchMetadataChanged =
        batchSymIdsHost_.size() != static_cast<size_t>(branchCount) ||
        batchZMirrorsHost_.size() != static_cast<size_t>(branchCount) ||
        !std::equal(batchSymIdsHost_.begin(), batchSymIdsHost_.end(), symIds) ||
        !std::equal(batchZMirrorsHost_.begin(), batchZMirrorsHost_.end(),
                    zMirrors);
    if (branchMetadataChanged) {
      dBatchSymIds_.copyFromHost(symIds, static_cast<size_t>(branchCount));
      dBatchZMirrors_.copyFromHost(zMirrors,
                                  static_cast<size_t>(branchCount));
      batchSymIdsHost_.assign(symIds, symIds + branchCount);
      batchZMirrorsHost_.assign(zMirrors, zMirrors + branchCount);
    }
    ensureProjDomainKernelsLoaded();

    dBatchOutput_.allocate(static_cast<size_t>(rows) *
                           static_cast<size_t>(batchSize));
    dFullBatchOutput_.allocate(static_cast<size_t>(cols) *
                               static_cast<size_t>(batchSize));

    TimedCudaStagePair forwardTiming;
    forwardTiming.recordStart(nullptr);
    executeColMajorBatchSpMVLocked(
        cached, CUSPARSE_OPERATION_NON_TRANSPOSE, dBatchInput_.get(), cols,
        dBatchOutput_.get(), rows, batchSize, nullptr,
        dFullBatchOutput_.get());
    forwardTiming.recordStop(nullptr);

    TimedCudaStagePair ratioTiming;
    ratioTiming.recordStart(nullptr);
    for (int bed = 0; bed < bedCount; ++bed) {
      const size_t offset = static_cast<size_t>(bed) *
                            static_cast<size_t>(branchCount) *
                            static_cast<size_t>(rows);
      launchCompactRatioBatchKernel(
          dBatchOutput_.get() + offset,
          dCompactProjectionInput_[static_cast<size_t>(cacheSlotBase + bed)]
              .get(),
          compactBackgroundDeviceLocked(cacheSlotBase + bed),
          fullProjectionCount, commonRatioParams, dBatchSymIds_.get(),
          dBatchZMirrors_.get(), branchCount, dBatchOutput_.get() + offset,
          static_cast<size_t>(rows), 1e-10);
    }
    ratioTiming.recordStop(nullptr);

    TimedCudaStagePair backwardTiming;
    backwardTiming.recordStart(nullptr);
    executeColMajorBatchSpMVLocked(
        cached, CUSPARSE_OPERATION_TRANSPOSE, dBatchOutput_.get(), rows,
        dFullBatchOutput_.get(), cols, batchSize);
    backwardTiming.recordStop(nullptr);

    TimedCudaStagePair transformTiming;
    transformTiming.recordStart(nullptr);
    for (int bed = 0; bed < bedCount; ++bed) {
      const size_t accumSlot = compactBackwardAccumulatorSlot(
          accumulatorBaseIndex + bed);
      const size_t offset = static_cast<size_t>(bed) *
                            static_cast<size_t>(branchCount) *
                            static_cast<size_t>(cols);
      launchCompactTransformBackwardAddBatchKernel(
          dFullBatchOutput_.get() + offset,
          dCompactBackwardAccum_[accumSlot].get(),
          compactBackwardAccumSize_[accumSlot], branchCount,
          commonTransformParams, dBatchSymIds_.get(), dBatchZMirrors_.get());
    }
    transformTiming.recordStop(nullptr);

    transformTiming.synchronizeStop();
    const double forwardSeconds = forwardTiming.elapsedSeconds();
    const double ratioSeconds = ratioTiming.elapsedSeconds();
    const double backwardSeconds = backwardTiming.elapsedSeconds();
    const double transformSeconds = transformTiming.elapsedSeconds();
    noteCudaCusparseForward(forwardSeconds, batchSize);
    noteCudaRatio(ratioSeconds, batchSize);
    noteCudaCusparseBackward(backwardSeconds, batchSize);
    noteCudaTransformBackward(transformSeconds, batchSize);
    if (ratioTimingOut != nullptr) {
      ratioTimingOut->forwardSeconds += forwardSeconds;
      ratioTimingOut->ratioSeconds += ratioSeconds;
      ratioTimingOut->backwardSeconds += backwardSeconds;
    }
    if (backwardTimingOut != nullptr) {
      backwardTimingOut->backwardSeconds += backwardSeconds;
      backwardTimingOut->transformBackwardSeconds += transformSeconds;
    }
  }

  void executeCompactSensitivityBranchBatchSpmvTAccumulateFloat(
      const io::CSRView &matrix,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const int *symIds, const int *zMirrors, int batchSize,
      cuda::CompactDeviceBackwardTiming *backwardTimingOut,
      int accumulatorIndex, int cacheSlot,
      const cuda::CompactDeviceRatioParams *commonRatioParams,
      const double *rowWeights, size_t rowWeightCount) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    if (cacheSlot < 0 || cacheSlot >= spect::core::kMaxBeds) {
      throw std::runtime_error(
          "compact sensitivity branch-batch cache slot out of range.");
    }
    if (batchSize <= 0) {
      throw std::runtime_error(
          "compact sensitivity branch-batch batchSize must be > 0.");
    }
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    const size_t accumSize = compactBackwardAccumSize_[slot];
    if (accumSize == 0 || accum.get() == nullptr) {
      throw std::runtime_error(
          "compact device accumulator was not initialized.");
    }
    if (commonTransformParams.symmetryMode != 45 &&
        commonTransformParams.symmetryMode != 90) {
      throw std::runtime_error(
          "compact sensitivity branch-batch helper requires symmetryMode 45 or 90.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    if (compactTransposeConfig().mode ==
        CompactTransposeMode::TrueCsrTranspose) {
      queueCompactTrueTransposeUploadLocked(matrix);
    }
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;
    if (static_cast<size_t>(cols) != accumSize) {
      throw std::runtime_error(
          "compact device accumulator size does not match matrix columns.");
    }

    dBatchSymIds_.copyFromHost(symIds, static_cast<size_t>(batchSize));
    dBatchZMirrors_.copyFromHost(zMirrors, static_cast<size_t>(batchSize));
    ensureProjDomainKernelsLoaded();

    const size_t batchedRows = static_cast<size_t>(rows) *
                               static_cast<size_t>(batchSize);
    dBatchOnes_.allocate(batchedRows);
    const double *dRowWeights =
        ensureCompactRowWeightsUploaded(rowWeights, rowWeightCount);
    if (dRowWeights != nullptr) {
      // S = A^T W 1: the batch input carries W per (row, branch), 0 for rows
      // without a full projection counterpart.
      if (commonRatioParams == nullptr) {
        throw std::runtime_error(
            "compact weighted sensitivity needs the row -> projection mapping.");
      }
      launchCompactUnitRatioBatchKernel(
          dRowWeights, rowWeightCount, *commonRatioParams, dBatchSymIds_.get(),
          dBatchZMirrors_.get(), batchSize, dBatchOnes_.get(),
          static_cast<size_t>(rows));
    } else {
      // Unweighted: all ones, the production default (kept as is so the
      // cached denominators stay bit-identical).
      std::vector<float> ones(batchedRows, 1.0f);
      dBatchOnes_.copyFromHost(ones.data(), batchedRows);
    }
    dFullBatchOutput_.allocate(static_cast<size_t>(cols) *
                               static_cast<size_t>(batchSize));

    CuSparseDnMatGuard matBBackward;
    CuSparseDnMatGuard matCBackward;
    matBBackward.create(rows, batchSize, rows, dBatchOnes_.get(), CUDA_R_32F,
                        CUSPARSE_ORDER_COL);
    matCBackward.create(cols, batchSize, cols, dFullBatchOutput_.get(),
                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    TimedCudaStagePair backwardTiming;
    backwardTiming.recordStart(nullptr);
    executeSpMMLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                      matBBackward.get(), matCBackward.get());
    backwardTiming.recordStop(nullptr);

    TimedCudaStagePair transformTiming;
    transformTiming.recordStart(nullptr);
    launchCompactTransformBackwardAddBatchKernel(
        dFullBatchOutput_.get(), accum.get(), accumSize, batchSize,
        commonTransformParams, dBatchSymIds_.get(), dBatchZMirrors_.get());
    transformTiming.recordStop(nullptr);

    transformTiming.synchronizeStop();
    backwardTiming.synchronizeStop();
    const double backwardSeconds = backwardTiming.elapsedSeconds();
    const double transformSeconds = transformTiming.elapsedSeconds();
    noteCudaCusparseBackward(backwardSeconds, batchSize);
    if (backwardTimingOut != nullptr) {
      backwardTimingOut->backwardSeconds += backwardSeconds;
      backwardTimingOut->transformBackwardSeconds += transformSeconds;
    }
  }



  void downloadCompactDeviceBackward(double *y, size_t voxelCount,
                                     int accumulatorIndex) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    const DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    if (voxelCount != compactBackwardAccumSize_[slot] ||
        accum.get() == nullptr) {
      throw std::runtime_error(
          "compact device-backward download size does not match accumulator.");
    }
    const auto outputStart = std::chrono::steady_clock::now();
    accum.copyToHost(y, voxelCount);
    noteCudaCompactOutputD2H(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      outputStart)
            .count(),
        voxelCount * sizeof(double), true);
  }

  // csrt experiment: derive host CSR(A^T) halves ahead of the iteration loop
  // so the one-time transpose cost lands in matrix prepare, not core time.
  bool prepareCompactTransposeHost(const io::CSRView &matrix) {
    if (compactTransposeConfig().mode !=
        CompactTransposeMode::TrueCsrTranspose) {
      return false;
    }
    if (!matrix.valid() || matrix.rows < 2 || matrix.nnz == 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    (void)ensureCompactTransposeHostEntry(
        compactTransposeFingerprint(matrix), matrix);
    return true;
  }

private:

  void stageInputH2D(const float *x, size_t count) {
    pinnedInputStage_.copyFromHost(x, count);
    dInput_.copyFromHostAsync(pinnedInputStage_.data(), count, nullptr);
  }

  template <typename DeviceArrayT>
  void stageOutputD2H(const DeviceArrayT &dev, float *y, size_t count) {
    pinnedOutputStage_.allocate(count);
    dev.copyToHostAsync(pinnedOutputStage_.data(), count, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    std::memcpy(y, pinnedOutputStage_.data(), count * sizeof(float));
  }

  // Double↔float host-device conversion helpers.
  // Pageable→device cudaMemcpy(Async) is host-synchronous, so a single
  // scratch buffer (hostFloatScratch_) is safe to reuse between calls.
  void uploadDouble(DeviceArray<float> &dev, const double *src, size_t count) {
    hostFloatScratch_.resize(count);
    for (size_t i = 0; i < count; ++i)
      hostFloatScratch_[i] = static_cast<float>(src[i]);
    dev.copyFromHost(hostFloatScratch_.data(), count);
  }

  void ensureCompactFullProjectionUploaded(const double *src, size_t count,
                                           int cacheSlot) {
    if (src == nullptr || count == 0) {
      throw std::runtime_error(
          "compact device-ratio CUDA helper received an empty projection.");
    }
    if (cacheSlot < 0 || cacheSlot >= kCompactProjectionSlots) {
      throw std::runtime_error("compact projection cache slot out of range.");
    }
    // The canonical host ratio divides the double measurement by a promoted
    // float forward estimate, then casts once to float. Keep the measurement
    // double on device so removing the round trip does not add an early cast.
    auto &dProjection = dCompactProjectionInput_[static_cast<size_t>(cacheSlot)];
    if (compactProjectionHostKey_[static_cast<size_t>(cacheSlot)] == src &&
        compactProjectionCount_[static_cast<size_t>(cacheSlot)] == count &&
        dProjection.get() != nullptr && dProjection.size() >= count) {
      return;
    }
    const auto uploadStart = std::chrono::steady_clock::now();
    dProjection.copyFromHost(src, count);
    compactProjectionHostKey_[static_cast<size_t>(cacheSlot)] = src;
    compactProjectionCount_[static_cast<size_t>(cacheSlot)] = count;
    noteCudaCompactProjectionH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      uploadStart)
            .count(),
        count * sizeof(double));
  }

  // Same caching contract as the projection, for the per-window background
  // (scatter estimate) the CSRQ ratio adds to the forward estimate.
  void ensureCompactBackgroundUploaded(const double *src, size_t count,
                                       int cacheSlot) {
    if (src == nullptr || count == 0) {
      throw std::runtime_error(
          "compact device-ratio CUDA helper received an empty background.");
    }
    if (cacheSlot < 0 || cacheSlot >= kCompactProjectionSlots) {
      throw std::runtime_error("compact background cache slot out of range.");
    }
    auto &dBackground = dCompactBackgroundInput_[static_cast<size_t>(cacheSlot)];
    if (compactBackgroundHostKey_[static_cast<size_t>(cacheSlot)] == src &&
        compactBackgroundCount_[static_cast<size_t>(cacheSlot)] == count &&
        dBackground.get() != nullptr && dBackground.size() >= count) {
      return;
    }
    const auto uploadStart = std::chrono::steady_clock::now();
    dBackground.copyFromHost(src, count);
    compactBackgroundHostKey_[static_cast<size_t>(cacheSlot)] = src;
    compactBackgroundCount_[static_cast<size_t>(cacheSlot)] = count;
    noteCudaCompactProjectionH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      uploadStart)
            .count(),
        count * sizeof(double));
  }

  // Sensitivity row weights W (detector masks, censoring, efficiency), one
  // per full projection row; uploaded once per host pointer. Returns nullptr
  // for "no weights".
  const double *ensureCompactRowWeightsUploaded(const double *rowWeights,
                                                size_t count) {
    if (rowWeights == nullptr || count == 0) {
      return nullptr;
    }
    if (compactRowWeightsHostKey_ != rowWeights ||
        compactRowWeightsCount_ != count || dCompactRowWeights_.get() == nullptr) {
      dCompactRowWeights_.copyFromHost(rowWeights, count);
      compactRowWeightsHostKey_ = rowWeights;
      compactRowWeightsCount_ = count;
    }
    return dCompactRowWeights_.get();
  }

  // Device background for a projection slot, or nullptr when the last
  // refreshCompactProjection for that slot carried none.
  const double *compactBackgroundDeviceLocked(int cacheSlot) const {
    const size_t slot = static_cast<size_t>(cacheSlot);
    return compactBackgroundHostKey_[slot] != nullptr
               ? dCompactBackgroundInput_[slot].get()
               : nullptr;
  }

  void uploadDoubleAsync(DeviceArray<float> &dev, const double *src,
                         size_t count, cudaStream_t stream) {
    hostFloatScratch_.resize(count);
    for (size_t i = 0; i < count; ++i)
      hostFloatScratch_[i] = static_cast<float>(src[i]);
    dev.copyFromHostAsync(hostFloatScratch_.data(), count, stream);
  }

  void downloadDouble(const DeviceArray<float> &dev, double *dst,
                      size_t count) {
    hostFloatScratch_.resize(count);
    dev.copyToHost(hostFloatScratch_.data(), count);
    for (size_t i = 0; i < count; ++i)
      dst[i] = static_cast<double>(hostFloatScratch_[i]);
  }

  enum class CachedMatrixValueType { Float };

  struct CachedMatrix {
    MatrixCacheKey key;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    CachedMatrixValueType valueType = CachedMatrixValueType::Float;
    DeviceArray<int32_t> dRowPtr;
    DeviceArray<int32_t> dColIdx;
    DeviceArray<float> dValues;
    CuSparseMatGuard matA;
    mutable std::array<size_t, 2> spmvBufferSizes{};
    mutable std::array<bool, 2> spmvBufferSizeValid{};
    mutable std::array<size_t, 2> spmmBufferSizes{};
    mutable std::array<bool, 2> spmmBufferSizeValid{};

    void prepareForUpload() {
      key = MatrixCacheKey{};
      rows = 0;
      cols = 0;
      nnz = 0;
      valueType = CachedMatrixValueType::Float;
      matA.reset();
      spmvBufferSizes = {};
      spmvBufferSizeValid = {};
      spmmBufferSizes = {};
      spmmBufferSizeValid = {};
      compactUploadPending = false;
    }

    void release() {
      prepareForUpload();
      dRowPtr.reset();
      dColIdx.reset();
      dValues.reset();
    }

    CudaEventGuard compactUploadReady;
    bool compactUploadPending = false;
    // Async upload staging for the (small, unregistered) rowPtr array: a
    // pageable-source cudaMemcpyAsync host-blocks until the stream drains,
    // which exposed whole-chunk copy times in the enqueue path.
    PinnedHostArray<int32_t> rowPtrStage;
    CudaEventGuard rowPtrStageDone;
    bool rowPtrStagePending = false;
  };

  // P3-D: Persistent device-side chunk cache.  Keeps as many chunks as fit
  // in VRAM across iterations so that their H2D upload can be skipped.
  // Uses fixed-residency assignment (not LRU) to avoid sequential-access
  // thrashing: the first N chunks that fit are permanently cached.
  struct DeviceChunkCache {
    std::vector<std::unique_ptr<CachedMatrix>> slots;
    std::vector<bool> valid;
    MatrixCacheKey matrixKey{};
    uint64_t nnzBudget = 0;
    size_t numCached = 0;
    bool initialized = false;

    bool isHit(size_t chunkIndex, const MatrixCacheKey &key) const {
      return initialized && matrixKey == key &&
             chunkIndex < valid.size() && valid[chunkIndex];
    }

    void invalidate() {
      slots.clear();
      valid.clear();
      numCached = 0;
      initialized = false;
    }
  };

  struct ResidentMatrixSetCache {
    std::vector<std::unique_ptr<CachedMatrix>> slots;
    std::vector<MatrixCacheKey> keys;
    std::vector<bool> valid;
    uint64_t reserveBytes = 0;
    size_t numCached = 0;
    bool initialized = false;

    bool matches(const std::vector<MatrixCacheKey> &otherKeys,
                 uint64_t otherReserveBytes) const {
      if (!initialized || reserveBytes != otherReserveBytes ||
          keys.size() != otherKeys.size()) {
        return false;
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        if (!(keys[i] == otherKeys[i])) {
          return false;
        }
      }
      return true;
    }

    const CachedMatrix *find(const MatrixCacheKey &key) const {
      if (!initialized) {
        return nullptr;
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        if (i < valid.size() && valid[i] && keys[i] == key &&
            i < slots.size() && slots[i] != nullptr) {
          return slots[i].get();
        }
      }
      return nullptr;
    }

    void invalidate() {
      slots.clear();
      keys.clear();
      valid.clear();
      reserveBytes = 0;
      numCached = 0;
      initialized = false;
    }
  };

  static size_t spmvBufferIndex(cusparseOperation_t operation) {
    return operation == CUSPARSE_OPERATION_TRANSPOSE ? 1u : 0u;
  }

  void executeSpMVDescriptorLocked(cusparseSpMatDescr_t mat,
                                   size_t &bufferSizeCache,
                                   bool &bufferSizeValid,
                                   cusparseOperation_t operation,
                                   cusparseDnVecDescr_t vecX,
                                   cusparseDnVecDescr_t vecY, float beta,
                                   bool profileAsBackward,
                                   cudaStream_t stream = nullptr) {
    const auto hostStart = std::chrono::steady_clock::now();
    const float alpha = 1.0f;
    constexpr cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG2;
    handle_.setStream(stream);
    size_t bufferSize = bufferSizeCache;
    if (!bufferSizeValid) {
      SPECT_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
          handle_.get(), operation, &alpha, mat, vecX, &beta, vecY,
          CUDA_R_32F, alg, &bufferSize));
      bufferSizeCache = bufferSize;
      bufferSizeValid = true;
    }
    dBuffer_.ensure(bufferSize);
    SPECT_CUSPARSE_CHECK(cusparseSpMV(
        handle_.get(), operation, &alpha, mat, vecX, &beta, vecY,
        CUDA_R_32F, alg, dBuffer_.get()));
    const double hostSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      hostStart)
            .count();
    if (profileAsBackward) {
      noteCudaCusparseBackwardHost(hostSeconds);
    } else {
      noteCudaCusparseForwardHost(hostSeconds);
    }
  }

  void executeSpMVLocked(const CachedMatrix &cached,
                         cusparseOperation_t operation,
                         cusparseDnVecDescr_t vecX,
                         cusparseDnVecDescr_t vecY, float beta,
                         cudaStream_t stream = nullptr) {
    const size_t bufferIndex = spmvBufferIndex(operation);
    executeSpMVDescriptorLocked(
        cached.matA.get(), cached.spmvBufferSizes[bufferIndex],
        cached.spmvBufferSizeValid[bufferIndex], operation, vecX, vecY, beta,
        operation == CUSPARSE_OPERATION_TRANSPOSE, stream);
  }

  // COL-major batch layout is one contiguous dense vector per RHS.  A 40-wide
  // SpMM with ldb=cols keeps a cols*batch working set (298 MiB on
  // Sampling_Approach) that misses L2 when CSR columns span the full Z grid.
  // Independent SpMVs keep each X/Y vector in L2 at the cost of re-reading A.
  // Rejected on Sampling 3-iter hit vs SpMV CSR_ALG2 27.076 s: 8-wide SpMM
  // tiles 68.362 s; ROW-major-B SpMM DEFAULT 48.790 s; ROW-major-B SpMM
  // CSR_ALG2 72.715 s; fused 4-RHS COL forward 35.433 s.
  // Forward tile is packed-K from a 60% L2 budget, K in {1,2,4}. packed-4
  // also requires sorted_hint (performance confidence, not correctness).
  // Transpose and packed remainder are per-RHS CSR_ALG2.
  void executeColMajorBatchSpMVLocked(const CachedMatrix &cached,
                                      cusparseOperation_t operation,
                                      float *matB, int64_t bRows, float *matC,
                                      int64_t cRows, int batchSize,
                                      cudaStream_t stream = nullptr,
                                      float *xPairScratch = nullptr) {
    if (batchSize <= 0 || bRows <= 0 || cRows <= 0) {
      throw std::runtime_error(
          "compact col-major batch SpMV requires positive sizes.");
    }
    if (matB == nullptr || matC == nullptr) {
      throw std::runtime_error(
          "compact col-major batch SpMV received a null dense buffer.");
    }
    int b = 0;
    const bool canPack =
        operation == CUSPARSE_OPERATION_NON_TRANSPOSE &&
        xPairScratch != nullptr &&
        bRows <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
        cRows <= static_cast<int64_t>(std::numeric_limits<int>::max());
    const int kMem =
        canPack ? cuda::compactPackedForwardKMem(
                      bRows, batchSize, cuda::getL2CacheBytes())
                : 1;
    const int tile =
        canPack ? cuda::compactPackedForwardTile(kMem, compactSortedHint_)
                : 1;
    if (canPack && !compactPackedKLogged_) {
      std::cerr << "[NOTE] compact packed forward K=" << tile
                << " k_mem=" << kMem
                << " sorted_hint=" << (compactSortedHint_ ? 1 : 0)
                << " l2_budget_frac=" << cuda::kCompactPackedXl2BudgetFraction
                << "\n";
      compactPackedKLogged_ = true;
    }
    if (canPack && tile >= 2) {
      const auto hostStart = std::chrono::steady_clock::now();
      const int cols = static_cast<int>(bRows);
      const int rows = static_cast<int>(cRows);
      if (tile >= 4) {
        for (; b + 3 < batchSize; b += 4) {
          const float *x0 =
              matB + static_cast<size_t>(b) * static_cast<size_t>(bRows);
          const float *x1 =
              matB + static_cast<size_t>(b + 1) * static_cast<size_t>(bRows);
          const float *x2 =
              matB + static_cast<size_t>(b + 2) * static_cast<size_t>(bRows);
          const float *x3 =
              matB + static_cast<size_t>(b + 3) * static_cast<size_t>(bRows);
          float *y0 =
              matC + static_cast<size_t>(b) * static_cast<size_t>(cRows);
          float *y1 =
              matC + static_cast<size_t>(b + 1) * static_cast<size_t>(cRows);
          float *y2 =
              matC + static_cast<size_t>(b + 2) * static_cast<size_t>(cRows);
          float *y3 =
              matC + static_cast<size_t>(b + 3) * static_cast<size_t>(cRows);
          launchCompactPackColQuadToFloat4Kernel(x0, x1, x2, x3, xPairScratch,
                                                 cols, stream);
          launchCompactCsrSpmvXQuadKernel(
              cached.dValues.get(), cached.dRowPtr.get(),
              cached.dColIdx.get(), xPairScratch, y0, y1, y2, y3, rows,
              stream);
        }
      }
      for (; b + 1 < batchSize; b += 2) {
        const float *x0 =
            matB + static_cast<size_t>(b) * static_cast<size_t>(bRows);
        const float *x1 =
            matB + static_cast<size_t>(b + 1) * static_cast<size_t>(bRows);
        float *y0 =
            matC + static_cast<size_t>(b) * static_cast<size_t>(cRows);
        float *y1 =
            matC + static_cast<size_t>(b + 1) * static_cast<size_t>(cRows);
        launchCompactPackColPairToFloat2Kernel(x0, x1, xPairScratch, cols,
                                               stream);
        launchCompactCsrSpmvXPairKernel(
            cached.dValues.get(), cached.dRowPtr.get(), cached.dColIdx.get(),
            xPairScratch, y0, y1, rows, stream);
      }
      noteCudaCusparseForwardHost(
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        hostStart)
              .count());
    }
    // Phase-B experiment: opt-in K-grouped transpose SpMM / SpMV preprocess.
    // Batch buffers are contiguous per-RHS slices, i.e. already COL-major
    // dense matrices (ld = vector length), so grouping needs no packing.
    const bool transposeOp = operation == CUSPARSE_OPERATION_TRANSPOSE;
    const CompactTransposeConfig &tCfg = compactTransposeConfig();
    if (transposeOp && tCfg.mode != CompactTransposeMode::KeepSpmvAlg2 &&
        !compactTransposeExpLogged_) {
      std::cerr << "[NOTE] experimental compact transpose path active: "
                << "SPECT_COMPACT_T_SPMM mode="
                << static_cast<int>(tCfg.mode) << " K=" << tCfg.groupK
                << "\n";
      compactTransposeExpLogged_ = true;
    }
    if (transposeOp && tCfg.mode == CompactTransposeMode::TrueCsrTranspose) {
      const bool slotsMatch =
          compactTransposeActiveValid_ &&
          compactTransposeActiveKey_ == cached.key &&
          compactTransposeSlots_[0].rows == cRows &&
          compactTransposeSlots_[0].rowBegin == 0 &&
          compactTransposeSlots_[1].rowEnd == static_cast<int>(bRows);
      if (slotsMatch) {
        const auto hostStart = std::chrono::steady_clock::now();
        if (!compactTransposeCsrtLogged_) {
          std::cerr << "[NOTE] csrt backward active: op=N SpMM on true "
                       "CSR(A^T) halves, K=" << batchSize << "\n";
          compactTransposeCsrtLogged_ = true;
        }
        handle_.setStream(stream);
        const float alpha = 1.0f;
        CuSparseDnMatGuard matBDescs[2];
        CuSparseDnMatGuard matCDescs[2];
        size_t spmmBytes = 0;
        for (int h = 0; h < 2; ++h) {
          CompactTransposeDeviceHalf &slot = compactTransposeSlots_[h];
          const float betaHalf = h == 0 ? 0.0f : 1.0f;
          matBDescs[h].create(slot.cols, batchSize, bRows,
                              matB + slot.rowBegin, CUDA_R_32F,
                              CUSPARSE_ORDER_COL);
          matCDescs[h].create(cRows, batchSize, cRows, matC, CUDA_R_32F,
                              CUSPARSE_ORDER_COL);
          size_t bytes = 0;
          SPECT_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
              handle_.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, slot.mat.get(),
              matBDescs[h].get(), &betaHalf, matCDescs[h].get(), CUDA_R_32F,
              CUSPARSE_SPMM_ALG_DEFAULT, &bytes));
          spmmBytes = std::max(spmmBytes, bytes);
        }
        dBuffer_.ensure(spmmBytes);
        for (int h = 0; h < 2; ++h) {
          CompactTransposeDeviceHalf &slot = compactTransposeSlots_[h];
          const float betaHalf = h == 0 ? 0.0f : 1.0f;
          if (slot.uploadPending) {
            slot.uploadReady.wait(stream);
            slot.uploadPending = false;
          }
          SPECT_CUSPARSE_CHECK(cusparseSpMM(
              handle_.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, slot.mat.get(),
              matBDescs[h].get(), &betaHalf, matCDescs[h].get(), CUDA_R_32F,
              CUSPARSE_SPMM_ALG_DEFAULT, dBuffer_.get()));
          slot.consumed.record(stream);
          slot.consumedRecorded = true;
        }
        noteCudaCusparseBackwardHost(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          hostStart)
                .count());
        b = batchSize;
      } else if (!compactTransposeFallbackLogged_) {
        std::cerr << "[NOTE] csrt: transpose call without queued A^T slots "
                     "(expected once for sensitivity/denominator); falling "
                     "back to per-RHS op=T.\n";
        compactTransposeFallbackLogged_ = true;
      }
    }
    if (transposeOp &&
        (tCfg.mode == CompactTransposeMode::SpmmCsrAlg1 ||
         tCfg.mode == CompactTransposeMode::SpmmCsrAlg1Preprocess ||
         tCfg.mode == CompactTransposeMode::SpmmDefault) &&
        batchSize >= tCfg.groupK) {
      const auto hostStart = std::chrono::steady_clock::now();
      const int k = tCfg.groupK;
      const cusparseSpMMAlg_t alg =
          tCfg.mode == CompactTransposeMode::SpmmDefault
              ? CUSPARSE_SPMM_ALG_DEFAULT
              : CUSPARSE_SPMM_CSR_ALG1;
      const float alpha = 1.0f;
      const float betaZero = 0.0f;
      CuSparseDnMatGuard matBDesc;
      CuSparseDnMatGuard matCDesc;
      matBDesc.create(bRows, k, bRows, matB, CUDA_R_32F, CUSPARSE_ORDER_COL);
      matCDesc.create(cRows, k, cRows, matC, CUDA_R_32F, CUSPARSE_ORDER_COL);
      handle_.setStream(stream);
      size_t spmmBufferSize = 0;
      SPECT_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
          handle_.get(), operation, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          cached.matA.get(), matBDesc.get(), &betaZero, matCDesc.get(),
          CUDA_R_32F, alg, &spmmBufferSize));
      dBuffer_.ensure(spmmBufferSize);
      if (tCfg.mode == CompactTransposeMode::SpmmCsrAlg1Preprocess) {
        SPECT_CUSPARSE_CHECK(cusparseSpMM_preprocess(
            handle_.get(), operation, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, cached.matA.get(), matBDesc.get(), &betaZero,
            matCDesc.get(), CUDA_R_32F, alg, dBuffer_.get()));
      }
      for (; b + k <= batchSize; b += k) {
        SPECT_CUSPARSE_CHECK(cusparseDnMatSetValues(
            matBDesc.get(),
            matB + static_cast<size_t>(b) * static_cast<size_t>(bRows)));
        SPECT_CUSPARSE_CHECK(cusparseDnMatSetValues(
            matCDesc.get(),
            matC + static_cast<size_t>(b) * static_cast<size_t>(cRows)));
        SPECT_CUSPARSE_CHECK(cusparseSpMM(
            handle_.get(), operation, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, cached.matA.get(), matBDesc.get(), &betaZero,
            matCDesc.get(), CUDA_R_32F, alg, dBuffer_.get()));
      }
      noteCudaCusparseBackwardHost(
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        hostStart)
              .count());
    }
    if (transposeOp && tCfg.mode == CompactTransposeMode::CscSpmvN &&
        b < batchSize) {
      // Control experiment: CSR(A) buffers are bit-identical to CSC(A^T)
      // [cRows x bRows]; run op=N SpMV per RHS on that descriptor.
      const auto hostStart = std::chrono::steady_clock::now();
      cusparseSpMatDescr_t cscMat = nullptr;
      SPECT_CUSPARSE_CHECK(cusparseCreateCsc(
          &cscMat, cRows, bRows, cached.nnz, cached.dRowPtr.get(),
          cached.dColIdx.get(), cached.dValues.get(), CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
      handle_.setStream(stream);
      const float alpha = 1.0f;
      const float betaZero = 0.0f;
      size_t cscBufferSize = 0;
      bool cscBufferSizeValid = false;
      for (; b < batchSize; ++b) {
        CuSparseVecGuard vecX;
        CuSparseVecGuard vecY;
        vecX.create(bRows, matB + static_cast<size_t>(b) *
                                      static_cast<size_t>(bRows));
        vecY.create(cRows, matC + static_cast<size_t>(b) *
                                      static_cast<size_t>(cRows));
        if (!cscBufferSizeValid) {
          SPECT_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
              handle_.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cscMat,
              vecX.get(), &betaZero, vecY.get(), CUDA_R_32F,
              CUSPARSE_SPMV_ALG_DEFAULT, &cscBufferSize));
          cscBufferSizeValid = true;
          std::cerr << "[NOTE] cscn control workspace bytes=" << cscBufferSize
                    << " (nnz=" << cached.nnz << ")\n";
          dBuffer_.ensure(cscBufferSize);
        }
        SPECT_CUSPARSE_CHECK(cusparseSpMV(
            handle_.get(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cscMat,
            vecX.get(), &betaZero, vecY.get(), CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_.get()));
      }
      SPECT_CUSPARSE_CHECK(cusparseDestroySpMat(cscMat));
      noteCudaCusparseBackwardHost(
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        hostStart)
              .count());
    }
    // Remainder and all transpose: per-RHS CSR_ALG2. Rejected custom T
    // (atomic 2-RHS, private-Y reduce) stays deleted.
    size_t bufferSize = 0;
    bool bufferSizeValid = false;
    const bool profileAsBackward = operation == CUSPARSE_OPERATION_TRANSPOSE;
    if (transposeOp &&
        tCfg.mode == CompactTransposeMode::SpmvAlg2Preprocess &&
        b < batchSize) {
      const auto hostStart = std::chrono::steady_clock::now();
      const float alpha = 1.0f;
      const float betaZero = 0.0f;
      CuSparseVecGuard vecX;
      CuSparseVecGuard vecY;
      vecX.create(bRows, matB);
      vecY.create(cRows, matC);
      handle_.setStream(stream);
      size_t preBufferSize = 0;
      SPECT_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
          handle_.get(), operation, &alpha, cached.matA.get(), vecX.get(),
          &betaZero, vecY.get(), CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2,
          &preBufferSize));
      bufferSize = preBufferSize;
      bufferSizeValid = true;
      dBuffer_.ensure(preBufferSize);
      SPECT_CUSPARSE_CHECK(cusparseSpMV_preprocess(
          handle_.get(), operation, &alpha, cached.matA.get(), vecX.get(),
          &betaZero, vecY.get(), CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2,
          dBuffer_.get()));
      noteCudaCusparseBackwardHost(
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        hostStart)
              .count());
    }
    for (; b < batchSize; ++b) {
      CuSparseVecGuard vecX;
      CuSparseVecGuard vecY;
      vecX.create(bRows, matB + static_cast<size_t>(b) *
                                    static_cast<size_t>(bRows));
      vecY.create(cRows, matC + static_cast<size_t>(b) *
                                    static_cast<size_t>(cRows));
      executeSpMVDescriptorLocked(
          cached.matA.get(), bufferSize, bufferSizeValid, operation,
          vecX.get(), vecY.get(), 0.0f, profileAsBackward, stream);
    }
  }

  void executeSpMMDescriptorLocked(
      cusparseSpMatDescr_t mat, size_t &bufferSizeCache,
      bool &bufferSizeValid, cusparseOperation_t opA,
      cusparseDnMatDescr_t matB, cusparseDnMatDescr_t matC,
      bool profileAsBackward, cudaStream_t stream = nullptr) {
    const auto hostStart = std::chrono::steady_clock::now();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    constexpr cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;
    handle_.setStream(stream);
    size_t bufferSize = bufferSizeCache;
    if (!bufferSizeValid) {
      SPECT_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
          handle_.get(), opA, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat,
          matB, &beta, matC, CUDA_R_32F, alg, &bufferSize));
      bufferSizeCache = bufferSize;
      bufferSizeValid = true;
    }
    dBuffer_.ensure(bufferSize);
    SPECT_CUSPARSE_CHECK(cusparseSpMM(
        handle_.get(), opA, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat,
        matB, &beta, matC, CUDA_R_32F, alg, dBuffer_.get()));
    const double hostSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      hostStart)
            .count();
    if (profileAsBackward) {
      noteCudaCusparseBackwardHost(hostSeconds);
    } else {
      noteCudaCusparseForwardHost(hostSeconds);
    }
  }

  void executeSpMMLocked(const CachedMatrix &cached,
                         cusparseOperation_t opA,
                         cusparseDnMatDescr_t matB,
                         cusparseDnMatDescr_t matC,
                         cudaStream_t stream = nullptr) {
    const size_t bufferIndex = spmvBufferIndex(opA);
    executeSpMMDescriptorLocked(
        cached.matA.get(), cached.spmmBufferSizes[bufferIndex],
        cached.spmmBufferSizeValid[bufferIndex], opA, matB, matC,
        opA == CUSPARSE_OPERATION_TRANSPOSE, stream);
  }

  void createMatA(CachedMatrix *cache) {
    cache->matA.create(cache->rows, cache->cols, cache->nnz,
                       cache->dRowPtr.get(), cache->dColIdx.get(),
                       cache->dValues.get());
  }

  void uploadValuesAndCreateMatA(CachedMatrix *cache, const float *hostValues,
                                 size_t valueCount) {
    cache->dValues.copyFromHost(hostValues, valueCount);
    createMatA(cache);
  }

  void uploadHostCSR(const HostCSRData &host, const MatrixCacheKey &key,
                     CachedMatrix *cache) {
    cache->key = key;
    cache->rows = host.rows;
    cache->cols = host.cols;
    cache->nnz = host.nnz;
    cache->dRowPtr.copyFromHost(host.rowPtr.data(), host.rowPtr.size());
    cache->dColIdx.copyFromHost(host.colIdx.data(), host.colIdx.size());
    uploadValuesAndCreateMatA(cache, host.values.data(), host.values.size());
  }

  void uploadHostCSR(const PinnedHostCSRData &host, const MatrixCacheKey &key,
                     CachedMatrix *cache) {
    cache->key = key;
    cache->rows = host.rows;
    cache->cols = host.cols;
    cache->nnz = host.nnz;
    cache->dRowPtr.copyFromHost(host.rowPtr.data(), host.rowPtrCount);
    cache->dColIdx.copyFromHost(host.colIdx.data(), host.colIdxCount);
    uploadValuesAndCreateMatA(cache, host.values.data(), host.valuesCount);
  }

  void uploadHostCSRAsync(const PinnedHostCSRData &host,
                          const MatrixCacheKey &key, CachedMatrix *cache,
                          cudaStream_t stream) {
    cache->key = key;
    cache->rows = host.rows;
    cache->cols = host.cols;
    cache->nnz = host.nnz;
    cache->dRowPtr.copyFromHostAsync(host.rowPtr.data(), host.rowPtrCount,
                                     stream);
    cache->dColIdx.copyFromHostAsync(host.colIdx.data(), host.colIdxCount,
                                     stream);
    cache->dValues.copyFromHostAsync(host.values.data(), host.valuesCount,
                                     stream);
    createMatA(cache);
  }

  void uploadIoCsrView(const io::CSRView &matrix, const MatrixCacheKey &key,
                       CachedMatrix *cache) {
    cache->prepareForUpload();
    cache->key = key;
    cache->rows = matrix.rows;
    cache->cols = matrix.cols;
    cache->nnz = static_cast<int64_t>(matrix.nnz);
    cache->valueType = CachedMatrixValueType::Float;
    cache->dRowPtr.copyFromHost(
        reinterpret_cast<const int32_t *>(matrix.row_ptr),
        static_cast<size_t>(matrix.rows) + 1u);
    cache->dColIdx.copyFromHost(
        reinterpret_cast<const int32_t *>(matrix.col_idx),
        static_cast<size_t>(matrix.nnz));
    uploadValuesAndCreateMatA(cache, matrix.val,
                              static_cast<size_t>(matrix.nnz));
  }

  void uploadIoCsrViewAsync(const io::CSRView &matrix, const MatrixCacheKey &key,
                            CachedMatrix *cache, cudaStream_t stream) {
    cache->prepareForUpload();
    cache->key = key;
    cache->rows = matrix.rows;
    cache->cols = matrix.cols;
    cache->nnz = static_cast<int64_t>(matrix.nnz);
    cache->valueType = CachedMatrixValueType::Float;
    const size_t rowPtrCount = static_cast<size_t>(matrix.rows) + 1u;
    if (cache->rowPtrStagePending) {
      CUDA_CHECK(cudaEventSynchronize(cache->rowPtrStageDone.get()));
      cache->rowPtrStagePending = false;
    }
    cache->rowPtrStage.copyFromHost(
        reinterpret_cast<const int32_t *>(matrix.row_ptr), rowPtrCount);
    cache->dRowPtr.copyFromHostAsync(cache->rowPtrStage.data(), rowPtrCount,
                                     stream);
    cache->rowPtrStageDone.record(stream);
    cache->rowPtrStagePending = true;
    cache->dColIdx.copyFromHostAsync(
        reinterpret_cast<const int32_t *>(matrix.col_idx),
        static_cast<size_t>(matrix.nnz), stream);
    cache->dValues.copyFromHostAsync(matrix.val, static_cast<size_t>(matrix.nnz),
                                     stream);
    createMatA(cache);
  }


  void validateIoCsrViewForUpload(const io::CSRView &matrix) const {
    if (!matrix.valid()) {
      throw std::runtime_error(
          "Invalid io::CSRView passed to CUDA sparse backend.");
    }
    const uint32_t nnz = matrix.row_ptr[static_cast<size_t>(matrix.rows)];
    if (nnz > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        matrix.nnz > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error(
          "CUDA sparse backend only supports nnz representable as int32.");
    }
    if (matrix.cols > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error(
          "CUDA sparse backend only supports column dimensions representable as int32.");
    }
  }


  template <typename MatrixType>
  const CachedMatrix &ensureMatrixUploaded(const MatrixType &matrix) {
    const MatrixCacheKey key = makeCacheKey(matrix);
    if (cache_.key == key && cache_.matA.get() != nullptr) {
      return cache_;
    }

    const HostCSRData host = convertCSR(matrix);
    uploadHostCSR(host, key, &cache_);
    return cache_;
  }

  const CachedMatrix &ensureMatrixUploaded(const io::CSRView &matrix) {
    const MatrixCacheKey key = makeCacheKey(matrix);
    if (const CachedMatrix *resident = residentMatrixSetCache_.find(key)) {
      return *resident;
    }
    if (cache_.key == key && cache_.matA.get() != nullptr) {
      return cache_;
    }
    validateIoCsrViewForUpload(matrix);
    uploadIoCsrView(matrix, key, &cache_);
    return cache_;
  }

  CachedMatrix *findCompactFloatSlot(const MatrixCacheKey &key) {
    if (residentMatrixSetCache_.initialized) {
      for (size_t i = 0; i < residentMatrixSetCache_.keys.size(); ++i) {
        if (i < residentMatrixSetCache_.valid.size() &&
            residentMatrixSetCache_.valid[i] &&
            residentMatrixSetCache_.keys[i] == key &&
            i < residentMatrixSetCache_.slots.size() &&
            residentMatrixSetCache_.slots[i] != nullptr) {
          return residentMatrixSetCache_.slots[i].get();
        }
      }
    }
    if (cache_.key == key && cache_.matA.get() != nullptr) {
      return &cache_;
    }
    return nullptr;
  }

  CachedMatrix *selectCompactPrefetchTarget(const MatrixCacheKey &currentKey) {
    if (!(cache_.key == currentKey && cache_.matA.get() != nullptr)) {
      return &cache_;
    }
    if (!residentMatrixSetCache_.initialized) {
      return nullptr;
    }
    for (size_t i = 0; i < residentMatrixSetCache_.slots.size(); ++i) {
      if (i < residentMatrixSetCache_.valid.size() &&
          residentMatrixSetCache_.valid[i] &&
          i < residentMatrixSetCache_.keys.size() &&
          !(residentMatrixSetCache_.keys[i] == currentKey) &&
          residentMatrixSetCache_.slots[i] != nullptr) {
        return residentMatrixSetCache_.slots[i].get();
      }
    }
    return nullptr;
  }

  void bindCompactSlotKey(CachedMatrix *slot, const MatrixCacheKey &key) {
    if (slot == nullptr || slot == &cache_) {
      return;
    }
    if (!residentMatrixSetCache_.initialized) {
      return;
    }
    for (size_t i = 0; i < residentMatrixSetCache_.slots.size(); ++i) {
      if (residentMatrixSetCache_.slots[i].get() == slot) {
        if (i < residentMatrixSetCache_.keys.size()) {
          residentMatrixSetCache_.keys[i] = key;
        }
        if (i < residentMatrixSetCache_.valid.size()) {
          residentMatrixSetCache_.valid[i] = true;
        }
        return;
      }
    }
  }

  // Do NOT cudaHostRegister io::CSRView arrays here: the chunk loader owns
  // those buffers and frees/reallocates them on host-cache eviction, while
  // pins are only dropped at release. A stale registration overlapping a
  // reissued allocation makes later cudaMemcpy fail with "invalid argument"
  // (seen at production scale on the 220 matrix with a 3/6-chunk host cache).
  // The loader's own pinned prefetch path already covers bandwidth.

  void waitCompactSlotUpload(CachedMatrix *slot, cudaStream_t stream) {
    if (slot == nullptr || !slot->compactUploadPending) {
      return;
    }
    slot->compactUploadReady.wait(stream);
    slot->compactUploadPending = false;
  }

  void unregisterCompactHostPinsLocked() {
    for (void *ptr : compactHostPins_) {
      const cudaError_t err = cudaHostUnregister(ptr);
      if (err != cudaSuccess) {
        (void)cudaGetLastError();
      }
    }
    compactHostPins_.clear();
  }

  const CachedMatrix &ensureCompactFloatMatrixUploaded(
      const io::CSRView &matrix) {
    const MatrixCacheKey key = makeCacheKey(matrix);
    if (CachedMatrix *slot = findCompactFloatSlot(key)) {
      waitCompactSlotUpload(slot, nullptr);
      noteCudaCompactMatrixReuseHit();
      return *slot;
    }
    validateIoCsrViewForUpload(matrix);
    waitCompactSlotUpload(&cache_, nullptr);
    const auto uploadStart = std::chrono::steady_clock::now();
    uploadIoCsrView(matrix, key, &cache_);
    noteCudaCompactMatrixUpload(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      uploadStart)
            .count());
    return cache_;
  }


  const CachedMatrix &ensureSubsetUploaded(const io::CSRView &matrix,
                                           const std::vector<int> &subset_rows,
                                           int threads) {
    const SubsetCacheKey key = makeSubsetCacheKey(matrix, subset_rows);
    for (size_t index = 0; index < subsetCache_.size(); ++index) {
      if (subsetCacheValid_[index] && subsetCacheKeys_[index] == key &&
          subsetCache_[index].matA.get() != nullptr) {
        return subsetCache_[index];
      }
    }

    const size_t slot = nextSubsetCacheSlot_;
    nextSubsetCacheSlot_ = (nextSubsetCacheSlot_ + 1) % subsetCache_.size();

    buildSubsetCSR(matrix, subset_rows, threads, subsetHostScratch_);
    uploadHostCSR(subsetHostScratch_, key.matrix, &subsetCache_[slot]);
    subsetCacheKeys_[slot] = key;
    subsetCacheValid_[slot] = true;
    return subsetCache_[slot];
  }

  const CachedMatrix &ensureStreamingRangeUploaded(const io::CSRView &matrix,
                                                   int rowBegin, int rowEnd,
                                                   int threads) {
    const RangeCacheKey key = makeRangeCacheKey(matrix, rowBegin, rowEnd);
    if (streamingRangeValid_ && streamingRangeKey_ == key &&
        streamingRangeCache_.matA.get() != nullptr) {
      return streamingRangeCache_;
    }

    buildContiguousSubsetCSR(matrix, rowBegin, rowEnd, threads,
                             subsetHostScratch_);
    uploadHostCSR(subsetHostScratch_, key.matrix, &streamingRangeCache_);
    streamingRangeKey_ = key;
    streamingRangeValid_ = true;
    return streamingRangeCache_;
  }

  const CachedMatrix &
  ensurePreparedStreamingRangeUploaded(const io::CSRView &matrix, int rowBegin,
                                       int rowEnd,
                                       const PinnedHostCSRData &hostChunk) {
    const RangeCacheKey key = makeRangeCacheKey(matrix, rowBegin, rowEnd);
    if (streamingRangeValid_ && streamingRangeKey_ == key &&
        streamingRangeCache_.matA.get() != nullptr) {
      return streamingRangeCache_;
    }

    uploadHostCSR(hostChunk, key.matrix, &streamingRangeCache_);
    streamingRangeKey_ = key;
    streamingRangeValid_ = true;
    return streamingRangeCache_;
  }

  // ---- csrt experiment (SPECT_COMPACT_T_SPMM=csrt): true CSR(A^T) ----
  // Host A^T is derived once per chunk (content-fingerprinted so host-cache
  // eviction/reload of A does not rebuild), split into two row halves so the
  // per-half upload hides under forward+ratio / the first backward SpMM.

  struct CompactTransposeHostHalf {
    int rowBegin = 0;
    int rowEnd = 0;
    int64_t nnz = 0;
    std::vector<int32_t> rowPtr; // cols(A)+1
    std::vector<int32_t> colIdx; // local A-row indices (r - rowBegin)
    std::vector<float> values;
  };

  struct CompactTransposeHostEntry {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    std::array<CompactTransposeHostHalf, 2> halves;
  };

  struct CompactTransposeDeviceHalf {
    DeviceArray<int32_t> dRowPtr;
    DeviceArray<int32_t> dColIdx;
    DeviceArray<float> dValues;
    CuSparseMatGuard mat;
    int64_t rows = 0; // = cols(A)
    int64_t cols = 0; // = rowEnd - rowBegin
    int64_t nnz = 0;
    int rowBegin = 0;
    int rowEnd = 0;
    CudaEventGuard uploadReady;
    CudaEventGuard consumed;
    bool uploadPending = false;
    bool consumedRecorded = false;
  };

  static uint64_t compactTransposeFingerprint(const io::CSRView &matrix) {
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](uint64_t v) {
      hash ^= v;
      hash *= 1099511628211ull;
    };
    mix(static_cast<uint64_t>(matrix.rows));
    mix(static_cast<uint64_t>(matrix.cols));
    mix(matrix.nnz);
    constexpr int kSamples = 64;
    for (int i = 0; i <= kSamples; ++i) {
      const size_t idx =
          matrix.rows > 0
              ? (static_cast<size_t>(matrix.rows) * static_cast<size_t>(i)) /
                    kSamples
              : 0;
      mix(matrix.row_ptr[idx]);
    }
    return hash;
  }

  void registerCompactTransposePinned(void *ptr, size_t bytes) {
    // Page-granular pinning of tiny heap blocks entangles neighbouring
    // allocations (toy tests); bandwidth only matters for the GB-scale
    // production arrays anyway.
    if (bytes < (1u << 20)) {
      return;
    }
    const cudaError_t err =
        cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
      compactHostPins_.push_back(ptr);
    } else {
      (void)cudaGetLastError();
      if (!compactTransposePinWarned_) {
        std::cerr << "[NOTE] csrt: pinning host A^T failed ("
                  << cudaGetErrorString(err)
                  << "); uploads fall back to pageable bandwidth.\n";
        compactTransposePinWarned_ = true;
      }
    }
  }

  const CompactTransposeHostEntry &ensureCompactTransposeHostEntry(
      uint64_t fingerprint, const io::CSRView &matrix) {
    auto it = compactTransposeHost_.find(fingerprint);
    if (it != compactTransposeHost_.end()) {
      const CompactTransposeHostEntry &entry = it->second;
      if (entry.rows != matrix.rows || entry.cols != matrix.cols ||
          entry.nnz != static_cast<int64_t>(matrix.nnz)) {
        throw std::runtime_error(
            "csrt: fingerprint collision between distinct chunks; aborting "
            "experiment.");
      }
      return entry;
    }

    const auto buildStart = std::chrono::steady_clock::now();
    CompactTransposeHostEntry entry;
    entry.rows = matrix.rows;
    entry.cols = matrix.cols;
    entry.nnz = static_cast<int64_t>(matrix.nnz);
    const int mid = matrix.rows / 2;
    for (int h = 0; h < 2; ++h) {
      CompactTransposeHostHalf &half = entry.halves[h];
      half.rowBegin = h == 0 ? 0 : mid;
      half.rowEnd = h == 0 ? mid : matrix.rows;
      const uint32_t nnzBase = matrix.row_ptr[half.rowBegin];
      half.nnz = static_cast<int64_t>(matrix.row_ptr[half.rowEnd]) -
                 static_cast<int64_t>(nnzBase);
      if (half.nnz > static_cast<int64_t>(
                         std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("csrt: half-chunk nnz exceeds int32.");
      }
      half.rowPtr.assign(static_cast<size_t>(matrix.cols) + 1u, 0);
      half.colIdx.resize(static_cast<size_t>(half.nnz));
      half.values.resize(static_cast<size_t>(half.nnz));
      {
        std::vector<int32_t> counts(static_cast<size_t>(matrix.cols), 0);
        const uint32_t *colBase = matrix.col_idx + nnzBase;
        for (int64_t k = 0; k < half.nnz; ++k) {
          ++counts[colBase[k]];
        }
        int64_t running = 0;
        for (int c = 0; c < matrix.cols; ++c) {
          half.rowPtr[static_cast<size_t>(c)] =
              static_cast<int32_t>(running);
          running += counts[static_cast<size_t>(c)];
        }
        half.rowPtr[static_cast<size_t>(matrix.cols)] =
            static_cast<int32_t>(running);
      }
      {
        std::unique_ptr<std::atomic<int32_t>[]> cursors(
            new std::atomic<int32_t>[static_cast<size_t>(matrix.cols)]);
        for (int c = 0; c < matrix.cols; ++c) {
          cursors[static_cast<size_t>(c)].store(
              half.rowPtr[static_cast<size_t>(c)],
              std::memory_order_relaxed);
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 256)
#endif
        for (int r = half.rowBegin; r < half.rowEnd; ++r) {
          for (uint32_t k = matrix.row_ptr[r]; k < matrix.row_ptr[r + 1];
               ++k) {
            const uint32_t c = matrix.col_idx[k];
            const int32_t pos =
                cursors[c].fetch_add(1, std::memory_order_relaxed);
            half.colIdx[static_cast<size_t>(pos)] =
                static_cast<int32_t>(r - half.rowBegin);
            half.values[static_cast<size_t>(pos)] = matrix.val[k];
          }
        }
      }
    }

    auto emplaced = compactTransposeHost_.emplace(fingerprint, std::move(entry));
    CompactTransposeHostEntry &stored = emplaced.first->second;
    for (int h = 0; h < 2; ++h) {
      CompactTransposeHostHalf &half = stored.halves[h];
      registerCompactTransposePinned(half.rowPtr.data(),
                                     half.rowPtr.size() * sizeof(int32_t));
      registerCompactTransposePinned(half.colIdx.data(),
                                     half.colIdx.size() * sizeof(int32_t));
      registerCompactTransposePinned(half.values.data(),
                                     half.values.size() * sizeof(float));
    }
    std::cerr << "[NOTE] csrt: host A^T halves built in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - buildStart)
                     .count()
              << " s (rows=" << stored.rows << " nnz=" << stored.nnz
              << " cached_chunks=" << compactTransposeHost_.size() << ")\n";
    return stored;
  }

  void queueCompactTrueTransposeUploadLocked(const io::CSRView &matrix) {
    if (matrix.rows < 2 || matrix.nnz == 0) {
      return; // degenerate halves; backward falls back to per-RHS op=T
    }
    const MatrixCacheKey key = makeCacheKey(matrix);
    const uint64_t fingerprint = compactTransposeFingerprint(matrix);
    // Pointer keys can alias after free/realloc; require the content
    // fingerprint to match too before trusting the resident A^T slots.
    if (compactTransposeActiveValid_ && compactTransposeActiveKey_ == key &&
        compactTransposeActiveFingerprint_ == fingerprint) {
      return;
    }
    const CompactTransposeHostEntry &entry =
        ensureCompactTransposeHostEntry(fingerprint, matrix);
    if (entry.halves[0].nnz == 0 || entry.halves[1].nnz == 0) {
      return;
    }
    cudaStream_t uploadStream = compactTransposeUploadStream_.get();
    const auto queueStart = std::chrono::steady_clock::now();
    for (int h = 0; h < 2; ++h) {
      CompactTransposeDeviceHalf &slot = compactTransposeSlots_[h];
      const CompactTransposeHostHalf &half = entry.halves[h];
      if (slot.consumedRecorded) {
        slot.consumed.wait(uploadStream);
        slot.consumedRecorded = false;
      }
      slot.dRowPtr.copyFromHostAsync(half.rowPtr.data(), half.rowPtr.size(),
                                     uploadStream);
      slot.dColIdx.copyFromHostAsync(half.colIdx.data(), half.colIdx.size(),
                                     uploadStream);
      slot.dValues.copyFromHostAsync(half.values.data(), half.values.size(),
                                     uploadStream);
      slot.rows = entry.cols;
      slot.cols = static_cast<int64_t>(half.rowEnd - half.rowBegin);
      slot.nnz = half.nnz;
      slot.rowBegin = half.rowBegin;
      slot.rowEnd = half.rowEnd;
      slot.mat.create(slot.rows, slot.cols, slot.nnz, slot.dRowPtr.get(),
                      slot.dColIdx.get(), slot.dValues.get());
      slot.uploadReady.record(uploadStream);
      slot.uploadPending = true;
    }
    compactTransposeActiveKey_ = key;
    compactTransposeActiveFingerprint_ = fingerprint;
    compactTransposeActiveValid_ = true;
    noteCudaCompactMatrixUpload(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      queueStart)
            .count());
  }

  void queueTransientRangeUpload(size_t slot, const MatrixCacheKey &key,
                                 const PinnedHostCSRData &hostChunk) {
    if (slot >= transientRangeCache_.size()) {
      throw std::runtime_error("CUDA transient range slot out of range.");
    }
    finalizeTransientRangeUploadTiming(slot);
    cudaStream_t stream = transientUploadStreams_[slot].get();
    if (transientRangeSlotUsed_[slot]) {
      transientRangeReusable_[slot].wait(stream);
    }
    transientRangeUploadStart_[slot].record(stream);
    uploadHostCSRAsync(hostChunk, key, &transientRangeCache_[slot], stream);
    transientRangeReady_[slot].record(stream);
    transientRangeValid_[slot] = true;
    transientRangeSlotUsed_[slot] = true;
    transientRangeUploadTimingPending_[slot] = true;
    transientRangeReusableReady_[slot] = false;
  }

  const CachedMatrix &waitTransientRangeUpload(
      size_t slot, TimedCudaStagePair *waitTiming = nullptr) {
    if (slot >= transientRangeCache_.size()) {
      throw std::runtime_error("CUDA transient range slot is out of range.");
    }
    if (!transientRangeValid_[slot]) {
      throw std::runtime_error("CUDA transient range slot is not ready.");
    }
    if (waitTiming != nullptr) {
      waitTiming->recordStart(computeStream_.get());
    }
    transientRangeReady_[slot].wait(computeStream_.get());
    if (waitTiming != nullptr) {
      waitTiming->recordStop(computeStream_.get());
    }
    return transientRangeCache_[slot];
  }

  void finalizeTransientRangeUploadTiming(size_t slot) {
    if (slot >= transientRangeCache_.size() ||
        !transientRangeUploadTimingPending_[slot]) {
      return;
    }
    const cudaError_t status = cudaEventQuery(transientRangeReady_[slot].get());
    if (status == cudaErrorNotReady) {
      cudaGetLastError();
      return;
    }
    CUDA_CHECK(status);
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs,
                                    transientRangeUploadStart_[slot].get(),
                                    transientRangeReady_[slot].get()));
    noteCudaH2DChunk(static_cast<double>(elapsedMs) / 1000.0);
    transientRangeUploadTimingPending_[slot] = false;
  }

  void waitTransientPinnedScratchReusable(size_t slot) {
    if (slot >= transientRangeCache_.size() || !transientRangeSlotUsed_[slot]) {
      return;
    }
    // The upload stream wait only orders future H2D work on that stream; it
    // does not stop the host from overwriting the pinned staging buffer early.
    // Reusing the slot-local pinned scratch therefore requires a host-side wait
    // until the previous upload on that slot has actually completed.
    CUDA_CHECK(cudaEventSynchronize(transientRangeReady_[slot].get()));
  }

  void markTransientRangeReusable(size_t slot) {
    if (slot >= transientRangeCache_.size() || !transientRangeValid_[slot]) {
      return;
    }
    finalizeTransientRangeUploadTiming(slot);
    transientRangeReusable_[slot].record(computeStream_.get());
    transientRangeReusableReady_[slot] = true;
  }

  void resetTransientRangeState() {
    transientRangeValid_.fill(false);
    transientRangeSlotUsed_.fill(false);
    transientRangeUploadTimingPending_.fill(false);
    transientRangeReusableReady_.fill(false);
  }



public:
  void refreshCompactBranchBatchInput(const float *input, size_t count) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (input == nullptr || count == 0) {
      throw std::runtime_error("compact branch-batch input is empty.");
    }
    const auto inputStart = std::chrono::steady_clock::now();
    dBatchInput_.copyFromHost(input, count);
    compactBatchInputCount_ = count;
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      inputStart)
            .count());
  }

  void prepareCompactBranchBatchInputFromGlobal(
      const double *globalImage, size_t globalImageCount, int globalNz,
      const int *bedZStarts, int bedCount, bool flipLocalZ,
      bool uploadGlobalImage,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      int branchCount, int zBranchCount,
      const std::vector<double> *castorKx,
      const std::vector<double> *castorKy,
      const std::vector<double> *castorKz) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (globalImage == nullptr || globalImageCount == 0 || globalNz <= 0 ||
        bedZStarts == nullptr || bedCount <= 0 ||
        bedCount > spect::core::kMaxBeds || branchCount <= 0 ||
        zBranchCount <= 0 || commonTransformParams.nx <= 0 ||
        commonTransformParams.ny <= 0 || commonTransformParams.nz <= 0) {
      throw std::runtime_error(
          "compact resident branch-batch input parameters are invalid.");
    }
    const bool useCastor =
        castorKx != nullptr || castorKy != nullptr || castorKz != nullptr;
    if (useCastor &&
        (castorKx == nullptr || castorKy == nullptr || castorKz == nullptr)) {
      throw std::runtime_error(
          "compact resident CASToR PSF requires all three kernels.");
    }
    const size_t plane =
        static_cast<size_t>(commonTransformParams.nx) *
        static_cast<size_t>(commonTransformParams.ny);
    const size_t nvox =
        plane * static_cast<size_t>(commonTransformParams.nz);
    if (globalImageCount != plane * static_cast<size_t>(globalNz)) {
      throw std::runtime_error(
          "compact resident global image dimensions do not match.");
    }
    for (int bed = 0; bed < bedCount; ++bed) {
      if (bedZStarts[bed] < 0 ||
          bedZStarts[bed] + commonTransformParams.nz > globalNz) {
        throw std::runtime_error(
            "compact resident bed range exceeds the global image.");
      }
    }

    if (uploadGlobalImage) {
      const auto inputStart = std::chrono::steady_clock::now();
      dCompactGlobalImage_.copyFromHost(globalImage, globalImageCount);
      compactResidentImageCount_ = globalImageCount;
      noteCudaCompactInputH2D(
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        inputStart)
              .count());
    } else if (compactResidentImageCount_ != globalImageCount ||
               dCompactGlobalImage_.get() == nullptr) {
      throw std::runtime_error(
          "compact resident image was not initialized for this iteration.");
    }
    dCompactBedPrimary_.allocate(nvox);
    if (useCastor)
      dCompactBedSecondary_.allocate(nvox);
    const size_t batchCount = nvox * static_cast<size_t>(branchCount) *
                              static_cast<size_t>(bedCount);
    dBatchInput_.allocate(batchCount);
    compactBatchInputCount_ = batchCount;
    ensureProjDomainKernelsLoaded();

    for (int bed = 0; bed < bedCount; ++bed) {
      launchCompactGatherBedDoubleToFloatKernel(
          dCompactGlobalImage_.get(), globalImageCount,
          dCompactBedPrimary_.get(), commonTransformParams.nx,
          commonTransformParams.ny, commonTransformParams.nz, globalNz,
          bedZStarts[bed], flipLocalZ);
      const float *bedInput = dCompactBedPrimary_.get();
      if (useCastor) {
        if (!cuda::enqueueCastorConv3DFloatDeviceCUDA(
                dCompactBedPrimary_.get(), dCompactBedSecondary_.get(),
                commonTransformParams.nx, commonTransformParams.ny,
                commonTransformParams.nz, *castorKx, *castorKy, *castorKz)) {
          throw std::runtime_error(
              "compact resident CASToR PSF operation is unavailable.");
        }
        bedInput = dCompactBedSecondary_.get();
      }
      float *bedBatch =
          dBatchInput_.get() + static_cast<size_t>(bed) *
                                   static_cast<size_t>(branchCount) * nvox;
      launchCompactTransformForwardBatchKernel(
          bedInput, bedBatch, nvox, branchCount, zBranchCount,
          commonTransformParams);
    }
  }

  // Shared tail of a resident iteration: per bed, convert the backward
  // accumulator, apply the CASToR PSF adjoint, and add the bed volume into the
  // global numerator q on the device. Refreshes the static denominator / mask
  // uploads when asked. Caller holds mutex_.
  void accumulateCompactResidentDataTermLocked(
      size_t globalImageCount, int globalNz, const int *bedZStarts,
      int bedCount, bool flipLocalZ,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const double *denom, const uint8_t *mask,
      const std::vector<double> *castorKx,
      const std::vector<double> *castorKy,
      const std::vector<double> *castorKz, bool refreshStaticInputs) {
    if (globalImageCount == 0 || globalNz <= 0 || bedZStarts == nullptr ||
        bedCount <= 0 || bedCount > spect::core::kMaxBeds || denom == nullptr ||
        compactResidentImageCount_ != globalImageCount ||
        dCompactGlobalImage_.get() == nullptr) {
      throw std::runtime_error(
          "compact resident data-term parameters are invalid.");
    }
    const bool useCastor =
        castorKx != nullptr || castorKy != nullptr || castorKz != nullptr;
    if (useCastor &&
        (castorKx == nullptr || castorKy == nullptr || castorKz == nullptr)) {
      throw std::runtime_error(
          "compact resident backward CASToR PSF requires all kernels.");
    }
    const size_t plane =
        static_cast<size_t>(commonTransformParams.nx) *
        static_cast<size_t>(commonTransformParams.ny);
    const size_t nvox =
        plane * static_cast<size_t>(commonTransformParams.nz);
    if (globalImageCount != plane * static_cast<size_t>(globalNz)) {
      throw std::runtime_error(
          "compact resident global dimensions do not match.");
    }
    for (int bed = 0; bed < bedCount; ++bed) {
      const size_t slot = compactBackwardAccumulatorSlot(bed);
      if (compactBackwardAccumSize_[slot] != nvox ||
          dCompactBackwardAccum_[slot].get() == nullptr ||
          bedZStarts[bed] < 0 ||
          bedZStarts[bed] + commonTransformParams.nz > globalNz) {
        throw std::runtime_error(
            "compact resident backward accumulator or bed range mismatch.");
      }
    }

    if (refreshStaticInputs) {
      dCompactDenom_.copyFromHost(denom, globalImageCount);
      if (mask != nullptr)
        dCompactMask_.copyFromHost(mask, globalImageCount);
      compactResidentStaticCount_ = globalImageCount;
      compactResidentHasMask_ = mask != nullptr;
    } else if (compactResidentStaticCount_ != globalImageCount ||
               dCompactDenom_.get() == nullptr ||
               compactResidentHasMask_ != (mask != nullptr)) {
      throw std::runtime_error(
          "compact resident denominator or mask was not initialized.");
    }

    dCompactGlobalNumer_.allocate(globalImageCount);
    CUDA_CHECK(cudaMemset(dCompactGlobalNumer_.get(), 0,
                          globalImageCount * sizeof(double)));
    dCompactBedPrimary_.allocate(nvox);
    if (useCastor)
      dCompactBedSecondary_.allocate(nvox);
    for (int bed = 0; bed < bedCount; ++bed) {
      const size_t slot = compactBackwardAccumulatorSlot(bed);
      launchCompactConvertDoubleToFloatKernel(
          dCompactBackwardAccum_[slot].get(), dCompactBedPrimary_.get(),
          nvox);
      const float *bedNumer = dCompactBedPrimary_.get();
      if (useCastor) {
        if (!cuda::enqueueCastorConv3DFloatDeviceCUDA(
                dCompactBedPrimary_.get(), dCompactBedSecondary_.get(),
                commonTransformParams.nx, commonTransformParams.ny,
                commonTransformParams.nz, *castorKx, *castorKy, *castorKz)) {
          throw std::runtime_error(
              "compact resident backward CASToR PSF is unavailable.");
        }
        bedNumer = dCompactBedSecondary_.get();
      }
      launchCompactAccumulateBedFloatToGlobalDoubleKernel(
          bedNumer, dCompactGlobalNumer_.get(), globalImageCount,
          commonTransformParams.nx, commonTransformParams.ny,
          commonTransformParams.nz, globalNz, bedZStarts[bed], flipLocalZ);
    }
  }

  // Reads the iteration flags written by an update kernel and turns them
  // into the shared error / all-zero contract.
  void checkCompactResidentUpdateFlagsLocked(bool requirePositiveNumerator,
                                             bool *imageAllZero,
                                             const char *what) {
    uint32_t flags = 0;
    dCompactIterationFlags_.copyToHost(&flags, 1);
    if ((flags & (1U | 4U)) != 0 ||
        (requirePositiveNumerator && (flags & 2U) == 0)) {
      throw std::runtime_error(
          "compact resident numerator is non-finite, zero, or non-positive.");
    }
    if ((flags & 8U) != 0) {
      throw std::runtime_error(std::string("compact resident ") + what +
                               " update produced a non-finite image.");
    }
    if (imageAllZero != nullptr)
      *imageAllZero = (flags & 16U) == 0;
  }

  void finishCompactResidentMlemIteration(
      size_t globalImageCount, int globalNz, const int *bedZStarts,
      int bedCount, bool flipLocalZ,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const double *denom, const uint8_t *mask,
      const std::vector<double> *castorKx,
      const std::vector<double> *castorKy,
      const std::vector<double> *castorKz,
      bool requirePositiveNumerator, bool refreshStaticInputs,
      bool *imageAllZero) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    accumulateCompactResidentDataTermLocked(
        globalImageCount, globalNz, bedZStarts, bedCount, flipLocalZ,
        commonTransformParams, denom, mask, castorKx, castorKy, castorKz,
        refreshStaticInputs);
    dCompactIterationFlags_.allocate(1);
    CUDA_CHECK(cudaMemset(dCompactIterationFlags_.get(), 0,
                          sizeof(uint32_t)));
    launchCompactMlemUpdateKernel(
        dCompactGlobalImage_.get(), dCompactGlobalNumer_.get(),
        dCompactDenom_.get(), mask == nullptr ? nullptr : dCompactMask_.get(),
        mask != nullptr, globalImageCount, dCompactIterationFlags_.get(),
        1e-12);
    checkCompactResidentUpdateFlagsLocked(requirePositiveNumerator,
                                          imageAllZero, "MLEM");
  }

  // Device pointer of the resident global image (valid until the next
  // releaseCompactExecution); the volume ops read it to form the regularizer
  // gradient without a host round trip.
  const double *compactResidentImageDevice(size_t count) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (count == 0 || count != compactResidentImageCount_ ||
        dCompactGlobalImage_.get() == nullptr) {
      throw std::runtime_error(
          "compact resident image is not available on the device.");
    }
    return dCompactGlobalImage_.get();
  }

  // Device scratch for the regularizer gradient of the resident image.
  double *compactResidentGradientDevice(size_t count) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (count == 0) {
      throw std::runtime_error("compact resident gradient scratch is empty.");
    }
    dCompactGradient_.allocate(count);
    return dCompactGradient_.get();
  }

  void finishCompactResidentBsremIteration(
      size_t globalImageCount, int globalNz, const int *bedZStarts,
      int bedCount, bool flipLocalZ,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const double *denom, const uint8_t *mask,
      const std::vector<double> *castorKx,
      const std::vector<double> *castorKy,
      const std::vector<double> *castorKz,
      const cuda::CompactBsremUpdateParams &params,
      bool requirePositiveNumerator, bool refreshStaticInputs,
      bool *imageAllZero) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    accumulateCompactResidentDataTermLocked(
        globalImageCount, globalNz, bedZStarts, bedCount, flipLocalZ,
        commonTransformParams, denom, mask, castorKx, castorKy, castorKz,
        refreshStaticInputs);

    const double *dKappa = nullptr;
    if (params.kappa != nullptr) {
      if (compactKappaHostKey_ != params.kappa ||
          compactKappaCount_ != globalImageCount ||
          dCompactKappa_.get() == nullptr) {
        dCompactKappa_.copyFromHost(params.kappa, globalImageCount);
        compactKappaHostKey_ = params.kappa;
        compactKappaCount_ = globalImageCount;
      }
      dKappa = dCompactKappa_.get();
    }
    const double *dCurvature = nullptr;
    if (params.useCurvature) {
      if (params.deviceGradient == nullptr) {
        throw std::runtime_error(
            "compact resident BSREM curvature needs the regularizer gradient.");
      }
      const bool needsRefresh =
          params.refreshCurvature ||
          dCompactCurvature_.get() == nullptr ||
          compactCurvatureCount_ != globalImageCount;
      dCompactCurvature_.allocate(globalImageCount);
      if (needsRefresh) {
        launchCompactBsremCurvatureKernel(
            dCompactGlobalImage_.get(), params.deviceGradient, params.curvEps,
            dCompactCurvature_.get(), globalImageCount);
        compactCurvatureCount_ = globalImageCount;
      }
      dCurvature = dCompactCurvature_.get();
    }

    dCompactIterationFlags_.allocate(1);
    CUDA_CHECK(cudaMemset(dCompactIterationFlags_.get(), 0,
                          sizeof(uint32_t)));
    launchCompactBsremUpdateKernel(
        dCompactGlobalImage_.get(), dCompactGlobalNumer_.get(),
        dCompactDenom_.get(), params.deviceGradient, dCurvature, dKappa,
        mask == nullptr ? nullptr : dCompactMask_.get(), mask != nullptr,
        globalImageCount, params, dCompactIterationFlags_.get());
    checkCompactResidentUpdateFlagsLocked(requirePositiveNumerator,
                                          imageAllZero, "BSREM");
  }

  void downloadCompactResidentImage(double *image, size_t count) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (image == nullptr || count == 0 || count != compactResidentImageCount_ ||
        dCompactGlobalImage_.get() == nullptr) {
      throw std::runtime_error(
          "compact resident image download parameters are invalid.");
    }
    dCompactGlobalImage_.copyToHost(image, count);
  }

  void releaseCompactExecution() {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    CUDA_CHECK(cudaDeviceSynchronize());
    unregisterCompactHostPinsLocked();

    residentMatrixSetCache_.invalidate();
    cache_.release();
    dInput_.reset();
    dOutput_.reset();
    dProjectionInput_.reset();
    dChunkInput_.reset();
    dAccumOutput_.reset();
    dFullOutput_.reset();
    dBatchInput_.reset();
    dBatchOutput_.reset();
    dFullBatchOutput_.reset();
    dBatchOnes_.reset();
    dBatchSymIds_.reset();
    dBatchZMirrors_.reset();
    dCompactGlobalImage_.reset();
    dCompactGradient_.reset();
    dCompactCurvature_.reset();
    dCompactKappa_.reset();
    dCompactBedPrimary_.reset();
    dCompactBedSecondary_.reset();
    dCompactGlobalNumer_.reset();
    dCompactDenom_.reset();
    dCompactMask_.reset();
    dCompactIterationFlags_.reset();
    for (auto &projection : dCompactProjectionInput_)
      projection.reset();
    for (auto &background : dCompactBackgroundInput_)
      background.reset();
    dCompactRowWeights_.reset();
    for (auto &accumulator : dCompactBackwardAccum_)
      accumulator.reset();
    csrbAttenChunks_.clear();
    dCsrbAttenBatch_.reset();
    dBuffer_.reset();
    releaseCsrqResidentLocked();

    compactResidentImageCount_ = 0;
    compactResidentStaticCount_ = 0;
    compactCurvatureCount_ = 0;
    compactKappaHostKey_ = nullptr;
    compactKappaCount_ = 0;
    compactResidentHasMask_ = false;
    compactBatchInputCount_ = 0;
    compactSortedHint_ = false;
    compactPackedKLogged_ = false;
    compactProjectionHostKey_ = {};
    compactProjectionCount_ = {};
    compactBackgroundHostKey_ = {};
    compactBackgroundCount_ = {};
    compactRowWeightsHostKey_ = nullptr;
    compactRowWeightsCount_ = 0;
    compactBackwardAccumSize_ = {};
    batchSymIdsHost_.clear();
    batchZMirrorsHost_.clear();
    hostFloatScratch_.clear();

    // csrt experiment state: host pins were just unregistered above and the
    // device buffers below would otherwise outlive this reset.
    for (auto &slot : compactTransposeSlots_) {
      slot.mat.reset();
      slot.dRowPtr.reset();
      slot.dColIdx.reset();
      slot.dValues.reset();
      slot.rows = 0;
      slot.cols = 0;
      slot.nnz = 0;
      slot.rowBegin = 0;
      slot.rowEnd = 0;
      slot.uploadPending = false;
      slot.consumedRecorded = false;
    }
    compactTransposeHost_.clear();
    compactTransposeActiveKey_ = MatrixCacheKey{};
    compactTransposeActiveFingerprint_ = 0;
    compactTransposeActiveValid_ = false;
  }

  void refreshCompactProjection(const double *projection, size_t count,
                                int cacheSlot, const double *background) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (cacheSlot < 0 || cacheSlot >= kCompactProjectionSlots) {
      throw std::runtime_error("compact projection cache slot out of range.");
    }
    // A recycled host address is not proof that a new reconstruction owns the
    // same data. Refresh once per numerator pass; inner branches still reuse it.
    compactProjectionHostKey_[static_cast<size_t>(cacheSlot)] = nullptr;
    ensureCompactFullProjectionUploaded(projection, count, cacheSlot);
    compactBackgroundHostKey_[static_cast<size_t>(cacheSlot)] = nullptr;
    if (background != nullptr) {
      ensureCompactBackgroundUploaded(background, count, cacheSlot);
    }
  }

  // ---- CSRQ matrices ---------------------------------------------------------
  // Up to kCsrqMaxMatrices energy windows share one voxel grid (same cols);
  // each matrix set is addressed by matrixIndex and is either resident (every
  // chunk on the device) or streamed (pinned host copies, two device chunk
  // slots refilled per pass). Both are walked chunk-outer by the passes below
  // so a streamed chunk is uploaded once per pass whatever the bed count.

  bool csrqResident(int matrixIndex) {
    std::lock_guard<std::mutex> lock(mutex_);
    return matrixIndex >= 0 &&
           static_cast<size_t>(matrixIndex) < csrqMatrices_.size() &&
           csrqMatrices_[static_cast<size_t>(matrixIndex)].ready;
  }

  static uint64_t csrqChunkDeviceBytes(uint64_t rows, uint64_t nnz,
                                       uint64_t runCount) {
    return 2ull * sizeof(uint32_t) * (rows + 1ull) + sizeof(float) * rows +
           sizeof(uint32_t) * runCount + sizeof(uint16_t) * nnz;
  }

  // Starts loading matrix matrixIndex (dropping whatever that slot held).
  // Decides resident vs streamed from the free device memory: resident when
  // the whole matrix plus scratch plus reserve fits, streamed when at least
  // two chunk slots do, error otherwise. Returns true when streamed.
  bool beginCsrqWarmup(int matrixIndex, int cols,
                       const std::vector<cuda::CsrqChunkShape> &shapes,
                       uint64_t reserveBytes, bool forceStreamed) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (shapes.empty() || cols <= 0) {
      throw std::runtime_error("CSRQ warmup received no chunks.");
    }
    if (matrixIndex < 0 || matrixIndex >= kCsrqMaxMatrices) {
      throw std::runtime_error("CSRQ matrix index out of range.");
    }
    for (const auto &resident : csrqMatrices_) {
      if (resident.ready && resident.cols != cols) {
        throw std::runtime_error(
            "CSRQ matrices must share the voxel grid (cols differ).");
      }
    }
    if (csrqMatrices_.size() <= static_cast<size_t>(matrixIndex)) {
      csrqMatrices_.resize(static_cast<size_t>(matrixIndex) + 1u);
    }
    CsrqResidentMatrix &target = csrqMatrices_[static_cast<size_t>(matrixIndex)];
    target = CsrqResidentMatrix{};

    uint64_t matrixBytes = 0;
    uint64_t totalRows = 0;
    uint64_t maxChunkBytes = 0;
    CsrqChunkDims maxDims;
    for (const auto &shape : shapes) {
      if (shape.rows <= 0 || shape.compactRowBase < 0 ||
          static_cast<uint64_t>(shape.compactRowBase) != totalRows) {
        throw std::runtime_error(
            "CSRQ warmup: chunks are not contiguous in compact row order.");
      }
      const uint64_t bytes = csrqChunkDeviceBytes(
          static_cast<uint64_t>(shape.rows), shape.nnz, shape.runCount);
      matrixBytes += bytes;
      maxChunkBytes = std::max(maxChunkBytes, bytes);
      totalRows += static_cast<uint64_t>(shape.rows);
      maxDims.rows = std::max(maxDims.rows, static_cast<size_t>(shape.rows));
      maxDims.nnz = std::max(maxDims.nnz, static_cast<size_t>(shape.nnz));
      maxDims.runCount =
          std::max(maxDims.runCount, static_cast<size_t>(shape.runCount));
    }
    if (totalRows > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("CSRQ warmup: too many compact rows.");
    }
    const size_t nvox = static_cast<size_t>(cols);
    const size_t ratioRows =
        std::max(csrqMaxRowsLocked(), static_cast<size_t>(totalRows));
    const uint64_t scratchBytes =
        (dCsrqX_.get() == nullptr ? 2ull * nvox * kCsrqK * sizeof(float) : 0ull) +
        (ratioRows > dCsrqRatio_.size() / kCsrqK
             ? (ratioRows - dCsrqRatio_.size() / kCsrqK) * kCsrqK * sizeof(float)
             : 0ull);
    uint64_t freeBytes = 0;
    try {
      freeBytes = static_cast<uint64_t>(cuda::getMemoryInfo().free);
    } catch (const std::exception &ex) {
      throw std::runtime_error(
          std::string("CSRQ warmup: CUDA memory query failed: ") + ex.what());
    }
    constexpr uint64_t kMiB = 1024ull * 1024ull;
    const uint64_t needResident = matrixBytes + scratchBytes + reserveBytes;
    const uint64_t needStreamed =
        static_cast<uint64_t>(kCsrqStreamSlots) * maxChunkBytes + scratchBytes +
        reserveBytes;
    if (!forceStreamed && freeBytes >= needResident) {
      target.streamed = false;
    } else if (freeBytes >= needStreamed) {
      target.streamed = true;
      std::cerr << "[NOTE] CSRQ matrix " << matrixIndex << " ("
                << (matrixBytes / kMiB) << " MiB) "
                << (forceStreamed ? "streaming forced"
                                  : "exceeds free device memory")
                << " (free " << (freeBytes / kMiB) << " MiB, reserve "
                << (reserveBytes / kMiB) << " MiB); streaming it through "
                << kCsrqStreamSlots << " chunk slots of "
                << (maxChunkBytes / kMiB) << " MiB from pinned host memory.\n";
    } else {
      target = CsrqResidentMatrix{};
      throw cuda::CsrqDeviceMemoryError(
          "CSRQ matrix does not fit on the device even streamed: two chunk "
          "slots " + std::to_string(2 * maxChunkBytes / kMiB) +
          " MiB + scratch " + std::to_string(scratchBytes / kMiB) +
          " MiB + reserve " + std::to_string(reserveBytes / kMiB) +
          " MiB > free " + std::to_string(freeBytes / kMiB) +
          " MiB. Use the CSRB directory or a GPU with more memory.");
    }
    target.cols = cols;
    target.totalRows = static_cast<size_t>(totalRows);
    target.bytes = matrixBytes;
    target.shapes = shapes;
    if (target.streamed) {
      target.host.resize(shapes.size());
      for (int slot = 0; slot < kCsrqStreamSlots; ++slot) {
        auto dev = std::make_unique<CsrqDeviceChunk>();
        dev->dRowNnzPtr.allocate(maxDims.rows + 1u);
        dev->dRowRunPtr.allocate(maxDims.rows + 1u);
        dev->dRowMax.allocate(maxDims.rows);
        dev->dRuns.allocate(maxDims.runCount);
        dev->dValuesQ.allocate(maxDims.nnz);
        target.chunks.push_back(std::move(dev));
      }
    } else {
      target.chunks.resize(shapes.size());
    }
    dCsrqX_.allocate(nvox * kCsrqK);
    dCsrqY_.allocate(nvox * kCsrqK);
    dCsrqRatio_.allocate(ratioRows * kCsrqK);
    target.loaded.assign(shapes.size(), false);
    csrqWarmupStart_ = std::chrono::steady_clock::now();
    return target.streamed;
  }

  // Uploads (resident) or pins (streamed) one chunk; the host view may be
  // released by the caller as soon as this returns.
  void warmupCsrqChunk(int matrixIndex, size_t chunkIndex,
                       const io::CompactRunU16View &view) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    CsrqResidentMatrix &target = csrqRequireWarmupTargetLocked(matrixIndex);
    if (chunkIndex >= target.shapes.size()) {
      throw std::runtime_error("CSRQ warmup chunk index out of range.");
    }
    const cuda::CsrqChunkShape &shape = target.shapes[chunkIndex];
    if (!view.valid() || view.cols != target.cols || view.rows != shape.rows ||
        view.nnz != shape.nnz || view.runCount != shape.runCount) {
      throw std::runtime_error(
          "CSRQ warmup chunk does not match the announced shape.");
    }
    const size_t ptrCount = static_cast<size_t>(view.rows) + 1u;
    if (target.streamed) {
      CsrqHostChunk &host = target.host[chunkIndex];
      host.rowNnzPtr.copyFromHost(view.rowNnzPtr, ptrCount);
      host.rowRunPtr.copyFromHost(view.rowRunPtr, ptrCount);
      host.rowMax.copyFromHost(view.rowMax, static_cast<size_t>(view.rows));
      host.runs.copyFromHost(view.runs, static_cast<size_t>(view.runCount));
      host.valuesQ.copyFromHost(view.valuesQ, static_cast<size_t>(view.nnz));
    } else {
      auto dev = std::make_unique<CsrqDeviceChunk>();
      dev->rows = view.rows;
      dev->compactRowBase = shape.compactRowBase;
      dev->nnz = view.nnz;
      dev->runCount = view.runCount;
      dev->dRowNnzPtr.copyFromHost(view.rowNnzPtr, ptrCount);
      dev->dRowRunPtr.copyFromHost(view.rowRunPtr, ptrCount);
      dev->dRowMax.copyFromHost(view.rowMax, static_cast<size_t>(view.rows));
      dev->dRuns.copyFromHost(view.runs, static_cast<size_t>(view.runCount));
      dev->dValuesQ.copyFromHost(view.valuesQ, static_cast<size_t>(view.nnz));
      target.chunks[chunkIndex] = std::move(dev);
    }
    target.loaded[chunkIndex] = true;
  }

  void endCsrqWarmup(int matrixIndex) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    CsrqResidentMatrix &target = csrqRequireWarmupTargetLocked(matrixIndex);
    for (size_t i = 0; i < target.loaded.size(); ++i) {
      if (!target.loaded[i]) {
        throw std::runtime_error("CSRQ warmup ended with chunk " +
                                 std::to_string(i) + " missing.");
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    target.ready = true;
    noteCudaCompactMatrixUpload(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      csrqWarmupStart_)
            .count());
    constexpr uint64_t kMiB = 1024ull * 1024ull;
    std::cerr << "[NOTE] CSRQ matrix " << matrixIndex << ": "
              << target.shapes.size() << " chunks, " << (target.bytes / kMiB)
              << " MiB " << (target.streamed ? "streamed (pinned host)" : "resident")
              << ", " << target.totalRows << " compact rows.\n";
  }

  // Forward + ratio + backward of matrix matrixIndex for beds
  // [0, bedCount) of the refreshed branch batch ([bed][branch][voxel]),
  // chunk-outer so a streamed chunk crosses PCIe once per pass. Bed b
  // accumulates weight * H_b^T(p_b / (weight * H_b x_b + background_b)) into
  // accumulator accumulatorBase + b; projection cache slot cacheSlotBase + b.
  // bedAttFactors (null, or one table per bed of attCount uint16) makes
  // H_b = A att_b^attPower: the chunk is then walked one compact angle at a
  // time with the factor applied at pack time and after the backward scatter
  // (attPower rescales the table to this window's photon energy).
  void csrqRunNumer(int matrixIndex, int bedCount, int branchCount,
                    const cuda::CompactDeviceRatioParams &commonRatioParams,
                    const cuda::CompactDeviceTransformParams &commonTransformParams,
                    const int *symIds, const int *zMirrors,
                    const double *const *fullProjections,
                    const double *const *fullBackgrounds,
                    size_t fullProjectionCount, double weight,
                    int accumulatorBase, int cacheSlotBase,
                    cuda::CompactDeviceRatioTiming *ratioTimingOut,
                    cuda::CompactDeviceBackwardTiming *backwardTimingOut,
                    const uint16_t *const *bedAttFactors, size_t attCount,
                    double attPower) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    CsrqResidentMatrix &matrix = csrqRequireReadyLocked(matrixIndex, branchCount);
    const size_t nvox = static_cast<size_t>(matrix.cols);
    const float power = csrqCheckedAttPower(attPower);
    if (bedCount <= 0 || bedCount > spect::core::kMaxBeds ||
        accumulatorBase < 0 || cacheSlotBase < 0 ||
        accumulatorBase + bedCount > kCompactDeviceBackwardMaxAccumulators ||
        cacheSlotBase + bedCount > kCompactProjectionSlots) {
      throw std::runtime_error("CSRQ numerator bed/slot range out of range.");
    }
    bool hasAtten = false;
    if (bedAttFactors != nullptr) {
      for (int bed = 0; bed < bedCount; ++bed) {
        hasAtten = hasAtten || bedAttFactors[bed] != nullptr;
      }
    }
    if (commonRatioParams.symmetryMode != 45 &&
        commonRatioParams.symmetryMode != 90) {
      throw std::runtime_error("CSRQ numerator requires symmetryMode 45 or 90.");
    }
    if (!(weight > 0.0)) {
      throw std::runtime_error("CSRQ window weight must be positive.");
    }
    if (fullProjections == nullptr) {
      throw std::runtime_error("CSRQ numerator received null projections.");
    }
    const size_t batchNeeded =
        static_cast<size_t>(bedCount) * static_cast<size_t>(kCsrqK) * nvox;
    if (dBatchInput_.get() == nullptr || compactBatchInputCount_ < batchNeeded) {
      throw std::runtime_error(
          "CSRQ branch-batch input was not refreshed for these beds.");
    }
    std::array<DeviceArray<double> *, spect::core::kMaxBeds> accums{};
    std::array<const double *, spect::core::kMaxBeds> dProjections{};
    std::array<const double *, spect::core::kMaxBeds> dBackgrounds{};
    for (int bed = 0; bed < bedCount; ++bed) {
      accums[bed] =
          &csrqRequireAccumulatorLocked(accumulatorBase + bed, matrix.cols);
      const int slot = cacheSlotBase + bed;
      ensureCompactFullProjectionUploaded(fullProjections[bed],
                                          fullProjectionCount, slot);
      dProjections[bed] = dCompactProjectionInput_[static_cast<size_t>(slot)].get();
      const double *background =
          fullBackgrounds == nullptr ? nullptr : fullBackgrounds[bed];
      if (background != nullptr) {
        ensureCompactBackgroundUploaded(background, fullProjectionCount, slot);
        dBackgrounds[bed] =
            dCompactBackgroundInput_[static_cast<size_t>(slot)].get();
      }
    }
    csrqEnsureBranchMetadataLocked(symIds, zMirrors);
    ensureProjDomainKernelsLoaded();

    // Timer slots: one per (chunk, bed) pass, or per (chunk, bed, compact
    // angle) launch group on the attenuated path.
    const CsrqAttenPlan *plan =
        hasAtten ? &csrqEnsureAttenPlanLocked(matrix, commonRatioParams) : nullptr;
    const size_t maxAngles =
        hasAtten ? csrqMaxAnglesPerChunkLocked(matrix, commonRatioParams) : 1u;
    const size_t passes = matrix.shapes.size() * static_cast<size_t>(bedCount);
    const size_t timerSlots = passes * maxAngles;
    csrqEnsureTimersLocked(timerSlots);
    size_t pass = 0;
    size_t usedTimers = 0;
    csrqForEachChunkLocked(matrix, [&](CsrqDeviceChunk &chunk, size_t chunkIndex) {
      cuda::CompactDeviceRatioParams ratioParams = commonRatioParams;
      ratioParams.compactRowBase = chunk.compactRowBase;
      float *ratio = dCsrqRatio_.get() +
                     static_cast<size_t>(chunk.compactRowBase) * kCsrqK;
      for (int bed = 0; bed < bedCount; ++bed, ++pass) {
        const float *bedInput =
            dBatchInput_.get() +
            static_cast<size_t>(bed) * static_cast<size_t>(kCsrqK) * nvox;
        const uint16_t *bedAtt =
            hasAtten ? bedAttFactors[bed] : nullptr;
        if (bedAtt == nullptr) {
          launchCsrqPackBranch8Kernel(bedInput, dCsrqX_.get(), nvox);

          // Forward and Poisson ratio in one kernel: the row's eight branch
          // sums never leave the block.
          const size_t t = usedTimers++;
          csrqForwardTimers_[t].recordStart(nullptr);
          launchCsrqForwardRatioK8Kernel(
              chunk.ptrs(), dCsrqX_.get(), ratio, dProjections[bed],
              dBackgrounds[bed], fullProjectionCount, weight, ratioParams,
              dBatchSymIds_.get(), dBatchZMirrors_.get(), 1e-10);
          csrqForwardTimers_[t].recordStop(nullptr);

          csrqBackwardTimers_[t].recordStart(nullptr);
          csrqBackwardChunkLocked(chunk, ratio, nvox, commonTransformParams,
                                  accums[bed]->get());
          csrqBackwardTimers_[t].recordStop(nullptr);
          continue;
        }
        // CTAC: stage this bed's factor slices for the chunk (overlapping the
        // previous pair's kernels), then one compact angle at a time so each
        // row group sees its own factors.
        const int slot = csrqStageAttenSlicesLocked(*plan, chunkIndex, bedAtt,
                                                    attCount, nvox);
        const uint16_t *dSlices = dCsrqAttSlices_[slot].get();
        const int *dSliceIndex =
            plan->dSliceIndex.get() + plan->chunkSliceIndexOffset[chunkIndex];
        const int angles = plan->chunkAngles[chunkIndex];
        const int rpa = commonRatioParams.compactRaysPerAngle;
        for (int i = 0; i < angles; ++i) {
          const int rowBase = i * rpa;
          const CsrqDeviceChunkPtrs rows = csrqRowRange(chunk, rowBase, rpa);
          const int *dSlice = dSliceIndex + i * kCsrqK;
          cuda::CompactDeviceRatioParams angleParams = ratioParams;
          angleParams.compactRowBase = chunk.compactRowBase + rowBase;
          float *angleRatio = ratio + static_cast<size_t>(rowBase) * kCsrqK;
          const size_t t = usedTimers++;
          csrqForwardTimers_[t].recordStart(nullptr);
          launchCsrqPackBranch8AttenKernel(
              bedInput, dSlices, dSlice, dCsrqX_.get(), nvox,
              commonTransformParams, dBatchSymIds_.get(),
              dBatchZMirrors_.get(), power);
          launchCsrqForwardRatioK8Kernel(
              rows, dCsrqX_.get(), angleRatio, dProjections[bed],
              dBackgrounds[bed], fullProjectionCount, weight, angleParams,
              dBatchSymIds_.get(), dBatchZMirrors_.get(), 1e-10);
          csrqForwardTimers_[t].recordStop(nullptr);

          csrqBackwardTimers_[t].recordStart(nullptr);
          csrqAttenBackwardAngleLocked(rows, angleRatio, nvox,
                                       commonTransformParams, dSlices, dSlice,
                                       power, accums[bed]->get());
          csrqBackwardTimers_[t].recordStop(nullptr);
        }
        csrqReleaseAttenSlotLocked(slot);
      }
    });
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const double forwardSeconds =
        accumulateTimedCudaPairs(csrqForwardTimers_, usedTimers);
    const double backwardSeconds =
        accumulateTimedCudaPairs(csrqBackwardTimers_, usedTimers);
    noteCudaCsrqForward(forwardSeconds, passes);
    noteCudaCsrqBackward(backwardSeconds, passes * csrqBackwardPassCount());
    if (ratioTimingOut != nullptr) {
      // The ratio is fused into the forward kernel; report it there.
      ratioTimingOut->forwardSeconds += forwardSeconds;
      ratioTimingOut->backwardSeconds += backwardSeconds;
    }
    if (backwardTimingOut != nullptr) {
      // The backward here includes the symmetry transform-add into the
      // accumulator.
      backwardTimingOut->backwardSeconds += backwardSeconds;
    }
  }

  // weight * H^T W 1 of matrix matrixIndex restricted to (row, branch) pairs
  // with a full projection counterpart, added to accumulatorIndex. rowWeights
  // (one per full projection row) may be null for W = 1; attFactors (one
  // bed's full-angle table, attCount uint16) may be null for H = A, otherwise
  // H = A att^attPower.
  void csrqRunSensitivity(
      int matrixIndex, int branchCount, double weight,
      const cuda::CompactDeviceRatioParams &commonRatioParams,
      const cuda::CompactDeviceTransformParams &commonTransformParams,
      const int *symIds, const int *zMirrors, int accumulatorIndex,
      const double *rowWeights, size_t rowWeightCount,
      size_t fullProjectionCount,
      cuda::CompactDeviceBackwardTiming *backwardTimingOut,
      const uint16_t *attFactors, size_t attCount, double attPower) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    CsrqResidentMatrix &matrix = csrqRequireReadyLocked(matrixIndex, branchCount);
    const size_t nvox = static_cast<size_t>(matrix.cols);
    const float power = csrqCheckedAttPower(attPower);
    if (rowWeights != nullptr && rowWeightCount != fullProjectionCount) {
      throw std::runtime_error(
          "CSRQ sensitivity row weights must hold one value per full row.");
    }
    const double *dRowWeights =
        ensureCompactRowWeightsUploaded(rowWeights, rowWeightCount);
    if (commonRatioParams.symmetryMode != 45 &&
        commonRatioParams.symmetryMode != 90) {
      throw std::runtime_error(
          "CSRQ sensitivity requires symmetryMode 45 or 90.");
    }
    if (!(weight > 0.0)) {
      throw std::runtime_error("CSRQ window weight must be positive.");
    }
    DeviceArray<double> &accum =
        csrqRequireAccumulatorLocked(accumulatorIndex, matrix.cols);
    csrqEnsureBranchMetadataLocked(symIds, zMirrors);
    ensureProjDomainKernelsLoaded();

    const bool hasAtten = attFactors != nullptr;
    const CsrqAttenPlan *plan =
        hasAtten ? &csrqEnsureAttenPlanLocked(matrix, commonRatioParams) : nullptr;
    const size_t maxAngles =
        hasAtten ? csrqMaxAnglesPerChunkLocked(matrix, commonRatioParams) : 1u;
    const size_t passes = matrix.shapes.size();
    csrqEnsureTimersLocked(passes * maxAngles);
    size_t usedTimers = 0;
    csrqForEachChunkLocked(matrix, [&](CsrqDeviceChunk &chunk, size_t chunkIndex) {
      cuda::CompactDeviceRatioParams ratioParams = commonRatioParams;
      ratioParams.compactRowBase = chunk.compactRowBase;
      float *ratio = dCsrqRatio_.get() +
                     static_cast<size_t>(chunk.compactRowBase) * kCsrqK;
      // The unit ratio covers the whole chunk in either mode.
      launchCsrqUnitRatioK8Kernel(ratio, static_cast<float>(weight),
                                  dRowWeights, fullProjectionCount, ratioParams,
                                  dBatchSymIds_.get(), dBatchZMirrors_.get(),
                                  chunk.rows);
      if (!hasAtten) {
        const size_t t = usedTimers++;
        csrqBackwardTimers_[t].recordStart(nullptr);
        csrqBackwardChunkLocked(chunk, ratio, nvox, commonTransformParams,
                                accum.get());
        csrqBackwardTimers_[t].recordStop(nullptr);
        return;
      }
      const int slot = csrqStageAttenSlicesLocked(*plan, chunkIndex, attFactors,
                                                  attCount, nvox);
      const uint16_t *dSlices = dCsrqAttSlices_[slot].get();
      const int *dSliceIndex =
          plan->dSliceIndex.get() + plan->chunkSliceIndexOffset[chunkIndex];
      const int angles = plan->chunkAngles[chunkIndex];
      const int rpa = commonRatioParams.compactRaysPerAngle;
      for (int i = 0; i < angles; ++i) {
        const int rowBase = i * rpa;
        const size_t t = usedTimers++;
        csrqBackwardTimers_[t].recordStart(nullptr);
        csrqAttenBackwardAngleLocked(
            csrqRowRange(chunk, rowBase, rpa),
            ratio + static_cast<size_t>(rowBase) * kCsrqK, nvox,
            commonTransformParams, dSlices, dSliceIndex + i * kCsrqK, power,
            accum.get());
        csrqBackwardTimers_[t].recordStop(nullptr);
      }
      csrqReleaseAttenSlotLocked(slot);
    });
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const double backwardSeconds =
        accumulateTimedCudaPairs(csrqBackwardTimers_, usedTimers);
    noteCudaCsrqBackward(backwardSeconds, passes * csrqBackwardPassCount());
    if (backwardTimingOut != nullptr) {
      backwardTimingOut->backwardSeconds += backwardSeconds;
    }
  }

  // Forward projection of matrix matrixIndex over the single-bed branch
  // batch (8 * nvox floats refreshed by refreshCompactBranchBatchInput);
  // hostYhatPacked receives [compactRow][branch] for every compact row.
  // With attFactors (one full-angle table) the projection is H x = A att x,
  // per compact angle like the numerator; the params / branch tables are
  // required then.
  void csrqRunForward(int matrixIndex, int branchCount, float *hostYhatPacked,
                      size_t count,
                      const cuda::CompactDeviceRatioParams *attenRatioParams,
                      const cuda::CompactDeviceTransformParams *attenTransformParams,
                      const int *symIds, const int *zMirrors,
                      const uint16_t *attFactors, size_t attCount,
                      double attPower) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    CsrqResidentMatrix &matrix = csrqRequireReadyLocked(matrixIndex, branchCount);
    const size_t nvox = static_cast<size_t>(matrix.cols);
    const float power = csrqCheckedAttPower(attPower);
    if (hostYhatPacked == nullptr ||
        count != matrix.totalRows * static_cast<size_t>(kCsrqK)) {
      throw std::runtime_error(
          "CSRQ forward output must hold compactRows * 8 floats.");
    }
    if (dBatchInput_.get() == nullptr ||
        compactBatchInputCount_ != static_cast<size_t>(kCsrqK) * nvox) {
      throw std::runtime_error(
          "CSRQ forward requires a refreshed single-bed branch batch input.");
    }
    const bool hasAtten = attFactors != nullptr;
    if (hasAtten && (attenRatioParams == nullptr ||
                     attenTransformParams == nullptr || symIds == nullptr ||
                     zMirrors == nullptr)) {
      throw std::runtime_error(
          "CSRQ attenuated forward needs the angle/branch parameters.");
    }
    ensureProjDomainKernelsLoaded();
    if (hasAtten) {
      csrqEnsureBranchMetadataLocked(symIds, zMirrors);
    } else {
      launchCsrqPackBranch8Kernel(dBatchInput_.get(), dCsrqX_.get(), nvox);
    }

    const CsrqAttenPlan *plan =
        hasAtten ? &csrqEnsureAttenPlanLocked(matrix, *attenRatioParams) : nullptr;
    const size_t maxAngles =
        hasAtten ? csrqMaxAnglesPerChunkLocked(matrix, *attenRatioParams) : 1u;
    const size_t passes = matrix.shapes.size();
    csrqEnsureTimersLocked(passes * maxAngles);
    size_t usedTimers = 0;
    csrqForEachChunkLocked(matrix, [&](CsrqDeviceChunk &chunk, size_t chunkIndex) {
      float *out =
          dCsrqRatio_.get() + static_cast<size_t>(chunk.compactRowBase) * kCsrqK;
      if (!hasAtten) {
        const size_t t = usedTimers++;
        csrqForwardTimers_[t].recordStart(nullptr);
        launchCsrqForwardK8Kernel(chunk.ptrs(), dCsrqX_.get(), out);
        csrqForwardTimers_[t].recordStop(nullptr);
        return;
      }
      const int slot = csrqStageAttenSlicesLocked(*plan, chunkIndex, attFactors,
                                                  attCount, nvox);
      const uint16_t *dSlices = dCsrqAttSlices_[slot].get();
      const int *dSliceIndex =
          plan->dSliceIndex.get() + plan->chunkSliceIndexOffset[chunkIndex];
      const int angles = plan->chunkAngles[chunkIndex];
      const int rpa = attenRatioParams->compactRaysPerAngle;
      for (int i = 0; i < angles; ++i) {
        const int rowBase = i * rpa;
        const size_t t = usedTimers++;
        csrqForwardTimers_[t].recordStart(nullptr);
        launchCsrqPackBranch8AttenKernel(
            dBatchInput_.get(), dSlices, dSliceIndex + i * kCsrqK,
            dCsrqX_.get(), nvox, *attenTransformParams, dBatchSymIds_.get(),
            dBatchZMirrors_.get(), power);
        launchCsrqForwardK8Kernel(csrqRowRange(chunk, rowBase, rpa),
                                  dCsrqX_.get(),
                                  out + static_cast<size_t>(rowBase) * kCsrqK);
        csrqForwardTimers_[t].recordStop(nullptr);
      }
      csrqReleaseAttenSlotLocked(slot);
    });
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    noteCudaCsrqForward(
        accumulateTimedCudaPairs(csrqForwardTimers_, usedTimers), passes);

    const auto outputStart = std::chrono::steady_clock::now();
    dCsrqRatio_.copyToHost(hostYhatPacked, count);
    noteCudaCompactOutputD2H(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      outputStart)
            .count(),
        count * sizeof(float), false);
  }

  void prefetchCompactFloatMatrix(const io::CSRView &next,
                                  const io::CSRView &current) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    validateIoCsrViewForUpload(next);
    validateIoCsrViewForUpload(current);
    const MatrixCacheKey nextKey = makeCacheKey(next);
    const MatrixCacheKey currentKey = makeCacheKey(current);
    if (findCompactFloatSlot(nextKey) != nullptr) {
      return;
    }
    CachedMatrix *target = selectCompactPrefetchTarget(currentKey);
    if (target == nullptr) {
      return;
    }
    const auto uploadStart = std::chrono::steady_clock::now();
    uploadIoCsrViewAsync(next, nextKey, target,
                         transientUploadStreams_[0].get());
    bindCompactSlotKey(target, nextKey);
    target->compactUploadReady.record(transientUploadStreams_[0].get());
    target->compactUploadPending = true;
    noteCudaCompactMatrixUpload(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      uploadStart)
            .count());
  }

  void noteCompactCsrLocality(bool sortedHint) {
    std::lock_guard<std::mutex> lock(mutex_);
    compactSortedHint_ = sortedHint;
  }

  bool warmupResidentMatrixSet(const std::vector<io::CSRView> &matrices,
                               uint64_t reserveBytes) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<MatrixCacheKey> keys;
    keys.reserve(matrices.size());
    for (const auto &matrix : matrices) {
      validateIoCsrViewForUpload(matrix);
      keys.push_back(makeCacheKey(matrix));
    }
    if (residentMatrixSetCache_.matches(keys, reserveBytes)) {
      return residentMatrixSetCache_.numCached > 0;
    }

    residentMatrixSetCache_.invalidate();
    if (matrices.empty()) {
      return false;
    }

    uint64_t freeBytes = 0;
    try {
      freeBytes = static_cast<uint64_t>(cuda::getMemoryInfo().free);
    } catch (...) {
      return false;
    }
    if (freeBytes <= reserveBytes) {
      return false;
    }
    const uint64_t availableForCache = freeBytes - reserveBytes;

    std::vector<uint64_t> matrixBytesPerEntry;
    matrixBytesPerEntry.reserve(matrices.size());
    uint64_t fitBytes = 0;
    size_t fitCount = 0;
    for (const auto &matrix : matrices) {
      const uint64_t bytes =
          estimateDeviceCsrBytes(matrix.rows, static_cast<int64_t>(matrix.nnz));
      matrixBytesPerEntry.push_back(bytes);
      if (bytes == 0 || fitBytes + bytes > availableForCache) {
        break;
      }
      fitBytes += bytes;
      ++fitCount;
    }
    const size_t requiredFitCount = std::min<size_t>(2, matrices.size());
    if (fitCount < requiredFitCount) {
      return false;
    }

    residentMatrixSetCache_.slots.resize(matrices.size());
    residentMatrixSetCache_.keys = keys;
    residentMatrixSetCache_.valid.resize(matrices.size(), false);
    residentMatrixSetCache_.reserveBytes = reserveBytes;

    uint64_t usedBytes = 0;
    for (size_t index = 0; index < matrices.size(); ++index) {
      const io::CSRView &matrix = matrices[index];
      const uint64_t matrixBytes = matrixBytesPerEntry[index];
      if (matrixBytes == 0 || usedBytes + matrixBytes > availableForCache) {
        break;
      }
      auto cached = std::make_unique<CachedMatrix>();
      uploadIoCsrViewAsync(matrix, keys[index], cached.get(),
                           computeStream_.get());
      residentMatrixSetCache_.slots[index] = std::move(cached);
      residentMatrixSetCache_.valid[index] = true;
      usedBytes += matrixBytes;
      ++residentMatrixSetCache_.numCached;
    }

    if (residentMatrixSetCache_.numCached == 0) {
      residentMatrixSetCache_.invalidate();
      return false;
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    residentMatrixSetCache_.initialized = true;
    return true;
  }

public:
private:
  size_t compactBackwardAccumulatorSlot(int accumulatorIndex) const {
    if (accumulatorIndex < 0 ||
        accumulatorIndex >= kCompactDeviceBackwardMaxAccumulators) {
      throw std::runtime_error(
          "compact device-backward accumulator index is out of range.");
    }
    return static_cast<size_t>(accumulatorIndex);
  }

  void releaseCsrqResidentLocked() {
    // Pending slice uploads must finish before their pinned sources are
    // unregistered and the slots freed.
    if (std::any_of(csrqAttSlotUsed_.begin(), csrqAttSlotUsed_.end(),
                    [](bool used) { return used; })) {
      CUDA_CHECK(cudaStreamSynchronize(csrqUploadStream_.get()));
      CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    csrqMatrices_.clear();
    dCsrqX_.reset();
    dCsrqY_.reset();
    dCsrqRatio_.reset();
    for (auto &slices : dCsrqAttSlices_) {
      slices.reset();
    }
    for (auto &used : csrqAttSlotUsed_) {
      used = false;
    }
    csrqAttNextSlot_ = 0;
    csrqReleasePinnedTablesLocked();
    csrqForwardTimers_.clear();
    csrqBackwardTimers_.clear();
    for (auto &used : csrqSlotUsed_) {
      used = false;
    }
  }

  // ---- CSRQ CTAC (attenuation factorised out of the nnz loop) -------------
  // h_ij = a_ij * att[fullAngle(i, branch), M_branch(j)]: the factor depends on
  // the row's full angle and the unrotated voxel only, so the passes below
  // walk a chunk one compact angle at a time and apply it in the image domain
  // (csrq_pack_branch8_atten before the forward, the attenuated transform-add
  // after the backward). The SpMV kernels themselves are untouched.

  // Host mirror of compact_atten_full_angle for the first row of a compact
  // angle; -1 when the branch has no full angle there (SYM_45Z edges).
  static int csrqAttenFullAngle(const cuda::CompactDeviceRatioParams &p,
                                int compactAngle, int symId) {
    int fullAngle = 0;
    if (p.symmetryMode == 45) {
      const int q = p.fullNumAngles / 8;
      if ((symId & 1) == 0) {
        if (compactAngle >= q) return -1;
        fullAngle = symId * q + compactAngle;
      } else {
        if (compactAngle <= 0) return -1;
        fullAngle = symId * q + (q - compactAngle);
      }
    } else {
      fullAngle = compactAngle + symId * p.compactNumAngles;
    }
    if (fullAngle < 0 || fullAngle >= p.fullNumAngles) return -1;
    return fullAngle;
  }

  static size_t csrqMaxAnglesPerChunkLocked(
      const CsrqResidentMatrix &matrix,
      const cuda::CompactDeviceRatioParams &p) {
    if (p.compactRaysPerAngle <= 0) {
      throw std::runtime_error("CSRQ CTAC requires compactRaysPerAngle > 0.");
    }
    size_t maxAngles = 1;
    for (const auto &shape : matrix.shapes) {
      maxAngles = std::max(
          maxAngles, static_cast<size_t>(shape.rows / p.compactRaysPerAngle));
    }
    return maxAngles;
  }

  // (compact angle, branch) -> slice table of one chunk of whole compact
  // angles [rowBase, rowBase + rows) for the current branch ids, appended to
  // sliceIndex ([angle][branchCount] ints, -1 = no full angle); the distinct
  // full angles the chunk reaches go to fullAngles in slice order. Returns
  // the angle count. Shared by the CSRQ and CSRB attenuated passes.
  int buildAttenChunkSliceTableLocked(const cuda::CompactDeviceRatioParams &p,
                                      int rowBase, int rows,
                                      std::vector<int> &sliceIndex,
                                      std::vector<int> &fullAngles) const {
    if (p.compactRaysPerAngle <= 0 || rows % p.compactRaysPerAngle != 0 ||
        rowBase % p.compactRaysPerAngle != 0) {
      throw std::runtime_error(
          "compact CTAC requires chunks made of whole compact angles.");
    }
    const int angles = rows / p.compactRaysPerAngle;
    const int firstAngle = rowBase / p.compactRaysPerAngle;
    const int branchCount = static_cast<int>(batchSymIdsHost_.size());
    fullAngles.clear();
    for (int i = 0; i < angles; ++i) {
      for (int k = 0; k < branchCount; ++k) {
        const int fullAngle =
            csrqAttenFullAngle(p, firstAngle + i, batchSymIdsHost_[k]);
        int slice = -1;
        if (fullAngle >= 0) {
          auto it = std::find(fullAngles.begin(), fullAngles.end(), fullAngle);
          if (it == fullAngles.end()) {
            fullAngles.push_back(fullAngle);
            it = fullAngles.end() - 1;
          }
          slice = static_cast<int>(it - fullAngles.begin());
        }
        sliceIndex.push_back(slice);
      }
    }
    return angles;
  }

  // Per-matrix CTAC plan: for every chunk the (compact angle, branch) -> slice
  // table and the distinct full angles it reaches (slice order). Built from
  // the announced chunk shapes and the current branch ids, uploaded once, so
  // no synchronous copy sits between the launches of a pass. Rebuilt when the
  // geometry parameters or the branch ids change.
  const CsrqAttenPlan &csrqEnsureAttenPlanLocked(
      CsrqResidentMatrix &matrix, const cuda::CompactDeviceRatioParams &p) {
    CsrqAttenPlan &plan = matrix.attenPlan;
    if (plan.valid && plan.symmetryMode == p.symmetryMode &&
        plan.compactRaysPerAngle == p.compactRaysPerAngle &&
        plan.compactNumAngles == p.compactNumAngles &&
        plan.fullNumAngles == p.fullNumAngles &&
        plan.symIds == batchSymIdsHost_) {
      return plan;
    }
    plan = CsrqAttenPlan{};
    plan.symmetryMode = p.symmetryMode;
    plan.compactRaysPerAngle = p.compactRaysPerAngle;
    plan.compactNumAngles = p.compactNumAngles;
    plan.fullNumAngles = p.fullNumAngles;
    plan.symIds = batchSymIdsHost_;
    std::vector<int> sliceIndex;
    for (const auto &shape : matrix.shapes) {
      std::vector<int> fullAngles;
      plan.chunkSliceIndexOffset.push_back(sliceIndex.size());
      plan.chunkAngles.push_back(buildAttenChunkSliceTableLocked(
          p, shape.compactRowBase, shape.rows, sliceIndex, fullAngles));
      plan.maxSlices = std::max(plan.maxSlices, fullAngles.size());
      plan.chunkFullAngles.push_back(std::move(fullAngles));
    }
    plan.dSliceIndex.copyFromHost(sliceIndex.data(), sliceIndex.size());
    plan.valid = true;
    return plan;
  }

  // CSRB counterpart for one uploaded chunk (row range [compactRowBase,
  // compactRowBase + rows)): slice table plus, per compact angle, the
  // zero-based sub-CSR row pointers and nnz range so cuSPARSE can address the
  // angle's rows as a matrix of their own. Cached per chunk row base and
  // revalidated against the chunk identity, geometry and branch ids.
  struct CsrbAttenChunk {
    MatrixCacheKey matrixKey;
    int rows = 0;
    int symmetryMode = 0;
    int compactRaysPerAngle = 0;
    int compactNumAngles = 0;
    int fullNumAngles = 0;
    std::vector<int> symIds;
    int angles = 0;
    std::vector<int> fullAngles;
    std::vector<int64_t> nnzOffset;   // per angle
    std::vector<int64_t> nnzCount;    // per angle
    DeviceArray<int> dSliceIndex;     // [angles][branchCount]
    DeviceArray<int32_t> dSubRowPtr;  // [angles][compactRaysPerAngle + 1]
  };

  // One cuSPARSE CSR descriptor per compact angle of a chunk (its rows as a
  // matrix of their own), with per-descriptor SpMV buffer-size caches for the
  // two operations. Lives for one pass; descriptors are host objects only.
  struct CsrbAngleMatrices {
    std::vector<std::unique_ptr<CuSparseMatGuard>> mats;
    std::vector<std::array<size_t, 2>> bufferSizes;
    std::vector<std::array<bool, 2>> bufferSizeValid;
  };

  CsrbAngleMatrices csrbCreateAngleMatricesLocked(const CachedMatrix &cached,
                                                  const CsrbAttenChunk &chunk) {
    CsrbAngleMatrices out;
    const int rpa = chunk.compactRaysPerAngle;
    out.mats.reserve(static_cast<size_t>(chunk.angles));
    out.bufferSizes.assign(static_cast<size_t>(chunk.angles), {});
    out.bufferSizeValid.assign(static_cast<size_t>(chunk.angles), {});
    for (int i = 0; i < chunk.angles; ++i) {
      auto mat = std::make_unique<CuSparseMatGuard>();
      const int64_t offset = chunk.nnzOffset[static_cast<size_t>(i)];
      mat->create(rpa, cached.cols, chunk.nnzCount[static_cast<size_t>(i)],
                  chunk.dSubRowPtr.get() + static_cast<size_t>(i) * (rpa + 1),
                  cached.dColIdx.get() + offset, cached.dValues.get() + offset);
      out.mats.push_back(std::move(mat));
    }
    return out;
  }

  // One angle's sub-matrix applied to a column-major batch: column b of matB
  // (bRows floats) -> column b of matC (cRows floats), overwriting matC.
  // Forward (op=N) uses the production packed-K row kernels on the sub-range
  // pointers (one block per row; cuSPARSE CSR_ALG2 needs ~20 ms per call on
  // 1152 rows of ~30k nnz, the packed kernels ~0.3 ms) with a per-RHS
  // cuSPARSE remainder; transpose is per-RHS cuSPARSE CSR_ALG2 as in the
  // plain batch path.
  // Column b of matC starts at matC + b * cStride (cStride >= cRows lets the
  // forward projection write one angle's rows into a [branch][chunkRow]
  // buffer).
  void csrbAngleSpMVLocked(CsrbAngleMatrices &angleMats, const CachedMatrix &cached,
                           const CsrbAttenChunk &chunk, int angle,
                           cusparseOperation_t operation, const float *matB,
                           size_t bRows, float *matC, size_t cRows,
                           size_t cStride, int batchSize, float *xPackScratch) {
    const size_t bufferIndex = spmvBufferIndex(operation);
    const size_t a = static_cast<size_t>(angle);
    int b = 0;
    if (operation == CUSPARSE_OPERATION_NON_TRANSPOSE && xPackScratch != nullptr &&
        bRows <= static_cast<size_t>(std::numeric_limits<int>::max())) {
      // The row kernels vectorise on t % 4 relative to the array base, so
      // they get the chunk's own arrays and the row pointers offset by the
      // angle's first row (absolute nnz indices); only cuSPARSE needs the
      // zero-based copies.
      const int rpa = chunk.compactRaysPerAngle;
      const float *values = cached.dValues.get();
      const int32_t *rowPtr = cached.dRowPtr.get() + a * static_cast<size_t>(rpa);
      const int32_t *colIdx = cached.dColIdx.get();
      const int cols = static_cast<int>(bRows);
      const int kMem = cuda::compactPackedForwardKMem(
          static_cast<int64_t>(bRows), batchSize, cuda::getL2CacheBytes());
      const int tile = cuda::compactPackedForwardTile(kMem, compactSortedHint_);
      if (tile >= 4) {
        for (; b + 3 < batchSize; b += 4) {
          launchCompactPackColQuadToFloat4Kernel(
              matB + static_cast<size_t>(b) * bRows,
              matB + static_cast<size_t>(b + 1) * bRows,
              matB + static_cast<size_t>(b + 2) * bRows,
              matB + static_cast<size_t>(b + 3) * bRows, xPackScratch, cols);
          launchCompactCsrSpmvXQuadKernel(
              values, rowPtr, colIdx, xPackScratch,
              matC + static_cast<size_t>(b) * cStride,
              matC + static_cast<size_t>(b + 1) * cStride,
              matC + static_cast<size_t>(b + 2) * cStride,
              matC + static_cast<size_t>(b + 3) * cStride, rpa);
        }
      }
      if (tile >= 2) {
        for (; b + 1 < batchSize; b += 2) {
          launchCompactPackColPairToFloat2Kernel(
              matB + static_cast<size_t>(b) * bRows,
              matB + static_cast<size_t>(b + 1) * bRows, xPackScratch, cols);
          launchCompactCsrSpmvXPairKernel(values, rowPtr, colIdx, xPackScratch,
                                          matC + static_cast<size_t>(b) * cStride,
                                          matC + static_cast<size_t>(b + 1) * cStride,
                                          rpa);
        }
      }
    }
    for (; b < batchSize; ++b) {
      CuSparseVecGuard vecX;
      CuSparseVecGuard vecY;
      vecX.create(static_cast<int64_t>(bRows),
                  const_cast<float *>(matB) + static_cast<size_t>(b) * bRows);
      vecY.create(static_cast<int64_t>(cRows),
                  matC + static_cast<size_t>(b) * cStride);
      executeSpMVDescriptorLocked(angleMats.mats[a]->get(),
                                  angleMats.bufferSizes[a][bufferIndex],
                                  angleMats.bufferSizeValid[a][bufferIndex],
                                  operation, vecX.get(), vecY.get(), 0.0f,
                                  operation == CUSPARSE_OPERATION_TRANSPOSE);
    }
  }

  void ensureBranchMetadataLocked(const int *symIds, const int *zMirrors,
                                  int branchCount) {
    if (symIds == nullptr || zMirrors == nullptr || branchCount <= 0) {
      throw std::runtime_error("compact branch symmetry tables are null.");
    }
    const size_t count = static_cast<size_t>(branchCount);
    const bool changed =
        batchSymIdsHost_.size() != count || batchZMirrorsHost_.size() != count ||
        !std::equal(batchSymIdsHost_.begin(), batchSymIdsHost_.end(), symIds) ||
        !std::equal(batchZMirrorsHost_.begin(), batchZMirrorsHost_.end(),
                    zMirrors);
    if (changed) {
      dBatchSymIds_.copyFromHost(symIds, count);
      dBatchZMirrors_.copyFromHost(zMirrors, count);
      batchSymIdsHost_.assign(symIds, symIds + count);
      batchZMirrorsHost_.assign(zMirrors, zMirrors + count);
    }
  }

  void csrbEnsureAttenTimersLocked(size_t stages) {
    auto grow = [stages](std::vector<TimedCudaStagePair> &timers) {
      while (timers.size() < stages) {
        timers.emplace_back();
      }
    };
    grow(csrbAttenForwardTimers_);
    grow(csrbAttenBackwardTimers_);
    grow(csrbAttenTransformTimers_);
  }

  const CsrbAttenChunk &csrbEnsureAttenChunkLocked(
      const io::CSRView &matrix, const CachedMatrix &cached,
      int compactRowBase, const cuda::CompactDeviceRatioParams &p) {
    CsrbAttenChunk &chunk = csrbAttenChunks_[compactRowBase];
    if (chunk.rows == static_cast<int>(cached.rows) &&
        chunk.matrixKey == cached.key && chunk.symmetryMode == p.symmetryMode &&
        chunk.compactRaysPerAngle == p.compactRaysPerAngle &&
        chunk.compactNumAngles == p.compactNumAngles &&
        chunk.fullNumAngles == p.fullNumAngles &&
        chunk.symIds == batchSymIdsHost_ && chunk.dSubRowPtr.get() != nullptr) {
      return chunk;
    }
    chunk.matrixKey = cached.key;
    chunk.rows = static_cast<int>(cached.rows);
    chunk.symmetryMode = p.symmetryMode;
    chunk.compactRaysPerAngle = p.compactRaysPerAngle;
    chunk.compactNumAngles = p.compactNumAngles;
    chunk.fullNumAngles = p.fullNumAngles;
    chunk.symIds = batchSymIdsHost_;
    std::vector<int> sliceIndex;
    chunk.angles = buildAttenChunkSliceTableLocked(p, compactRowBase, chunk.rows,
                                                   sliceIndex, chunk.fullAngles);
    chunk.dSliceIndex.copyFromHost(sliceIndex.data(), sliceIndex.size());
    const int rpa = p.compactRaysPerAngle;
    std::vector<int32_t> subRowPtr(static_cast<size_t>(chunk.angles) *
                                   (static_cast<size_t>(rpa) + 1u));
    chunk.nnzOffset.assign(static_cast<size_t>(chunk.angles), 0);
    chunk.nnzCount.assign(static_cast<size_t>(chunk.angles), 0);
    for (int i = 0; i < chunk.angles; ++i) {
      const int64_t base = matrix.row_ptr[static_cast<size_t>(i) * rpa];
      const int64_t end = matrix.row_ptr[static_cast<size_t>(i + 1) * rpa];
      chunk.nnzOffset[static_cast<size_t>(i)] = base;
      chunk.nnzCount[static_cast<size_t>(i)] = end - base;
      for (int j = 0; j <= rpa; ++j) {
        const int64_t shifted =
            static_cast<int64_t>(
                matrix.row_ptr[static_cast<size_t>(i) * rpa + j]) -
            base;
        if (shifted < 0 || shifted > std::numeric_limits<int32_t>::max()) {
          throw std::runtime_error(
              "compact CTAC sub-CSR row pointer out of int32 range.");
        }
        subRowPtr[static_cast<size_t>(i) * (static_cast<size_t>(rpa) + 1u) +
                  static_cast<size_t>(j)] = static_cast<int32_t>(shifted);
      }
    }
    chunk.dSubRowPtr.copyFromHost(subRowPtr.data(), subRowPtr.size());
    return chunk;
  }

  // Page-locks a caller-owned factor table once (cached by pointer) so its
  // slices can be copied asynchronously. Returns false when registration is
  // refused (the table is then copied synchronously). Registrations are
  // released in releaseCsrqResidentLocked, i.e. before the caller frees the
  // tables at the end of a reconstruction.
  bool csrqEnsureHostTablePinnedLocked(const uint16_t *table, size_t bytes) {
    for (const auto &entry : csrqPinnedTables_) {
      if (entry.ptr == table && entry.bytes == bytes) {
        return entry.pinned;
      }
    }
    CsrqPinnedHostTable entry;
    entry.ptr = table;
    entry.bytes = bytes;
    const auto start = std::chrono::steady_clock::now();
    const cudaError_t status = cudaHostRegister(
        const_cast<uint16_t *>(table), bytes, cudaHostRegisterPortable);
    if (status == cudaSuccess) {
      entry.pinned = true;
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count();
      if (bytes >= (64ull << 20)) {
        std::cerr << "[NOTE] CSRQ CTAC: page-locked a " << (bytes >> 20)
                  << " MiB attenuation table in " << seconds
                  << " s (slices now stream asynchronously).\n";
      }
    } else {
      (void)cudaGetLastError();
      entry.pinned = false;
      std::cerr << "[NOTE] CSRQ CTAC: could not page-lock a " << (bytes >> 20)
                << " MiB attenuation table (" << cudaGetErrorString(status)
                << "); its slices are copied synchronously.\n";
    }
    csrqPinnedTables_.push_back(entry);
    return entry.pinned;
  }

  void csrqReleasePinnedTablesLocked() {
    for (const auto &entry : csrqPinnedTables_) {
      if (entry.pinned) {
        (void)cudaHostUnregister(const_cast<void *>(entry.ptr));
        (void)cudaGetLastError();
      }
    }
    csrqPinnedTables_.clear();
  }

  // Stages the factor slices chunk chunkIndex needs from one bed's table
  // ([fullAngle][nvox] uint16) into one of kCsrqAttSlots device slots and
  // returns the slot. Pinned tables go through the upload stream, double
  // buffered against the compute of the previous (chunk, bed): the copy for
  // this pair overlaps the kernels of the last one and the slot is refilled
  // only after those kernels released it (csrqAttFree_). The default stream
  // waits for the slot's ready event before its first launch. Contiguous
  // full-angle runs are copied in one piece each.
  int csrqStageAttenSlicesLocked(const CsrqAttenPlan &plan, size_t chunkIndex,
                                 const uint16_t *attFactors, size_t attCount,
                                 size_t nvox) {
    return stageAttenSlicesLocked(plan.chunkFullAngles[chunkIndex],
                                  plan.maxSlices, plan.fullNumAngles,
                                  attFactors, attCount, nvox);
  }

  int stageAttenSlicesLocked(const std::vector<int> &fullAngles,
                             size_t maxSlices, int fullNumAngles,
                             const uint16_t *attFactors, size_t attCount,
                             size_t nvox) {
    if (attFactors == nullptr ||
        attCount != static_cast<size_t>(fullNumAngles) * nvox) {
      throw std::runtime_error(
          "compact CTAC table must hold fullNumAngles * voxels uint16 factors.");
    }
    const int slot = csrqAttNextSlot_;
    csrqAttNextSlot_ = (slot + 1) % kCsrqAttSlots;
    DeviceArray<uint16_t> &dSlices = dCsrqAttSlices_[slot];
    dSlices.allocate(std::max<size_t>(maxSlices, fullAngles.size()) * nvox);
    if (fullAngles.empty()) {
      csrqAttSlotUsed_[slot] = true;
      return slot;
    }
    const bool pinned =
        csrqEnsureHostTablePinnedLocked(attFactors, attCount * sizeof(uint16_t));
    cudaStream_t upload = csrqUploadStream_.get();
    if (pinned && csrqAttSlotUsed_[slot]) {
      csrqAttFree_[slot].wait(upload);
    }
    const auto start = std::chrono::steady_clock::now();
    size_t begin = 0;
    while (begin < fullAngles.size()) {
      size_t end = begin + 1;
      while (end < fullAngles.size() &&
             fullAngles[end] == fullAngles[end - 1] + 1) {
        ++end;
      }
      const size_t bytes = (end - begin) * nvox * sizeof(uint16_t);
      const uint16_t *src =
          attFactors + static_cast<size_t>(fullAngles[begin]) * nvox;
      uint16_t *dst = dSlices.get() + begin * nvox;
      if (pinned) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                   upload));
      } else {
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
      }
      begin = end;
    }
    if (pinned) {
      csrqAttReady_[slot].record(upload);
      csrqAttReady_[slot].wait(nullptr);
    }
    noteCudaCompactInputH2D(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    csrqAttSlotUsed_[slot] = true;
    return slot;
  }

  // Marks the slot's slices as consumed by everything enqueued so far on the
  // default stream (the upload stream waits on this before refilling it).
  void csrqReleaseAttenSlotLocked(int slot) {
    csrqAttFree_[slot].record(nullptr);
  }

  // Rows [rowBase, rowBase + rows) of a chunk as a chunk view of their own:
  // the row pointer arrays are absolute within the chunk, so offsetting them
  // is enough (runs / values stay whole).
  static CsrqDeviceChunkPtrs csrqRowRange(const CsrqDeviceChunk &chunk,
                                          int rowBase, int rows) {
    CsrqDeviceChunkPtrs p = chunk.ptrs();
    p.rowNnzPtr += rowBase;
    p.rowRunPtr += rowBase;
    p.rowMax += rowBase;
    p.rows = rows;
    return p;
  }

  // Attenuated backward of one compact angle's rows into accum:
  // scatter -> [8][nvox], then transform-add with the branch's factor slice
  // raised to the window's energy exponent.
  void csrqAttenBackwardAngleLocked(const CsrqDeviceChunkPtrs &rows,
                                    const float *ratio, size_t nvox,
                                    const cuda::CompactDeviceTransformParams &tp,
                                    const uint16_t *dSlices,
                                    const int *dSliceIndex, float attPower,
                                    double *accum) {
    CUDA_CHECK(cudaMemsetAsync(dCsrqY_.get(), 0, nvox * kCsrqK * sizeof(float),
                               nullptr));
    for (int kFirst = 0; kFirst < kCsrqK;
         kFirst += kCsrqBackwardBranchesPerPass) {
      const int kCount = std::min(kCsrqBackwardBranchesPerPass, kCsrqK - kFirst);
      launchCsrqBackwardK8Kernel(rows, ratio, dCsrqY_.get(), nvox, kFirst,
                                 kCount);
    }
    launchCompactTransformBackwardAddBatchAttenKernel(
        dCsrqY_.get(), accum, nvox, kCsrqK, tp, dBatchSymIds_.get(),
        dBatchZMirrors_.get(), dSlices, dSliceIndex, attPower);
  }

  static float csrqCheckedAttPower(double attPower) {
    if (!(attPower > 0.0) || !std::isfinite(attPower)) {
      throw std::runtime_error(
          "CSRQ CTAC energy exponent must be a positive finite number.");
    }
    return static_cast<float>(attPower);
  }

  size_t csrqMaxRowsLocked() const {
    size_t rows = 0;
    for (const auto &matrix : csrqMatrices_) {
      rows = std::max(rows, matrix.totalRows);
    }
    return rows;
  }

  CsrqResidentMatrix &csrqRequireWarmupTargetLocked(int matrixIndex) {
    if (matrixIndex < 0 ||
        static_cast<size_t>(matrixIndex) >= csrqMatrices_.size() ||
        csrqMatrices_[static_cast<size_t>(matrixIndex)].shapes.empty()) {
      throw std::runtime_error("CSRQ warmup was not started for matrix " +
                               std::to_string(matrixIndex) + ".");
    }
    return csrqMatrices_[static_cast<size_t>(matrixIndex)];
  }

  // The CSRQ kernels are compiled for exactly kCsrqK = 8 right-hand sides
  // (4 xy symmetries x 2 z branches).
  CsrqResidentMatrix &csrqRequireReadyLocked(int matrixIndex, int branchCount) {
    if (matrixIndex < 0 ||
        static_cast<size_t>(matrixIndex) >= csrqMatrices_.size() ||
        !csrqMatrices_[static_cast<size_t>(matrixIndex)].ready ||
        dCsrqX_.get() == nullptr) {
      throw std::runtime_error(
          "CSRQ matrix " + std::to_string(matrixIndex) +
          " is not loaded on the device (warmup missing).");
    }
    if (branchCount != kCsrqK) {
      throw std::runtime_error(
          "CSRQ kernels require exactly " + std::to_string(kCsrqK) +
          " symmetry branches, got " + std::to_string(branchCount) + ".");
    }
    return csrqMatrices_[static_cast<size_t>(matrixIndex)];
  }

  static constexpr size_t csrqBackwardPassCount() {
    return static_cast<size_t>(
        (kCsrqK + kCsrqBackwardBranchesPerPass - 1) / kCsrqBackwardBranchesPerPass);
  }

  void csrqEnsureTimersLocked(size_t passes) {
    auto grow = [passes](std::vector<TimedCudaStagePair> &timers) {
      while (timers.size() < passes) {
        timers.emplace_back();
      }
    };
    grow(csrqForwardTimers_);
    grow(csrqBackwardTimers_);
  }

  // A^T of one chunk added into accum (unrotated frame): scatter into the
  // branch-major dCsrqY_ ([8][nvox]) a few branches per pass so the atomic
  // target stays L2-resident, then the symmetry transform-add kernel. (A
  // shared-memory slice-tile variant was measured 2x slower on sm_89 — no
  // native fp32 shared atomics; see reports/attachments/2026-09-04-*.)
  void csrqBackwardChunkLocked(const CsrqDeviceChunk &chunk, const float *ratio,
                               size_t nvox,
                               const cuda::CompactDeviceTransformParams &tp,
                               double *accum) {
    CUDA_CHECK(cudaMemsetAsync(dCsrqY_.get(), 0, nvox * kCsrqK * sizeof(float),
                               nullptr));
    for (int kFirst = 0; kFirst < kCsrqK;
         kFirst += kCsrqBackwardBranchesPerPass) {
      const int kCount = std::min(kCsrqBackwardBranchesPerPass, kCsrqK - kFirst);
      launchCsrqBackwardK8Kernel(chunk.ptrs(), ratio, dCsrqY_.get(), nvox,
                                 kFirst, kCount);
    }
    launchCompactTransformBackwardAddBatchKernel(
        dCsrqY_.get(), accum, nvox, kCsrqK, tp, dBatchSymIds_.get(),
        dBatchZMirrors_.get());
  }

  // Visits every chunk of a matrix in compact row order on the default
  // stream. Resident chunks are handed over directly; streamed chunks are
  // copied from their pinned host mirror into one of kCsrqStreamSlots device
  // slots on the upload stream, double-buffered: the copy of chunk c+1 runs
  // while chunk c computes, and a slot is refilled only after the default
  // stream has finished reading it.
  template <typename Fn>
  void csrqForEachChunkLocked(CsrqResidentMatrix &matrix, Fn &&fn) {
    if (!matrix.streamed) {
      for (size_t chunkIndex = 0; chunkIndex < matrix.chunks.size(); ++chunkIndex) {
        fn(*matrix.chunks[chunkIndex], chunkIndex);
      }
      return;
    }
    cudaStream_t upload = csrqUploadStream_.get();
    auto enqueueUpload = [&](size_t chunkIndex) {
      const int slot = static_cast<int>(chunkIndex % kCsrqStreamSlots);
      if (csrqSlotUsed_[slot]) {
        CUDA_CHECK(cudaStreamWaitEvent(upload, csrqSlotFree_[slot].get(), 0));
      }
      const CsrqHostChunk &host = matrix.host[chunkIndex];
      const cuda::CsrqChunkShape &shape = matrix.shapes[chunkIndex];
      CsrqDeviceChunk &dev = *matrix.chunks[slot];
      const size_t ptrCount = static_cast<size_t>(shape.rows) + 1u;
      dev.dRowNnzPtr.copyFromHostAsync(host.rowNnzPtr.data(), ptrCount, upload);
      dev.dRowRunPtr.copyFromHostAsync(host.rowRunPtr.data(), ptrCount, upload);
      dev.dRowMax.copyFromHostAsync(host.rowMax.data(),
                                    static_cast<size_t>(shape.rows), upload);
      dev.dRuns.copyFromHostAsync(host.runs.data(),
                                  static_cast<size_t>(shape.runCount), upload);
      dev.dValuesQ.copyFromHostAsync(host.valuesQ.data(),
                                     static_cast<size_t>(shape.nnz), upload);
      csrqUploadReady_[slot].record(upload);
      csrqSlotUsed_[slot] = true;
    };
    const size_t chunkCount = matrix.shapes.size();
    enqueueUpload(0);
    for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
      const int slot = static_cast<int>(chunkIndex % kCsrqStreamSlots);
      if (chunkIndex + 1 < chunkCount) {
        enqueueUpload(chunkIndex + 1);
      }
      CsrqDeviceChunk &dev = *matrix.chunks[slot];
      const cuda::CsrqChunkShape &shape = matrix.shapes[chunkIndex];
      dev.rows = shape.rows;
      dev.compactRowBase = shape.compactRowBase;
      dev.nnz = shape.nnz;
      dev.runCount = shape.runCount;
      csrqUploadReady_[slot].wait(nullptr);
      fn(dev, chunkIndex);
      csrqSlotFree_[slot].record(nullptr);
    }
  }

  DeviceArray<double> &csrqRequireAccumulatorLocked(int accumulatorIndex,
                                                    int cols) {
    const size_t slot = compactBackwardAccumulatorSlot(accumulatorIndex);
    DeviceArray<double> &accum = dCompactBackwardAccum_[slot];
    if (accum.get() == nullptr ||
        compactBackwardAccumSize_[slot] != static_cast<size_t>(cols)) {
      throw std::runtime_error(
          "CSRQ backward accumulator was not initialized for this matrix.");
    }
    return accum;
  }

  static size_t csrqBackwardLaunchCountLocked(const CsrqResidentMatrix &matrix) {
    constexpr int passes =
        (kCsrqK + kCsrqBackwardBranchesPerPass - 1) / kCsrqBackwardBranchesPerPass;
    return matrix.chunks.size() * static_cast<size_t>(passes);
  }

  // A^T over every chunk of one matrix: dCsrqRatio_ ([compactRow][8])
  // scatters into the branch-major dCsrqY_ ([8][nvox]), a few branches per
  // matrix pass so the atomic target stays L2-resident.
  void csrqBackwardAllChunksLocked(const CsrqResidentMatrix &matrix,
                                   size_t nvox) {
    for (int kFirst = 0; kFirst < kCsrqK;
         kFirst += kCsrqBackwardBranchesPerPass) {
      const int kCount = std::min(kCsrqBackwardBranchesPerPass, kCsrqK - kFirst);
      for (const auto &chunk : matrix.chunks) {
        launchCsrqBackwardK8Kernel(
            chunk->ptrs(),
            dCsrqRatio_.get() +
                static_cast<size_t>(chunk->compactRowBase) * kCsrqK,
            dCsrqY_.get(), nvox, kFirst, kCount);
      }
    }
  }

  void csrqEnsureBranchMetadataLocked(const int *symIds, const int *zMirrors) {
    if (symIds == nullptr || zMirrors == nullptr) {
      throw std::runtime_error("CSRQ branch symmetry tables are null.");
    }
    const size_t count = static_cast<size_t>(kCsrqK);
    const bool changed =
        batchSymIdsHost_.size() != count || batchZMirrorsHost_.size() != count ||
        !std::equal(batchSymIdsHost_.begin(), batchSymIdsHost_.end(), symIds) ||
        !std::equal(batchZMirrorsHost_.begin(), batchZMirrorsHost_.end(),
                    zMirrors);
    if (changed) {
      dBatchSymIds_.copyFromHost(symIds, count);
      dBatchZMirrors_.copyFromHost(zMirrors, count);
      batchSymIdsHost_.assign(symIds, symIds + count);
      batchZMirrorsHost_.assign(zMirrors, zMirrors + count);
    }
  }

  CuSparseHandleGuard handle_;
  CachedMatrix cache_;
  std::array<CachedMatrix, 4> subsetCache_;
  std::array<SubsetCacheKey, 4> subsetCacheKeys_{};
  std::array<bool, 4> subsetCacheValid_{};
  size_t nextSubsetCacheSlot_ = 0;
  CachedMatrix streamingRangeCache_;
  RangeCacheKey streamingRangeKey_{};
  bool streamingRangeValid_ = false;
  std::array<CachedMatrix, kTransientRangeSlotCount> transientRangeCache_;
  std::array<bool, kTransientRangeSlotCount> transientRangeValid_{};
  std::array<bool, kTransientRangeSlotCount> transientRangeSlotUsed_{};
  std::array<bool, kTransientRangeSlotCount> transientRangeReusableReady_{};
  std::array<PinnedHostCSRData, kTransientRangeSlotCount>
      transientRangePinnedScratch_;
  std::array<CudaEventGuard, kTransientRangeSlotCount> transientRangeUploadStart_;
  std::array<CudaEventGuard, kTransientRangeSlotCount> transientRangeReady_;
  std::array<CudaEventGuard, kTransientRangeSlotCount> transientRangeReusable_;
  std::array<bool, kTransientRangeSlotCount> transientRangeUploadTimingPending_{};
  DeviceChunkCache deviceChunkCache_;            // P3-D
  ResidentMatrixSetCache residentMatrixSetCache_;
  HostCSRData subsetHostScratch_;
  DeviceArray<float> dInput_;
  DeviceArray<float> dOutput_;
  DeviceArray<float> dProjectionInput_;
  std::array<DeviceArray<double>, kCompactProjectionSlots>
      dCompactProjectionInput_;
  DeviceArray<float> dChunkInput_;
  PinnedHostArray<float> pinnedInputStage_;
  PinnedHostArray<float> pinnedOutputStage_;
  DeviceArray<float> dAccumOutput_;
  DeviceArray<float> dFullOutput_;
  // Branch-batch compact-symmetric buffers (cols x batchSize / rows x batchSize).
  DeviceArray<float> dBatchInput_;
  DeviceArray<float> dBatchOutput_;
  DeviceArray<float> dFullBatchOutput_;
  DeviceArray<float> dBatchOnes_;
  DeviceArray<int> dBatchSymIds_;
  DeviceArray<int> dBatchZMirrors_;
  DeviceArray<double> dCompactGlobalImage_;
  DeviceArray<float> dCompactBedPrimary_;
  DeviceArray<float> dCompactBedSecondary_;
  size_t compactResidentImageCount_ = 0;
  DeviceArray<double> dCompactGlobalNumer_;
  DeviceArray<double> dCompactDenom_;
  DeviceArray<uint8_t> dCompactMask_;
  DeviceArray<uint32_t> dCompactIterationFlags_;
  size_t compactResidentStaticCount_ = 0;
  bool compactResidentHasMask_ = false;
  std::vector<int> batchSymIdsHost_;
  std::vector<int> batchZMirrorsHost_;
  size_t compactBatchInputCount_ = 0;
  bool compactSortedHint_ = false;
  bool compactPackedKLogged_ = false;
  bool compactTransposeExpLogged_ = false;
  std::array<DeviceArray<double>, kCompactDeviceBackwardMaxAccumulators>
      dCompactBackwardAccum_;
  std::vector<float> hostFloatScratch_;
  std::array<const double *, kCompactProjectionSlots> compactProjectionHostKey_{};
  std::array<size_t, kCompactProjectionSlots> compactProjectionCount_{};
  // Per-window background (scatter estimate) next to each projection slot;
  // both formats' ratio kernels add it to the forward estimate.
  std::array<DeviceArray<double>, kCompactProjectionSlots>
      dCompactBackgroundInput_;
  std::array<const double *, kCompactProjectionSlots> compactBackgroundHostKey_{};
  std::array<size_t, kCompactProjectionSlots> compactBackgroundCount_{};
  // Sensitivity row weights (one per full projection row), shared by all
  // channels of a run.
  DeviceArray<double> dCompactRowWeights_;
  const double *compactRowWeightsHostKey_ = nullptr;
  size_t compactRowWeightsCount_ = 0;
  // Resident BSREM-II state: regularizer gradient scratch (filled by the
  // volume ops), curvature surrogate, kappa map.
  DeviceArray<double> dCompactGradient_;
  DeviceArray<double> dCompactCurvature_;
  size_t compactCurvatureCount_ = 0;
  DeviceArray<double> dCompactKappa_;
  const double *compactKappaHostKey_ = nullptr;
  size_t compactKappaCount_ = 0;
  std::array<size_t, kCompactDeviceBackwardMaxAccumulators>
      compactBackwardAccumSize_{};
  // CSRQ scratch shared by all resident matrices: dCsrqX_ is the [voxel][8]
  // packed forward input, dCsrqY_ the branch-major [8][voxel] backward
  // output, and dCsrqRatio_ the [compactRow][8] forward/ratio buffer sized
  // for the largest matrix (3.3 MB at 103680 rows), so forward/ratio/backward
  // run as three whole-matrix stages instead of interleaving per chunk.
  std::vector<CsrqResidentMatrix> csrqMatrices_;
  DeviceArray<float> dCsrqX_;
  DeviceArray<float> dCsrqY_;
  DeviceArray<float> dCsrqRatio_;
  // Streaming: upload stream plus per-slot ready/free events (shared by all
  // streamed matrices; passes run one matrix at a time).
  CudaStreamGuard csrqUploadStream_;
  std::array<CudaEventGuard, kCsrqStreamSlots> csrqUploadReady_;
  std::array<CudaEventGuard, kCsrqStreamSlots> csrqSlotFree_;
  std::array<bool, kCsrqStreamSlots> csrqSlotUsed_{};
  // Per-(chunk, bed) stage timers reused across passes.
  std::vector<TimedCudaStagePair> csrqForwardTimers_;
  std::vector<TimedCudaStagePair> csrqBackwardTimers_;
  std::chrono::steady_clock::time_point csrqWarmupStart_{};
  // CSRQ CTAC staging: two device slots holding the factor slices of the
  // full angles one chunk reaches ([slice][nvox] uint16, a few hundred MB
  // instead of the 2.3 GB table), filled on the upload stream and handed
  // over with ready/free events like the streamed matrix chunks; the
  // caller-owned tables are page-locked once for that.
  std::array<DeviceArray<uint16_t>, kCsrqAttSlots> dCsrqAttSlices_;
  std::array<CudaEventGuard, kCsrqAttSlots> csrqAttReady_;
  std::array<CudaEventGuard, kCsrqAttSlots> csrqAttFree_;
  std::array<bool, kCsrqAttSlots> csrqAttSlotUsed_{};
  int csrqAttNextSlot_ = 0;
  std::vector<CsrqPinnedHostTable> csrqPinnedTables_;
  // CSRB CTAC: per-chunk slice tables and sub-CSR row pointers, keyed by the
  // chunk's compact row base (revalidated against the uploaded chunk).
  std::map<int, CsrbAttenChunk> csrbAttenChunks_;
  // CSRB CTAC scratch: attenuated [branch][voxel] batch input of one bed and
  // per-(bed, compact angle) stage timers.
  DeviceArray<float> dCsrbAttenBatch_;
  std::vector<TimedCudaStagePair> csrbAttenForwardTimers_;
  std::vector<TimedCudaStagePair> csrbAttenBackwardTimers_;
  std::vector<TimedCudaStagePair> csrbAttenTransformTimers_;
  DeviceBuffer dBuffer_;
  CudaStreamGuard computeStream_;
  std::array<CudaStreamGuard, kTransientRangeSlotCount> transientUploadStreams_;
  std::vector<void *> compactHostPins_;
  // csrt experiment state (see queueCompactTrueTransposeUploadLocked).
  std::unordered_map<uint64_t, CompactTransposeHostEntry>
      compactTransposeHost_;
  std::array<CompactTransposeDeviceHalf, 2> compactTransposeSlots_;
  MatrixCacheKey compactTransposeActiveKey_{};
  uint64_t compactTransposeActiveFingerprint_ = 0;
  bool compactTransposeActiveValid_ = false;
  CudaStreamGuard compactTransposeUploadStream_;
  bool compactTransposePinWarned_ = false;
  bool compactTransposeCsrtLogged_ = false;
  bool compactTransposeFallbackLogged_ = false;
  std::mutex mutex_;
};

SparseOpsCUDAState &cudaState() {
  // CUDA runtime teardown order at process exit is not stable across driver and
  // MSVC CRT versions. Keep the backend state alive until process termination;
  // the OS/driver reclaims these resources and we avoid use-after-runtime
  // crashes from static destructors.
  static SparseOpsCUDAState *state = new SparseOpsCUDAState();
  return *state;
}
