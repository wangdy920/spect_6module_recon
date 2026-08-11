#pragma once

constexpr int kCompactDeviceBackwardMaxAccumulators =
  cuda::kCompactDeviceBackwardMaxAccumulators;

cusparseSpMVAlg_t parseSpmvAlgName(const char *text,
                                   cusparseSpMVAlg_t fallback) {
  if (text == nullptr || *text == '\0') {
    return fallback;
  }
  std::string value(text);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  if (value == "default") {
    return CUSPARSE_SPMV_ALG_DEFAULT;
  }
  if (value == "csr1" || value == "csr_alg1" || value == "alg1") {
    return CUSPARSE_SPMV_CSR_ALG1;
  }
  if (value == "csr2" || value == "csr_alg2" || value == "alg2") {
    return CUSPARSE_SPMV_CSR_ALG2;
  }
  return fallback;
}

cusparseSpMVAlg_t selectedSpmvAlg(bool backward) {
  static const cusparseSpMVAlg_t globalAlg = parseSpmvAlgName(
      std::getenv("SPECT_CUDA_SPMV_ALG"), CUSPARSE_SPMV_ALG_DEFAULT);
  static const cusparseSpMVAlg_t forwardAlg = parseSpmvAlgName(
      std::getenv("SPECT_CUDA_SPMV_FWD_ALG"), globalAlg);
  static const cusparseSpMVAlg_t backwardAlg = parseSpmvAlgName(
      std::getenv("SPECT_CUDA_SPMV_BWD_ALG"), globalAlg);
  return backward ? backwardAlg : forwardAlg;
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

double accumulateTimedCudaPairs(const std::vector<TimedCudaStagePair> &pairs) {
  double seconds = 0.0;
  for (const auto &pair : pairs) {
    seconds += pair.elapsedSeconds();
  }
  return seconds;
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
public:
  template <typename MatrixType>
  void executeSpMV(const MatrixType &matrix, const double *x, double *y,
                   cusparseOperation_t operation) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const CachedMatrix &cached = ensureMatrixUploaded(matrix);
    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;

    uploadDouble(dInput_, x, static_cast<size_t>(inputSize));
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f);
    downloadDouble(dOutput_, y, static_cast<size_t>(outputSize));
  }

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

  void executeCompactAttenSpMVFloat(const io::CSRView &matrix,
                                    const uint16_t *attFactors,
                                    size_t attCount,
                                    const cuda::CompactAttenBakeParams &params,
                                    const float *x, float *y,
                                    cusparseOperation_t operation,
                                    int cacheSlot) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    if (cacheSlot < 0 || cacheSlot >= spect::core::kMaxBeds) {
      throw std::runtime_error("compact attenuation cache slot out of range.");
    }
    const size_t slot = static_cast<size_t>(cacheSlot);
    if (params.symmetryMode != 45 && params.symmetryMode != 90) {
      throw std::runtime_error(
          "compact fused CTAC SpMV requires symmetryMode 45 or 90.");
    }
    if (attFactors == nullptr || attCount == 0) {
      throw std::runtime_error(
          "compact fused CTAC SpMV received an empty factor table.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);

    // Upload the full-angle attenuation table once per bed slot; cache by host
    // pointer so repeated branches/chunks for the same bed reuse it.
    auto &dAtten = dAttenTable_[slot];
    if (attenTableHostPtr_[slot] != attFactors ||
        attenTableCount_[slot] != attCount) {
      dAtten.copyFromHost(attFactors, attCount);
      attenTableHostPtr_[slot] = attFactors;
      attenTableCount_[slot] = attCount;
      bakedValid_[slot] = false;
    }
    if (compactCtacBakedReferenceEnabled()) {
      const bool bakeHit =
          bakedValid_[slot] && bakedMatrixKey_[slot] == cached.key &&
          bakedRowPtrDev_[slot] == static_cast<const void *>(cached.dRowPtr.get()) &&
          bakedColIdxDev_[slot] == static_cast<const void *>(cached.dColIdx.get()) &&
          bakedAttPtr_[slot] == static_cast<const void *>(attFactors) &&
          bakedParams_[slot].symmetryMode == params.symmetryMode &&
          bakedParams_[slot].compactRowBase == params.compactRowBase &&
          bakedParams_[slot].symId == params.symId &&
          bakedParams_[slot].zMirror == params.zMirror;
      if (!bakeHit) {
        auto &dBaked = dBakedValues_[slot];
        dBaked.allocate(static_cast<size_t>(cached.nnz));
        launchCompactBakeAttenKernel(
            cached.dValues.get(), cached.dRowPtr.get(), cached.dColIdx.get(),
            dAtten.get(), dBaked.get(),
            static_cast<int>(cached.rows), params, computeStream_.get());
        auto &bakedMat = bakedMat_[slot];
        bakedMat.create(cached.rows, cached.cols, cached.nnz,
                        cached.dRowPtr.get(), cached.dColIdx.get(),
                        dBaked.get());
        bakedBufferSizeValid_[slot] = {};
        bakedMatrixKey_[slot] = cached.key;
        bakedRowPtrDev_[slot] = cached.dRowPtr.get();
        bakedColIdxDev_[slot] = cached.dColIdx.get();
        bakedAttPtr_[slot] = attFactors;
        bakedParams_[slot] = params;
        bakedValid_[slot] = true;
        CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
      }

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
      const size_t bufferIndex = spmvBufferIndex(operation);
      executeSpMVDescriptorLocked(bakedMat_[slot].get(),
                                  bakedBufferSizes_[slot][bufferIndex],
                                  bakedBufferSizeValid_[slot][bufferIndex],
                                  operation, vecX.get(), vecY.get(), 0.0f,
                                  backward);
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
      return;
    }

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
    if (backward && outputSize > 0) {
      CUDA_CHECK(cudaMemset(dOutput_.get(), 0,
                            static_cast<size_t>(outputSize) * sizeof(float)));
    }

    TimedCudaStagePair deviceTiming;
    deviceTiming.recordStart(nullptr);
    if (backward) {
      launchCompactAttenFusedBackwardKernel(
          cached.dValues.get(), cached.dRowPtr.get(), cached.dColIdx.get(),
          dAtten.get(), dInput_.get(), dOutput_.get(),
          static_cast<int>(cached.rows), params);
    } else {
      launchCompactAttenFusedForwardKernel(
          cached.dValues.get(), cached.dRowPtr.get(), cached.dColIdx.get(),
          dAtten.get(), dInput_.get(), dOutput_.get(),
          static_cast<int>(cached.rows), params);
    }
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

  void executeCompactDeviceRatioSpmvTFloat(
      const io::CSRView &matrix, const float *x, const double *fullProjection,
      size_t fullProjectionCount,
      const cuda::CompactDeviceRatioParams &params, float *y,
      cuda::CompactDeviceRatioTiming *timing, int cacheSlot) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    if (cacheSlot < 0 || cacheSlot >= spect::core::kMaxBeds) {
      throw std::runtime_error("compact device-ratio cache slot out of range.");
    }
    const size_t slot = static_cast<size_t>(cacheSlot);
    if (params.symmetryMode != 45 && params.symmetryMode != 90) {
      throw std::runtime_error(
          "compact device-ratio CUDA helper requires symmetryMode 45 or 90.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;

    const auto inputStart = std::chrono::steady_clock::now();
    stageInputH2D(x, static_cast<size_t>(cols));
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
    launchCompactRatioKernel(dOutput_.get(), dCompactProjectionInput_[slot].get(),
                             fullProjectionCount, params, dChunkInput_.get(),
                             static_cast<size_t>(rows), 1e-10);
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

    const auto outputStart = std::chrono::steady_clock::now();
    stageOutputD2H(dFullOutput_, y, static_cast<size_t>(cols));
    const double outputSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      outputStart)
            .count();
    noteCudaCompactOutputD2H(outputSeconds,
                             static_cast<size_t>(cols) * sizeof(float),
                             true);
    const double forwardSeconds = forwardTiming.elapsedSeconds();
    const double ratioSeconds = ratioTiming.elapsedSeconds();
    const double backwardSeconds = backwardTiming.elapsedSeconds();
    noteCudaCusparseForward(forwardSeconds, 1);
    noteCudaRatio(ratioSeconds, 1);
    noteCudaCusparseBackward(backwardSeconds, 1);
    if (timing != nullptr) {
      timing->forwardSeconds += forwardSeconds;
      timing->ratioSeconds += ratioSeconds;
      timing->backwardSeconds += backwardSeconds;
    }
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

  void executeCompactDeviceRatioAttenSpmvTAccumulateFloat(
      const io::CSRView &matrix, const float *x, const uint16_t *attFactors,
      size_t attCount, const cuda::CompactAttenBakeParams &attenParams,
      const double *fullProjection, size_t fullProjectionCount,
      const cuda::CompactDeviceRatioParams &ratioParams,
      const cuda::CompactDeviceTransformParams &transformParams,
      cuda::CompactDeviceRatioTiming *ratioTimingOut,
      cuda::CompactDeviceBackwardTiming *backwardTimingOut,
      int accumulatorIndex, int cacheSlot) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    if (cacheSlot < 0 || cacheSlot >= spect::core::kMaxBeds) {
      throw std::runtime_error(
          "compact device-ratio attenuation cache slot out of range.");
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
          "compact device-ratio attenuation helper requires symmetryMode 45 or 90.");
    }
    const CachedMatrix &cached = ensureCompactFloatMatrixUploaded(matrix);
    const int64_t rows = cached.rows;
    const int64_t cols = cached.cols;
    if (static_cast<size_t>(cols) != accumSize) {
      throw std::runtime_error(
          "compact device accumulator size does not match matrix columns.");
    }

    // Upload/cache the per-bed attenuation table by host pointer.
    auto &dAtten = dAttenTable_[projectionSlot];
    if (attenTableHostPtr_[projectionSlot] != attFactors ||
        attenTableCount_[projectionSlot] != attCount) {
      dAtten.copyFromHost(attFactors, attCount);
      attenTableHostPtr_[projectionSlot] = attFactors;
      attenTableCount_[projectionSlot] = attCount;
      bakedValid_[projectionSlot] = false;
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
    if (cols > 0) {
      CUDA_CHECK(cudaMemset(dFullOutput_.get(), 0,
                            static_cast<size_t>(cols) * sizeof(float)));
    }

    TimedCudaStagePair forwardTiming;
    forwardTiming.recordStart(nullptr);
    launchCompactAttenFusedForwardKernel(
        cached.dValues.get(), cached.dRowPtr.get(), cached.dColIdx.get(),
        dAtten.get(), dInput_.get(), dOutput_.get(),
        static_cast<int>(cached.rows), attenParams);
    forwardTiming.recordStop(nullptr);

    TimedCudaStagePair ratioTiming;
    ratioTiming.recordStart(nullptr);
    launchCompactRatioKernel(dOutput_.get(),
                             dCompactProjectionInput_[projectionSlot].get(),
                             fullProjectionCount, ratioParams,
                             dChunkInput_.get(), static_cast<size_t>(rows),
                             1e-10);
    ratioTiming.recordStop(nullptr);

    TimedCudaStagePair backwardTiming;
    backwardTiming.recordStart(nullptr);
    launchCompactAttenFusedBackwardKernel(
        cached.dValues.get(), cached.dRowPtr.get(), cached.dColIdx.get(),
        dAtten.get(), dChunkInput_.get(), dFullOutput_.get(),
        static_cast<int>(cached.rows), attenParams);
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

    CuSparseDnMatGuard matBForward;
    CuSparseDnMatGuard matCForward;
    matBForward.create(cols, batchSize, cols, dBatchInput_.get(), CUDA_R_32F,
                       CUSPARSE_ORDER_COL);
    matCForward.create(rows, batchSize, rows, dBatchOutput_.get(), CUDA_R_32F,
                       CUSPARSE_ORDER_COL);
    TimedCudaStagePair forwardTiming;
    forwardTiming.recordStart(nullptr);
    executeSpMMLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      matBForward.get(), matCForward.get());
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
          fullProjectionCount, commonRatioParams, dBatchSymIds_.get(),
          dBatchZMirrors_.get(), branchCount, dBatchOutput_.get() + offset,
          static_cast<size_t>(rows), 1e-10);
    }
    ratioTiming.recordStop(nullptr);

    CuSparseDnMatGuard matBBackward;
    CuSparseDnMatGuard matCBackward;
    matBBackward.create(rows, batchSize, rows, dBatchOutput_.get(), CUDA_R_32F,
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
      int accumulatorIndex, int cacheSlot) {
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
    {
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

  void executeSubsetSpMV(const io::CSRView &matrix,
                         const std::vector<int> &subset_rows, const double *x,
                         double *y, cusparseOperation_t operation,
                         int threads) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const CachedMatrix &cached = ensureSubsetUploaded(matrix, subset_rows, threads);
    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;

    uploadDouble(dInput_, x, static_cast<size_t>(inputSize));
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f);
    downloadDouble(dOutput_, y, static_cast<size_t>(outputSize));
  }

  void executeAttenuatedSpMV(const io::CSRView &matrix,
                             const std::vector<uint16_t> &atten,
                             int numAngles, const double *x, double *y,
                             cusparseOperation_t operation,
                             int threads) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const CachedMatrix &cached =
        ensureAttenuatedUploaded(matrix, atten, numAngles, threads);
    const int64_t inputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                  ? cached.rows
                                  : cached.cols;
    const int64_t outputSize = (operation == CUSPARSE_OPERATION_TRANSPOSE)
                                   ? cached.cols
                                   : cached.rows;

    uploadDouble(dInput_, x, static_cast<size_t>(inputSize));
    dOutput_.allocate(static_cast<size_t>(outputSize));

    CuSparseVecGuard vecX;
    CuSparseVecGuard vecY;
    vecX.create(inputSize, dInput_.get());
    vecY.create(outputSize, dOutput_.get());
    executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f);
    downloadDouble(dOutput_, y, static_cast<size_t>(outputSize));
  }

private:

  static bool compactCtacBakedReferenceEnabled() {
    static const bool enabled = []() {
      const char *v = std::getenv("SPECT_COMPACT_CTAC_BAKED_REFERENCE");
      return v != nullptr && v[0] == '1';
    }();
    return enabled;
  }

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
    if (cacheSlot < 0 || cacheSlot >= spect::core::kMaxBeds) {
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
    }

    void release() {
      prepareForUpload();
      dRowPtr.reset();
      dColIdx.reset();
      dValues.reset();
    }
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
    const cusparseSpMVAlg_t alg = selectedSpmvAlg(profileAsBackward);
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
    cache->dRowPtr.copyFromHostAsync(
        reinterpret_cast<const int32_t *>(matrix.row_ptr),
        static_cast<size_t>(matrix.rows) + 1u, stream);
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

  const CachedMatrix &ensureCompactFloatMatrixUploaded(
      const io::CSRView &matrix) {
    const MatrixCacheKey key = makeCacheKey(matrix);
    if (const CachedMatrix *resident = residentMatrixSetCache_.find(key)) {
      noteCudaCompactMatrixReuseHit();
      return *resident;
    }
    if (cache_.key == key && cache_.matA.get() != nullptr) {
      noteCudaCompactMatrixReuseHit();
      return cache_;
    }
    validateIoCsrViewForUpload(matrix);
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

  const CachedMatrix &ensureAttenuatedUploaded(const io::CSRView &matrix,
                                               const std::vector<uint16_t> &atten,
                                               int numAngles,
                                               int threads) {
    const AttenCacheKey key = makeAttenCacheKey(matrix, atten, numAngles);
    for (size_t index = 0; index < attenCache_.size(); ++index) {
      if (attenCacheValid_[index] && attenCacheKeys_[index] == key &&
          attenCache_[index].matA.get() != nullptr) {
        return attenCache_[index];
      }
    }

    const size_t slot = nextAttenCacheSlot_;
    nextAttenCacheSlot_ = (nextAttenCacheSlot_ + 1) % attenCache_.size();

    buildAttenuatedCSR(matrix, atten, numAngles, threads, attenHostScratch_);
    uploadHostCSR(attenHostScratch_, key.matrix, &attenCache_[slot]);
    attenCacheKeys_[slot] = key;
    attenCacheValid_[slot] = true;
    return attenCache_[slot];
  }

  // P3-D: Populate device-side chunk cache from prepared host cache.
  // Called during warmup.  Uploads as many chunks as available VRAM
  // permits while reserving space for transient slots and workspace.
  void populateDeviceChunkCacheLocked(
      const PreparedChunkedHostCSR &prepared,
      const MatrixCacheKey &matrixKey,
      uint64_t nnzBudget) {
    if (deviceChunkCache_.initialized &&
        deviceChunkCache_.matrixKey == matrixKey) {
      return;  // Already populated for this matrix.
    }
    if (deviceChunkCache_.initialized) {
      deviceChunkCache_.invalidate();
    }

    const auto &ranges = prepared.ranges;
    if (ranges.empty()) {
      return;
    }

    // Estimate the largest per-chunk device CSR size.
    uint64_t maxChunkBytes = 0;
    for (size_t i = 0; i < ranges.size(); ++i) {
      if (i < prepared.chunks.size() && prepared.ready[i]) {
        const uint64_t csrBytes = estimateDeviceCsrBytes(
            prepared.chunks[i].rows, prepared.chunks[i].nnz);
        maxChunkBytes = std::max(maxChunkBytes, csrBytes);
      }
    }
    if (maxChunkBytes == 0) {
      return;
    }

    // Reserve space for 2 transient slots + cuSPARSE/cuFFT workspace.
    constexpr uint64_t kDeviceCacheReserveBytes =
      cuda::kDeviceChunkCacheReserveBytes;  // ~6 GiB headroom

    uint64_t freeBytes = 0;
    try {
      freeBytes = static_cast<uint64_t>(cuda::getMemoryInfo().free);
    } catch (...) {
      return;
    }
    if (freeBytes <= kDeviceCacheReserveBytes) {
      return;
    }
    const uint64_t availableForCache = freeBytes - kDeviceCacheReserveBytes;
    size_t numToCacheMax =
        static_cast<size_t>(availableForCache / maxChunkBytes);
    numToCacheMax = std::min(numToCacheMax, ranges.size());
    if (numToCacheMax == 0) {
      return;
    }

    deviceChunkCache_.slots.resize(numToCacheMax);
    deviceChunkCache_.valid.resize(numToCacheMax, false);
    deviceChunkCache_.matrixKey = matrixKey;
    deviceChunkCache_.nnzBudget = nnzBudget;

    for (size_t i = 0; i < numToCacheMax; ++i) {
      if (i >= prepared.chunks.size() || !prepared.ready[i]) {
        continue;
      }
      deviceChunkCache_.slots[i] = std::make_unique<CachedMatrix>();
      uploadHostCSRAsync(prepared.chunks[i], matrixKey,
                         deviceChunkCache_.slots[i].get(), computeStream_.get());
      deviceChunkCache_.valid[i] = true;
      ++deviceChunkCache_.numCached;
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    deviceChunkCache_.initialized = true;

    const uint64_t usedMiB =
        deviceChunkCache_.numCached * maxChunkBytes / cuda::kCudaMiB;
    // cerr (not fprintf) so the process tee log captures cache residency.
    std::cerr << "[P3-D] Device chunk cache: " << deviceChunkCache_.numCached
              << " / " << ranges.size() << " chunks cached (" << usedMiB
              << " MiB VRAM)\n";
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
    if (globalImageCount == 0 || globalNz <= 0 || bedZStarts == nullptr ||
        bedCount <= 0 || bedCount > spect::core::kMaxBeds || denom == nullptr ||
        compactResidentImageCount_ != globalImageCount ||
        dCompactGlobalImage_.get() == nullptr) {
      throw std::runtime_error(
          "compact resident MLEM update parameters are invalid.");
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
          "compact resident MLEM global dimensions do not match.");
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

    dCompactIterationFlags_.allocate(1);
    CUDA_CHECK(cudaMemset(dCompactIterationFlags_.get(), 0,
                          sizeof(uint32_t)));
    launchCompactMlemUpdateKernel(
        dCompactGlobalImage_.get(), dCompactGlobalNumer_.get(),
        dCompactDenom_.get(), mask == nullptr ? nullptr : dCompactMask_.get(),
        mask != nullptr, globalImageCount, dCompactIterationFlags_.get(),
        1e-12);

    uint32_t flags = 0;
    dCompactIterationFlags_.copyToHost(&flags, 1);
    if ((flags & (1U | 4U)) != 0 ||
        (requirePositiveNumerator && (flags & 2U) == 0)) {
      throw std::runtime_error(
          "compact resident numerator is non-finite, zero, or non-positive.");
    }
    if ((flags & 8U) != 0) {
      throw std::runtime_error(
          "compact resident MLEM update produced a non-finite image.");
    }
    if (imageAllZero != nullptr)
      *imageAllZero = (flags & 16U) == 0;
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
    dCompactBedPrimary_.reset();
    dCompactBedSecondary_.reset();
    dCompactGlobalNumer_.reset();
    dCompactDenom_.reset();
    dCompactMask_.reset();
    dCompactIterationFlags_.reset();
    for (auto &projection : dCompactProjectionInput_)
      projection.reset();
    for (auto &accumulator : dCompactBackwardAccum_)
      accumulator.reset();
    for (size_t bed = 0; bed < dAttenTable_.size(); ++bed) {
      bakedMat_[bed].reset();
      dAttenTable_[bed].reset();
      dBakedValues_[bed].reset();
    }
    dBuffer_.reset();

    compactResidentImageCount_ = 0;
    compactResidentStaticCount_ = 0;
    compactResidentHasMask_ = false;
    compactBatchInputCount_ = 0;
    compactProjectionHostKey_ = {};
    compactProjectionCount_ = {};
    compactBackwardAccumSize_ = {};
    attenTableHostPtr_ = {};
    attenTableCount_ = {};
    bakedValid_ = {};
    batchSymIdsHost_.clear();
    batchZMirrorsHost_.clear();
    hostFloatScratch_.clear();
  }

  void refreshCompactProjection(const double *projection, size_t count,
                                int cacheSlot) {
    ensureCudaRuntimeAvailable();
    std::lock_guard<std::mutex> lock(mutex_);
    if (cacheSlot < 0 || cacheSlot >= spect::core::kMaxBeds) {
      throw std::runtime_error("compact projection cache slot out of range.");
    }
    // A recycled host address is not proof that a new reconstruction owns the
    // same data. Refresh once per numerator pass; inner branches still reuse it.
    compactProjectionHostKey_[static_cast<size_t>(cacheSlot)] = nullptr;
    ensureCompactFullProjectionUploaded(projection, count, cacheSlot);
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
    std::cerr << "[CompactChunkCUDA] Resident matrix cache: "
              << residentMatrixSetCache_.numCached << " / " << matrices.size()
              << " matrices cached (" << (usedBytes / cuda::kCudaMiB)
              << " MiB reserve=" << (reserveBytes / cuda::kCudaMiB)
              << " MiB)\n";
    return true;
  }

public:
  bool warmupFullSubsetChunkCache(
      const io::CSRView &matrix, int bedCount,
      const std::vector<std::vector<uint16_t>> *bedAttenFactors, int numAngles,
      int threads) {
    ensureCudaRuntimeAvailable();
    if (!needsWholeMatrixChunking(matrix)) {
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t nnzBudget = selectCudaChunkNnzBudget();
    const bool useAttenuation =
        bedAttenFactors != nullptr && !bedAttenFactors->empty();
    if (!useAttenuation) {
      const auto *prepared =
          getPreparedContiguousHostChunks(matrix, nnzBudget, threads, true);
      if (prepared != nullptr) {
        // P3-D: populate device-side chunk cache from the prepared host data.
        const MatrixCacheKey matrixKey = makeCacheKey(matrix);
        populateDeviceChunkCacheLocked(*prepared, matrixKey, nnzBudget);
        return true;
      }
      const auto &chunks = getContiguousRowChunks(matrix, nnzBudget);
      const MatrixCacheKey matrixKey = makeCacheKey(matrix);
      for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        const auto buildStart = std::chrono::steady_clock::now();
        waitTransientPinnedScratchReusable(0);
        buildContiguousSubsetCSR(matrix, chunk.rowBegin, chunk.rowEnd, threads,
                                 transientRangePinnedScratch_[0]);
        noteCudaChunkBuild(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          buildStart)
                .count());
        queueTransientRangeUpload(0, matrixKey, transientRangePinnedScratch_[0]);
        (void)waitTransientRangeUpload(0);
        markTransientRangeReusable(0);
      }
      CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
      return true;
    }

    if (bedCount <= 0 || static_cast<size_t>(bedCount) != bedAttenFactors->size()) {
      throw std::runtime_error(
          "CUDA chunk warmup received inconsistent attenuation inputs.");
    }

    bool preparedAny = true;
    const uint64_t perBedPreparedHostCacheBudgetBytes =
        selectPreparedChunkHostCacheBudgetBytes(
            matrix, static_cast<size_t>(bedCount));
    for (int bed = 0; bed < bedCount; ++bed) {
      const auto *prepared = getPreparedAttenuatedHostChunks(
          matrix, (*bedAttenFactors)[static_cast<size_t>(bed)], numAngles,
          nnzBudget, threads, true, perBedPreparedHostCacheBudgetBytes);
      preparedAny = preparedAny && (prepared != nullptr);
    }
    if (preparedAny) {
      return true;
    }
    const auto &chunks = getContiguousRowChunks(matrix, nnzBudget);
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
      for (int bed = 0; bed < bedCount; ++bed) {
        for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
          const RowChunkRange &chunk = chunks[chunkIndex];
          const auto buildStart = std::chrono::steady_clock::now();
          waitTransientPinnedScratchReusable(0);
          buildAttenuatedContiguousSubsetCSR(
              matrix, (*bedAttenFactors)[static_cast<size_t>(bed)], numAngles,
              chunk.rowBegin, chunk.rowEnd, threads,
              transientRangePinnedScratch_[0]);
          noteCudaChunkBuild(
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            buildStart)
                  .count());
          queueTransientRangeUpload(0, matrixKey, transientRangePinnedScratch_[0]);
          (void)waitTransientRangeUpload(0);
          markTransientRangeReusable(0);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    return true;
  }

  void executeChunkedWholeMatrixSpMV(const io::CSRView &matrix, const double *x,
                                     double *y,
                                     cusparseOperation_t operation,
                                     int threads) {
    ensureCudaRuntimeAvailable();

    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t nnzBudget = selectCudaChunkNnzBudget();
    PreparedChunkedHostCSR *prepared =
        getPreparedContiguousHostChunks(matrix, nnzBudget, threads);
    const auto &chunks =
        prepared ? prepared->ranges : getContiguousRowChunks(matrix, nnzBudget);
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
    const size_t transientSlotCount = transientRangeCache_.size();
    resetTransientRangeState();

    const auto queueChunkUpload = [&](size_t chunkIndex, size_t slot) {
      const RowChunkRange &chunk = chunks[chunkIndex];
      if (prepared != nullptr && prepared->ready[chunkIndex]) {
        touchPreparedChunk(*prepared, chunkIndex);
        queueTransientRangeUpload(slot, matrixKey, prepared->chunks[chunkIndex]);
        return;
      }
      const auto buildStart = std::chrono::steady_clock::now();
      waitTransientPinnedScratchReusable(slot);
      buildContiguousSubsetCSR(matrix, chunk.rowBegin, chunk.rowEnd, threads,
                               transientRangePinnedScratch_[slot]);
      noteCudaChunkBuild(
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        buildStart)
              .count());
      queueTransientRangeUpload(slot, matrixKey, transientRangePinnedScratch_[slot]);
    };

    if (!chunks.empty()) {
      queueChunkUpload(0, 0);
      if (chunks.size() > 1 && transientSlotCount > 1) {
        queueChunkUpload(1, 1);
      }
    }

    if (operation == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      std::vector<TimedCudaStagePair> uploadWaitTimings;
      std::vector<TimedCudaStagePair> forwardTimings;
      uploadWaitTimings.reserve(chunks.size());
      forwardTimings.reserve(chunks.size());
      uploadDoubleAsync(dInput_, x, static_cast<size_t>(matrix.cols),
                        computeStream_.get());
      for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        const size_t slot =
            transientSlotCount > 0 ? chunkIndex % transientSlotCount : 0;
        uploadWaitTimings.emplace_back();
        const CachedMatrix &cached =
            waitTransientRangeUpload(slot, &uploadWaitTimings.back());
        const size_t chunkRows =
            static_cast<size_t>(chunk.rowEnd - chunk.rowBegin);
        dOutput_.allocate(chunkRows);

        CuSparseVecGuard vecX;
        CuSparseVecGuard vecY;
        vecX.create(cached.cols, dInput_.get());
        vecY.create(cached.rows, dOutput_.get());
        forwardTimings.emplace_back();
        forwardTimings.back().recordStart(computeStream_.get());
        executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f,
                          computeStream_.get());
        forwardTimings.back().recordStop(computeStream_.get());
        markTransientRangeReusable(slot);
        if (chunkIndex + transientSlotCount < chunks.size()) {
          queueChunkUpload(chunkIndex + transientSlotCount, slot);
        }
        hostFloatScratch_.resize(chunkRows);
        dOutput_.copyToHostAsync(hostFloatScratch_.data(), chunkRows,
                                 computeStream_.get());
        CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
        for (size_t i = 0; i < chunkRows; ++i)
          y[static_cast<size_t>(chunk.rowBegin) + i] =
              static_cast<double>(hostFloatScratch_[i]);
      }
      noteCudaUploadWaitOnCompute(accumulateTimedCudaPairs(uploadWaitTimings),
                                  uploadWaitTimings.size());
      noteCudaCusparseForward(accumulateTimedCudaPairs(forwardTimings),
                              forwardTimings.size());
      return;
    }

    std::vector<TimedCudaStagePair> uploadWaitTimings;
    std::vector<TimedCudaStagePair> backwardTimings;
    uploadWaitTimings.reserve(chunks.size());
    backwardTimings.reserve(chunks.size());
    uploadDoubleAsync(dInput_, x, static_cast<size_t>(matrix.rows),
                      computeStream_.get());
    dAccumOutput_.allocate(static_cast<size_t>(matrix.cols));
    if (matrix.cols > 0) {
      CUDA_CHECK(cudaMemsetAsync(dAccumOutput_.get(), 0,
                                 static_cast<size_t>(matrix.cols) *
                                     sizeof(float),
                                 computeStream_.get()));
    }

    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
      const RowChunkRange &chunk = chunks[chunkIndex];
      const size_t slot =
          transientSlotCount > 0 ? chunkIndex % transientSlotCount : 0;
      uploadWaitTimings.emplace_back();
      const CachedMatrix &cached =
          waitTransientRangeUpload(slot, &uploadWaitTimings.back());

      CuSparseVecGuard vecX;
      CuSparseVecGuard vecY;
      vecX.create(cached.rows,
                  dInput_.get() + static_cast<size_t>(chunk.rowBegin));
      vecY.create(cached.cols, dAccumOutput_.get());
      backwardTimings.emplace_back();
      backwardTimings.back().recordStart(computeStream_.get());
      executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 1.0f,
                        computeStream_.get());
      backwardTimings.back().recordStop(computeStream_.get());
      markTransientRangeReusable(slot);
      if (chunkIndex + transientSlotCount < chunks.size()) {
        queueChunkUpload(chunkIndex + transientSlotCount, slot);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    noteCudaUploadWaitOnCompute(accumulateTimedCudaPairs(uploadWaitTimings),
                                uploadWaitTimings.size());
    noteCudaCusparseBackward(accumulateTimedCudaPairs(backwardTimings),
                             backwardTimings.size());
    downloadDouble(dAccumOutput_, y, static_cast<size_t>(matrix.cols));
  }

  void executeChunkedFullMultiBedSpMV(
      const io::CSRView &matrix, const double *x, double *y,
      const std::vector<long long> &bedOffsets, int bedCount,
      cusparseOperation_t operation, int threads) {
    ensureCudaRuntimeAvailable();
    if (bedCount <= 0 || static_cast<size_t>(bedCount) != bedOffsets.size()) {
      throw std::runtime_error(
          "CUDA full multi-bed chunked SpMV received inconsistent bed offsets.");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t nnzBudget = selectCudaChunkNnzBudget();
    PreparedChunkedHostCSR *prepared =
        getPreparedContiguousHostChunks(matrix, nnzBudget, threads);
    const auto &chunks =
        prepared ? prepared->ranges : getContiguousRowChunks(matrix, nnzBudget);
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
    const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
    const size_t baseCols = static_cast<size_t>(matrix.cols);
    const size_t totalRows = rowsPerBed * static_cast<size_t>(bedCount);
    const size_t outputSize = static_cast<size_t>(bedOffsets.back()) + baseCols;
    const auto queueChunkUpload = [&](size_t chunkIndex, size_t slot) {
      const RowChunkRange &chunk = chunks[chunkIndex];
      if (prepared != nullptr && prepared->ready[chunkIndex]) {
        touchPreparedChunk(*prepared, chunkIndex);
        queueTransientRangeUpload(
            slot, matrixKey, prepared->chunks[chunkIndex]);
        return;
      }
      waitTransientPinnedScratchReusable(slot);
      buildContiguousSubsetCSR(matrix, chunk.rowBegin, chunk.rowEnd, threads,
                               transientRangePinnedScratch_[slot]);
      queueTransientRangeUpload(slot, matrixKey,
                                transientRangePinnedScratch_[slot]);
    };

    if (!chunks.empty()) {
      queueChunkUpload(0, 0);
      if (chunks.size() > 1) {
        queueChunkUpload(1, 1);
      }
    }

    if (operation == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      uploadDoubleAsync(dInput_, x, outputSize, computeStream_.get());
      dFullOutput_.allocate(totalRows);

      for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        const size_t slot = chunkIndex % transientRangeCache_.size();
        const CachedMatrix &cached = waitTransientRangeUpload(slot);
        const size_t chunkRows =
            static_cast<size_t>(chunk.rowEnd - chunk.rowBegin);

        for (int bed = 0; bed < bedCount; ++bed) {
          const size_t colOffset =
              static_cast<size_t>(bedOffsets[static_cast<size_t>(bed)]);
          const size_t rowOffset =
              static_cast<size_t>(bed) * rowsPerBed +
              static_cast<size_t>(chunk.rowBegin);

          CuSparseVecGuard vecX;
          CuSparseVecGuard vecY;
          vecX.create(cached.cols, dInput_.get() + colOffset);
          vecY.create(cached.rows, dFullOutput_.get() + rowOffset);
          executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f,
                            computeStream_.get());
        }
        markTransientRangeReusable(slot);
        if (chunkIndex + transientRangeCache_.size() < chunks.size()) {
          queueChunkUpload(chunkIndex + transientRangeCache_.size(), slot);
        }
      }

      CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
      downloadDouble(dFullOutput_, y, totalRows);
      return;
    }

    uploadDoubleAsync(dInput_, x, totalRows, computeStream_.get());
    dAccumOutput_.allocate(outputSize);
    if (outputSize > 0) {
      CUDA_CHECK(cudaMemsetAsync(dAccumOutput_.get(), 0,
                                 outputSize * sizeof(float),
                                 computeStream_.get()));
    }

    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
      const RowChunkRange &chunk = chunks[chunkIndex];
      const size_t slot = chunkIndex % transientRangeCache_.size();
      const CachedMatrix &cached = waitTransientRangeUpload(slot);

      for (int bed = 0; bed < bedCount; ++bed) {
        const size_t rowOffset =
            static_cast<size_t>(bed) * rowsPerBed +
            static_cast<size_t>(chunk.rowBegin);
        const size_t colOffset =
            static_cast<size_t>(bedOffsets[static_cast<size_t>(bed)]);

        CuSparseVecGuard vecX;
        CuSparseVecGuard vecY;
        vecX.create(cached.rows, dInput_.get() + rowOffset);
        vecY.create(cached.cols, dAccumOutput_.get() + colOffset);
        executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 1.0f,
                          computeStream_.get());
      }
      markTransientRangeReusable(slot);
      if (chunkIndex + transientRangeCache_.size() < chunks.size()) {
        queueChunkUpload(chunkIndex + transientRangeCache_.size(), slot);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    downloadDouble(dAccumOutput_, y, outputSize);
  }

  void executeChunkedFullMultiBedAttenSpMV(
      const io::CSRView &matrix, const double *x, double *y,
      const std::vector<long long> &bedOffsets,
      const std::vector<std::vector<uint16_t>> &bedAttenFactors,
      int numAngles, int bedCount, cusparseOperation_t operation,
      int threads) {
    ensureCudaRuntimeAvailable();
    if (bedCount <= 0 || static_cast<size_t>(bedCount) != bedOffsets.size() ||
        static_cast<size_t>(bedCount) != bedAttenFactors.size()) {
      throw std::runtime_error(
          "CUDA full multi-bed attenuated chunked SpMV received inconsistent bed data.");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t nnzBudget = selectCudaChunkNnzBudget();
    std::vector<PreparedChunkedHostCSR *> preparedPerBed(
        static_cast<size_t>(bedCount), nullptr);
    const uint64_t perBedPreparedHostCacheBudgetBytes =
        selectPreparedChunkHostCacheBudgetBytes(
            matrix, static_cast<size_t>(bedCount));
    for (int bed = 0; bed < bedCount; ++bed) {
      preparedPerBed[static_cast<size_t>(bed)] = getPreparedAttenuatedHostChunks(
          matrix, bedAttenFactors[static_cast<size_t>(bed)], numAngles,
          nnzBudget, threads, false, perBedPreparedHostCacheBudgetBytes);
    }
    const auto &chunks = getContiguousRowChunks(matrix, nnzBudget);
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
    const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
    const size_t baseCols = static_cast<size_t>(matrix.cols);
    const size_t totalRows = rowsPerBed * static_cast<size_t>(bedCount);
    const size_t outputSize = static_cast<size_t>(bedOffsets.back()) + baseCols;

    struct AttenTask {
      size_t chunkIndex = 0;
      int bed = 0;
    };
    std::vector<AttenTask> tasks;
    tasks.reserve(chunks.size() * static_cast<size_t>(bedCount));
    for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
      for (int bed = 0; bed < bedCount; ++bed) {
        tasks.push_back(AttenTask{chunkIndex, bed});
      }
    }

    const auto queueTaskUpload = [&](size_t taskIndex, size_t slot) {
      const AttenTask &task = tasks[taskIndex];
      const RowChunkRange &chunk = chunks[task.chunkIndex];
      PreparedChunkedHostCSR *prepared =
          preparedPerBed[static_cast<size_t>(task.bed)];
      if (prepared != nullptr && prepared->ready[task.chunkIndex]) {
        touchPreparedChunk(*prepared, task.chunkIndex);
        queueTransientRangeUpload(
            slot, matrixKey,
            prepared->chunks[task.chunkIndex]);
        return;
      }
      waitTransientPinnedScratchReusable(slot);
      buildAttenuatedContiguousSubsetCSR(
          matrix, bedAttenFactors[static_cast<size_t>(task.bed)], numAngles,
          chunk.rowBegin, chunk.rowEnd, threads,
          transientRangePinnedScratch_[slot]);
      queueTransientRangeUpload(slot, matrixKey,
                                transientRangePinnedScratch_[slot]);
    };

    if (!tasks.empty()) {
      queueTaskUpload(0, 0);
      if (tasks.size() > 1) {
        queueTaskUpload(1, 1);
      }
    }

    if (operation == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      uploadDoubleAsync(dInput_, x, outputSize, computeStream_.get());
      dFullOutput_.allocate(totalRows);

      for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
        const AttenTask &task = tasks[taskIndex];
        const RowChunkRange &chunk = chunks[task.chunkIndex];
        const size_t slot = taskIndex % transientRangeCache_.size();
        const CachedMatrix &cached = waitTransientRangeUpload(slot);
        const size_t colOffset =
            static_cast<size_t>(bedOffsets[static_cast<size_t>(task.bed)]);
        const size_t rowOffset =
            static_cast<size_t>(task.bed) * rowsPerBed +
            static_cast<size_t>(chunk.rowBegin);

        CuSparseVecGuard vecX;
        CuSparseVecGuard vecY;
        vecX.create(cached.cols, dInput_.get() + colOffset);
        vecY.create(cached.rows, dFullOutput_.get() + rowOffset);
        executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 0.0f,
                          computeStream_.get());

        markTransientRangeReusable(slot);
        if (taskIndex + transientRangeCache_.size() < tasks.size()) {
          queueTaskUpload(taskIndex + transientRangeCache_.size(), slot);
        }
      }

      CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
      downloadDouble(dFullOutput_, y, totalRows);
      return;
    }

    uploadDoubleAsync(dInput_, x, totalRows, computeStream_.get());
    dAccumOutput_.allocate(outputSize);
    if (outputSize > 0) {
      CUDA_CHECK(cudaMemsetAsync(dAccumOutput_.get(), 0,
                                 outputSize * sizeof(float),
                                 computeStream_.get()));
    }

    for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
      const AttenTask &task = tasks[taskIndex];
      const RowChunkRange &chunk = chunks[task.chunkIndex];
      const size_t slot = taskIndex % transientRangeCache_.size();
      const CachedMatrix &cached = waitTransientRangeUpload(slot);
      const size_t rowOffset =
          static_cast<size_t>(task.bed) * rowsPerBed +
          static_cast<size_t>(chunk.rowBegin);
      const size_t colOffset =
          static_cast<size_t>(bedOffsets[static_cast<size_t>(task.bed)]);

      CuSparseVecGuard vecX;
      CuSparseVecGuard vecY;
      vecX.create(cached.rows, dInput_.get() + rowOffset);
      vecY.create(cached.cols, dAccumOutput_.get() + colOffset);
      executeSpMVLocked(cached, operation, vecX.get(), vecY.get(), 1.0f,
                        computeStream_.get());

      markTransientRangeReusable(slot);
      if (taskIndex + transientRangeCache_.size() < tasks.size()) {
        queueTaskUpload(taskIndex + transientRangeCache_.size(), slot);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    downloadDouble(dAccumOutput_, y, outputSize);
  }

  void runFullSubsetEmNumer(const io::CSRView &matrix,
                            const std::vector<double> &projections,
                            const double *image, int bedCount,
                            const std::vector<long long> *bedOffsets,
                            const std::vector<std::vector<uint16_t>> *bedAttenFactors,
                            int numAngles, int threads, double *numer) {
    ensureCudaRuntimeAvailable();
    if (image == nullptr || numer == nullptr) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper requires valid input/output buffers.");
    }
    if (bedCount <= 0) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper requires a positive bed count.");
    }
    if (bedOffsets != nullptr &&
        bedOffsets->size() != static_cast<size_t>(bedCount)) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper received inconsistent bed offsets.");
    }
    if (bedAttenFactors != nullptr && !bedAttenFactors->empty() &&
        (numAngles <= 0 ||
         bedAttenFactors->size() != static_cast<size_t>(bedCount))) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper received inconsistent attenuation inputs.");
    }

    const size_t rowsPerBed = static_cast<size_t>(matrix.rows);
    const size_t baseCols = static_cast<size_t>(matrix.cols);
    const size_t totalRows = rowsPerBed * static_cast<size_t>(bedCount);
    if (projections.size() != totalRows) {
      throw std::runtime_error(
          "CUDA fused EM full-subset helper received a projection vector with unexpected size.");
    }

    const auto resolveColOffset = [&](int bed) -> size_t {
      if (bedOffsets != nullptr && !bedOffsets->empty()) {
        return static_cast<size_t>((*bedOffsets)[static_cast<size_t>(bed)]);
      }
      return static_cast<size_t>(bed) * baseCols;
    };
    const size_t outputSize =
        resolveColOffset(bedCount - 1) + baseCols;

    std::lock_guard<std::mutex> lock(mutex_);
    // P3-D: Reuse the device cache's nnzBudget when available to keep
    // chunk boundaries consistent with cached device-side data.
    const MatrixCacheKey matrixKey = makeCacheKey(matrix);
    const uint64_t nnzBudget =
        (deviceChunkCache_.initialized &&
         deviceChunkCache_.matrixKey == matrixKey)
            ? deviceChunkCache_.nnzBudget
            : selectCudaChunkNnzBudget();
    const bool useAttenuation =
        bedAttenFactors != nullptr && !bedAttenFactors->empty();
    PreparedChunkedHostCSR *preparedHost = nullptr;
    if (!useAttenuation) {
      preparedHost = getPreparedContiguousHostChunks(matrix, nnzBudget, threads);
    }
    std::vector<PreparedChunkedHostCSR *> preparedPerBed;
    if (useAttenuation) {
      preparedPerBed.resize(static_cast<size_t>(bedCount), nullptr);
      const uint64_t perBedPreparedHostCacheBudgetBytes =
          selectPreparedChunkHostCacheBudgetBytes(
              matrix, static_cast<size_t>(bedCount));
      for (int bed = 0; bed < bedCount; ++bed) {
        preparedPerBed[static_cast<size_t>(bed)] =
            getPreparedAttenuatedHostChunks(
                matrix, (*bedAttenFactors)[static_cast<size_t>(bed)], numAngles,
                nnzBudget, threads, false,
                perBedPreparedHostCacheBudgetBytes);
      }
    }
    const auto &chunks = getContiguousRowChunks(matrix, nnzBudget);
    std::vector<TimedCudaStagePair> uploadWaitTimings;
    std::vector<TimedCudaStagePair> bedCycleTimings;
    std::vector<TimedCudaStagePair> forwardTimings;
    std::vector<TimedCudaStagePair> ratioTimings;
    std::vector<TimedCudaStagePair> backwardTimings;
    if (!useAttenuation) {
      const size_t opCount = chunks.size() * static_cast<size_t>(bedCount);
      uploadWaitTimings.reserve(chunks.size());
      bedCycleTimings.reserve(opCount);
      forwardTimings.reserve(opCount);
      ratioTimings.reserve(opCount);
      backwardTimings.reserve(opCount);
    }

    uploadDoubleAsync(dInput_, image, outputSize, computeStream_.get());
    uploadDoubleAsync(dProjectionInput_, projections.data(), projections.size(),
                                        computeStream_.get());
    dAccumOutput_.allocate(outputSize);
    if (outputSize > 0) {
      CUDA_CHECK(cudaMemsetAsync(dAccumOutput_.get(), 0,
                                 outputSize * sizeof(float),
                                 computeStream_.get()));
    }
    resetTransientRangeState();

    constexpr float eps = 1e-12f;
    if (!useAttenuation) {
      const size_t transientSlotCount = transientRangeCache_.size();
      const auto tryQueueChunkUploadToSlot = [&](size_t chunkIndex,
                                                  size_t slot) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        if (slot >= transientRangeCache_.size()) {
          return false;
        }
        if (transientRangeSlotUsed_[slot] &&
            !transientRangeReusableReady_[slot]) {
          return false;
        }
        // Use prepared cache for resident chunks — avoids rebuilding.
        // Non-resident chunks go through the transient scratch path
        // to prevent LRU thrashing (sequential access would evict
        // chunks needed in the next iteration).
        if (preparedHost != nullptr &&
            preparedHost->ready[chunkIndex]) {
          touchPreparedChunk(*preparedHost, chunkIndex);
          queueTransientRangeUpload(
              slot, matrixKey,
              preparedHost->chunks[chunkIndex]);
          return true;
        }
        const auto buildStart = std::chrono::steady_clock::now();
        waitTransientPinnedScratchReusable(slot);
        buildContiguousSubsetCSR(matrix, chunk.rowBegin, chunk.rowEnd, threads,
                                 transientRangePinnedScratch_[slot]);
        noteCudaChunkBuild(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          buildStart)
                .count());
        queueTransientRangeUpload(slot, matrixKey,
                                  transientRangePinnedScratch_[slot]);
        return true;
      };
      const auto runChunkCompute = [&](size_t chunkIndex,
                                       const CachedMatrix &cached) {
        const RowChunkRange &chunk = chunks[chunkIndex];
        const size_t chunkRows =
            static_cast<size_t>(chunk.rowEnd - chunk.rowBegin);
        dOutput_.allocate(chunkRows);
        dChunkInput_.allocate(chunkRows);

        for (int bed = 0; bed < bedCount; ++bed) {
          bedCycleTimings.emplace_back();
          bedCycleTimings.back().recordStart(computeStream_.get());
          const size_t colOffset = resolveColOffset(bed);
          const size_t projOffset =
              static_cast<size_t>(bed) * rowsPerBed +
              static_cast<size_t>(chunk.rowBegin);

          CuSparseVecGuard vecXForward;
          CuSparseVecGuard vecYForward;
          vecXForward.create(cached.cols, dInput_.get() + colOffset);
          vecYForward.create(cached.rows, dOutput_.get());
          forwardTimings.emplace_back();
          forwardTimings.back().recordStart(computeStream_.get());
          executeSpMVLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            vecXForward.get(), vecYForward.get(), 0.0f,
                            computeStream_.get());
          forwardTimings.back().recordStop(computeStream_.get());

          ratioTimings.emplace_back();
          ratioTimings.back().recordStart(computeStream_.get());
          launchProjDomainRatioKernel(
              dOutput_.get(), dProjectionInput_.get(), projOffset,
              dChunkInput_.get(), chunkRows, eps, computeStream_.get());
          ratioTimings.back().recordStop(computeStream_.get());

          CuSparseVecGuard vecXTranspose;
          CuSparseVecGuard vecYTranspose;
          vecXTranspose.create(cached.rows, dChunkInput_.get());
          vecYTranspose.create(cached.cols, dAccumOutput_.get() + colOffset);
          backwardTimings.emplace_back();
          backwardTimings.back().recordStart(computeStream_.get());
          executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                            vecXTranspose.get(), vecYTranspose.get(), 1.0f,
                            computeStream_.get());
          backwardTimings.back().recordStop(computeStream_.get());
          bedCycleTimings.back().recordStop(computeStream_.get());
        }
      };

      // P3-C+D: Build interleaved schedule that overlaps cached-chunk
      // compute with uncached-chunk uploads.  When device-cached and
      // transient chunks coexist, processing them in original order
      // (all cached first) wastes the GPU compute time of cached chunks
      // that could hide upload latency.  Instead, we interleave:
      //   [prime U0] [C0] [wait U0, prime U1] [U0] [C1] [wait U1, ...]
      // Each cached chunk's compute time (~0.043s) overlaps with the
      // next uncached chunk's upload (~0.058s), hiding most of the
      // transfer cost.
      std::vector<size_t> cachedChunks;
      std::vector<size_t> uncachedChunks;
      cachedChunks.reserve(chunks.size());
      uncachedChunks.reserve(chunks.size());
      for (size_t ci = 0; ci < chunks.size(); ++ci) {
        if (deviceChunkCache_.isHit(ci, matrixKey)) {
          cachedChunks.push_back(ci);
        } else {
          uncachedChunks.push_back(ci);
        }
      }

      // Build interleaved schedule: alternate uncached→cached so that
      // each cached-chunk compute overlaps with the next upload.
      // Pattern: U, C, U, C, ... then remaining of whichever is longer.
      std::vector<size_t> schedule;
      schedule.reserve(chunks.size());
      size_t ci = 0, ui = 0;
      while (ui < uncachedChunks.size() || ci < cachedChunks.size()) {
        if (ui < uncachedChunks.size()) schedule.push_back(uncachedChunks[ui++]);
        if (ci < cachedChunks.size())   schedule.push_back(cachedChunks[ci++]);
      }

      // Prime transient slots with first uncached chunks from schedule.
      size_t nextUncachedIdx = 0;  // index into uncachedChunks
      while (nextUncachedIdx < uncachedChunks.size() &&
             nextUncachedIdx < transientSlotCount) {
        const size_t slot = nextUncachedIdx % transientSlotCount;
        const bool queued =
            tryQueueChunkUploadToSlot(uncachedChunks[nextUncachedIdx], slot);
        if (!queued) {
          throw std::runtime_error(
              "CUDA chunk upload failed to initialize stable slots.");
        }
        ++nextUncachedIdx;
      }

      // Process chunks in interleaved order.
      size_t transientCounter = 0;  // counts uncached chunks processed
      for (const size_t chunkIndex : schedule) {
        if (deviceChunkCache_.isHit(chunkIndex, matrixKey)) {
          // Device cache hit — compute directly, no upload needed.
          // GPU compute here overlaps with any in-flight upload.
          ++g_cudaChunkStageTiming.deviceCacheHitCalls;
          runChunkCompute(chunkIndex, *deviceChunkCache_.slots[chunkIndex]);
        } else {
          // Transient upload path.
          const size_t slot = transientCounter % transientSlotCount;
          uploadWaitTimings.emplace_back();
          const CachedMatrix &cached =
              waitTransientRangeUpload(slot, &uploadWaitTimings.back());
          runChunkCompute(chunkIndex, cached);

          markTransientRangeReusable(slot);
          ++transientCounter;

          // Queue next uncached chunk if available.
          if (nextUncachedIdx < uncachedChunks.size()) {
            const size_t nextSlot = nextUncachedIdx % transientSlotCount;
            const bool queued = tryQueueChunkUploadToSlot(
                uncachedChunks[nextUncachedIdx], nextSlot);
            if (!queued) {
              throw std::runtime_error(
                  "CUDA chunk upload lost stable slot sequencing.");
            }
            ++nextUncachedIdx;
          }
        }
      }
    } else {
      struct AttenTask {
        size_t chunkIndex = 0;
        int bed = 0;
      };
      std::vector<AttenTask> tasks;
      tasks.reserve(chunks.size() * static_cast<size_t>(bedCount));
      for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        for (int bed = 0; bed < bedCount; ++bed) {
          tasks.push_back(AttenTask{chunkIndex, bed});
        }
      }

      uploadWaitTimings.reserve(tasks.size());
      bedCycleTimings.reserve(tasks.size());
      forwardTimings.reserve(tasks.size());
      ratioTimings.reserve(tasks.size());
      backwardTimings.reserve(tasks.size());
      const auto tryQueueTaskUpload = [&](size_t taskIndex) {
        const size_t slot = taskIndex % transientRangeCache_.size();
        const AttenTask &task = tasks[taskIndex];
        const RowChunkRange &chunk = chunks[task.chunkIndex];
        if (transientRangeSlotUsed_[slot] &&
            !transientRangeReusableReady_[slot]) {
          return false;
        }
        // Use prepared cache for resident chunks; otherwise build
        // into transient scratch to avoid LRU thrashing.
        PreparedChunkedHostCSR *preparedAtten =
            preparedPerBed[static_cast<size_t>(task.bed)];
        if (preparedAtten != nullptr &&
            preparedAtten->ready[task.chunkIndex]) {
          touchPreparedChunk(*preparedAtten, task.chunkIndex);
          queueTransientRangeUpload(
              slot, matrixKey,
              preparedAtten->chunks[task.chunkIndex]);
          return true;
        }
        const auto buildStart = std::chrono::steady_clock::now();
        waitTransientPinnedScratchReusable(slot);
        buildAttenuatedContiguousSubsetCSR(
            matrix, (*bedAttenFactors)[static_cast<size_t>(task.bed)],
            numAngles, chunk.rowBegin, chunk.rowEnd, threads,
            transientRangePinnedScratch_[slot]);
        noteCudaChunkBuild(
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          buildStart)
                .count());
        queueTransientRangeUpload(slot, matrixKey,
                                  transientRangePinnedScratch_[slot]);
        return true;
      };

      size_t nextTaskToQueue = 0;
      while (nextTaskToQueue < tasks.size() &&
             nextTaskToQueue < transientRangeCache_.size()) {
        const bool queued = tryQueueTaskUpload(nextTaskToQueue);
        if (!queued) {
          throw std::runtime_error(
              "CUDA attenuated chunk upload failed to initialize stable slots.");
        }
        ++nextTaskToQueue;
      }

      for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
        const AttenTask &task = tasks[taskIndex];
        const RowChunkRange &chunk = chunks[task.chunkIndex];
        const size_t slot = taskIndex % transientRangeCache_.size();
        uploadWaitTimings.emplace_back();
        const CachedMatrix &cached =
            waitTransientRangeUpload(slot, &uploadWaitTimings.back());
        const size_t chunkRows =
            static_cast<size_t>(chunk.rowEnd - chunk.rowBegin);
        dOutput_.allocate(chunkRows);
        dChunkInput_.allocate(chunkRows);

        bedCycleTimings.emplace_back();
        bedCycleTimings.back().recordStart(computeStream_.get());
        const size_t colOffset = resolveColOffset(task.bed);
        const size_t projOffset =
            static_cast<size_t>(task.bed) * rowsPerBed +
            static_cast<size_t>(chunk.rowBegin);

        CuSparseVecGuard vecXForward;
        CuSparseVecGuard vecYForward;
        vecXForward.create(cached.cols, dInput_.get() + colOffset);
        vecYForward.create(cached.rows, dOutput_.get());
        forwardTimings.emplace_back();
        forwardTimings.back().recordStart(computeStream_.get());
        executeSpMVLocked(cached, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          vecXForward.get(), vecYForward.get(), 0.0f,
                          computeStream_.get());
        forwardTimings.back().recordStop(computeStream_.get());

        ratioTimings.emplace_back();
        ratioTimings.back().recordStart(computeStream_.get());
        launchProjDomainRatioKernel(dOutput_.get(), dProjectionInput_.get(),
                                    projOffset, dChunkInput_.get(), chunkRows,
                                    eps, computeStream_.get());
        ratioTimings.back().recordStop(computeStream_.get());

        CuSparseVecGuard vecXTranspose;
        CuSparseVecGuard vecYTranspose;
        vecXTranspose.create(cached.rows, dChunkInput_.get());
        vecYTranspose.create(cached.cols, dAccumOutput_.get() + colOffset);
        backwardTimings.emplace_back();
        backwardTimings.back().recordStart(computeStream_.get());
        executeSpMVLocked(cached, CUSPARSE_OPERATION_TRANSPOSE,
                          vecXTranspose.get(), vecYTranspose.get(), 1.0f,
                          computeStream_.get());
        backwardTimings.back().recordStop(computeStream_.get());
        bedCycleTimings.back().recordStop(computeStream_.get());

        markTransientRangeReusable(slot);
        if (nextTaskToQueue < tasks.size()) {
          const bool queued = tryQueueTaskUpload(nextTaskToQueue);
          if (!queued) {
            throw std::runtime_error(
                "CUDA attenuated chunk upload lost stable slot sequencing.");
          }
          ++nextTaskToQueue;
        }
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream_.get()));
    noteCudaUploadWaitOnCompute(accumulateTimedCudaPairs(uploadWaitTimings),
                                uploadWaitTimings.size());
    noteCudaFusedBedCycle(accumulateTimedCudaPairs(bedCycleTimings),
                          bedCycleTimings.size());
    noteCudaCusparseForward(accumulateTimedCudaPairs(forwardTimings),
                            forwardTimings.size());
    noteCudaRatio(accumulateTimedCudaPairs(ratioTimings), ratioTimings.size());
    noteCudaCusparseBackward(accumulateTimedCudaPairs(backwardTimings),
                             backwardTimings.size());
    downloadDouble(dAccumOutput_, numer, outputSize);
  }

  void runFullSubsetBsrem(const io::CSRView &matrix,
                          const std::vector<double> &projections,
                          const double *image, int bedCount,
                          const std::vector<long long> *bedOffsets,
                          const std::vector<std::vector<uint16_t>> *bedAttenFactors,
                          int numAngles, int threads, double *numer) {
    runFullSubsetEmNumer(matrix, projections, image, bedCount, bedOffsets,
                         bedAttenFactors, numAngles, threads, numer);
  }

private:
  size_t compactBackwardAccumulatorSlot(int accumulatorIndex) const {
    if (accumulatorIndex < 0 ||
        accumulatorIndex >= kCompactDeviceBackwardMaxAccumulators) {
      throw std::runtime_error(
          "compact device-backward accumulator index is out of range.");
    }
    return static_cast<size_t>(accumulatorIndex);
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
  std::array<CachedMatrix, 4> attenCache_;
  std::array<AttenCacheKey, 4> attenCacheKeys_{};
  std::array<bool, 4> attenCacheValid_{};
  size_t nextAttenCacheSlot_ = 0;
  HostCSRData attenHostScratch_;
  DeviceArray<float> dInput_;
  DeviceArray<float> dOutput_;
  DeviceArray<float> dProjectionInput_;
  std::array<DeviceArray<double>, spect::core::kMaxBeds>
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
  std::array<DeviceArray<double>, kCompactDeviceBackwardMaxAccumulators>
      dCompactBackwardAccum_;
  std::vector<float> hostFloatScratch_;
  std::array<const double *, spect::core::kMaxBeds> compactProjectionHostKey_{};
  std::array<size_t, spect::core::kMaxBeds> compactProjectionCount_{};
  std::array<size_t, kCompactDeviceBackwardMaxAccumulators>
      compactBackwardAccumSize_{};
  DeviceBuffer dBuffer_;
  // Compact CTAC: resident attenuation table plus explicit debug/reference
  // baked-values scratch. Default CTAC uses fused kernels above.
  std::array<DeviceArray<uint16_t>, spect::core::kMaxBeds> dAttenTable_;
  std::array<const void *, spect::core::kMaxBeds> attenTableHostPtr_{};
  std::array<size_t, spect::core::kMaxBeds> attenTableCount_{};
  std::array<DeviceArray<float>, spect::core::kMaxBeds> dBakedValues_;
  std::array<CuSparseMatGuard, spect::core::kMaxBeds> bakedMat_;
  std::array<MatrixCacheKey, spect::core::kMaxBeds> bakedMatrixKey_{};
  std::array<cuda::CompactAttenBakeParams, spect::core::kMaxBeds> bakedParams_{};
  std::array<const void *, spect::core::kMaxBeds> bakedAttPtr_{};
  std::array<const void *, spect::core::kMaxBeds> bakedRowPtrDev_{};
  std::array<const void *, spect::core::kMaxBeds> bakedColIdxDev_{};
  std::array<bool, spect::core::kMaxBeds> bakedValid_{};
  std::array<std::array<size_t, 2>, spect::core::kMaxBeds> bakedBufferSizes_{};
  std::array<std::array<bool, 2>, spect::core::kMaxBeds> bakedBufferSizeValid_{};
  CudaStreamGuard computeStream_;
  std::array<CudaStreamGuard, kTransientRangeSlotCount> transientUploadStreams_;
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
