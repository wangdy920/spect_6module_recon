// Device-only helper kernels. Compiled with nvcc --fatbin for sm_86/89/120.
static constexpr float kAttenUint16Scale = 65535.0f;

// CSRQ resident-matrix kernels (io/CompactChunkCSR.h, CompactChunkFormat::
// CsrqRunU16). One block per compact row, 256 threads. Runs are consumed in
// tiles of 256; each tile is expanded in shared memory to a per-nnz run id so
// threads stride over nnz (coalesced u16 value loads, contiguous x within a
// run). K = 8 right-hand sides (4 xy symmetries x 2 z branches). The forward
// gathers x as [voxel][8] floats (one 32-byte sector per nnz); the backward
// scatters into branch-major [8][voxel] so warp atomics coalesce.
#define CSRQ_K 8
#define CSRQ_BLOCK 256
#define CSRQ_RUN_LEN_BITS 6
#define CSRQ_MAX_RUN_LEN 64
#define CSRQ_RUN_TILE 256
#define CSRQ_TILE_MAX_NNZ (CSRQ_RUN_TILE * CSRQ_MAX_RUN_LEN)
static constexpr float kCsrqInvQScale = 1.0f / 65535.0f;

extern "C" __device__ __forceinline__ int mirror_index(int idx, int size) {
  if (size <= 1) {
    return 0;
  }
  while (idx < 0 || idx >= size) {
    if (idx < 0) {
      idx = -idx;
    } else {
      idx = 2 * size - 2 - idx;
    }
  }
  return idx;
}

// Poisson ratio r = p / max(yhat + b, eps) for one symmetry branch of a compact
// chunk. fullBackground (additive scatter estimate, same layout as the
// measurement) may be null; the check is per row, not per nnz.
extern "C" __global__ void compute_compact_ratio_chunk(
    const float *yhat, const double *fullProjection,
    const double *fullBackground,
    unsigned long long fullProjectionCount, int symmetryMode,
    int compactRowBase, int compactRaysPerAngle, int compactNumAngles,
    int detNx, int detNz, int compactDetZ, int fullNumAngles,
    int fullRaysPerAngle, int symId, int zMirror, float *ratio,
    unsigned long long count, double eps) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count || compactRaysPerAngle <= 0 || detNx <= 0) {
    return;
  }

  const int compactRow = compactRowBase + static_cast<int>(index);
  const int compactAngle = compactRow / compactRaysPerAngle;
  const int rr = compactRow - compactAngle * compactRaysPerAngle;
  const int ix = rr % detNx;
  const int izCompact = rr / detNx;
  bool valid = true;
  int fullAngle = 0;
  if (symmetryMode == 45) {
    const int q = fullNumAngles / 8;
    if ((symId & 1) == 0) {
      if (compactAngle >= q) {
        valid = false;
      } else {
        fullAngle = symId * q + compactAngle;
      }
    } else {
      if (compactAngle <= 0) {
        valid = false;
      } else {
        fullAngle = symId * q + (q - compactAngle);
      }
    }
  } else {
    fullAngle = compactAngle + symId * compactNumAngles;
  }

  int izFull = izCompact;
  if (zMirror != 0) {
    izFull = detNz - 1 - izCompact;
    if (izFull < compactDetZ) {
      valid = false;
    }
  }
  if (!valid || fullAngle < 0 || fullAngle >= fullNumAngles ||
      izFull < 0 || izFull >= detNz) {
    ratio[index] = 0.0f;
    return;
  }

  const unsigned long long fullIndex =
      static_cast<unsigned long long>(fullAngle) *
          static_cast<unsigned long long>(fullRaysPerAngle) +
      static_cast<unsigned long long>(izFull) *
          static_cast<unsigned long long>(detNx) +
      static_cast<unsigned long long>(ix);
  if (fullIndex >= fullProjectionCount) {
    ratio[index] = 0.0f;
    return;
  }
  double denom = static_cast<double>(yhat[index]);
  if (fullBackground != nullptr) {
    denom += fullBackground[fullIndex];
  }
  if (denom < eps) {
    denom = eps;
  }
  ratio[index] = static_cast<float>(fullProjection[fullIndex] / denom);
}

extern "C" __global__ void compute_compact_ratio_chunk_batch(
    const float *yhatBatch, const double *fullProjection,
    const double *fullBackground,
    unsigned long long fullProjectionCount, int symmetryMode,
    int compactRowBase, int compactRaysPerAngle, int compactNumAngles,
    int detNx, int detNz, int compactDetZ, int fullNumAngles,
    int fullRaysPerAngle, const int *symIds, const int *zMirrors,
    int batchSize, float *ratioBatch, unsigned long long rows, double eps) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      rows * static_cast<unsigned long long>(batchSize);
  if (index >= total || compactRaysPerAngle <= 0 || detNx <= 0) {
    return;
  }
  const int branch = static_cast<int>(index / rows);
  const unsigned long long rowInBranch = index % rows;
  const int symId = symIds[branch];
  const int zMirror = zMirrors[branch];

  const int compactRow = compactRowBase + static_cast<int>(rowInBranch);
  const int compactAngle = compactRow / compactRaysPerAngle;
  const int rr = compactRow - compactAngle * compactRaysPerAngle;
  const int ix = rr % detNx;
  const int izCompact = rr / detNx;
  bool valid = true;
  int fullAngle = 0;
  if (symmetryMode == 45) {
    const int q = fullNumAngles / 8;
    if ((symId & 1) == 0) {
      if (compactAngle >= q) {
        valid = false;
      } else {
        fullAngle = symId * q + compactAngle;
      }
    } else {
      if (compactAngle <= 0) {
        valid = false;
      } else {
        fullAngle = symId * q + (q - compactAngle);
      }
    }
  } else {
    if (compactAngle < 0 || compactAngle >= compactNumAngles) {
      valid = false;
    } else {
      fullAngle = compactAngle + symId * compactNumAngles;
    }
  }
  int izFull = izCompact;
  if (zMirror != 0) {
    izFull = detNz - 1 - izCompact;
    if (izFull < compactDetZ) {
      valid = false;
    }
  }
  if (!valid || fullAngle < 0 || fullAngle >= fullNumAngles ||
      izFull < 0 || izFull >= detNz) {
    ratioBatch[index] = 0.0f;
    return;
  }

  const unsigned long long fullIndex =
      static_cast<unsigned long long>(fullAngle) *
          static_cast<unsigned long long>(fullRaysPerAngle) +
      static_cast<unsigned long long>(izFull) *
          static_cast<unsigned long long>(detNx) +
      static_cast<unsigned long long>(ix);
  if (fullIndex >= fullProjectionCount) {
    ratioBatch[index] = 0.0f;
    return;
  }
  // Reuse the exact ratio row mapping to prepare a projection-activity mask.
  // Keep this as a double zero comparison: tiny positive counts stay active.
  if (yhatBatch == nullptr) {
    ratioBatch[index] = fullProjection[fullIndex] == 0.0 ? 0.0f : 1.0f;
    return;
  }
  double denom = static_cast<double>(yhatBatch[index]);
  if (fullBackground != nullptr) {
    denom += fullBackground[fullIndex];
  }
  if (denom < eps) {
    denom = eps;
  }
  ratioBatch[index] =
      static_cast<float>(fullProjection[fullIndex] / denom);
}

// Sensitivity input for the branch batch: ratioBatch[branch*rows + row] =
// rowWeights[fullIndex] (1 when rowWeights is null) where the (row, branch)
// maps to a full projection row, else 0. S = A^T W 1 then falls out of the
// ordinary batched backward.
extern "C" __global__ void compact_unit_ratio_chunk_batch(
    const double *rowWeights, unsigned long long fullProjectionCount,
    int symmetryMode, int compactRowBase, int compactRaysPerAngle,
    int compactNumAngles, int detNx, int detNz, int compactDetZ,
    int fullNumAngles, int fullRaysPerAngle, const int *symIds,
    const int *zMirrors, int batchSize, float *ratioBatch,
    unsigned long long rows) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      rows * static_cast<unsigned long long>(batchSize);
  if (index >= total || compactRaysPerAngle <= 0 || detNx <= 0) {
    return;
  }
  const int branch = static_cast<int>(index / rows);
  const unsigned long long rowInBranch = index % rows;
  const int symId = symIds[branch];
  const int zMirror = zMirrors[branch];

  const int compactRow = compactRowBase + static_cast<int>(rowInBranch);
  const int compactAngle = compactRow / compactRaysPerAngle;
  const int rr = compactRow - compactAngle * compactRaysPerAngle;
  const int ix = rr % detNx;
  const int izCompact = rr / detNx;
  bool valid = true;
  int fullAngle = 0;
  if (symmetryMode == 45) {
    const int q = fullNumAngles / 8;
    if ((symId & 1) == 0) {
      if (compactAngle >= q) {
        valid = false;
      } else {
        fullAngle = symId * q + compactAngle;
      }
    } else {
      if (compactAngle <= 0) {
        valid = false;
      } else {
        fullAngle = symId * q + (q - compactAngle);
      }
    }
  } else {
    if (compactAngle < 0 || compactAngle >= compactNumAngles) {
      valid = false;
    } else {
      fullAngle = compactAngle + symId * compactNumAngles;
    }
  }
  int izFull = izCompact;
  if (zMirror != 0) {
    izFull = detNz - 1 - izCompact;
    if (izFull < compactDetZ) {
      valid = false;
    }
  }
  if (!valid || fullAngle < 0 || fullAngle >= fullNumAngles || izFull < 0 ||
      izFull >= detNz) {
    ratioBatch[index] = 0.0f;
    return;
  }
  const unsigned long long fullIndex =
      static_cast<unsigned long long>(fullAngle) *
          static_cast<unsigned long long>(fullRaysPerAngle) +
      static_cast<unsigned long long>(izFull) *
          static_cast<unsigned long long>(detNx) +
      static_cast<unsigned long long>(ix);
  if (fullIndex >= fullProjectionCount) {
    ratioBatch[index] = 0.0f;
    return;
  }
  ratioBatch[index] = rowWeights == nullptr
                          ? 1.0f
                          : static_cast<float>(rowWeights[fullIndex]);
}

extern "C" __global__ void compact_gather_bed_double_to_float(
    const double *globalImage, unsigned long long globalCount,
    float *bedImage, int nx, int ny, int nz, int globalNz, int zStart,
    int flipLocalZ, unsigned long long nvox) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= nvox || nx <= 0 || ny <= 0 || nz <= 0 || globalNz <= 0) {
    return;
  }
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const int localZ = static_cast<int>(index / plane);
  const unsigned long long xy = index % plane;
  const int globalZ =
      zStart + (flipLocalZ != 0 ? nz - 1 - localZ : localZ);
  const unsigned long long globalIndex =
      static_cast<unsigned long long>(globalZ) * plane + xy;
  bedImage[index] =
      globalZ >= 0 && globalZ < globalNz && globalIndex < globalCount
          ? static_cast<float>(globalImage[globalIndex])
          : 0.0f;
}

extern "C" __global__ void compact_convert_double_to_float(
    const double *input, float *output, unsigned long long count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = static_cast<float>(input[index]);
  }
}

extern "C" __global__ void compact_accumulate_bed_float_to_global_double(
    const float *bedImage, double *globalImage, unsigned long long globalCount,
    int nx, int ny, int nz, int globalNz, int zStart, int flipLocalZ,
    unsigned long long nvox) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= nvox || nx <= 0 || ny <= 0 || nz <= 0 || globalNz <= 0) {
    return;
  }
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const int localZ = static_cast<int>(index / plane);
  const unsigned long long xy = index % plane;
  const int globalZ =
      zStart + (flipLocalZ != 0 ? nz - 1 - localZ : localZ);
  const unsigned long long globalIndex =
      static_cast<unsigned long long>(globalZ) * plane + xy;
  if (globalZ >= 0 && globalZ < globalNz && globalIndex < globalCount) {
    globalImage[globalIndex] += static_cast<double>(bedImage[index]);
  }
}

extern "C" __global__ void compact_mlem_update(
    double *image, const double *numer, const double *denom,
    const unsigned char *mask, int hasMask, unsigned long long count,
    unsigned int *flags, double eps) {
  __shared__ unsigned int blockFlags;
  if (threadIdx.x == 0) blockFlags = 0;
  __syncthreads();

  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned int localFlags = 0;
  if (index < count) {
    const double n = numer[index];
    if (!isfinite(n)) localFlags |= 1U;
    else if (n > 0.0) localFlags |= 2U;
    else if (n < 0.0) localFlags |= 4U;

    double value = 0.0;
    if ((hasMask == 0 || mask[index] != 0) && denom[index] > 0.0) {
      const double d = denom[index] > eps ? denom[index] : eps;
      value = image[index] * (n / d);
      if (value < 0.0) value = 0.0;
    }
    image[index] = value;
    if (!isfinite(value)) localFlags |= 8U;
    else if (value != 0.0) localFlags |= 16U;
  }
  if (localFlags != 0) atomicOr(&blockFlags, localFlags);
  __syncthreads();
  if (threadIdx.x == 0 && blockFlags != 0) atomicOr(flags, blockFlags);
}

// BSREM curvature surrogate D_reg = |grad R| / max(x, eps) (the host
// bsremUseCurvDenom heuristic), refreshed only on the iterations the host
// asks for so a frozen curvature stays frozen.
extern "C" __global__ void compact_bsrem_curvature(
    const double *image, const double *gradient, double eps, double *curvature,
    unsigned long long count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const double base = image[index] > eps ? image[index] : eps;
  curvature[index] = fabs(gradient[index]) / base;
}

// BSREM-II relative update, one voxel per thread, same arithmetic as the host
// applyBsremUpdate (subset = 1, so the subset and global sensitivities are the
// same array):
//   lambda = beta * kappa;  v = (q - S - lambda * r) / (S + lambda * max(c,0) + eps)
//   u = clip(alpha * v);     x <- max(x, 1e-12) * (1 + u), floored per nonneg
// gradient / curvature / kappa / mask may be null. Flags share the MLEM
// layout: 1 non-finite numerator, 2 positive numerator seen, 4 negative
// numerator seen, 8 non-finite image, 16 nonzero image.
extern "C" __global__ void compact_bsrem_update(
    double *image, const double *numer, const double *denom,
    const double *gradient, const double *curvature, const double *kappa,
    const unsigned char *mask, int hasMask, unsigned long long count,
    double alpha, double beta, double denomEps, double sMin, int clip,
    double clipMin, double clipMax, int nonneg, int zeroLowSensitivity,
    unsigned int *flags) {
  __shared__ unsigned int blockFlags;
  if (threadIdx.x == 0) blockFlags = 0;
  __syncthreads();

  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned int localFlags = 0;
  if (index < count) {
    const double n = numer[index];
    if (!isfinite(n)) localFlags |= 1U;
    else if (n > 0.0) localFlags |= 2U;
    else if (n < 0.0) localFlags |= 4U;

    const double imageEps = 1e-12;
    double value = image[index];
    if (hasMask != 0 && mask[index] == 0) {
      value = 0.0;
    } else {
      const double s = denom[index];
      if (!(s > sMin)) {
        if (zeroLowSensitivity != 0) value = 0.0;
      } else {
        const double k = kappa != nullptr ? kappa[index] : 1.0;
        const double lambda = beta * k;
        const double r = gradient != nullptr ? gradient[index] : 0.0;
        double c = curvature != nullptr ? curvature[index] : 0.0;
        if (c < 0.0) c = 0.0;
        const double v = (n - s - lambda * r) / (s + lambda * c + denomEps);
        double u = alpha * v;
        if (clip != 0) {
          if (u > clipMax) u = clipMax;
          if (u < clipMin) u = clipMin;
        }
        double f = value < imageEps ? imageEps : value;
        f = f * (1.0 + u);
        if (f < imageEps) f = nonneg != 0 ? 0.0 : imageEps;
        value = f;
      }
    }
    image[index] = value;
    if (!isfinite(value)) localFlags |= 8U;
    else if (value != 0.0) localFlags |= 16U;
  }
  if (localFlags != 0) atomicOr(&blockFlags, localFlags);
  __syncthreads();
  if (threadIdx.x == 0 && blockFlags != 0) atomicOr(flags, blockFlags);
}

extern "C" __global__ void compact_transform_forward_batch(
    const float *bedImage, float *rotatedBatch, unsigned long long nvox,
    int branchCount, int zBranchCount, int symmetryMode,
    int nx, int ny, int nz) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      nvox * static_cast<unsigned long long>(branchCount);
  if (index >= total || branchCount <= 0 || zBranchCount <= 0 ||
      nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const int branch = static_cast<int>(index / nvox);
  const unsigned long long voxel = index % nvox;
  const int symId = branch / zBranchCount;
  const int zMirror = branch % zBranchCount != 0;
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const int iz = static_cast<int>(voxel / plane);
  const unsigned long long rem =
      voxel - static_cast<unsigned long long>(iz) * plane;
  const int iy = static_cast<int>(rem / static_cast<unsigned long long>(nx));
  const int ix = static_cast<int>(
      rem - static_cast<unsigned long long>(iy) *
                static_cast<unsigned long long>(nx));
  int x = ix;
  int y = iy;
  if (symmetryMode == 45) {
    switch (symId) {
    case 1: x = iy; y = ix; break;
    case 2: x = nx - 1 - iy; y = ix; break;
    case 3: x = nx - 1 - ix; y = iy; break;
    case 4: x = nx - 1 - ix; y = ny - 1 - iy; break;
    case 5: x = nx - 1 - iy; y = ny - 1 - ix; break;
    case 6: x = iy; y = ny - 1 - ix; break;
    case 7: x = ix; y = ny - 1 - iy; break;
    default: break;
    }
  } else {
    switch (symId) {
    case 1: x = nx - 1 - iy; y = ix; break;
    case 2: x = nx - 1 - ix; y = ny - 1 - iy; break;
    case 3: x = iy; y = ny - 1 - ix; break;
    default: break;
    }
  }
  const int z = zMirror != 0 ? nz - 1 - iz : iz;
  if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
    rotatedBatch[index] = 0.0f;
    return;
  }
  const unsigned long long mapped =
      static_cast<unsigned long long>(z) * plane +
      static_cast<unsigned long long>(y) * static_cast<unsigned long long>(nx) +
      static_cast<unsigned long long>(x);
  rotatedBatch[index] = bedImage[mapped];
}

extern "C" __global__ void compact_transform_backward_add(
    const float *rotated, double *accum, unsigned long long nvox,
    int symmetryMode, int nx, int ny, int nz, int symId, int zMirror) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= nvox || nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const int iz = static_cast<int>(index / plane);
  const unsigned long long rem =
      index - static_cast<unsigned long long>(iz) * plane;
  const int iy = static_cast<int>(rem / static_cast<unsigned long long>(nx));
  const int ix = static_cast<int>(rem - static_cast<unsigned long long>(iy) *
                                            static_cast<unsigned long long>(nx));
  int x = ix;
  int y = iy;
  if (symmetryMode == 45) {
    switch (symId) {
    case 1:
      x = iy;
      y = ix;
      break;
    case 2:
      x = nx - 1 - iy;
      y = ix;
      break;
    case 3:
      x = nx - 1 - ix;
      y = iy;
      break;
    case 4:
      x = nx - 1 - ix;
      y = ny - 1 - iy;
      break;
    case 5:
      x = nx - 1 - iy;
      y = ny - 1 - ix;
      break;
    case 6:
      x = iy;
      y = ny - 1 - ix;
      break;
    case 7:
      x = ix;
      y = ny - 1 - iy;
      break;
    default:
      break;
    }
  } else {
    switch (symId) {
    case 1:
      x = nx - 1 - iy;
      y = ix;
      break;
    case 2:
      x = nx - 1 - ix;
      y = ny - 1 - iy;
      break;
    case 3:
      x = iy;
      y = ny - 1 - ix;
      break;
    default:
      break;
    }
  }
  int z = zMirror != 0 ? (nz - 1 - iz) : iz;
  if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
    return;
  }
  const unsigned long long mapped =
      static_cast<unsigned long long>(z) * plane +
      static_cast<unsigned long long>(y) * static_cast<unsigned long long>(nx) +
      static_cast<unsigned long long>(x);
  accum[mapped] += static_cast<double>(rotated[index]);
}

extern "C" __global__ void compact_transform_backward_add_batch(
    const float *rotatedBatch, double *accum, unsigned long long nvox,
    int batchSize, int symmetryMode, int nx, int ny, int nz,
    const int *symIds, const int *zMirrors) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      nvox * static_cast<unsigned long long>(batchSize);
  if (index >= total || nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const int branch = static_cast<int>(index / nvox);
  const unsigned long long voxel = index % nvox;
  const int symId = symIds[branch];
  const int zMirror = zMirrors[branch];
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const int iz = static_cast<int>(voxel / plane);
  const unsigned long long rem =
      voxel - static_cast<unsigned long long>(iz) * plane;
  const int iy = static_cast<int>(rem / static_cast<unsigned long long>(nx));
  const int ix = static_cast<int>(
      rem - static_cast<unsigned long long>(iy) *
                static_cast<unsigned long long>(nx));
  int x = ix;
  int y = iy;
  if (symmetryMode == 45) {
    switch (symId) {
    case 1:
      x = iy;
      y = ix;
      break;
    case 2:
      x = nx - 1 - iy;
      y = ix;
      break;
    case 3:
      x = nx - 1 - ix;
      y = iy;
      break;
    case 4:
      x = nx - 1 - ix;
      y = ny - 1 - iy;
      break;
    case 5:
      x = nx - 1 - iy;
      y = ny - 1 - ix;
      break;
    case 6:
      x = iy;
      y = ny - 1 - ix;
      break;
    case 7:
      x = ix;
      y = ny - 1 - iy;
      break;
    default:
      break;
    }
  } else {
    switch (symId) {
    case 1:
      x = nx - 1 - iy;
      y = ix;
      break;
    case 2:
      x = nx - 1 - ix;
      y = ny - 1 - iy;
      break;
    case 3:
      x = iy;
      y = ny - 1 - ix;
      break;
    default:
      break;
    }
  }
  const int z = zMirror != 0 ? (nz - 1 - iz) : iz;
  if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
    return;
  }
  const unsigned long long mapped =
      static_cast<unsigned long long>(z) * plane +
      static_cast<unsigned long long>(y) * static_cast<unsigned long long>(nx) +
      static_cast<unsigned long long>(x);
  atomicAdd(&accum[mapped], static_cast<double>(rotatedBatch[index]));
}

// Rotated/mirrored voxel index of a branch -> unrotated voxel (the CTAC table
// frame); the row -> full angle mapping lives on the host
// (csrqAttenFullAngle) since the factor is applied per compact angle.
extern "C" __device__ __forceinline__ unsigned long long
compact_atten_mapped_voxel(unsigned long long index, int symmetryMode,
                           int symId, int zMirror, int nx, int ny, int nz,
                           unsigned long long plane, int *valid) {
  const int iz = static_cast<int>(index / plane);
  const unsigned long long rem =
      index - static_cast<unsigned long long>(iz) * plane;
  const int iy = static_cast<int>(rem / static_cast<unsigned long long>(nx));
  const int ix = static_cast<int>(
      rem - static_cast<unsigned long long>(iy) *
                static_cast<unsigned long long>(nx));
  int x = ix;
  int y = iy;
  if (symmetryMode == 45) {
    switch (symId) {
    case 1:
      x = iy;
      y = ix;
      break;
    case 2:
      x = nx - 1 - iy;
      y = ix;
      break;
    case 3:
      x = nx - 1 - ix;
      y = iy;
      break;
    case 4:
      x = nx - 1 - ix;
      y = ny - 1 - iy;
      break;
    case 5:
      x = nx - 1 - iy;
      y = ny - 1 - ix;
      break;
    case 6:
      x = iy;
      y = ny - 1 - ix;
      break;
    case 7:
      x = ix;
      y = ny - 1 - iy;
      break;
    default:
      break;
    }
  } else {
    switch (symId) {
    case 1:
      x = nx - 1 - iy;
      y = ix;
      break;
    case 2:
      x = nx - 1 - ix;
      y = ny - 1 - iy;
      break;
    case 3:
      x = iy;
      y = ny - 1 - ix;
      break;
    default:
      break;
    }
  }
  const int z = zMirror != 0 ? (nz - 1 - iz) : iz;
  if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
    *valid = 0;
    return 0;
  }
  *valid = 1;
  return static_cast<unsigned long long>(z) * plane +
         static_cast<unsigned long long>(y) *
             static_cast<unsigned long long>(nx) +
         static_cast<unsigned long long>(x);
}

// Decoded CTAC factor with the per-window energy exponent: att^power, where
// att is the u16 table value / 65535 (power 1 = the table itself; power =
// mu_w(E_window)/mu_w(E_map) rescales the line integral to another photon
// energy under the water-scaled mu model, since att = exp(-integral mu)).
__device__ __forceinline__ float compact_atten_decode(unsigned short q,
                                                      float power) {
  const float att =
      static_cast<float>(q) * (1.0f / static_cast<float>(kAttenUint16Scale));
  return power == 1.0f ? att : __powf(att, power);
}

// ROW-packed 2-RHS forward CSR. Not a revival of compact_csr_spmm_col_tile4:
// that kernel gathered four COL-major B columns (stride = n_cols) and lost to
// CSR_ALG2 (35.433 s vs 27.076 s). Here each 2-RHS tile is packed to float2 so
// a sorted CSR row walks X sequentially, A is float4/int4, and reduce is
// warp-then-block.
extern "C" __global__ void compact_pack_colpair_to_float2(const float *x0,
                                                          const float *x1,
                                                          float2 *out,
                                                          int cols) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < cols) {
    out[col] = make_float2(x0[col], x1[col]);
  }
}

extern "C" __device__ __forceinline__ float2 compact_warp_sum_float2(float2 v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v.x += __shfl_down_sync(0xffffffffu, v.x, offset);
    v.y += __shfl_down_sync(0xffffffffu, v.y, offset);
  }
  return v;
}

extern "C" __global__ void compact_csr_spmv_xpair(const float *values,
                                                 const int *rowPtr,
                                                 const int *colIdx,
                                                 const float2 *xpair,
                                                 float *y0, float *y1,
                                                 int rows, int zeroProjectionMask) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  // Numerator only: y initially holds 0/1 activity, then Ax overwrites it.
  // A zero pair already contains the required zeros; ratio later reads p.
  if (zeroProjectionMask != 0 && y0[row] == 0.0f && y1[row] == 0.0f) {
    return;
  }
  const int start = rowPtr[row];
  const int end = rowPtr[row + 1];
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);
  float2 sum = make_float2(0.0f, 0.0f);

  const int aligned = (start + 3) & ~3;
  for (int t = start + tid; t < aligned && t < end; t += stride) {
    const float2 x = xpair[colIdx[t]];
    const float a = values[t];
    sum.x += a * x.x;
    sum.y += a * x.y;
  }
  for (int t = aligned + tid * 4; t + 3 < end; t += stride * 4) {
    const float4 a = *reinterpret_cast<const float4 *>(values + t);
    const int4 c = *reinterpret_cast<const int4 *>(colIdx + t);
    const float2 x0 = xpair[c.x];
    const float2 x1 = xpair[c.y];
    const float2 x2 = xpair[c.z];
    const float2 x3 = xpair[c.w];
    sum.x += a.x * x0.x + a.y * x1.x + a.z * x2.x + a.w * x3.x;
    sum.y += a.x * x0.y + a.y * x1.y + a.z * x2.y + a.w * x3.y;
  }
  const int vecEnd =
      aligned < end ? aligned + ((end - aligned) & ~3) : end;
  for (int t = vecEnd + tid; t < end; t += stride) {
    const float2 x = xpair[colIdx[t]];
    const float a = values[t];
    sum.x += a * x.x;
    sum.y += a * x.y;
  }

  sum = compact_warp_sum_float2(sum);
  __shared__ float2 warpSums[32];
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) {
    warpSums[warp] = sum;
  }
  __syncthreads();
  if (tid == 0) {
    const int nwarps = stride >> 5;
    float2 total = warpSums[0];
    for (int w = 1; w < nwarps; ++w) {
      total.x += warpSums[w].x;
      total.y += warpSums[w].y;
    }
    y0[row] = total.x;
    y1[row] = total.y;
  }
}

// 4-RHS extension of compact_csr_spmv_xpair. Not a revival of
// compact_csr_spmm_col_tile4 (COL-major stride gathers). Pack four COL
// vectors to float4, stream A once, four accumulators, same row walk.
extern "C" __global__ void compact_pack_colquad_to_float4(const float *x0,
                                                          const float *x1,
                                                          const float *x2,
                                                          const float *x3,
                                                          float4 *out,
                                                          int cols) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < cols) {
    out[col] = make_float4(x0[col], x1[col], x2[col], x3[col]);
  }
}

extern "C" __device__ __forceinline__ float4 compact_warp_sum_float4(float4 v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v.x += __shfl_down_sync(0xffffffffu, v.x, offset);
    v.y += __shfl_down_sync(0xffffffffu, v.y, offset);
    v.z += __shfl_down_sync(0xffffffffu, v.z, offset);
    v.w += __shfl_down_sync(0xffffffffu, v.w, offset);
  }
  return v;
}

extern "C" __global__ void compact_csr_spmv_xquad(const float *values,
                                                 const int *rowPtr,
                                                 const int *colIdx,
                                                 const float4 *xquad,
                                                 float *y0, float *y1,
                                                 float *y2, float *y3,
                                                 int rows) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int start = rowPtr[row];
  const int end = rowPtr[row + 1];
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);
  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  const int aligned = (start + 3) & ~3;
  for (int t = start + tid; t < aligned && t < end; t += stride) {
    const float4 x = xquad[colIdx[t]];
    const float a = values[t];
    sum.x += a * x.x;
    sum.y += a * x.y;
    sum.z += a * x.z;
    sum.w += a * x.w;
  }
  for (int t = aligned + tid * 4; t + 3 < end; t += stride * 4) {
    const float4 a = *reinterpret_cast<const float4 *>(values + t);
    const int4 c = *reinterpret_cast<const int4 *>(colIdx + t);
    const float4 x0 = xquad[c.x];
    const float4 x1 = xquad[c.y];
    const float4 x2 = xquad[c.z];
    const float4 x3 = xquad[c.w];
    sum.x += a.x * x0.x + a.y * x1.x + a.z * x2.x + a.w * x3.x;
    sum.y += a.x * x0.y + a.y * x1.y + a.z * x2.y + a.w * x3.y;
    sum.z += a.x * x0.z + a.y * x1.z + a.z * x2.z + a.w * x3.z;
    sum.w += a.x * x0.w + a.y * x1.w + a.z * x2.w + a.w * x3.w;
  }
  const int vecEnd =
      aligned < end ? aligned + ((end - aligned) & ~3) : end;
  for (int t = vecEnd + tid; t < end; t += stride) {
    const float4 x = xquad[colIdx[t]];
    const float a = values[t];
    sum.x += a * x.x;
    sum.y += a * x.y;
    sum.z += a * x.z;
    sum.w += a * x.w;
  }

  sum = compact_warp_sum_float4(sum);
  __shared__ float4 warpSums[32];
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) {
    warpSums[warp] = sum;
  }
  __syncthreads();
  if (tid == 0) {
    const int nwarps = stride >> 5;
    float4 total = warpSums[0];
    for (int w = 1; w < nwarps; ++w) {
      total.x += warpSums[w].x;
      total.y += warpSums[w].y;
      total.z += warpSums[w].z;
      total.w += warpSums[w].w;
    }
    y0[row] = total.x;
    y1[row] = total.y;
    y2[row] = total.z;
    y3[row] = total.w;
  }
}

// Transpose (rays -> voxels) scatter for a pair of RHS with zero-row
// skipping: y0[col] += a_t * r0, y1[col] += a_t * r1 for every nnz t of row
// (one block per row, the same float4/int4 row walk as the forward pair
// kernel). Rows whose two ratios are both exactly zero contribute nothing to
// the accumulation and are skipped before any matrix byte is read; the
// production numerator ratio is exactly 0 for zero-count bins, so on the
// five-bed Lu-177 input ~62 % of the nnz work disappears. The outputs are
// accumulated (the caller zeroes them first). Independent forward /
// sensitivity semantics are untouched: this only replaces the per-RHS
// cuSPARSE transpose inside the numerator.
extern "C" __global__ void compact_csr_spmvt_pair(const float *values,
                                                  const int *rowPtr,
                                                  const int *colIdx,
                                                  const float *x0,
                                                  const float *x1, float *y0,
                                                  float *y1, int rows) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float r0 = x0[row];
  const float r1 = x1[row];
  if (r0 == 0.0f && r1 == 0.0f) {
    return;
  }
  const int start = rowPtr[row];
  const int end = rowPtr[row + 1];
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);
  const bool both = r0 != 0.0f && r1 != 0.0f;
  float *ySingle = r0 != 0.0f ? y0 : y1;
  const float rSingle = r0 != 0.0f ? r0 : r1;

  const int aligned = (start + 3) & ~3;
  for (int t = start + tid; t < aligned && t < end; t += stride) {
    const int c = colIdx[t];
    const float a = values[t];
    if (both) {
      atomicAdd(y0 + c, a * r0);
      atomicAdd(y1 + c, a * r1);
    } else {
      atomicAdd(ySingle + c, a * rSingle);
    }
  }
  for (int t = aligned + tid * 4; t + 3 < end; t += stride * 4) {
    const float4 a = *reinterpret_cast<const float4 *>(values + t);
    const int4 c = *reinterpret_cast<const int4 *>(colIdx + t);
    if (both) {
      atomicAdd(y0 + c.x, a.x * r0);
      atomicAdd(y1 + c.x, a.x * r1);
      atomicAdd(y0 + c.y, a.y * r0);
      atomicAdd(y1 + c.y, a.y * r1);
      atomicAdd(y0 + c.z, a.z * r0);
      atomicAdd(y1 + c.z, a.z * r1);
      atomicAdd(y0 + c.w, a.w * r0);
      atomicAdd(y1 + c.w, a.w * r1);
    } else {
      atomicAdd(ySingle + c.x, a.x * rSingle);
      atomicAdd(ySingle + c.y, a.y * rSingle);
      atomicAdd(ySingle + c.z, a.z * rSingle);
      atomicAdd(ySingle + c.w, a.w * rSingle);
    }
  }
  const int vecEnd =
      aligned < end ? aligned + ((end - aligned) & ~3) : end;
  for (int t = vecEnd + tid; t < end; t += stride) {
    const int c = colIdx[t];
    const float a = values[t];
    if (both) {
      atomicAdd(y0 + c, a * r0);
      atomicAdd(y1 + c, a * r1);
    } else {
      atomicAdd(ySingle + c, a * rSingle);
    }
  }
}

// Same for four RHS at once (one matrix read per four columns); a row is
// skipped only when all four ratios are zero, so fewer rows skip than with
// pairs — the bench decides which grouping wins on the real input.
extern "C" __global__ void compact_csr_spmvt_quad(
    const float *values, const int *rowPtr, const int *colIdx, const float *x0,
    const float *x1, const float *x2, const float *x3, float *y0, float *y1,
    float *y2, float *y3, int rows) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float r0 = x0[row];
  const float r1 = x1[row];
  const float r2 = x2[row];
  const float r3 = x3[row];
  if (r0 == 0.0f && r1 == 0.0f && r2 == 0.0f && r3 == 0.0f) {
    return;
  }
  const int start = rowPtr[row];
  const int end = rowPtr[row + 1];
  const int tid = static_cast<int>(threadIdx.x);
  const int stride = static_cast<int>(blockDim.x);
  const bool z0 = r0 != 0.0f;
  const bool z1 = r1 != 0.0f;
  const bool z2 = r2 != 0.0f;
  const bool z3 = r3 != 0.0f;

  const int aligned = (start + 3) & ~3;
  for (int t = start + tid; t < aligned && t < end; t += stride) {
    const int c = colIdx[t];
    const float a = values[t];
    if (z0) atomicAdd(y0 + c, a * r0);
    if (z1) atomicAdd(y1 + c, a * r1);
    if (z2) atomicAdd(y2 + c, a * r2);
    if (z3) atomicAdd(y3 + c, a * r3);
  }
  for (int t = aligned + tid * 4; t + 3 < end; t += stride * 4) {
    const float4 a = *reinterpret_cast<const float4 *>(values + t);
    const int4 c = *reinterpret_cast<const int4 *>(colIdx + t);
    const int cs[4] = {c.x, c.y, c.z, c.w};
    const float as[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (z0) atomicAdd(y0 + cs[q], as[q] * r0);
      if (z1) atomicAdd(y1 + cs[q], as[q] * r1);
      if (z2) atomicAdd(y2 + cs[q], as[q] * r2);
      if (z3) atomicAdd(y3 + cs[q], as[q] * r3);
    }
  }
  const int vecEnd =
      aligned < end ? aligned + ((end - aligned) & ~3) : end;
  for (int t = vecEnd + tid; t < end; t += stride) {
    const int c = colIdx[t];
    const float a = values[t];
    if (z0) atomicAdd(y0 + c, a * r0);
    if (z1) atomicAdd(y1 + c, a * r1);
    if (z2) atomicAdd(y2 + c, a * r2);
    if (z3) atomicAdd(y3 + c, a * r3);
  }
}

// ---------------------------------------------------------------------------
// CSRQ kernels
// ---------------------------------------------------------------------------

struct CsrqTile {
  unsigned int start[CSRQ_RUN_TILE];       // first column of each run
  unsigned short off[CSRQ_RUN_TILE];       // nnz offset of each run in the tile
  unsigned char runId[CSRQ_TILE_MAX_NNZ];  // run index for every nnz in the tile
  unsigned int warpSums[CSRQ_BLOCK / 32];
};

// Exclusive scan over the 256 block threads; returns this thread's prefix and
// broadcasts the block total. Callers must __syncthreads() before reusing
// warpSums (the tile loop ends with a barrier).
__device__ __forceinline__ unsigned int csrq_block_exclusive_scan(
    unsigned int value, unsigned int *warpSums, unsigned int *total) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  unsigned int inclusive = value;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const unsigned int n = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += n;
  }
  if (lane == 31) warpSums[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    unsigned int w = lane < (CSRQ_BLOCK / 32) ? warpSums[lane] : 0u;
    for (int offset = 1; offset < (CSRQ_BLOCK / 32); offset <<= 1) {
      const unsigned int n = __shfl_up_sync(0xffffffffu, w, offset);
      if (lane >= offset) w += n;
    }
    if (lane < (CSRQ_BLOCK / 32)) warpSums[lane] = w;
  }
  __syncthreads();
  const unsigned int warpPrefix = warp == 0 ? 0u : warpSums[warp - 1];
  *total = warpSums[(CSRQ_BLOCK / 32) - 1];
  return warpPrefix + inclusive - value;
}

// Loads up to 256 run words starting at tileBase, scans their lengths and
// expands the per-nnz run id table. Returns the tile nnz count.
__device__ __forceinline__ unsigned int csrq_load_tile(
    const unsigned int *runs, unsigned int tileBase, unsigned int tileRuns,
    CsrqTile &s) {
  unsigned int len = 0;
  unsigned int start = 0;
  if (threadIdx.x < tileRuns) {
    const unsigned int word = runs[tileBase + threadIdx.x];
    len = (word & (CSRQ_MAX_RUN_LEN - 1u)) + 1u;
    start = word >> CSRQ_RUN_LEN_BITS;
  }
  unsigned int total = 0;
  const unsigned int off = csrq_block_exclusive_scan(len, s.warpSums, &total);
  if (threadIdx.x < tileRuns) {
    s.start[threadIdx.x] = start;
    s.off[threadIdx.x] = static_cast<unsigned short>(off);
    for (unsigned int k = 0; k < len; ++k) {
      s.runId[off + k] = static_cast<unsigned char>(threadIdx.x);
    }
  }
  __syncthreads();
  return total;
}

// Same row -> full projection mapping as compute_compact_ratio_chunk_batch.
__device__ __forceinline__ int csrq_full_projection_index(
    int compactRow, int symmetryMode, int compactRaysPerAngle,
    int compactNumAngles, int detNx, int detNz, int compactDetZ,
    int fullNumAngles, int fullRaysPerAngle, int symId, int zMirror,
    unsigned long long *fullIndex) {
  const int compactAngle = compactRow / compactRaysPerAngle;
  const int rr = compactRow - compactAngle * compactRaysPerAngle;
  const int ix = rr % detNx;
  const int izCompact = rr / detNx;
  int fullAngle = 0;
  if (symmetryMode == 45) {
    const int q = fullNumAngles / 8;
    if ((symId & 1) == 0) {
      if (compactAngle >= q) return 0;
      fullAngle = symId * q + compactAngle;
    } else {
      if (compactAngle <= 0) return 0;
      fullAngle = symId * q + (q - compactAngle);
    }
  } else {
    if (compactAngle < 0 || compactAngle >= compactNumAngles) return 0;
    fullAngle = compactAngle + symId * compactNumAngles;
  }
  int izFull = izCompact;
  if (zMirror != 0) {
    izFull = detNz - 1 - izCompact;
    if (izFull < compactDetZ) return 0;
  }
  if (fullAngle < 0 || fullAngle >= fullNumAngles || izFull < 0 ||
      izFull >= detNz) {
    return 0;
  }
  *fullIndex = static_cast<unsigned long long>(fullAngle) *
                   static_cast<unsigned long long>(fullRaysPerAngle) +
               static_cast<unsigned long long>(izFull) *
                   static_cast<unsigned long long>(detNx) +
               static_cast<unsigned long long>(ix);
  return 1;
}

// yhat[row*8 + k] = sum_t v_t * xPacked[col_t*8 + k]. Value decode matches
// tools/convert_csrb_to_csrq.py dequantize_u16_sqrt bit for bit (fp32).
//
// ratioMode == 0 stores yhat. ratioMode == 1 stores the Poisson ratio
//   scale * projection[fullIndex] / max(scale * yhat + background, eps)
// for the (row, branch) pairs that map to a full projection row and 0 for the
// others, straight from the reduced sums (the ratio inputs are ignored in
// mode 0). scale is the window weight of a joint multi-window model
// (mean = scale * A x + background); scale = 1 and no background is the plain
// MLEM ratio. fullBackground may be null.
extern "C" __global__ void __launch_bounds__(CSRQ_BLOCK) csrq_forward_k8(
    const unsigned int *rowNnzPtr, const unsigned int *rowRunPtr,
    const float *rowMax, const unsigned int *runs,
    const unsigned short *valuesQ, const float4 *xPacked, float *yPacked,
    int rows, int ratioMode, const double *fullProjection,
    const double *fullBackground, unsigned long long fullProjectionCount,
    double scale, int symmetryMode, int compactRowBase,
    int compactRaysPerAngle, int compactNumAngles, int detNx, int detNz,
    int compactDetZ, int fullNumAngles, int fullRaysPerAngle,
    const int *symIds, const int *zMirrors, double eps) {
  __shared__ CsrqTile s;
  __shared__ float sRed[CSRQ_BLOCK / 32][CSRQ_K];
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const unsigned int runBegin = rowRunPtr[row];
  const unsigned int runEnd = rowRunPtr[row + 1];
  unsigned int valBase = rowNnzPtr[row];
  const float m = rowMax[row];
  float acc[CSRQ_K];
#pragma unroll
  for (int k = 0; k < CSRQ_K; ++k) acc[k] = 0.0f;

  for (unsigned int tileBase = runBegin; tileBase < runEnd;
       tileBase += CSRQ_RUN_TILE) {
    const unsigned int tileRuns = min(static_cast<unsigned int>(CSRQ_RUN_TILE),
                                      runEnd - tileBase);
    const unsigned int tileNnz = csrq_load_tile(runs, tileBase, tileRuns, s);
    for (unsigned int p = threadIdx.x; p < tileNnz; p += CSRQ_BLOCK) {
      const unsigned int j = s.runId[p];
      const unsigned int col = s.start[j] + (p - s.off[j]);
      const float t = static_cast<float>(valuesQ[valBase + p]) * kCsrqInvQScale;
      const float v = m * (t * t);
      const float4 x0 = xPacked[2u * col];
      const float4 x1 = xPacked[2u * col + 1u];
      acc[0] = fmaf(v, x0.x, acc[0]);
      acc[1] = fmaf(v, x0.y, acc[1]);
      acc[2] = fmaf(v, x0.z, acc[2]);
      acc[3] = fmaf(v, x0.w, acc[3]);
      acc[4] = fmaf(v, x1.x, acc[4]);
      acc[5] = fmaf(v, x1.y, acc[5]);
      acc[6] = fmaf(v, x1.z, acc[6]);
      acc[7] = fmaf(v, x1.w, acc[7]);
    }
    valBase += tileNnz;
    __syncthreads();
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
  for (int k = 0; k < CSRQ_K; ++k) {
    float v = acc[k];
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if (lane == 0) sRed[warp][k] = v;
  }
  __syncthreads();
  if (threadIdx.x < CSRQ_K) {
    float sum = 0.0f;
    for (int w = 0; w < CSRQ_BLOCK / 32; ++w) sum += sRed[w][threadIdx.x];
    const size_t out = static_cast<size_t>(row) * CSRQ_K + threadIdx.x;
    if (ratioMode == 0) {
      yPacked[out] = sum;
      return;
    }
    const int b = static_cast<int>(threadIdx.x);
    unsigned long long fullIndex = 0;
    float ratio = 0.0f;
    if (csrq_full_projection_index(compactRowBase + row, symmetryMode,
                                   compactRaysPerAngle, compactNumAngles,
                                   detNx, detNz, compactDetZ, fullNumAngles,
                                   fullRaysPerAngle, symIds[b], zMirrors[b],
                                   &fullIndex) &&
        fullIndex < fullProjectionCount) {
      double denom = scale * static_cast<double>(sum);
      if (fullBackground != nullptr) {
        denom += fullBackground[fullIndex];
      }
      if (denom < eps) {
        denom = eps;
      }
      ratio = static_cast<float>(scale * fullProjection[fullIndex] / denom);
    }
    yPacked[out] = ratio;
  }
}

// yBranch[k*nvox + col] += sum_rows v_t * ratioPacked[row*8 + k] (atomic
// float) for k in [kFirst, kFirst + kCount). The output is branch-major
// ([8][nvox], the layout compact_transform_backward_add_batch consumes) on
// purpose: for one k the 32 lanes of a warp hit 32 consecutive floats, so the
// L2 coalesces each warp atomic into four 32-byte sectors instead of 32
// scattered ones as a [nvox][8] layout would. The branch subset lets the
// caller keep the touched output (kCount * nvox floats) inside L2; re-reading
// the matrix per pass is cheap next to the atomic traffic. Rows whose
// selected ratios are all zero (no valid full projection index, or a masked
// row) are skipped entirely.
extern "C" __global__ void __launch_bounds__(CSRQ_BLOCK) csrq_backward_k8(
    const unsigned int *rowNnzPtr, const unsigned int *rowRunPtr,
    const float *rowMax, const unsigned int *runs,
    const unsigned short *valuesQ, const float4 *ratioPacked, float *yBranch,
    unsigned long long nvox, int rows, int kFirst, int kCount) {
  __shared__ CsrqTile s;
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float4 r0 = ratioPacked[2 * row];
  const float4 r1 = ratioPacked[2 * row + 1];
  const float r[CSRQ_K] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
  const int kEnd = kFirst + kCount;
  // Selected ratios into registers (a runtime index into r[] would spill to
  // local memory); the two-branch pass is the production configuration.
  float ra = 0.0f;
  float rb = 0.0f;
  const bool twoBranch = kCount == 2;
  if (twoBranch) {
    switch (kFirst) {
    case 0: ra = r0.x; rb = r0.y; break;
    case 2: ra = r0.z; rb = r0.w; break;
    case 4: ra = r1.x; rb = r1.y; break;
    case 6: ra = r1.z; rb = r1.w; break;
    default: ra = 0.0f; rb = 0.0f; break;
    }
    if (ra == 0.0f && rb == 0.0f) {
      return;
    }
  } else {
    bool anyNonZero = false;
#pragma unroll
    for (int k = 0; k < CSRQ_K; ++k) {
      anyNonZero = anyNonZero || (k >= kFirst && k < kEnd && r[k] != 0.0f);
    }
    if (!anyNonZero) {
      return;
    }
  }
  const unsigned int runBegin = rowRunPtr[row];
  const unsigned int runEnd = rowRunPtr[row + 1];
  unsigned int valBase = rowNnzPtr[row];
  const float m = rowMax[row];
  float *dstA = yBranch + static_cast<unsigned long long>(kFirst) * nvox;
  float *dstB = dstA + nvox;

  for (unsigned int tileBase = runBegin; tileBase < runEnd;
       tileBase += CSRQ_RUN_TILE) {
    const unsigned int tileRuns = min(static_cast<unsigned int>(CSRQ_RUN_TILE),
                                      runEnd - tileBase);
    const unsigned int tileNnz = csrq_load_tile(runs, tileBase, tileRuns, s);
    if (twoBranch) {
      for (unsigned int p = threadIdx.x; p < tileNnz; p += CSRQ_BLOCK) {
        const unsigned int j = s.runId[p];
        const unsigned int col = s.start[j] + (p - s.off[j]);
        const float t =
            static_cast<float>(valuesQ[valBase + p]) * kCsrqInvQScale;
        const float v = m * (t * t);
        atomicAdd(dstA + col, v * ra);
        atomicAdd(dstB + col, v * rb);
      }
    } else {
      for (unsigned int p = threadIdx.x; p < tileNnz; p += CSRQ_BLOCK) {
        const unsigned int j = s.runId[p];
        const unsigned int col = s.start[j] + (p - s.off[j]);
        const float t =
            static_cast<float>(valuesQ[valBase + p]) * kCsrqInvQScale;
        const float v = m * (t * t);
        float *dst = yBranch + col;
#pragma unroll
        for (int k = 0; k < CSRQ_K; ++k) {
          if (k >= kFirst && k < kEnd) {
            atomicAdd(dst + static_cast<unsigned long long>(k) * nvox,
                      v * r[k]);
          }
        }
      }
    }
    valBase += tileNnz;
    __syncthreads();
  }
}

// Vector atomic add of four floats. sm_90+ has a native float4 global atomic
// (red.global.add.v4.f32): one instruction and one 16-byte sector per four
// branches. Older parts fall back to four scalar atomics on the same
// addresses, so the packed layout below computes the same sums everywhere
// and only the instruction (and the speed) differ.
__device__ __forceinline__ void csrq_atomic_add_float4(float4 *dst, float4 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  atomicAdd(dst, v);
#else
  float *f = reinterpret_cast<float *>(dst);
  atomicAdd(f, v.x);
  atomicAdd(f + 1, v.y);
  atomicAdd(f + 2, v.z);
  atomicAdd(f + 3, v.w);
#endif
}

// Packed-output backward: yPacked[col][8] += v_t * ratio[row][8] for all
// eight branches in one pass over the matrix (two float4 atomics per nnz)
// instead of four branch-major passes of two scalar atomics each. Opt-in
// (SPECT_CSRQ_BACKWARD_PACKED=1): on sm_86/89 the scalar fallback is the
// [voxel][8] layout that measured 715 ms per pass (32 scattered sectors per
// warp atomic); it exists for the sm_120 vector-atomic trial.
extern "C" __global__ void __launch_bounds__(CSRQ_BLOCK) csrq_backward_k8_packed(
    const unsigned int *rowNnzPtr, const unsigned int *rowRunPtr,
    const float *rowMax, const unsigned int *runs,
    const unsigned short *valuesQ, const float4 *ratioPacked, float4 *yPacked,
    int rows) {
  __shared__ CsrqTile s;
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float4 r0 = ratioPacked[2 * row];
  const float4 r1 = ratioPacked[2 * row + 1];
  if (r0.x == 0.0f && r0.y == 0.0f && r0.z == 0.0f && r0.w == 0.0f &&
      r1.x == 0.0f && r1.y == 0.0f && r1.z == 0.0f && r1.w == 0.0f) {
    return;
  }
  const unsigned int runBegin = rowRunPtr[row];
  const unsigned int runEnd = rowRunPtr[row + 1];
  unsigned int valBase = rowNnzPtr[row];
  const float m = rowMax[row];
  for (unsigned int tileBase = runBegin; tileBase < runEnd;
       tileBase += CSRQ_RUN_TILE) {
    const unsigned int tileRuns = min(static_cast<unsigned int>(CSRQ_RUN_TILE),
                                      runEnd - tileBase);
    const unsigned int tileNnz = csrq_load_tile(runs, tileBase, tileRuns, s);
    for (unsigned int p = threadIdx.x; p < tileNnz; p += CSRQ_BLOCK) {
      const unsigned int j = s.runId[p];
      const unsigned int col = s.start[j] + (p - s.off[j]);
      const float t = static_cast<float>(valuesQ[valBase + p]) * kCsrqInvQScale;
      const float v = m * (t * t);
      csrq_atomic_add_float4(yPacked + 2u * col,
                             make_float4(v * r0.x, v * r0.y, v * r0.z, v * r0.w));
      csrq_atomic_add_float4(yPacked + 2u * col + 1u,
                             make_float4(v * r1.x, v * r1.y, v * r1.z, v * r1.w));
    }
    valBase += tileNnz;
    __syncthreads();
  }
}

// Consumer of the packed backward output: accum[M_k(v)] += y[v][k] (times the
// branch's CTAC factor^power when attSlices is given), one thread per voxel
// reading its two float4 and issuing eight double atomics.
extern "C" __global__ void compact_transform_backward_add_packed8(
    const float4 *yPacked, double *accum, unsigned long long nvox,
    int symmetryMode, int nx, int ny, int nz, const int *symIds,
    const int *zMirrors, const unsigned short *attSlices, const int *sliceIndex,
    float power) {
  const unsigned long long v =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= nvox || nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const float4 a = yPacked[2 * v];
  const float4 b = yPacked[2 * v + 1];
  const float vals[CSRQ_K] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
#pragma unroll
  for (int k = 0; k < CSRQ_K; ++k) {
    float value = vals[k];
    if (value == 0.0f) {
      continue;
    }
    int valid = 0;
    const unsigned long long mapped = compact_atten_mapped_voxel(
        v, symmetryMode, symIds[k], zMirrors[k], nx, ny, nz, plane, &valid);
    if (valid == 0) {
      continue;
    }
    if (attSlices != nullptr) {
      const int slice = sliceIndex[k];
      if (slice >= 0) {
        value *= compact_atten_decode(
            attSlices[static_cast<unsigned long long>(slice) * nvox + mapped],
            power);
      }
    }
    atomicAdd(&accum[mapped], static_cast<double>(value));
  }
}

// [8][nvox] branch-major input (one bed of the branch batch) -> [nvox][8].
extern "C" __global__ void csrq_pack_branch8(const float *branchMajor,
                                             float4 *packed,
                                             unsigned long long nvox) {
  const unsigned long long v =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= nvox) {
    return;
  }
  packed[2 * v] = make_float4(branchMajor[v], branchMajor[nvox + v],
                              branchMajor[2 * nvox + v],
                              branchMajor[3 * nvox + v]);
  packed[2 * v + 1] = make_float4(branchMajor[4 * nvox + v],
                                  branchMajor[5 * nvox + v],
                                  branchMajor[6 * nvox + v],
                                  branchMajor[7 * nvox + v]);
}

// CTAC pack for one compact angle: packed[v][k] = branchMajor[k*nvox + v] *
// (att[slice_k * nvox + M_k(v)] / 65535)^power, where M_k maps the rotated
// voxel v of branch k back to the unrotated frame (compact_atten_mapped_voxel)
// and slice_k = sliceIndex[k] is the staged factor slice of the branch's full
// angle (-1: the branch has no full angle at this compact angle -> factor 1).
// The factor is applied once per voxel in the image domain instead of once
// per nnz inside the SpMV.
extern "C" __global__ void csrq_pack_branch8_atten(
    const float *branchMajor, const unsigned short *attSlices,
    const int *sliceIndex, float4 *packed, unsigned long long nvox,
    int symmetryMode, int nx, int ny, int nz, const int *symIds,
    const int *zMirrors, float power) {
  const unsigned long long v =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= nvox) {
    return;
  }
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  float f[CSRQ_K];
#pragma unroll
  for (int k = 0; k < CSRQ_K; ++k) {
    float factor = 1.0f;
    const int slice = sliceIndex[k];
    if (slice >= 0) {
      int valid = 0;
      const unsigned long long mapped = compact_atten_mapped_voxel(
          v, symmetryMode, symIds[k], zMirrors[k], nx, ny, nz, plane, &valid);
      if (valid != 0) {
        factor = compact_atten_decode(
            attSlices[static_cast<unsigned long long>(slice) * nvox + mapped],
            power);
      }
    }
    f[k] = branchMajor[static_cast<unsigned long long>(k) * nvox + v] * factor;
  }
  packed[2 * v] = make_float4(f[0], f[1], f[2], f[3]);
  packed[2 * v + 1] = make_float4(f[4], f[5], f[6], f[7]);
}

// CTAC pre-scaling of a branch-major batch (the CSRB layout, one dense column
// per branch): out[k*nvox + v] = in[k*nvox + v] * (att[slice_k][M_k(v)] /
// 65535)^power for k < batchSize; factor 1 where sliceIndex[k] < 0. Same
// factor and index as csrq_pack_branch8_atten, without the [voxel][8]
// interleave, so cuSPARSE consumes the result unchanged.
extern "C" __global__ void compact_branch_batch_atten_scale(
    const float *branchMajor, const unsigned short *attSlices,
    const int *sliceIndex, float *out, unsigned long long nvox, int batchSize,
    int symmetryMode, int nx, int ny, int nz, const int *symIds,
    const int *zMirrors, float power) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      nvox * static_cast<unsigned long long>(batchSize);
  if (index >= total || nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const int branch = static_cast<int>(index / nvox);
  const unsigned long long voxel = index % nvox;
  float factor = 1.0f;
  const int slice = sliceIndex[branch];
  if (slice >= 0) {
    const unsigned long long plane =
        static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
    int valid = 0;
    const unsigned long long mapped = compact_atten_mapped_voxel(
        voxel, symmetryMode, symIds[branch], zMirrors[branch], nx, ny, nz,
        plane, &valid);
    if (valid != 0) {
      factor = compact_atten_decode(
          attSlices[static_cast<unsigned long long>(slice) * nvox + mapped],
          power);
    }
  }
  out[index] = branchMajor[index] * factor;
}

// compact_transform_backward_add_batch with the CTAC factor applied at the
// unrotated voxel: accum[M_k(v)] += rotated[k][v] * (att[slice_k * nvox +
// M_k(v)] / 65535)^power. Together with csrq_pack_branch8_atten this is the
// exact adjoint of the attenuated forward (same table, same index, same
// exponent on both sides).
extern "C" __global__ void compact_transform_backward_add_batch_atten(
    const float *rotatedBatch, double *accum, unsigned long long nvox,
    int batchSize, int symmetryMode, int nx, int ny, int nz,
    const int *symIds, const int *zMirrors, const unsigned short *attSlices,
    const int *sliceIndex, float power) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      nvox * static_cast<unsigned long long>(batchSize);
  if (index >= total || nx <= 0 || ny <= 0 || nz <= 0) {
    return;
  }
  const int branch = static_cast<int>(index / nvox);
  const unsigned long long voxel = index % nvox;
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  int valid = 0;
  const unsigned long long mapped = compact_atten_mapped_voxel(
      voxel, symmetryMode, symIds[branch], zMirrors[branch], nx, ny, nz, plane,
      &valid);
  if (valid == 0) {
    return;
  }
  float value = rotatedBatch[index];
  if (value == 0.0f) {
    return;
  }
  const int slice = sliceIndex[branch];
  if (slice >= 0) {
    value *= compact_atten_decode(
        attSlices[static_cast<unsigned long long>(slice) * nvox + mapped],
        power);
  }
  atomicAdd(&accum[mapped], static_cast<double>(value));
}

// ratioPacked[row*8 + b] = value * rowWeights[fullIndex] (value alone when
// rowWeights is null) where the (row, branch) maps to a full projection row,
// else 0: the sensitivity value * A^T W 1 over valid rows only (value is the
// window weight of a joint multi-window model, 1 otherwise; W carries detector
// masks and efficiency weights).
extern "C" __global__ void csrq_unit_ratio_k8(
    float *ratioPacked, float value, const double *rowWeights,
    unsigned long long fullProjectionCount, int symmetryMode,
    int compactRowBase, int compactRaysPerAngle, int compactNumAngles,
    int detNx, int detNz, int compactDetZ, int fullNumAngles,
    int fullRaysPerAngle, const int *symIds, const int *zMirrors, int rows) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      static_cast<unsigned long long>(rows) * CSRQ_K;
  if (index >= total || compactRaysPerAngle <= 0 || detNx <= 0) {
    return;
  }
  const int row = static_cast<int>(index / CSRQ_K);
  const int b = static_cast<int>(index % CSRQ_K);
  unsigned long long fullIndex = 0;
  float out = 0.0f;
  if (csrq_full_projection_index(compactRowBase + row, symmetryMode,
                                 compactRaysPerAngle, compactNumAngles, detNx,
                                 detNz, compactDetZ, fullNumAngles,
                                 fullRaysPerAngle, symIds[b], zMirrors[b],
                                 &fullIndex) &&
      fullIndex < fullProjectionCount) {
    out = rowWeights == nullptr
              ? value
              : static_cast<float>(static_cast<double>(value) *
                                   rowWeights[fullIndex]);
  }
  ratioPacked[index] = out;
}
