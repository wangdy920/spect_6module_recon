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

extern "C" __device__ __forceinline__ int compact_atten_full_angle(
    int symmetryMode, int compactRow, int compactRaysPerAngle,
    int compactNumAngles, int fullNumAngles, int symId) {
  const int compactAngle = compactRow / compactRaysPerAngle;
  int fullAngle = 0;
  if (symmetryMode == 45) {
    const int q = fullNumAngles / 8;
    if ((symId & 1) == 0) {
      if (compactAngle >= q) {
        return -1;
      }
      fullAngle = symId * q + compactAngle;
    } else {
      if (compactAngle <= 0) {
        return -1;
      }
      fullAngle = symId * q + (q - compactAngle);
    }
  } else {
    fullAngle = compactAngle + symId * compactNumAngles;
  }
  if (fullAngle < 0 || fullAngle >= fullNumAngles) {
    return -1;
  }
  return fullAngle;
}

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

extern "C" __device__ __forceinline__ float compact_atten_factor(
    const unsigned short *att, int fullAngle, unsigned long long col,
    int symmetryMode, int symId, int zMirror, int nx, int ny, int nz,
    unsigned long long plane, unsigned long long nvox) {
  if (fullAngle < 0) {
    return 1.0f;
  }
  int validVoxel = 0;
  const unsigned long long mapped = compact_atten_mapped_voxel(
      col, symmetryMode, symId, zMirror, nx, ny, nz, plane, &validVoxel);
  if (validVoxel == 0) {
    return 1.0f;
  }
  const unsigned long long attIndex =
      static_cast<unsigned long long>(fullAngle) * nvox + mapped;
  return static_cast<float>(att[attIndex]) *
         (1.0f / static_cast<float>(kAttenUint16Scale));
}

// Bake per-angle attenuation factors into compact chunk matrix values:
// baked[t] = values[t] * att[fullAngle(row(t)) * nvox + M_b(col(t))] / 65535,
// where M_b maps the rotated/mirrored voxel index back to the unrotated frame
// (same mapping as compact_transform_backward_add). One block per compact row;
// rows whose compact angle has no valid full angle (SYM_45Z edges) copy the
// original values (their ratios are forced to zero elsewhere).
extern "C" __global__ void compact_bake_atten_values(
    const float *values, const int *rowPtr, const int *colIdx,
    const unsigned short *att, float *baked, int rows, int symmetryMode,
    int compactRowBase, int compactRaysPerAngle, int compactNumAngles,
    int fullNumAngles, int symId, int zMirror, int nx, int ny, int nz) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows || compactRaysPerAngle <= 0 || nx <= 0 || ny <= 0 ||
      nz <= 0) {
    return;
  }
  const int compactRow = compactRowBase + row;
  const int fullAngle = compact_atten_full_angle(
      symmetryMode, compactRow, compactRaysPerAngle, compactNumAngles,
      fullNumAngles, symId);
  const int start = rowPtr[row];
  const int end = rowPtr[row + 1];
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const unsigned long long nvox = plane * static_cast<unsigned long long>(nz);

  for (int t = start + static_cast<int>(threadIdx.x); t < end;
       t += static_cast<int>(blockDim.x)) {
    const float factor = compact_atten_factor(
        att, fullAngle, static_cast<unsigned long long>(colIdx[t]),
        symmetryMode, symId, zMirror, nx, ny, nz, plane, nvox);
    baked[t] = values[t] * factor;
  }
}

extern "C" __global__ void compact_atten_fused_spmv(
    const float *values, const int *rowPtr, const int *colIdx,
    const unsigned short *att, const float *x, float *y, int rows,
    int symmetryMode, int compactRowBase, int compactRaysPerAngle,
    int compactNumAngles, int fullNumAngles, int symId, int zMirror, int nx,
    int ny, int nz) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows || compactRaysPerAngle <= 0 || nx <= 0 || ny <= 0 ||
      nz <= 0) {
    return;
  }
  const int compactRow = compactRowBase + row;
  const int fullAngle = compact_atten_full_angle(
      symmetryMode, compactRow, compactRaysPerAngle, compactNumAngles,
      fullNumAngles, symId);
  const int start = rowPtr[row];
  const int end = rowPtr[row + 1];
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const unsigned long long nvox = plane * static_cast<unsigned long long>(nz);

  float sum = 0.0f;
  for (int t = start + static_cast<int>(threadIdx.x); t < end;
       t += static_cast<int>(blockDim.x)) {
    const int col = colIdx[t];
    const float factor = compact_atten_factor(
        att, fullAngle, static_cast<unsigned long long>(col), symmetryMode,
        symId, zMirror, nx, ny, nz, plane, nvox);
    sum += values[t] * factor * x[col];
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    y[row] = partial[0];
  }
}

extern "C" __global__ void compact_atten_fused_spmvt(
    const float *values, const int *rowPtr, const int *colIdx,
    const unsigned short *att, const float *x, float *y, int rows,
    int symmetryMode, int compactRowBase, int compactRaysPerAngle,
    int compactNumAngles, int fullNumAngles, int symId, int zMirror, int nx,
    int ny, int nz) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows || compactRaysPerAngle <= 0 || nx <= 0 || ny <= 0 ||
      nz <= 0) {
    return;
  }
  const float rowInput = x[row];
  if (rowInput == 0.0f) {
    return;
  }
  const int compactRow = compactRowBase + row;
  const int fullAngle = compact_atten_full_angle(
      symmetryMode, compactRow, compactRaysPerAngle, compactNumAngles,
      fullNumAngles, symId);
  const int start = rowPtr[row];
  const int end = rowPtr[row + 1];
  const unsigned long long plane =
      static_cast<unsigned long long>(nx) * static_cast<unsigned long long>(ny);
  const unsigned long long nvox = plane * static_cast<unsigned long long>(nz);

  for (int t = start + static_cast<int>(threadIdx.x); t < end;
       t += static_cast<int>(blockDim.x)) {
    const int col = colIdx[t];
    const float factor = compact_atten_factor(
        att, fullAngle, static_cast<unsigned long long>(col), symmetryMode,
        symId, zMirror, nx, ny, nz, plane, nvox);
    atomicAdd(&y[col], values[t] * factor * rowInput);
  }
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
                                                 int rows) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
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
