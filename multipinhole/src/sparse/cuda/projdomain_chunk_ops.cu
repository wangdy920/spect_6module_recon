// Device-only helper kernels. Compiled with nvcc --fatbin for sm_86/89/120.
static constexpr float kAttenUint16Scale = 65535.0f;

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

extern "C" __global__ void compute_ratio_chunk(
    const float *yhat, const float *projections, unsigned long long projOffset,
    float *ratio, unsigned long long count, float eps) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  float denom = yhat[index];
  if (denom < eps) {
    denom = eps;
  }
  ratio[index] = projections[projOffset + index] / denom;
}

extern "C" __global__ void compute_compact_ratio_chunk(
    const float *yhat, const double *fullProjection,
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
  if (denom < eps) {
    denom = eps;
  }
  ratio[index] = static_cast<float>(fullProjection[fullIndex] / denom);
}

extern "C" __global__ void compute_compact_ratio_chunk_batch(
    const float *yhatBatch, const double *fullProjection,
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
  if (denom < eps) {
    denom = eps;
  }
  ratioBatch[index] =
      static_cast<float>(fullProjection[fullIndex] / denom);
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

