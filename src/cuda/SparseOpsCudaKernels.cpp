#include "cuda/SparseOpsCudaKernels.h"

#include <string>

namespace spect::cuda {

const char *projectionDomainKernelSource() {
  static const std::string source = std::string(R"CU(
// NVRTC source must be self-contained; host header constants are not visible here.
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
    const float *yhat, const float *fullProjection,
    unsigned long long fullProjectionCount, int symmetryMode,
    int compactRowBase, int compactRaysPerAngle, int compactNumAngles,
    int detNx, int detNz, int compactDetZ, int fullNumAngles,
    int fullRaysPerAngle, int symId, int zMirror, float *ratio,
    unsigned long long count, float eps) {
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
  float denom = yhat[index];
  if (denom < eps) {
    denom = eps;
  }
  ratio[index] = fullProjection[fullIndex] / denom;
}

extern "C" __global__ void compute_compact_ratio_chunk_batch(
    const float *yhatBatch, const float *fullProjection,
    unsigned long long fullProjectionCount, int symmetryMode,
    int compactRowBase, int compactRaysPerAngle, int compactNumAngles,
    int detNx, int detNz, int compactDetZ, int fullNumAngles,
    int fullRaysPerAngle, const int *symIds, const int *zMirrors,
    int batchSize, float *ratioBatch, unsigned long long rows, float eps) {
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
  float denom = yhatBatch[index];
  if (denom < eps) {
    denom = eps;
  }
  ratioBatch[index] = fullProjection[fullIndex] / denom;
}

)CU") + R"CU(
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

)CU" + R"CU(
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

)CU" + R"CU(
extern "C" __global__ void conv2d_y_mirror_frames(
    const float *input, float *output, const float *kernel, int radius,
    int detNz, int detNx, unsigned long long frameCount) {
  const unsigned long long frameSize =
      static_cast<unsigned long long>(detNz) * static_cast<unsigned long long>(detNx);
  const unsigned long long total = frameCount * frameSize;
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  const int x = static_cast<int>(index % static_cast<unsigned long long>(detNx));
  const int y = static_cast<int>((index / static_cast<unsigned long long>(detNx)) %
                                 static_cast<unsigned long long>(detNz));
  const unsigned long long frame = index / frameSize;
  const unsigned long long base = frame * frameSize;

  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    const int yy = mirror_index(y + k, detNz);
    sum += kernel[k + radius] *
           input[base + static_cast<unsigned long long>(yy) *
                            static_cast<unsigned long long>(detNx) +
                 static_cast<unsigned long long>(x)];
  }
  output[index] = sum;
}

extern "C" __global__ void conv2d_x_mirror_frames(
    const float *input, float *output, const float *kernel, int radius,
    int detNz, int detNx, unsigned long long frameCount) {
  const unsigned long long frameSize =
      static_cast<unsigned long long>(detNz) * static_cast<unsigned long long>(detNx);
  const unsigned long long total = frameCount * frameSize;
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  const int x = static_cast<int>(index % static_cast<unsigned long long>(detNx));
  const int y = static_cast<int>((index / static_cast<unsigned long long>(detNx)) %
                                 static_cast<unsigned long long>(detNz));
  const unsigned long long frame = index / frameSize;
  const unsigned long long base = frame * frameSize;

  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    const int xx = mirror_index(x + k, detNx);
    sum += kernel[k + radius] *
           input[base + static_cast<unsigned long long>(y) *
                            static_cast<unsigned long long>(detNx) +
                 static_cast<unsigned long long>(xx)];
  }
  output[index] = sum;
}

)CU";
  return source.c_str();
}

} // namespace spect::cuda
