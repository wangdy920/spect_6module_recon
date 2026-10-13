// Device-only helper kernels. Compiled with nvcc --fatbin for sm_86/89/120.

extern "C" __device__ int spect_mirror_index(int idx, int dim) {
  if (dim <= 1) {
    return 0;
  }
  const int period = 2 * dim - 2;
  idx %= period;
  if (idx < 0) idx += period;
  return idx < dim ? idx : period - idx;
}

extern "C" __global__ void conv_x_mirror_3d(const double* input,
                                             double* output,
                                             int Nx, int Ny, int Nz,
                                             const double* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int xx = spect_mirror_index(x + t, Nx);
    const unsigned long long src =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_y_mirror_3d(const double* input,
                                             double* output,
                                             int Nx, int Ny, int Nz,
                                             const double* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int yy = spect_mirror_index(y + t, Ny);
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_z_mirror_3d(const double* input,
                                             double* output,
                                             int Nx, int Ny, int Nz,
                                             const double* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int zz = spect_mirror_index(z + t, Nz);
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_x_zero_3d(const double* input,
                                           double* output,
                                           int Nx, int Ny, int Nz,
                                           const double* kernel,
                                           int radius,
                                           unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int xx = x + t;
    if (xx < 0 || xx >= Nx) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_y_zero_3d(const double* input,
                                           double* output,
                                           int Nx, int Ny, int Nz,
                                           const double* kernel,
                                           int radius,
                                           unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int yy = y + t;
    if (yy < 0 || yy >= Ny) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_z_zero_3d(const double* input,
                                           double* output,
                                           int Nx, int Ny, int Nz,
                                           const double* kernel,
                                           int radius,
                                           unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int zz = z + t;
    if (zz < 0 || zz >= Nz) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

// CASToR z-boundary compensation: divide each voxel by znorm[z] (length Nz).
extern "C" __global__ void div_znorm_3d(double* data,
                                         const double* znorm,
                                         int Nx, int Ny, int Nz,
                                         unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  double d = znorm[z];
  if (!isfinite(d) || d < 1e-12) d = 1e-12;
  data[index] /= d;
}

extern "C" __global__ void conv_mirror_adjoint_3d(
    const double* input, double* output, int Nx, int Ny, int Nz,
    const double* kernel, int radius, int axis, unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  const int coord = axis == 0 ? x : (axis == 1 ? y : z);
  const int dim = axis == 0 ? Nx : (axis == 1 ? Ny : Nz);

  if (dim <= 1) {
    double kernelSum = 0.0;
    for (int t = -radius; t <= radius; ++t)
      kernelSum += kernel[t + radius];
    output[index] = kernelSum * input[index];
    return;
  }

  const int period = 2 * dim - 2;
  double sum = 0.0;
  for (int t = -radius; t <= radius; ++t) {
    const int lower = t;
    const int upper = t + dim - 1;
    int q = coord;
    while (q < lower) q += period;
    while (q - period >= lower) q -= period;
    for (; q <= upper; q += period) {
      const int sourceCoord = q - t;
      const unsigned long long src =
          axis == 0
              ? static_cast<unsigned long long>(sourceCoord) +
                    static_cast<unsigned long long>(Nx) * y +
                    static_cast<unsigned long long>(Nx) * Ny * z
              : (axis == 1
                     ? static_cast<unsigned long long>(x) +
                           static_cast<unsigned long long>(Nx) * sourceCoord +
                           static_cast<unsigned long long>(Nx) * Ny * z
                     : static_cast<unsigned long long>(x) +
                           static_cast<unsigned long long>(Nx) * y +
                           static_cast<unsigned long long>(Nx) * Ny *
                               sourceCoord);
      sum += kernel[t + radius] * input[src];
    }
    if (coord != 0 && coord != dim - 1) {
      q = period - coord;
      while (q < lower) q += period;
      while (q - period >= lower) q -= period;
      for (; q <= upper; q += period) {
        const int sourceCoord = q - t;
        const unsigned long long src =
            axis == 0
                ? static_cast<unsigned long long>(sourceCoord) +
                      static_cast<unsigned long long>(Nx) * y +
                      static_cast<unsigned long long>(Nx) * Ny * z
                : (axis == 1
                       ? static_cast<unsigned long long>(x) +
                             static_cast<unsigned long long>(Nx) * sourceCoord +
                             static_cast<unsigned long long>(Nx) * Ny * z
                       : static_cast<unsigned long long>(x) +
                             static_cast<unsigned long long>(Nx) * y +
                             static_cast<unsigned long long>(Nx) * Ny *
                                 sourceCoord);
        sum += kernel[t + radius] * input[src];
      }
    }
  }
  output[index] = sum;
}

extern "C" __global__ void conv_x_mirror_3d_f(const float* input,
                                               float* output,
                                               int Nx, int Ny, int Nz,
                                               const float* kernel,
                                               int radius,
                                               unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int xx = spect_mirror_index(x + t, Nx);
    const unsigned long long src =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_y_mirror_3d_f(const float* input,
                                               float* output,
                                               int Nx, int Ny, int Nz,
                                               const float* kernel,
                                               int radius,
                                               unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int yy = spect_mirror_index(y + t, Ny);
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_z_mirror_3d_f(const float* input,
                                               float* output,
                                               int Nx, int Ny, int Nz,
                                               const float* kernel,
                                               int radius,
                                               unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int zz = spect_mirror_index(z + t, Nz);
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_x_zero_3d_f(const float* input,
                                             float* output,
                                             int Nx, int Ny, int Nz,
                                             const float* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int xx = x + t;
    if (xx < 0 || xx >= Nx) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_y_zero_3d_f(const float* input,
                                             float* output,
                                             int Nx, int Ny, int Nz,
                                             const float* kernel,
                                             int radius,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int yy = y + t;
    if (yy < 0 || yy >= Ny) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(z);
    sum += kernel[t + radius] * input[src];
  }
  output[index] = sum;
}

extern "C" __global__ void conv_z_zero_3d_f(const float* input,
                                             float* output,
                                             int Nx, int Ny, int Nz,
                                             const float* kernel,
                                             int radius,
                                             const float* znorm,
                                             unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int zz = z + t;
    if (zz < 0 || zz >= Nz) continue;
    const unsigned long long src =
        static_cast<unsigned long long>(x) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(y) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    sum += kernel[t + radius] * input[src];
  }
  float d = znorm[z];
  if (!isfinite(d) || d < 1e-12f) d = 1e-12f;
  output[index] = sum / d;
}

extern "C" __global__ void conv_mirror_adjoint_3d_f(
    const float* input, float* output, int Nx, int Ny, int Nz,
    const float* kernel, int radius, int axis, unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));
  const int coord = axis == 0 ? x : (axis == 1 ? y : z);
  const int dim = axis == 0 ? Nx : (axis == 1 ? Ny : Nz);

  if (dim <= 1) {
    float kernelSum = 0.0f;
    for (int t = -radius; t <= radius; ++t)
      kernelSum += kernel[t + radius];
    output[index] = kernelSum * input[index];
    return;
  }

  const int period = 2 * dim - 2;
  float sum = 0.0f;
  for (int t = -radius; t <= radius; ++t) {
    const int lower = t;
    const int upper = t + dim - 1;
    int q = coord;
    while (q < lower) q += period;
    while (q - period >= lower) q -= period;
    for (; q <= upper; q += period) {
      const int sourceCoord = q - t;
      const unsigned long long src =
          axis == 0
              ? static_cast<unsigned long long>(sourceCoord) +
                    static_cast<unsigned long long>(Nx) * y +
                    static_cast<unsigned long long>(Nx) * Ny * z
              : (axis == 1
                     ? static_cast<unsigned long long>(x) +
                           static_cast<unsigned long long>(Nx) * sourceCoord +
                           static_cast<unsigned long long>(Nx) * Ny * z
                     : static_cast<unsigned long long>(x) +
                           static_cast<unsigned long long>(Nx) * y +
                           static_cast<unsigned long long>(Nx) * Ny *
                               sourceCoord);
      sum += kernel[t + radius] * input[src];
    }
    if (coord != 0 && coord != dim - 1) {
      q = period - coord;
      while (q < lower) q += period;
      while (q - period >= lower) q -= period;
      for (; q <= upper; q += period) {
        const int sourceCoord = q - t;
        const unsigned long long src =
            axis == 0
                ? static_cast<unsigned long long>(sourceCoord) +
                      static_cast<unsigned long long>(Nx) * y +
                      static_cast<unsigned long long>(Nx) * Ny * z
                : (axis == 1
                       ? static_cast<unsigned long long>(x) +
                             static_cast<unsigned long long>(Nx) * sourceCoord +
                             static_cast<unsigned long long>(Nx) * Ny * z
                       : static_cast<unsigned long long>(x) +
                             static_cast<unsigned long long>(Nx) * y +
                             static_cast<unsigned long long>(Nx) * Ny *
                                 sourceCoord);
        sum += kernel[t + radius] * input[src];
      }
    }
  }
  output[index] = sum;
}

extern "C" __device__ double ct_edge_weight(const double* ct,
                                              bool hasCt,
                                              unsigned long long a,
                                              unsigned long long b,
                                              double sigma,
                                              double floorWeight) {
  if (!hasCt) return 1.0;
  const double d = (ct[a] - ct[b]) / sigma;
  return floorWeight + (1.0 - floorWeight) * exp(-0.5 * d * d);
}

extern "C" __global__ void atten_line_integral_3d(
    const double* mu, const double* stepFxByAngle,
    const double* stepFyByAngle, int Nx, int Ny, int Nz, int numAngles,
    double step, int maxSteps, unsigned short* output,
    unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const unsigned long long plane =
      static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
  const unsigned long long voxels = plane * static_cast<unsigned long long>(Nz);
  const int angle = static_cast<int>(index / voxels);
  const unsigned long long voxel = index - static_cast<unsigned long long>(angle) * voxels;
  const int z = static_cast<int>(voxel / plane);
  const unsigned long long inPlane = voxel - static_cast<unsigned long long>(z) * plane;
  const int y = static_cast<int>(inPlane / static_cast<unsigned long long>(Nx));
  const int x = static_cast<int>(inPlane - static_cast<unsigned long long>(y) * Nx);
  const double stepFx = stepFxByAngle[angle];
  const double stepFy = stepFyByAngle[angle];
  double fx = static_cast<double>(x) + 0.5 * stepFx;
  double fy = static_cast<double>(y) + 0.5 * stepFy;
  double integral = 0.0;
  const double* slice = mu + static_cast<unsigned long long>(z) * plane;
  for (int k = 0; k < maxSteps; ++k) {
    if (fx <= -1.0 || fy <= -1.0 || fx >= static_cast<double>(Nx) ||
        fy >= static_cast<double>(Ny))
      break;
    const double cx = fmin(fmax(fx, 0.0), static_cast<double>(Nx - 1));
    const double cy = fmin(fmax(fy, 0.0), static_cast<double>(Ny - 1));
    const int x0 = static_cast<int>(floor(cx));
    const int iy0 = static_cast<int>(floor(cy));
    const int x1 = x0 + 1 < Nx ? x0 + 1 : Nx - 1;
    const int iy1 = iy0 + 1 < Ny ? iy0 + 1 : Ny - 1;
    const double tx = cx - static_cast<double>(x0);
    const double ty = cy - static_cast<double>(iy0);
    const double v00 = slice[static_cast<unsigned long long>(iy0) * Nx + x0];
    const double v10 = slice[static_cast<unsigned long long>(iy0) * Nx + x1];
    const double v01 = slice[static_cast<unsigned long long>(iy1) * Nx + x0];
    const double v11 = slice[static_cast<unsigned long long>(iy1) * Nx + x1];
    const double value = (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) +
                         ty * ((1.0 - tx) * v01 + tx * v11);
    if (value > 0.0) integral += value * step;
    fx += stepFx;
    fy += stepFy;
  }
  const double factor = fmin(fmax(exp(-integral), 0.0), 1.0);
  output[index] = static_cast<unsigned short>(floor(factor * 65535.0 + 0.5));
}

extern "C" __global__ void rdp_grad_3d(const double* input,
                                        const unsigned char* mask,
                                        const double* ct,
                                        bool hasMask,
                                        bool hasCt,
                                        bool skipOutsideMask,
                                        int Nx, int Ny, int Nz,
                                        double eps,
                                        double gamma,
                                        double ctSigma,
                                        double ctWeightFloor,
                                        double* output,
                                        unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  if (hasMask && mask[index] == 0) {
    output[index] = 0.0;
    return;
  }

  const double fj = input[index];
  double g = 0.0;

  const int dx[6] = {-1, 1, 0, 0, 0, 0};
  const int dy[6] = {0, 0, -1, 1, 0, 0};
  const int dz[6] = {0, 0, 0, 0, -1, 1};

  for (int n = 0; n < 6; ++n) {
    const int xx = x + dx[n];
    const int yy = y + dy[n];
    const int zz = z + dz[n];
    if (xx < 0 || xx >= Nx || yy < 0 || yy >= Ny || zz < 0 || zz >= Nz) {
      continue;
    }
    const unsigned long long neighbor =
        static_cast<unsigned long long>(xx) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(yy) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(zz);
    if (skipOutsideMask && hasMask && mask[neighbor] == 0) {
      continue;
    }
    const double fk = input[neighbor];
    const double diff = fj - fk;
    const double absDiff = fabs(diff);
    const double signDiff = diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
    double denom = fj + fk + gamma * absDiff + 2.0 * eps;
    if (denom < 1.0e-9) {
      denom = 1.0e-9;
    }
    const double term =
        2.0 * diff * denom - diff * diff * (1.0 + gamma * signDiff);
    g += ct_edge_weight(ct, hasCt, index, neighbor, ctSigma,
                        ctWeightFloor) * term / (denom * denom);
  }

  output[index] = g;
}

extern "C" __device__ void tv_compute_p(const double* input,
                                         const unsigned char* mask,
                                         const double* ct,
                                         bool hasMask,
                                         bool hasCt,
                                         bool skipOutsideMask,
                                         int px, int py, int pz,
                                         int Nx, int Ny, int Nz,
                                         double eps,
                                         double ctSigma,
                                         double ctWeightFloor,
                                         double* out_gx, double* out_gy, double* out_gz) {
  const unsigned long long idx =
      static_cast<unsigned long long>(px) +
      static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py) +
      static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
          static_cast<unsigned long long>(pz);

  if (hasMask && mask[idx] == 0) {
    *out_gx = 0.0;
    *out_gy = 0.0;
    *out_gz = 0.0;
    return;
  }

  const double fv = input[idx];
  double gx = 0.0, gy = 0.0, gz = 0.0;
  double mag2 = eps * eps;

  if (px + 1 < Nx) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px + 1) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0) {
      const double d = input[nidx] - fv;
      const double w = ct_edge_weight(ct, hasCt, idx, nidx, ctSigma,
                                      ctWeightFloor);
      gx = w * d;
      mag2 += w * d * d;
    }
  }
  if (py + 1 < Ny) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py + 1) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0) {
      const double d = input[nidx] - fv;
      const double w = ct_edge_weight(ct, hasCt, idx, nidx, ctSigma,
                                      ctWeightFloor);
      gy = w * d;
      mag2 += w * d * d;
    }
  }
  if (pz + 1 < Nz) {
    const unsigned long long nidx =
        static_cast<unsigned long long>(px) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(py) +
        static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny) *
            static_cast<unsigned long long>(pz + 1);
    if (!skipOutsideMask || !hasMask || mask[nidx] != 0) {
      const double d = input[nidx] - fv;
      const double w = ct_edge_weight(ct, hasCt, idx, nidx, ctSigma,
                                      ctWeightFloor);
      gz = w * d;
      mag2 += w * d * d;
    }
  }

  const double mag = sqrt(mag2);
  *out_gx = gx / mag;
  *out_gy = gy / mag;
  *out_gz = gz / mag;
}

extern "C" __global__ void tv_grad_3d(const double* input,
                                       const unsigned char* mask,
                                       const double* ct,
                                       bool hasMask,
                                       bool hasCt,
                                       bool skipOutsideMask,
                                       int Nx, int Ny, int Nz,
                                       double eps,
                                       double ctSigma,
                                       double ctWeightFloor,
                                       double* output,
                                       unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;

  const int x = static_cast<int>(index % static_cast<unsigned long long>(Nx));
  const unsigned long long yz = index / static_cast<unsigned long long>(Nx);
  const int y = static_cast<int>(yz % static_cast<unsigned long long>(Ny));
  const int z = static_cast<int>(yz / static_cast<unsigned long long>(Ny));

  if (hasMask && mask[index] == 0) {
    output[index] = 0.0;
    return;
  }

  /* Compute normalized gradient p(x,y,z) = nabla f / |nabla f| */
  double px, py, pz;
  tv_compute_p(input, mask, ct, hasMask, hasCt, skipOutsideMask,
               x, y, z, Nx, Ny, Nz, eps, ctSigma, ctWeightFloor,
               &px, &py, &pz);

  double diverg = 0.0;

  /* X divergence */
  if (x > 0 && (!skipOutsideMask || !hasMask ||
                mask[index - 1] != 0)) {
    double prev_px, prev_py, prev_pz;
    tv_compute_p(input, mask, ct, hasMask, hasCt, skipOutsideMask,
                 x - 1, y, z, Nx, Ny, Nz, eps, ctSigma, ctWeightFloor,
                 &prev_px, &prev_py, &prev_pz);
    diverg += px - prev_px;
  } else {
    diverg += px;
  }

  /* Y divergence */
  if (y > 0 && (!skipOutsideMask || !hasMask ||
                mask[index - static_cast<unsigned long long>(Nx)] != 0)) {
    double prev_px, prev_py, prev_pz;
    tv_compute_p(input, mask, ct, hasMask, hasCt, skipOutsideMask,
                 x, y - 1, z, Nx, Ny, Nz, eps, ctSigma, ctWeightFloor,
                 &prev_px, &prev_py, &prev_pz);
    diverg += py - prev_py;
  } else {
    diverg += py;
  }

  /* Z divergence */
  const unsigned long long slice =
      static_cast<unsigned long long>(Nx) * static_cast<unsigned long long>(Ny);
  if (z > 0 && (!skipOutsideMask || !hasMask ||
                mask[index - slice] != 0)) {
    double prev_px, prev_py, prev_pz;
    tv_compute_p(input, mask, ct, hasMask, hasCt, skipOutsideMask,
                 x, y, z - 1, Nx, Ny, Nz, eps, ctSigma, ctWeightFloor,
                 &prev_px, &prev_py, &prev_pz);
    diverg += pz - prev_pz;
  } else {
    diverg += pz;
  }

  output[index] = -diverg;
}
