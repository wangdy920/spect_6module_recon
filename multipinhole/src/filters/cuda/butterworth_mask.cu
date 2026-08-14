// Device-only helper kernels. Compiled with nvcc --fatbin for sm_86/89/120.
#include <math.h>

extern "C" __global__ void apply_butterworth_mask_f(
    float2* data, const float* mask, unsigned long long total) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const float scale = mask[index];
  data[index].x *= scale;
  data[index].y *= scale;
}

// Embed the unpadded input at the center of a zero-filled padded volume
// (output-stage filter: 2x zero padding per axis, mirroring the 6module
// prototype k_butter_embed_centered).
extern "C" __global__ void butter_embed_centered_f(
    const float* input, float* padded,
    int nx, int ny, int nz,
    int padded_nx, int padded_ny,
    int offset_x, int offset_y, int offset_z,
    unsigned long long input_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= input_count) return;
  const int z = static_cast<int>(
      index / (static_cast<unsigned long long>(nx) * ny));
  const unsigned long long rem =
      index - static_cast<unsigned long long>(z) * nx * ny;
  const int y = static_cast<int>(rem / nx);
  const int x = static_cast<int>(rem - static_cast<unsigned long long>(y) * nx);
  const unsigned long long padded_index =
      static_cast<unsigned long long>(z + offset_z) * padded_nx * padded_ny +
      static_cast<unsigned long long>(y + offset_y) * padded_nx + x + offset_x;
  padded[padded_index] = input[index];
}

// Crop the centered region back out of the padded volume and apply the inverse
// FFT 1/N_padded scale. Final output filtering is linear, so negative
// Butterworth sidelobes are preserved exactly as in the 6module prototype.
extern "C" __global__ void butter_crop_centered_f(
    const float* padded, float* output,
    int nx, int ny, int nz,
    int padded_nx, int padded_ny,
    int offset_x, int offset_y, int offset_z,
    float inverse_fft_size,
    unsigned long long input_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= input_count) return;
  const int z = static_cast<int>(
      index / (static_cast<unsigned long long>(nx) * ny));
  const unsigned long long rem =
      index - static_cast<unsigned long long>(z) * nx * ny;
  const int y = static_cast<int>(rem / nx);
  const int x = static_cast<int>(rem - static_cast<unsigned long long>(y) * nx);
  const unsigned long long padded_index =
      static_cast<unsigned long long>(z + offset_z) * padded_nx * padded_ny +
      static_cast<unsigned long long>(y + offset_y) * padded_nx + x + offset_x;
  float value = padded[padded_index] * inverse_fft_size;
  if (!isfinite(value)) value = 0.0f;
  output[index] = value;
}
