/**
 * Tattletale — Positron FWHT-128 CUDA kernel
 *
 * Incoherence-processing Hadamard transform for EXL3 quantization.
 * Each block is one warp (32 threads), each thread processes 4 half values
 * via a half4 load.
 *
 * Matches exllamav3's had_hf_r_128_inner exactly:
 *   1. Load half4 (4 half) per thread → 128 elements/warp.
 *   2. Optional pre-scale via __hmul2 on two half2.
 *   3. 4-element butterfly (fp32, per-thread).
 *   4. 32-element butterfly via __shfl_xor_sync (warp shuffle).
 *   5. Convert back to fp16 with norm factor.
 *   6. Optional post-scale via __hmul2.
 *   7. Store half4 per thread.
 *
 * Grid: (rows, cols/128).  Block: (32).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FULL_MASK 0xffffffff

// ─── 4-element butterfly (in-register, fp32) ──────────────────────────

__device__ __forceinline__ void butterfly4(
    float& h0, float& h1, float& h2, float& h3) {
  float s0 = h0 + h1;  float d0 = h0 - h1;
  float s1 = h2 + h3;  float d1 = h2 - h3;
  h0 = s0 + s1;  h1 = d0 + d1;
  h2 = s0 - s1;  h3 = d0 - d1;
}

// ─── 32-element warp-shuffle butterfly (fp32) ────────────────────────
// Matches shuffle_had_f4x32 in exllamav3's hadamard_inner.cuh.

__device__ __forceinline__ void shuffle_butterfly_f4x32(
    float& h0, float& h1, float& h2, float& h3, int lane_id) {
  #pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    uint32_t i0 = __float_as_uint(h0);
    uint32_t i1 = __float_as_uint(h1);
    uint32_t i2 = __float_as_uint(h2);
    uint32_t i3 = __float_as_uint(h3);
    uint64_t h01 = (uint64_t)i0 | (((uint64_t)i1) << 32);
    uint64_t h23 = (uint64_t)i2 | (((uint64_t)i3) << 32);
    uint64_t ph01 = __shfl_xor_sync(FULL_MASK, h01, i);
    uint64_t ph23 = __shfl_xor_sync(FULL_MASK, h23, i);
    float ph0 = __uint_as_float((uint32_t)(ph01 & 0xffffffff));
    float ph1 = __uint_as_float((uint32_t)(ph01 >> 32));
    float ph2 = __uint_as_float((uint32_t)(ph23 & 0xffffffff));
    float ph3 = __uint_as_float((uint32_t)(ph23 >> 32));
    int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
    i0 ^= sfm & 0x80000000;
    i1 ^= sfm & 0x80000000;
    i2 ^= sfm & 0x80000000;
    i3 ^= sfm & 0x80000000;
    h0 = __uint_as_float(i0) + ph0;
    h1 = __uint_as_float(i1) + ph1;
    h2 = __uint_as_float(i2) + ph2;
    h3 = __uint_as_float(i3) + ph3;
  }
}

// ─── Main kernel (half I/O, half scales) ──────────────────────────────
// Each thread: 4 half values (1 half4 = 2 half2).
// 32 threads = 128 elements per block.

__global__ __launch_bounds__(32) void hadamard_rotate_128_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    const half* __restrict__ pre_scale, const half* __restrict__ post_scale,
    float r_scale) {
  int row = blockIdx.x;
  int blk = blockIdx.y;
  int t   = threadIdx.x;      // 0..31

  // Pointer to this thread's 4 consecutive half values.
  size_t offset = (size_t)row * gridDim.y * 128 + blk * 128 + (size_t)t * 4;
  const half* in_ptr  = input  + offset;
  half*       out_ptr = output + offset;

  // Load 2 half2 = 4 half values per thread.
  const half2* src = reinterpret_cast<const half2*>(in_ptr);
  half2 vx = src[0];  // elements t*4+0, t*4+1
  half2 vy = src[1];  // elements t*4+2, t*4+3

  // Pre-scale (fp16, __hmul2 matches EXL3's __hmul2).
  if (pre_scale) {
    const half2* ps = reinterpret_cast<const half2*>(pre_scale);
    int sidx = blk * 32 + t;
    vx = __hmul2(vx, ps[sidx * 2 + 0]);
    vy = __hmul2(vy, ps[sidx * 2 + 1]);
  }

  // Convert to fp32.
  float h0 = __half2float(__low2half(vx));
  float h1 = __half2float(__high2half(vx));
  float h2 = __half2float(__low2half(vy));
  float h3 = __half2float(__high2half(vy));

  // 4-element butterfly (per-thread, fp32).
  butterfly4(h0, h1, h2, h3);

  // 32-element warp-shuffle butterfly.
  shuffle_butterfly_f4x32(h0, h1, h2, h3, t);

  // Apply norm factor and convert back to half2 (round-to-nearest-even).
  vx = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
  vy = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

  // Post-scale (fp16).
  if (post_scale) {
    const half2* ps = reinterpret_cast<const half2*>(post_scale);
    int sidx = blk * 32 + t;
    vx = __hmul2(vx, ps[sidx * 2 + 0]);
    vy = __hmul2(vy, ps[sidx * 2 + 1]);
  }

  // Store 2 half2 per thread.
  half2* dst = reinterpret_cast<half2*>(out_ptr);
  dst[0] = vx;
  dst[1] = vy;
}

// ─── Host entry point (extern "C" for Nim FFI) ────────────────────────

extern "C" {

int pkl_hadamard_rotate_128_cuda(
    const half* input, half* output,
    const half* pre_scale, const half* post_scale,
    float r_scale,
    int rows, int cols) {
  if (cols % 128 != 0) return -1;
  if (rows <= 0 || cols <= 0) return -1;
  int blocks = cols / 128;
  dim3 grid(rows, blocks);
  dim3 block(32);
  hadamard_rotate_128_kernel<<<grid, block, 0, 0>>>(
      input, output, pre_scale, post_scale, r_scale);
  return 0;
}

}  // extern "C"
