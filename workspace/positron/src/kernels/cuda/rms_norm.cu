/**
 * Tattletale — Positron RMSNorm CUDA kernel
 * For now this kernel is made intentionally compatible
 * with Exllamav3 RMSNorm to facilitate exact testing
 *
 * RMSNorm: y = x / sqrt(mean(x²) + ε) * w
 *
 * Three kernel strategies selected at launch time by hidden dimension:
 *
 * 1. Warp-only path  (dim ≤ 256): Single warp, no __syncthreads.
 *    ~2–3× faster for QK norms (head_dim = 64/128).
 *
 * 2. Medium block    (256 < dim ≤ 2048): 1024 threads, no smem caching.
 *    Bit-identical to the exllamav3 production kernel.
 *
 * 3. Wide block      (dim > 2048): 1024 threads, shared-memory input cache.
 *    Halves HBM reads for x (smem read in pass 2 instead of global re-read).
 *
 * Numerical compatibility:
 *   Paths 2 and 3 use 1024 threads and the exact same FMA order as exllamav3
 *   when run with half I/O and half weights — bit-identical results.
 *   Path 1 (warp-only) differs only by FP32 addition associativity.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

// ========================================================================
// Configuration Constants
// ========================================================================

/// Maximum threads per block (sm_90+ limit).
#define MAX_THREADS 1024

/// Warp size on all NVIDIA GPUs.
#define WARP_SIZE   32

/// Below this threshold use the warp-only path (no __syncthreads).
#define WARP_ONLY_DIM_THRESHOLD  256

/// Above this threshold enable shared-memory input caching.
#define SMEM_CACHE_DIM_THRESHOLD  2048

// ========================================================================
// Type helpers (match exllamav3 layout for bit-exact loads/stores)
// ========================================================================

/// Packed half4 — two half2 values side by side, 64-bit aligned.
/// Enables coalesced 64-bit reads from global memory.
typedef struct alignas(8) { half2 x, y; } HalfPair4;

/// Cast a half pointer to HalfPair4.
static inline __device__ const HalfPair4* as_half_pair4(const half* p) {
  return reinterpret_cast<const HalfPair4*>(p);
}

// ========================================================================
// Reduction Primitives
// ========================================================================

/**
 * Cross-warp block reduction via warp shuffle + shared memory.
 *
 * Algorithm:
 *   1. Intra-warp butterfly shuffle (16 -> 8 -> 4 -> 2 -> 1).
 *   2. Lane 0 of each warp writes partial sum to shared.
 *   3. __syncthreads() -- all warps converge.
 *   4. One warp reads all partials from shared and finalises.
 *
 * @param partial   Per-thread partial sum of x[i]².
 * @param warp_id   threadIdx.x / 32
 * @param lane_id   threadIdx.x % 32
 * @param nwarps    Total warps in block (compile-time template arg).
 * @return          Fully reduced sum, broadcast to all threads.
 */
template <int NWARPS>
__device__ __forceinline__ float
block_reduce_sum(float partial, int warp_id, int lane_id) {
  // Stage 1: intra-warp butterfly reduction.
  // __shfl_xor_sync with 0xffffffff mask includes all active lanes.
  // No synchronisation needed -- shuffles are warp-synchronous.
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
  }

  // Stage 2: inter-warp exchange via shared memory.
  __shared__ float warp_partials[NWARPS];
  if (lane_id == 0) {
    warp_partials[warp_id] = partial;
  }
  __syncthreads();

  // Stage 3: one warp collects and finalises.
  partial = (lane_id < NWARPS) ? warp_partials[lane_id] : 0.0f;
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
  }
  return partial;
}

/**
 * Intra-warp reduction only -- no shared memory, no __syncthreads.
 * Used for the warp-only path (dim <= 256) where one warp is one block.
 */
__device__ __forceinline__ float
warp_reduce_sum(float partial) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
  }
  return partial;
}

// ========================================================================
// Vectorised half <-> float conversion helpers
// ========================================================================

/**
 * Load four consecutive half values from global memory into a float4.
 * Uses two 64-bit coalesced loads via HalfPair4.
 */
__device__ __forceinline__ void
load_half4_as_float4(float4& out, const half* addr) {
  const HalfPair4* src = as_half_pair4(addr);
  HalfPair4 tmp;
  tmp.x = __ldg(&src->x);     // 64-bit coalesced load
  tmp.y = __ldg(&src->y);     // 64-bit coalesced load
  out.x = __half2float(tmp.x.x);
  out.y = __half2float(tmp.x.y);
  out.z = __half2float(tmp.y.x);
  out.w = __half2float(tmp.y.y);
}

/**
 * Store a float4 as four consecutive half values.
 * Uses two 64-bit coalesced writes.
 */
__device__ __forceinline__ void
store_float4_as_half4(const float4& src, half* addr) {
  HalfPair4* dst = reinterpret_cast<HalfPair4*>(addr);
  dst->x = __halves2half2(__float2half_rn(src.x), __float2half_rn(src.y));
  dst->y = __halves2half2(__float2half_rn(src.z), __float2half_rn(src.w));
}

// ========================================================================
// Arithmetic helpers (EXL3-compatible FMA order)
// ========================================================================

/**
 * Accumulate sum of squares of a float4 into a running total.
 * Uses fmaf in the same order as exllamav3's sum_sq4().
 */
__device__ __forceinline__ float
accumulate_sum_of_squares(float running, const float4& v) {
  running = fmaf(v.x, v.x, running);
  running = fmaf(v.y, v.y, running);
  running = fmaf(v.z, v.z, running);
  running = fmaf(v.w, v.w, running);
  return running;
}

/**
 * Apply RMS normalisation and weight scaling to one float4.
 * out[i] = x[i] * w[i] * (1 / rms)
 *
 * Multiplication order: (x * w) * inv_rms -- matches EXL3 exactly.
 * This order is the dominant source of numerical variation between
 * EXL3 fixtures and HF-reference fixtures.
 */
__device__ __forceinline__ void
scale_by_weight_and_rms(float4& x, const float4& w, float inv_rms) {
  x.x = x.x * w.x * inv_rms;
  x.y = x.y * w.y * inv_rms;
  x.z = x.z * w.z * inv_rms;
  x.w = x.w * w.w * inv_rms;
}
// ========================================================================
// Kernel 1: Warp-only RMSNorm  (dim <= 256 -- QK norms)
// ========================================================================
/**
 * Warp-only RMSNorm kernel.
 *
 * Each block is exactly one warp (32 threads) processing one row.
 * No shared memory, no __syncthreads -- pure shuffle reduction.
 *
 * @param x         Input  (rows x dim)  half-precision
 * @param w         Weight (dim,)          half-precision
 * @param y         Output (rows x dim)  half-precision
 * @param epsilon   Numerical stability constant
 * @param dim       Hidden dimension (must be multiple of 4)
 */
__global__ void __launch_bounds__(WARP_SIZE)
rms_norm_warp_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y,
    float epsilon,
    int dim)
{
  int row     = blockIdx.x;
  int lane    = threadIdx.x;                   // 0..31
  int columns = dim / 4;                       // float4 columns
  float sum   = 0.0f;

  // Pass 1: sum of squares.
  // With 32 lanes x 4 elements = 128 elements per iteration.
  // dim=256 -> 64 columns -> one iteration covers all.
  for (int col = lane; col < columns; col += WARP_SIZE) {
    float4 xf;
    load_half4_as_float4(xf, x + (size_t)row * dim + col * 4);
    sum = accumulate_sum_of_squares(sum, xf);
  }

  // Warp-only reduction.
  sum = warp_reduce_sum(sum);

  float inv_rms = rsqrtf(sum / (float)dim + epsilon);

  // Pass 2: normalise and scale.
  for (int col = lane; col < columns; col += WARP_SIZE) {
    float4 xf, wf;
    load_half4_as_float4(xf, x + (size_t)row * dim + col * 4);
    load_half4_as_float4(wf, w + col * 4);
    scale_by_weight_and_rms(xf, wf, inv_rms);
    store_float4_as_half4(xf, y + (size_t)row * dim + col * 4);
  }
}

// ========================================================================
// Kernel 2: Medium-block RMSNorm  (256 < dim <= 2048)
// ========================================================================
/**
 * Standard block-reduction RMSNorm.
 *
 * Uses 1024 threads (the exllamav3 default).  No shared-memory input caching.
 * For dim <= 2048 the per-thread fragment is small enough that L2 cache
 * serves the second read of x with near-zero latency.
 *
 * This kernel is bit-identical to exllamav3's production kernel for the
 * same (half I/O, half weight, 1024 threads) configuration.
 */
template <int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
rms_norm_medium_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y,
    float epsilon,
    int rows,
    int dim)
{
  int t       = threadIdx.x;
  int warp_id = t / WARP_SIZE;
  int lane_id = t % WARP_SIZE;
  int row     = blockIdx.x;
  int columns = dim / 4;

  float sum = 0.0f;

  // Pass 1: sum of squares.
  for (int col = t; col < columns; col += BLOCK_THREADS) {
    float4 xf;
    load_half4_as_float4(xf, x + (size_t)row * dim + col * 4);
    sum = accumulate_sum_of_squares(sum, xf);
  }

  // Block reduction.
  constexpr int NWARPS = BLOCK_THREADS / WARP_SIZE;
  sum = block_reduce_sum<NWARPS>(sum, warp_id, lane_id);

  float inv_rms = rsqrtf(sum / (float)dim + epsilon);

  // Pass 2: normalise and scale.
  for (int col = t; col < columns; col += BLOCK_THREADS) {
    float4 xf, wf;
    load_half4_as_float4(xf, x + (size_t)row * dim + col * 4);
    load_half4_as_float4(wf, w + col * 4);
    scale_by_weight_and_rms(xf, wf, inv_rms);
    store_float4_as_half4(xf, y + (size_t)row * dim + col * 4);
  }
}

// ========================================================================
// Kernel 3: Wide-block RMSNorm  (dim > 2048 -- smem-cached input)
// ========================================================================
/**
 * Wide RMSNorm with shared-memory input caching.
 *
 * Optimisation: cache x in shared memory during Pass 1 so that Pass 2
 * reads from smem instead of re-traversing HBM -> L2 -> register.
 *
 * For batch_size > L2 capacity (~3K rows at dim=8192 half-precision)
 * this halves HBM traffic for x (from 2 reads to 1).
 *
 * Memory cost: dim x sizeof(float).
 *   dim=8192   -> 32 KB smem  (fits in 228 KB H100 capacity)
 *   dim=16384  -> 64 KB smem  (reduces occupancy by ~25%)
 *
 * Dynamic shared memory is allocated at launch time.
 */
template <int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
rms_norm_wide_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y,
    float epsilon,
    int rows,
    int dim)
{
  int t       = threadIdx.x;
  int warp_id = t / WARP_SIZE;
  int lane_id = t % WARP_SIZE;
  int row     = blockIdx.x;
  int columns = dim / 4;

  // Shared memory input cache: one float4 per column.
  extern __shared__ float shared_x[];
  float4* cache = reinterpret_cast<float4*>(shared_x);

  float sum = 0.0f;

  // Pass 1: sum of squares + cache input.
  for (int col = t; col < columns; col += BLOCK_THREADS) {
    float4 xf;
    load_half4_as_float4(xf, x + (size_t)row * dim + col * 4);
    cache[col] = xf;
    sum = accumulate_sum_of_squares(sum, xf);
  }
  __syncthreads();

  // Block reduction.
  constexpr int NWARPS = BLOCK_THREADS / WARP_SIZE;
  sum = block_reduce_sum<NWARPS>(sum, warp_id, lane_id);

  float inv_rms = rsqrtf(sum / (float)dim + epsilon);

  // Pass 2: normalise and scale (read x from smem cache).
  for (int col = t; col < columns; col += BLOCK_THREADS) {
    float4 xf = cache[col];  // smem read ~20 cycles vs ~200 for HBM
    float4 wf;
    load_half4_as_float4(wf, w + col * 4);
    scale_by_weight_and_rms(xf, wf, inv_rms);
    store_float4_as_half4(xf, y + (size_t)row * dim + col * 4);
  }
}

// ========================================================================
// Host launch dispatcher
// ========================================================================

/**
 * pkl_rms_norm_fp16_cuda -- RMSNorm for FP16 activations and weights.
 *
 * Selects optimal kernel strategy by hidden dimension:
 *   dim <= 256:     warp-only           (single warp, no barrier)
 *   256 < dim <= 2048:  medium block    (1024 threads, no smem cache)
 *   dim > 2048:    wide block          (1024 threads, smem-cached input)
 *
 * @param x       Input tensor  (rows x dim)  -- contiguous FP16 on device.
 * @param w       Weight tensor (dim,)         -- contiguous FP16 on device.
 * @param y       Output tensor (rows x dim) -- contiguous FP16 on device.
 * @param epsilon Small constant for numerical stability (e.g. 1e-6).
 * @param rows    Number of rows (batch size x sequence length).
 * @param dim     Hidden dimension.
 * @param stream  CUDA stream.
 * @return        0 on success, -1 on error.
 */
extern "C" {

int pkl_rms_norm_fp16_cuda(
    const half* x,
    const half* w,
    half* y,
    float epsilon,
    int rows,
    int dim,
    cudaStream_t stream)
{
  if (rows <= 0 || dim <= 0) return -1;
  if (dim % 4 != 0) return -1;
  dim3 grid(rows, 1, 1);

  // ----------------------------------------------------------------------
  // Strategy 1: warp-only path for small dimensions
  // ----------------------------------------------------------------------
  if (dim <= WARP_ONLY_DIM_THRESHOLD) {
    dim3 warp_block(WARP_SIZE, 1, 1);
    rms_norm_warp_kernel<<<grid, warp_block, 0, stream>>>(
        x, w, y, epsilon, dim);
    return 0;
  }

  // ----------------------------------------------------------------------
  // Strategies 2 & 3: block-level reduction with 1024 threads
  // ----------------------------------------------------------------------
  dim3 block(MAX_THREADS, 1, 1);  // 1024 threads (same as exllamav3)

  if (dim <= SMEM_CACHE_DIM_THRESHOLD) {
    // Medium path: no smem caching (L2 cache handles re-reads).
    rms_norm_medium_kernel<MAX_THREADS><<<grid, block, 0, stream>>>(
        x, w, y, epsilon, rows, dim);
  } else {
    // Wide path: share-memory input caching.
    size_t smem_bytes = (size_t)dim * sizeof(float);
    rms_norm_wide_kernel<MAX_THREADS><<<grid, block, smem_bytes, stream>>>(
        x, w, y, epsilon, rows, dim);
  }

  return 0;
}

}  // extern "C"
