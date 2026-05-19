# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Pure-Nim RMSNorm CUDA kernel — compiled with nvcc via `--cc:nvcc`.

# TODO:
#   Unfortunately, it is not possible to compile this kernel as a Nim static library at the moment
#   and import it due to Nim core symbols being defined multiple times

import workspace/positron/experimental/platforms/cuda

const NUM_THREADS = 1024

# ─── Warp-aware reduction ──

{.push checks: off, overflowChecks: off, rangeChecks: off, boundChecks: off.}

proc reduceSum(sum: float32, warpId, laneId: int32): float32 {.
  exportc, codegenDecl: "__device__ __forceinline__ $# $#$#".} =
  ## Cross-warp reduction using warp shuffle + shared memory.
  var sums {.exportc, codegenDecl: "__shared__ $# $#", noinit.}: array[NUM_THREADS div 32, float32]
  result = sum
  # First shuffle across warp lanes (mask, val, lane_id = matches CUDA API)
  var offset = 16'i32
  while offset > 0:
    result += shflXorSync(0xFFFFFFFF'u32, result, offset)
    offset = offset div 2
  # Cross-warp via shared memory
  if laneId == 0: sums[warpId] = result
  syncthreads()
  result = if laneId < 32.int32:
              sums[laneId]
           else:
            0.0'f32
  # Second shuffle across warp lanes
  offset = 16
  while offset > 0:
    result += shflXorSync(0xFFFFFFFF'u32, result, offset)
    offset = offset div 2

# ─── Device helpers ────────────────────────────────────────────

proc readHalf4(f4: var Float4, p: pHalf) {.
  inline, exportc, codegenDecl: "__device__ __forceinline__ $# $#$#".} =
  f4.x = half2float(p[0]); f4.y = half2float(p[1])
  f4.z = half2float(p[2]); f4.w = half2float(p[3])

proc writeHalf4(f4: Float4, p: pHalf) {.
  inline, exportc, codegenDecl: "__device__ __forceinline__ $# $#$#".} =
  p[0] = float2halfRn(f4.x); p[1] = float2halfRn(f4.y)
  p[2] = float2halfRn(f4.z); p[3] = float2halfRn(f4.w)

proc sumSquares4(lsum: float32, f4: Float4): float32 {.
  inline, exportc, codegenDecl: "__device__ __forceinline__ $# $#$#".} =
  result = fmaf(f4.x, f4.x, lsum)
  result = fmaf(f4.y, f4.y, result)
  result = fmaf(f4.z, f4.z, result)
  result = fmaf(f4.w, f4.w, result)

proc rescale4(x4: var Float4, w4: Float4, rmf: float32) {.
  inline, exportc, codegenDecl: "__device__ __forceinline__ $# $#$#".} =
  x4.x = x4.x * w4.x * rmf; x4.y = x4.y * w4.y * rmf
  x4.z = x4.z * w4.z * rmf; x4.w = x4.w * w4.w * rmf

# ─── Kernel ───────────────────────────────────────────────────

proc rmsNormFp16Kernel(x {.noalias.}: pHalf,
                        w {.noalias.}: pHalf,
                        y {.noalias.}: pHalf,
                        epsilon: float32, rows: int32, dim: int32) {.
  exportc, codegenDecl: "__global__ $# $#$#".} =

  let t = threadIdx.x.int32
  let warpId = t div 32
  let laneId = t mod 32
  let row = blockIdx.x.int32
  let columns = dim div 4

  # Row base pointer
  let xRow = x +% (row * dim)
  let yRow = y +% (row * dim)

  # Sum of squares
  var sum: float32 = 0.0
  for col in countup(t, columns - 1, NUM_THREADS):
    var x4: Float4
    readHalf4(x4, xRow +% col*4)
    sum = sumSquares4(sum, x4)

  # Reduction (warp shuffle + shared memory)
  sum = reduceSum(sum, warpId, laneId)
  let rmf = rsqrtf(sum / float32(dim) + epsilon)

  # Normalize + scale
  for col in countup(t, columns - 1, NUM_THREADS):
    var x4, w4: Float4
    readHalf4(x4, xRow +% col*4)
    readHalf4(w4, w +% col*4)
    rescale4(x4, w4, rmf)
    writeHalf4(x4, yRow +% col*4)

{.pop.}

# ─── Host wrapper ─────────────────────────────────────────────

proc rmsNormCuda_fp16*(
       x: pHalf, w: pHalf,
       y: pHalf, epsilon: float32,
       rows: int32, dim: int32,
       stream: cudaStream_t = nil) {.exportc: "pkl_$1".} =
  if dim mod 4 != 0:
    stderr.writeLine("rmsNormCuda: dim must be divisible by 4, got ", dim)
    quit(1)
  var args: array[6, pointer]
  args[0] = addr x;    args[1] = addr w
  args[2] = addr y;    args[3] = addr epsilon
  args[4] = addr rows; args[5] = addr dim
  let gridDim = dim3(rows)
  let blockDim = dim3(NUM_THREADS)
  check cudaLaunchKernel(rmsNormFp16Kernel,
    gridDim, blockDim, args[0].addr, 0.csize_t, stream)
  check cudaDeviceSynchronize()
