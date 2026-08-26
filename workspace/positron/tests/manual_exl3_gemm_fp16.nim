## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_exl3_gemm_fp16.nim

import std/strformat
import workspace/crucible
import workspace/ceramic
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/exl3_gemm_fwd
import ./exl3_test_utils

const exl3GemmMsl = metal:
  proc exl3GemmB3C0(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_gemm_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 0, 128)
  proc exl3GemmB5C0(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_gemm_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 0, 128)
  proc exl3GemmB8C0(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_gemm_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 0, 128)
  proc exl3GemmB5C2(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_gemm_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 2, 128)

proc exl3GemmKernel(xBits: seq[uint16], trellis: seq[int16],
                    suh, svh: seq[uint16], M, K, N, bits, cb: int): F.Tensor =
  ## Runs the fused prefill-GEMM kernel on the fp16 inputs and returns
  ## its fp16 output as fp32.
  var engine = bkMetal.init()
  engine.ingest(exl3GemmMsl)
  var outO = newSeq[uint16](M * N)
  engine.run << (grid: (N div 128, (M + 31) div 32, 1), blk: (32, 1)) >> (
    "exl3GemmB" & $bits & "C" & $cb, outO,
    (xBits, trellis, suh, svh, int32(M), int32(K), int32(N)))
  var outF = newSeq[float32](M * N)
  for i in 0 ..< M * N:
    outF[i] = fp16ToFp32(outO[i])
  result = toTensor(outF).reshape(M, N)

proc exl3GemmReference(xBits, suh, svh: seq[uint16], wBits: seq[uint16],
                       M, K, N: int): F.Tensor =
  ## The torch reference, identical to the linear forward: fp16(x·suh),
  ## FWHT-128 per K-block, x @ Wᵀ, fp16 round, FWHT-128 per N-block, fp16(y·svh).
  let xT = toTensor(fp16sToF32(xBits)).reshape(M, K)
  let suhT = toTensor(fp16sToF32(suh)).reshape(1, K)
  let svhT = toTensor(fp16sToF32(svh)).reshape(1, N)
  let wT = toTensor(fp16sToF32(wBits)).reshape(K, N).t()
  let xf = fwhtBlocks((xT * suhT).to(kFloat16).to(kFloat32), K).to(kFloat16)
  let d = F.linear(xf.to(kFloat32), wT).to(kFloat16)
  let y = fwhtBlocks(d.to(kFloat32), N).to(kFloat16)
  result = (y.to(kFloat32) * svhT).to(kFloat16).to(kFloat32)

proc checkExl3Gemm(): bool =
  ## Five (M, K, N, bits, cb) cases against the torch reference: the
  ## M ≤ 32 decode shapes, the M = 64 grid.y boundary and the cb2
  ## codebook. The 5e-3 bound is ~40 fp16 ulps at the ~1e-1 output
  ## magnitude, far above the fp32 accumulation-order noise of the torch matmuls.
  let cases = [(16, 128, 128, 5, 0), (8, 256, 256, 3, 0), (32, 256, 128, 8, 0),
               (64, 128, 256, 5, 0), (16, 128, 256, 5, 2)]
  var worstAll = 0.0'f32
  for d in 0 ..< cases.len:
    let (M, K, N, bits, cb) = cases[d]
    let trellis = genTrellis(K div 16, N div 16, bits, seed = 100 + d)
    let wBits = dequantWeights(trellis, K div 16, N div 16, bits, cb)
    let inputs = buildExl3Inputs(M, K, N, seed = 100 + d)
    let actual = exl3GemmKernel(inputs.xBits, trellis, inputs.suh, inputs.svh, M, K, N, bits, cb)
    let expected = exl3GemmReference(inputs.xBits, inputs.suh, inputs.svh, wBits, M, K, N)
    let w = worstAbsDiff(actual, expected)
    if w > worstAll: worstAll = w
    echo &"  M={M} K={K} N={N} bits={bits} cb={cb}: worst |Δ| = {w}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  echo &"  worst |Δ| across {cases.len} cases = {worstAll} (tolerance 5e-3)"
  result = true

when isMainModule:
  runCppTest("exl3_gemm_fwd vs the torch reference", checkExl3Gemm)
