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
##   workspace/positron/tests/manual_exl3_gemv_fp16.nim

import std/strformat
import workspace/crucible
import workspace/ceramic
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/exl3_gemv
import ./exl3_test_utils

const exl3GemvMsl = metal:
  proc exl3GemvM0B3(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 0, 0, 128)
  proc exl3GemvM0B5(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 0, 0, 128)
  proc exl3GemvM1B5(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 0, 1, 128)
  proc exl3GemvM1B8(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 0, 1, 128)

proc exl3GemvKernel(xBits: seq[uint16], trellis: seq[int16],
                    suh, svh: seq[uint16], M, K, N, bits, mmode: int): F.Tensor =
  ## Runs the fused decode-GEMV kernel on the fp16 inputs and returns
  ## its fp16 output as fp32. MMODE 0 is the m = 1 fast path, MMODE 1 the m ≤ 8 path.
  var engine = bkMetal.init()
  engine.ingest(exl3GemvMsl)
  var outO = newSeq[uint16](M * N)
  engine.run << (grid: (N div 128, (M + 31) div 32, 1), blk: (32, 1)) >> (
    "exl3GemvM" & $mmode & "B" & $bits, outO,
    (xBits, trellis, suh, svh, int32(M), int32(K), int32(N)))
  var outF = newSeq[float32](M * N)
  for i in 0 ..< M * N:
    outF[i] = fp16ToFp32(outO[i])
  result = toTensor(outF).reshape(M, N)

proc exl3GemvReference(xBits, suh, svh: seq[uint16], wBits: seq[uint16],
                       M, K, N: int): F.Tensor =
  ## The torch reference, the same math as the fused linear forward:
  ## fp16(x·suh), FWHT-128 per K-block, x @ Wᵀ, fp16 round, FWHT-128 per N-block, fp16(y·svh).
  let xT = toTensor(fp16sToF32(xBits)).reshape(M, K)
  let suhT = toTensor(fp16sToF32(suh)).reshape(1, K)
  let svhT = toTensor(fp16sToF32(svh)).reshape(1, N)
  let wT = toTensor(fp16sToF32(wBits)).reshape(K, N).t()
  let xf = fwhtBlocks((xT * suhT).to(kFloat16).to(kFloat32), K).to(kFloat16)
  let d = F.linear(xf.to(kFloat32), wT).to(kFloat16)
  let y = fwhtBlocks(d.to(kFloat32), N).to(kFloat16)
  result = (y.to(kFloat32) * svhT).to(kFloat16).to(kFloat32)

proc checkExl3Gemv(): bool =
  ## Four (M, K, N, bits, mmode) cases against the torch reference: the
  ## MMODE 0 m = 1 fast path and the MMODE 1 m ≤ 8 path. The 5e-3 bound
  ## is ~40 fp16 ulps at the ~1e-1 output magnitude, far above the fp32
  ## accumulation-order noise of the torch matmuls vs the Metal mma.
  let cases = [(1, 256, 256, 5, 0), (1, 128, 128, 3, 0), (8, 256, 128, 5, 1),
               (5, 128, 256, 8, 1)]
  var worstAll = 0.0'f32
  for d in 0 ..< cases.len:
    let (M, K, N, bits, mmode) = cases[d]
    let trellis = genTrellis(K div 16, N div 16, bits, seed = 100 + d)
    let wBits = dequantWeights(trellis, K div 16, N div 16, bits, cb = 0)
    let inputs = buildExl3Inputs(M, K, N, seed = 100 + d)
    let actual = exl3GemvKernel(inputs.xBits, trellis, inputs.suh, inputs.svh, M, K, N, bits, mmode)
    let expected = exl3GemvReference(inputs.xBits, inputs.suh, inputs.svh, wBits, M, K, N)
    let w = worstAbsDiff(actual, expected)
    if w > worstAll: worstAll = w
    echo &"  M={M} K={K} N={N} bits={bits} mmode={mmode}: worst |Δ| = {w}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  echo &"  worst |Δ| across {cases.len} cases = {worstAll} (tolerance 5e-3)"
  result = true

when isMainModule:
  runCppTest("exl3_gemv vs the torch reference", checkExl3Gemv)
