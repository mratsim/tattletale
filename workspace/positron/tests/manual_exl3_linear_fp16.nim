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
##   workspace/positron/tests/manual_exl3_linear_fp16.nim

import std/strformat
import workspace/crucible
import workspace/ceramic
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/exl3_linear_fwd
import ./exl3_test_utils

const exl3LinearMsl = metal:
  proc exl3LinearB3(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_linear_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 128)
  proc exl3LinearB5(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_linear_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 128)
  proc exl3LinearB8(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16], suh, svh: ptr UncheckedArray[float16], M, K, N: int32) {.global.} =
    exl3_linear_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 128)

proc exl3LinearKernel(xBits: seq[uint16], trellis: seq[int16],
                      suh, svh: seq[uint16], M, K, N, bits: int): F.Tensor =
  ## Runs the fused linear kernel on the fp16 inputs and returns the
  ## fp16 output converted to fp32.
  var engine = bkMetal.init()
  engine.ingest(exl3LinearMsl)
  var outO = newSeq[uint16](M * N)
  engine.run << (grid: (N div 128, (M + 31) div 32, 1), blk: (32, 1)) >> (
    "exl3LinearB" & $bits, outO, (xBits, trellis, suh, svh, int32(M), int32(K), int32(N)))
  var outF = newSeq[float32](M * N)
  for i in 0 ..< M * N:
    outF[i] = fp16ToFp32(outO[i])
  result = toTensor(outF).reshape(M, N)

proc exl3LinearReference(xBits, suh, svh: seq[uint16], wBits: seq[uint16],
                         M, K, N: int): F.Tensor =
  ## The torch reference, linear.nim's F.linear over the dequantized
  ## weight: fp16(x·suh), FWHT-128 per K-block, x @ Wᵀ, fp16 round,
  ## FWHT-128 per N-block, fp16(y·svh).
  let xT = toTensor(fp16sToF32(xBits)).reshape(M, K)
  let suhT = toTensor(fp16sToF32(suh)).reshape(1, K)
  let svhT = toTensor(fp16sToF32(svh)).reshape(1, N)
  let wT = toTensor(fp16sToF32(wBits)).reshape(K, N).t()
  let xf = fwhtBlocks((xT * suhT).to(kFloat16).to(kFloat32), K).to(kFloat16)
  let d = F.linear(xf.to(kFloat32), wT).to(kFloat16)
  let y = fwhtBlocks(d.to(kFloat32), N).to(kFloat16)
  result = (y.to(kFloat32) * svhT).to(kFloat16).to(kFloat32)

proc checkExl3Linear(): bool =
  ## Five (M, K, N, bits) cases against the torch reference. The 5e-3
  ## bound is ~40 fp16 ulps at the ~1e-1 output magnitude, far above
  ## the fp32 accumulation-order noise between the torch matmuls and
  ## the Metal mma, while a broken FWHT scale, dequant placement or
  ## codebook shows up as O(1) errors.
  let cases = [(1, 128, 128, 5), (32, 256, 256, 5), (64, 128, 256, 5),
               (8, 256, 128, 3), (16, 128, 128, 8)]
  var worstAll = 0.0'f32
  for d in 0 ..< cases.len:
    let (M, K, N, bits) = cases[d]
    let trellis = genTrellis(K div 16, N div 16, bits, seed = 100 + d)
    let wBits = dequantWeights(trellis, K div 16, N div 16, bits, cb = 0)
    let inputs = buildExl3Inputs(M, K, N, seed = 100 + d)
    let actual = exl3LinearKernel(inputs.xBits, trellis, inputs.suh, inputs.svh, M, K, N, bits)
    let expected = exl3LinearReference(inputs.xBits, inputs.suh, inputs.svh, wBits, M, K, N)
    let w = worstAbsDiff(actual, expected)
    if w > worstAll: worstAll = w
    echo &"  M={M} K={K} N={N} bits={bits}: worst |Δ| = {w}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  echo &"  worst |Δ| across {cases.len} cases = {worstAll} (tolerance 5e-3)"
  result = true

when isMainModule:
  runCppTest("exl3_linear_fwd vs the torch reference", checkExl3Linear)
