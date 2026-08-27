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
##   workspace/positron/tests/manual_gguf_linear_fp16.nim

import std/strformat, workspace/crucible, workspace/ceramic
import workspace/libtorch, workspace/libtorch as F
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/gguf_linear_fwd
import ./gguf_test_utils

const ggufLinearMsl = metal:
  proc ggufLinearQ8(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, 0)
  proc ggufLinearQ4K(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, 1)
  proc ggufLinearIQ4XS(Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, 2)

proc ggufLinearKernel(engine: var auto, launcher: string, xBits: seq[uint16],
    packed: seq[uint8], M, K, N, rowBytes: int): F.Tensor =
  var outO = newSeq[uint16](M * N)
  engine.run << (grid: (N div 128, (M + 31) div 32, 1), blk: (32, 1)) >> (
    launcher, outO, (xBits, packed, int32(M), int32(K), int32(N), int32(rowBytes)))
  var outF = newSeq[float32](M * N)
  for i in 0 ..< M * N: outF[i] = fp16ToFp32(outO[i])
  toTensor(outF).reshape(M, N)

proc ggufLinearReference(xBits, wBits: seq[uint16], M, K, N: int): F.Tensor =
  ## Torch reference: fp32 F.linear over the fp16-rounded inputs, weight in (N, K) file order, one fp16 round.
  F.linear(toTensor(fp16sToF32(xBits)).reshape(M, K),
           toTensor(fp16sToF32(wBits)).reshape(N, K)).to(kFloat16).to(kFloat32)

proc worstAbsDiff(a, b: F.Tensor): float32 =
  let ap = a.contiguous().data_ptr(float32)
  let bp = b.contiguous().data_ptr(float32)
  for i in 0 ..< a.numel():
    if abs(ap[i] - bp[i]) > result: result = abs(ap[i] - bp[i])
proc checkGGUFLinear(): bool =
  # per-scheme x scale keeps the output O(1) (weight RMS ~2e2..3e3)
  var engine = bkMetal.init()
  engine.ingest(ggufLinearMsl)
  let cases = [
    (0, 32, 128, 128, 200, 1.0'f32 / 16384.0'f32, "ggufLinearQ8", genQ8_0, decodeWeightsQ8_0),
    (0, 64, 256, 256, 201, 1.0'f32 / 16384.0'f32, "ggufLinearQ8", genQ8_0, decodeWeightsQ8_0),
    (1, 17, 256, 128, 202, 1.0'f32 / 65536.0'f32, "ggufLinearQ4K", genQ4_K, decodeWeightsQ4_K),
    (1, 64, 512, 256, 203, 1.0'f32 / 65536.0'f32, "ggufLinearQ4K", genQ4_K, decodeWeightsQ4_K),
    (2, 64, 256, 128, 204, 1.0'f32 / 262144.0'f32, "ggufLinearIQ4XS", genIQ4_XS, decodeWeightsIQ4_XS),
    (2, 8, 512, 256, 205, 1.0'f32 / 262144.0'f32, "ggufLinearIQ4XS", genIQ4_XS, decodeWeightsIQ4_XS),
  ]
  var worstAll = 0.0'f32
  for d in 0 ..< cases.len:
    let (scheme, M, K, N, seed, xScale, launcher, gen, decode) = cases[d]
    let packed = gen(K, N, seed)
    let xBits = buildX(M, K, seed, xScale)
    let actual = ggufLinearKernel(engine, launcher, xBits, packed, M, K, N, packed.len div N)
    let expected = ggufLinearReference(xBits, decode(packed, K, N), M, K, N)
    let w = worstAbsDiff(actual, expected)
    if w > worstAll: worstAll = w
    echo &"  scheme {scheme} M={M} K={K} N={N}: worst |Δ| = {w}, {M * N} elements"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  echo &"  worst |Δ| across {cases.len} cases = {worstAll} (tolerance 5e-3)"
  result = true

when isMainModule:
  runCppTest("GGUF quantized linear fwd vs the torch F.linear reference", checkGGUFLinear)
