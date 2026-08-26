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
##   workspace/positron/tests/manual_silu_and_mul_fp16.nim

import std/strformat
import workspace/crucible
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/silu_and_mul

const siluMsl = metal:
  proc siluAndMulKernel(
      Out: ptr UncheckedArray[float16],
      X: ptr UncheckedArray[float16],
      M, N: int32,
      actLimit: float32) {.global.} =
    silu_and_mul_fwd(Out, X, M, N, actLimit, 64)

proc scaledRand(rows, cols: int, scaleF: float32): seq[float32] =
  ## rand(rows, cols) shifted host-side to [-scaleF, scaleF). The
  ## kernel buffer and the reference tensors derive from the same seq.
  let t = rand(rows, cols)
  let p = t.contiguous().data_ptr(float32)
  result = newSeq[float32](rows * cols)
  for i in 0 ..< rows * cols:
    result[i] = (p[i] - 0.5'f32) * (2.0'f32 * scaleF)

proc worstAbsDiff(a, b: F.Tensor): float32 =
  ## Largest |a[i] - b[i]| over all elements.
  let ap = a.contiguous().data_ptr(float32)
  let bp = b.contiguous().data_ptr(float32)
  result = 0.0'f32
  for i in 0 ..< a.numel():
    let d = abs(ap[i] - bp[i])
    if d > result: result = d

proc siluAndMulKernel(gs, us: seq[float32], M, N: int): F.Tensor =
  ## Runs siluAndMulKernel on the fp16-rounded (g, u) interleaved
  ## buffer; returns the fp16 output converted to fp32.
  var engine = bkMetal.init()
  engine.ingest(siluMsl)
  var X = newSeq[uint16](M * 2 * N)
  for r in 0 ..< M:
    for c in 0 ..< N:
      X[r * 2 * N + c] = fp32ToFp16(gs[r * N + c])
      X[r * 2 * N + N + c] = fp32ToFp16(us[r * N + c])
  var outO = newSeq[uint16](M * N)
  engine.run << (grid: (N div 64, (M + 7) div 8, 1), blk: (32, 1)) >> ("siluAndMulKernel", outO, (X, int32(M), int32(N), 0.0'f32))
  var outF = newSeq[float32](M * N)
  for i in 0 ..< M * N:
    outF[i] = fp16ToFp32(outO[i])
  result = toTensor(outF).reshape(M, N)

proc siluAndMulReference(gs, us: seq[float32], M, N: int): F.Tensor =
  ## The mlp.nim:82 reference act = silu(g)·u, composed in fp32 over
  ## the same fp16-rounded inputs.
  let gT = toTensor(gs).reshape(M, N).to(kFloat16)
  let uT = toTensor(us).reshape(M, N).to(kFloat16)
  result = F.silu(gT.to(kFloat32)) * uT.to(kFloat32)

proc checkSiluAndMul(): bool =
  ## One random (M, N) batch: kernel output vs the torch reference.
  Torch.manual_seed(0x5EED'u64)
  let M = 32
  let N = 128
  let gs = scaledRand(M, N, 2.0'f32)
  let us = scaledRand(M, N, 2.0'f32)
  let actual = siluAndMulKernel(gs, us, M, N)
  let expected = siluAndMulReference(gs, us, M, N)
  echo &"  worst |Δ| = {worstAbsDiff(actual, expected)} (tolerance 5e-3)"
  assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  result = true

when isMainModule:
  runCppTest("silu_and_mul vs the torch reference (mlp.nim:82)", checkSiluAndMul)
