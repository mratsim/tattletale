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
##   workspace/positron/tests/manual_rms_norm_res_in_fp16.nim

import std/strformat
import workspace/crucible
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/rms_norm_res_in

const rmsMsl = metal:
  proc rmsNormResInKernel(
      Out: ptr UncheckedArray[float16],
      X, R, G: ptr UncheckedArray[float16],
      M, C: int32,
      eps: float32) {.global.} =
    rms_norm_res_in_fwd(Out, X, R, G, M, C, eps, 128)

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

proc rmsNormResInKernel(Xh, Rh, Gh: seq[uint16], M, C: int, eps: float32): F.Tensor =
  ## Runs rmsNormResInKernel on the fp16 buffers; returns the fp16
  ## output converted to fp32.
  var engine = bkMetal.init()
  engine.ingest(rmsMsl)
  var outO = newSeq[uint16](M * C)
  engine.run << (grid: (1, (M + 7) div 8), blk: (32, 1)) >> ("rmsNormResInKernel", outO, (Xh, Rh, Gh, int32(M), int32(C), eps))
  var outF = newSeq[float32](M * C)
  for i in 0 ..< M * C:
    outF[i] = fp16ToFp32(outO[i])
  result = toTensor(outF).reshape(M, C)

proc rmsNormResInReference(M, C: int, Xh, Rh, Gh: seq[uint16], eps: float32): F.Tensor =
  ## y = rms_norm(x + res, γ, ε) via libtorch: the fp16 residual add
  ## (one RNE round, the same intermediate the kernel materializes),
  ## then fp32 rms_norm on the widened sum.
  var xf = newSeq[float32](M * C)
  var rf = newSeq[float32](M * C)
  var gf = newSeq[float32](C)
  for i in 0 ..< M * C:
    xf[i] = fp16ToFp32(Xh[i])
    rf[i] = fp16ToFp32(Rh[i])
  for c in 0 ..< C:
    gf[c] = fp16ToFp32(Gh[c])
  let sum = toTensor(xf).reshape(M, C).to(kFloat16) + toTensor(rf).reshape(M, C).to(kFloat16)
  result = rms_norm(sum.to(kFloat32), C, toTensor(gf), float64(eps))

proc checkRmsNormResIn(): bool =
  ## One random (M, C) batch: kernel output vs torch rms_norm(x + res).
  Torch.manual_seed(0x5EED'u64)
  let M = 19            # partial last row tile
  let C = 2048          # 16 column blocks of 128
  let eps = 1e-2'f32
  let n = M * C
  let xs = scaledRand(M, C, 2.0'f32)
  let rs = scaledRand(M, C, 2.0'f32)
  let gs = scaledRand(1, C, 1.0'f32)
  var Xh = newSeq[uint16](n)
  var Rh = newSeq[uint16](n)
  var Gh = newSeq[uint16](C)
  for i in 0 ..< n:
    Xh[i] = fp32ToFp16(xs[i])
    Rh[i] = fp32ToFp16(rs[i])
  for c in 0 ..< C:
    Gh[c] = fp32ToFp16(gs[c])
  let actual = rmsNormResInKernel(Xh, Rh, Gh, M, C, eps)
  let expected = rmsNormResInReference(M, C, Xh, Rh, Gh, eps)
  echo &"  worst |Δ| = {worstAbsDiff(actual, expected)} (tolerance 5e-3)"
  assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  result = true

when isMainModule:
  runCppTest("rms_norm_res_in vs torch rms_norm(x + res)", checkRmsNormResIn)
