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
##   workspace/positron/tests/manual_gated_mlp_silu_fp16.nim

import std/strformat
import workspace/crucible
import workspace/ceramic
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/gated_mlp_silu

const mlpMsl = metal:
  proc gatedMlpSiluKernel(
      Out: ptr UncheckedArray[float16],
      X, wg, wu, wd: ptr UncheckedArray[float16],
      M, K, NIntm, NOut: int32) {.global.} =
    gated_mlp_silu_fwd(Out, X, wg, wu, wd, M, K, NIntm, NOut, 32)

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

proc gatedMlpSiluKernel(xf, wgf, wuf, wdf: seq[float32], M, K, NIntm, NOut: int): F.Tensor =
  ## Runs gatedMlpSiluKernel on the fp16-rounded inputs; returns the
  ## fp16 output converted to fp32.
  var engine = bkMetal.init()
  engine.ingest(mlpMsl)
  var xb = newSeq[uint16](M * K)
  var wgb = newSeq[uint16](K * NIntm)
  var wub = newSeq[uint16](K * NIntm)
  var wdb = newSeq[uint16](NIntm * NOut)
  for i in 0 ..< M * K:
    xb[i] = fp32ToFp16(xf[i])
  for i in 0 ..< K * NIntm:
    wgb[i] = fp32ToFp16(wgf[i])
    wub[i] = fp32ToFp16(wuf[i])
  for i in 0 ..< NIntm * NOut:
    wdb[i] = fp32ToFp16(wdf[i])
  var outO = newSeq[uint16](M * NOut)
  let args = (xb, wgb, wub, wdb, int32(M), int32(K), int32(NIntm), int32(NOut))
  engine.run << (grid: (NOut div 32, (M + 31) div 32, 1), blk: (32, 1)) >> ("gatedMlpSiluKernel", outO, args)
  var outF = newSeq[float32](M * NOut)
  for i in 0 ..< M * NOut:
    outF[i] = fp16ToFp32(outO[i])
  result = toTensor(outF).reshape(M, NOut)

proc gatedMlpSiluReference(xf, wgf, wuf, wdf: seq[float32], M, K, NIntm, NOut: int): F.Tensor =
  ## The mlp.nim:82 reference out = down(silu(x·wg)·(x·wu)): torch
  ## linear over the same fp16-rounded inputs, weights transposed to
  ## linear's (out, in) layout.
  let xh = toTensor(xf).reshape(M, K).to(kFloat16)
  let wgh = toTensor(wgf).reshape(K, NIntm).to(kFloat16)
  let wuh = toTensor(wuf).reshape(K, NIntm).to(kFloat16)
  let wdh = toTensor(wdf).reshape(NIntm, NOut).to(kFloat16)
  let x32 = xh.to(kFloat32)
  let g = F.linear(x32, wgh.to(kFloat32).transpose(0, 1))
  let u = F.linear(x32, wuh.to(kFloat32).transpose(0, 1))
  let act = F.silu(g) * u
  result = F.linear(act, wdh.to(kFloat32).transpose(0, 1))

proc checkGatedMlpSilu(): bool =
  ## One random batch: kernel output vs the torch reference.
  Torch.manual_seed(0x5EED'u64)
  let M = 32
  let K = 64
  let NIntm = 128
  let NOut = 128
  let xf = scaledRand(M, K, 0.5'f32)
  let wgf = scaledRand(K, NIntm, 0.5'f32)
  let wuf = scaledRand(K, NIntm, 0.5'f32)
  let wdf = scaledRand(NIntm, NOut, 0.5'f32)
  let actual = gatedMlpSiluKernel(xf, wgf, wuf, wdf, M, K, NIntm, NOut)
  let expected = gatedMlpSiluReference(xf, wgf, wuf, wdf, M, K, NIntm, NOut)
  echo &"  worst |Δ| = {worstAbsDiff(actual, expected)} (tolerance 5e-3)"
  assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  result = true

when isMainModule:
  runCppTest("gated_mlp_silu vs the torch reference (mlp.nim:82)", checkGatedMlpSilu)
