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
##   workspace/positron/tests/manual_qk_norm_rope_fp16.nim

import std/[math, strformat]
import workspace/[crucible, ceramic, libtorch]
import workspace/libtorch as F
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/qk_norm_rope, ./exl3_test_utils

const qkNormRopeMsl = metal:
  proc qkNormRope(Out: ptr UncheckedArray[float16], X: ptr UncheckedArray[float16],
      G: ptr UncheckedArray[float16], Cos: ptr UncheckedArray[float32], Sin: ptr UncheckedArray[float32],
      xTokenStride, cosTokenStride, headBlocks, xColBase: int32, eps: float32) {.global.} =
    qk_norm_rope_fwd(Out, X, G, Cos, Sin, xTokenStride, cosTokenStride, headBlocks, xColBase, eps)

proc gammaVal(c, seed: int): float32 =
  ## Deterministic fp16-exact norm weight in [0.5, 1.5).
  let k = (7 * c + 11 * seed + 13 * (c div 32)) mod 16
  0.5'f32 + float32(k) / 16.0'f32

proc buildRopeCosSin(rows, D: int; theta: float32): tuple[cosT, sinT: seq[float32]] =
  ## NEOX fp32 cos/sin tables, row r carries cos/sin of inv_freq[t]·r.
  let half = D shr 1
  result.cosT = newSeq[float32](rows * half)
  result.sinT = newSeq[float32](rows * half)
  for r in 0 ..< rows:
    for t in 0 ..< half:
      let ang = pow(float64(theta), -float64(2 * t) / float64(D)).float32 * float32(r)
      result.cosT[r * half + t] = cos(ang)
      result.sinT[r * half + t] = sin(ang)

proc qkNormRopeKernel(x, gamma: seq[uint16], cosT, sinT: seq[float32],
                      rows: int, eps: float32): F.Tensor =
  var engine = bkMetal.init()
  engine.ingest(qkNormRopeMsl)
  var outO = newSeq[uint16](rows * 128)
  engine.run << (grid: (1, rows div 8, 1), blk: (32, 1)) >> (
    "qkNormRope", outO, (x, gamma, cosT, sinT, int32(8 * 128), int32(8), int32(1), int32(0), eps))
  var outF = newSeq[float32](rows * 128)
  for i in 0 ..< rows * 128: outF[i] = fp16ToFp32(outO[i])
  result = toTensor(outF).reshape(rows, 128)

proc qkNormRopeReference(x, gamma: seq[uint16], cosT, sinT: seq[float32],
                         rows: int, eps: float32): F.Tensor =
  ## The torch reference: fp32 rms_norm, the two-rounding fp16(x·rmf) then fp16(x·γ16), then the NEOX rotation.
  let xT = toTensor(fp16sToF32(x)).reshape(rows, 128)
  let normed = rms_norm(xT, 128, F.ones(128, kFloat32), float64(eps))
  let xr = normed.to(kFloat16)
  let gT = toTensor(fp16sToF32(gamma)).reshape(1, 128).to(kFloat16)
  let xg = (xr.to(kFloat32) * gT.to(kFloat32)).to(kFloat16).to(kFloat32)
  let cosTns = toTensor(cosT).reshape(rows, 64)
  let sinTns = toTensor(sinT).reshape(rows, 64)
  let t1 = (xg.narrow(1, 0, 64) * cosTns - xg.narrow(1, 64, 64) * sinTns).to(kFloat16)
  let t2 = (xg.narrow(1, 64, 64) * cosTns + xg.narrow(1, 0, 64) * sinTns).to(kFloat16)
  result = F.cat([t1, t2], 1).to(kFloat32)

proc checkQkNormRope(): bool =
  ## Three flat-contract cases (real rows padded to the 8-row tile grid)
  ## vs the torch reference. The 5e-3 bound is ~5 fp16 ulps at the ~1
  ## output magnitude, above the rms_norm sum-order and the non-fma rotation noise.
  let cases = [(real: 8, rows: 8, eps: 1e-6'f32), (real: 17, rows: 24, eps: 1e-2'f32),
               (real: 100, rows: 104, eps: 1e-6'f32)]
  for d in cases:
    var x = newSeq[uint16](d.rows * 128)
    for r in 0 ..< d.real:
      for c in 0 ..< 128: x[r * 128 + c] = fp32ToFp16(exl3Val(r, c, 3, 0.3'f32))
    var gamma = newSeq[uint16](128)
    for c in 0 ..< 128: gamma[c] = fp32ToFp16(gammaVal(c, 5))
    let (cosT, sinT) = buildRopeCosSin(d.rows, 128, 1000000.0'f32)
    let actual = qkNormRopeKernel(x, gamma, cosT, sinT, d.rows, d.eps)
    let expected = qkNormRopeReference(x, gamma, cosT, sinT, d.rows, d.eps)
    echo &"  rows={d.real} (padded {d.rows}) eps={d.eps}: worst |Δ| = {worstAbsDiff(actual, expected)}"
    assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  result = true

when isMainModule:
  runCppTest("qk_norm_rope vs the torch reference", checkQkNormRope)
