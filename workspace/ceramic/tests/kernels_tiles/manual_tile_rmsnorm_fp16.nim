## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## On-device tile-layer RMSNorm (manual, Metal): `rms_single_row` vs
## libtorch fp32 `rms_norm` (tolerance 1e-5, 40× the deterministic 2.4e-7)
## over hash-randomized M×C shapes, C arbitrary (1..128), plus one unpadded
## C == 128 draw. The ε varies across the draws (1e-5, 1e-2):
## a kernel that drops the ε shift fails the gate loudly at ε = 1e-2.
## The loads are branchless: the host zero-pads the sources to width 128.
##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_rmsnorm_fp16 \
##   --nimcache:nimcache/tests/manual_tile_rmsnorm_fp16 \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_rmsnorm_fp16.nim

import std/[strformat, strutils]
import workspace/crucible
import workspace/libtorch
import ../tile_test_utils
import ../../src/kernels/k_tile_rmsnorm

# ═════════════════════════════════════════════════════════════════════════
#  The thin {.global.} launcher. First param = the output buffer Out.
# ═════════════════════════════════════════════════════════════════════════

const rmsnormMsl = metal:
  proc rmsnorm(Out: ptr UncheckedArray[float32], X, G: ptr UncheckedArray[float16],
               C: int32, eps: float32) {.global.} =
    rms_single_row(Out, X, G, C, eps)

# ═════════════════════════════════════════════════════════════════════════
#  Host reference: libtorch rms_norm in fp32 over the real shape
# ═════════════════════════════════════════════════════════════════════════

const Ct = 128          # the static tile width (the kernel's rt_r(float32, 8, 128))

func torchRmsnorm(M, C: int; Xh, Gh: seq[uint16]; eps: float32): seq[float32] =
  ## y = rms_norm(x, γ, ε) via libtorch in fp32 over the real M×C shape,
  ## the dumb reference sharing no machinery with the kernel. fp16→fp32
  ## is exact, so the f32 tensor carries the same values the kernel reads.
  ## `Xh` is the 128-wide row-padded buffer: data in the first C columns
  ## per row, zeros beyond (the kernel reads unconditionally).
  var xf = newSeq[float32](M * C)
  var gf = newSeq[float32](C)
  for m in 0 ..< M:
    for c in 0 ..< C:
      xf[m * C + c] = fp16ToFp32(Xh[m * 128 + c])
  for c in 0 ..< C:
    gf[c] = fp16ToFp32(Gh[c])
  let y = rms_norm(toTensor(xf).reshape(M, C), C, toTensor(gf), float64(eps))
  result = newSeq[float32](M * C)
  let p = y.data_ptr(float32)
  for i in 0 ..< M * C:
    result[i] = p[i]

# ═════════════════════════════════════════════════════════════════════════
#  Tests
# ═════════════════════════════════════════════════════════════════════════

proc checkRmsnorm(engine: var auto; kernel: string; M, C: int; eps: float32) =
  ## Fills the 128-wide padded X (Mp rows × Ct, data in the first C
  ## columns, zeros beyond) and γ (Ct, data in [0, C)), runs the kernel
  ## with grid (1, Mp div 8) × blk 32, then compares the real M×C
  ## outputs against torch rms_norm with a 1e-5 tolerance (40× the deterministic 2.4e-7).
  ## The gate catches an fp16 output store (7.8e-4) and a dropped-ε kernel
  ## at ε = 1e-2. The padding zeros the reads, so the loads
  ## need no per-element guard.
  let Mp = ((M + 7) div 8) * 8
  var Xh = newSeq[uint16](Mp * Ct)
  var Gh = newSeq[uint16](Ct)
  for r in 0 ..< M:
    for c in 0 ..< C:
      Xh[r * Ct + c] = fp32ToFp16(float32((r * 7 + c * 13) mod 64) * 0.1'f32 - 2.0'f32)
  for c in 0 ..< C:
    Gh[c] = fp32ToFp16(0.5'f32 + 0.01'f32 * float32(c))

  var outF = newSeq[float32](Mp * Ct)
  engine.run << (grid: (1, Mp div 8), blk: (32, 1)) >> (
    kernel, outF, (Xh, Gh, int32(C), eps))

  let refOut = torchRmsnorm(M, C, Xh, Gh, eps)
  var ok = true
  var worst = 0.0'f32
  var firstBad = ""
  var nBad = 0
  for m in 0 ..< M:
    for c in 0 ..< C:
      let d = abs(outF[m * Ct + c] - refOut[m * C + c])
      if d > worst: worst = d
      if d > 1e-5'f32:
        ok = false
        inc nBad
        if firstBad.len == 0:
          firstBad = &"(r={m}, c={c}): got {outF[m * Ct + c]}, want {refOut[m * C + c]}"
  echo &"RMSNorm {M}×{C} ({kernel}, grid (1,{Mp div 8}), tile width {Ct}, " &
       &"C div 8 = {C div 8}): ",
       (if ok: &"PASS — worst |Δ| = {worst} (tolerance 1e-5)"
        else: &"FAIL — {nBad} bad, e.g. {firstBad}, worst |Δ| = {worst}")
  if not ok:
    quit 1

proc runTests() =   # engines are RAII, so keep them function-local
  checkDrawPins()          # the hash draws must not drift
  var engine = bkMetal.init()
  engine.ingest(rmsnormMsl)
  echo rmsnormMsl           # keep the generated MSL inspectable

  var draws = initShapeDraws("rmsnorm")
  for draw in 0 ..< 2:
    let b = draws.nextBytes()
    let M = drawInRange(b, 0, 1, 64)
    let C = drawInRange(b, 1, 1, 128)
    doAssert C <= Ct, "C must fit the static tile width (Ct)"
    # ε varies across the draws: at ε = 1e-2 a kernel that drops the ε shift
    # (rstd without the add) errs orders above the 1e-5 gate.
    let eps = if draw == 0: 1e-5'f32 else: 1e-2'f32
    checkRmsnorm(engine, "rmsnorm", M, C, eps)
  # The C == 128 draw exercises the unpadded path: every source column
  # carries real data, so the zero-padding never contributes.
  let b = draws.nextBytes()
  let M = drawInRange(b, 0, 1, 64)
  checkRmsnorm(engine, "rmsnorm", M, Ct, eps = 1e-2'f32)

when isMainModule:
  runTests()
