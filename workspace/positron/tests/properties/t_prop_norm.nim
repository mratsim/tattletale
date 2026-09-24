# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_properties
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/positron/tests/properties/p_norm.nim
##
## Property suite over the fused fp16 residual RMSNorm, the idempotence invariant,
## `rms_norm_res_in_fwd` of `src/kernels/ceramic/norm.nim`:
##
##   norm(norm(x)) ≈ norm(x)
##
## With a zero residual and the all-ones weight, the normalized row stands as a fixed
## point of the norm, the second application reproduces the first
##   within the fp16 rounding allowance.
##
## Double-normalize drift is the judged quantity. The bound is stated before
## measurement and judged per element over fp16 values,
##   u16 = 2⁻¹¹ the fp16 unit roundoff:
##
## | term        | bound                                                                            |
## | ----------- | -------------------------------------------------------------------------------- |
## | rstd drift  | the second pass's rstd deviates from 1 by ≤ 2u16 + C·2⁻²⁴ + 4·2⁻²¹               |
## | elementwise | abs(out2 − out1) ≤ abs(out1)·(3u16 + C·2⁻²⁴ + 4·2⁻²¹) + ½ulp16(abs(out1)) + 2⁻²⁵ |
##
## Derivation:
## - the first pass returns out1 = fp16(s·rstd1), rstd1 = rsqrt(mean(s²) + ε),
##   every element within u16 relative of s·rstd1
## - the second pass's mean of squares sits within 2u16 (the per-element rounding, squared)
##   + C·2⁻²⁴ (the fp32 accumulation) + 4·2⁻²¹ of 1, the 4·2⁻²¹ term carrying the rsqrt
##   form error of the first pass through rstd1²·mean(s²)
## - rstd2 = rsqrt(mean2 + ε) then deviates from 1 by half the mean's relative deviation,
##   the same rsqrt form running in both passes, so the form error enters only
##   through this carried term
## - the element judgment bounds one fp16 store round (½ulp) plus the fp16 subnormal
##   floor 2⁻²⁵ on top of the rstd drift
##
## Shapes (C = 128 = the static tile width, grid (1, ceil(M/8)), 32 lanes):
##
## | shape  | M  | grid tail              | cases |
## | ------ | --- | ---------------------- | ----- |
## | exact  | 8  | one full tile          | 16    |
## | tail   | 13 | 3 rows zero-filled     | 16    |
## | multi  | 29 | 3 blocks, 2 tail       | 8     |
## | single | 1  | one row, 7 zero-filled | 8     |
##
## - the zero-filled rows are skipped on store, out reads back exactly M rows
## - eps = 1e-6, negligible against the near-one means of squares

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/norm
import properties
import ../ceramic/ceramic_pagebuf

# ─── Device entry ─────────────────────────────────────────────────────

const NormMsl = metal:
  proc prop_norm_res_in_fp16(
      outp: ptr UncheckedArray[float16],
      x, r, w: ptr UncheckedArray[float16],
      M, C: int32, eps: float32) {.global.} =
    rms_norm_res_in_fwd(outp, x, r, w, M, C, eps, 128)

# ─── Allowance constants ─────────────────────────────────────────────

const
  U16 = 4.8828125e-4                 # 2⁻¹¹, the fp16 unit roundoff
  RStdCarry = 4.0 * 4.76837158203125e-7  # 2·2⁻²¹, the rsqrt form error class
  FloorSub = 2.9802322387695312e-8   # 2⁻²⁵, half the constant fp16 subnormal ulp

var suiteCases, suiteLaunches = 0
var suiteWorstUse, suiteWorstAbs, suiteWorstUlp = 0.0'f64
var suiteExact, suiteTotal = 0

proc runShape(engine: HwEngine, M, cases: int, seed: uint64, label: string) =
  ## One shape over `cases` independent seeded runs, norm applied once
  ## then again on its own output, judged per element against
  ## the idempotence allowance above.
  const C = 128
  let nElems = M * C
  let eps = 1.0e-6'f32
  var xB = allocPageBuf[uint16](nElems)
  var rB = allocPageBuf[uint16](nElems)
  var wB = allocPageBuf[uint16](C)
  var outB = allocPageBuf[uint16](nElems)
  defer:
    freePageBuf(xB); freePageBuf(rB); freePageBuf(wB); freePageBuf(outB)
  var xPA = xB.pa()
  var rPA = rB.pa()
  var wPA = wB.pa()
  var outPA = outB.pa()

  var judge1 = Judge()
  var launches = 0

  # zero residual, all-ones weight:
  #   the fixed-point configuration
  for i in 0 ..< nElems: rB.hostPtr[i] = fp32ToFp16(0.0'f32)
  for c in 0 ..< C: wB.hostPtr[c] = fp32ToFp16(1.0'f32)

  proc load(xBits: seq[uint16]) =
    for i in 0 ..< nElems: xB.hostPtr[i] = xBits[i]

  proc launch() =
    engine.run << (grid: (1, int((M + 7) div 8), 1), blk: (32, 1, 1)) >>
      ("prop_norm_res_in_fp16", outPA, (xPA, rPA, wPA, int32(M), int32(C), eps))
    inc launches

  var rng = initPropRng(seed)
  for caseId in 0 ..< cases:
    var xBits = newSeq[uint16](nElems)
    for i in 0 ..< nElems:
      xBits[i] = rng.nextF32(-2.0'f32, 2.0'f32).narrowTo(kFloat16)
    load(xBits)
    launch()
    # kernel-read buffers stay bit-identical, the host memory is the device memory
    for i in 0 ..< nElems:
      doAssert xB.hostPtr[i] == xBits[i], "kernel-read x modified"
    for i in 0 ..< nElems:
      doAssert rB.hostPtr[i] == fp32ToFp16(0.0'f32), "kernel-read r modified"
    for c in 0 ..< C:
      doAssert wB.hostPtr[c] == fp32ToFp16(1.0'f32), "kernel-read w modified"

    var out1 = newSeq[uint16](nElems)
    for i in 0 ..< nElems: out1[i] = outB.hostPtr[i]

    # second application on the first output, fresh input buffer holding out1
    for i in 0 ..< nElems: xB.hostPtr[i] = out1[i]
    launch()
    for i in 0 ..< nElems:
      doAssert xB.hostPtr[i] == out1[i], "kernel-read x modified"

    # the per-row allowance, from the input row's exact fp32 squares
    for m in 0 ..< M:
      var sumSq = 0.0'f64
      for c in 0 ..< C:
        let xv = fp16ToFp32(xBits[m * C + c]).float64
        sumSq += xv * xv
      let meanSq = sumSq / C.float64
      for c in 0 ..< C:
        let o1 = fp16ToFp32(out1[m * C + c]).float64
        let o2 = fp16ToFp32(outB.hostPtr[m * C + c]).float64
        let allow = abs(o1) * (3.0 * U16 + C.float64 * U32 + RStdCarry) +
          0.5 * ulpStepAt(ulpFp16, abs(o1)) + FloorSub
        judge1.judge(&"out2−out1 (m {m}, c {c}, case {caseId})", o2, o1, allow,
          ulpStepAt(ulpFp16, abs(o1)))
        # meanSq stays in the derivation's checked regime, near 1
        doAssert meanSq > 1.0e-2 and meanSq < 1.0e2,
          "mean of squares left the near-1 regime the allowance derivation assumes"

  echo &"[{label} M={M}] cases={cases} launches={launches} " &
    &"worst |Δ| {judge1.worstAbs:.3e}, worst usage {judge1.worstUse:.3f}, " &
    &"worst {judge1.worstUlp:.2f} ulp, bit-exact {judge1.exact}/{judge1.total}"
  suiteCases += cases
  suiteLaunches += launches
  suiteWorstUse = max(suiteWorstUse, judge1.worstUse)
  suiteWorstAbs = max(suiteWorstAbs, judge1.worstAbs)
  suiteWorstUlp = max(suiteWorstUlp, judge1.worstUlp)
  suiteExact += judge1.exact
  suiteTotal += judge1.total

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(NormMsl)
  runShape(engine, 8, 16, 0xC04D0601'u64, "exact one tile")
  runShape(engine, 13, 16, 0xC04D0602'u64, "tail rows")
  runShape(engine, 29, 8, 0xC04D0603'u64, "multi block tail")
  runShape(engine, 1, 8, 0xC04D0604'u64, "single row")
  echo &"PROPERTIES NORM IDEMPOTENCE VERDICT: cases={suiteCases} launches={suiteLaunches} " &
    &"worst |Δ| {suiteWorstAbs:.3e}, worst usage {suiteWorstUse:.3f}, " &
    &"worst {suiteWorstUlp:.2f} ulp, bit-exact {suiteExact}/{suiteTotal}"

main()
