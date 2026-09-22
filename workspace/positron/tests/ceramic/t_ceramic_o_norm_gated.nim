## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_o_norm_gated.nim
##
## Ceramic SiLU-gated RMSNorm suite, `src/kernels/ceramic/o_norm_gated.nim` judged per
## element against the closed-form naive reference below.
##
## | subject        | contract                                                                               |
## | -------------- | -------------------------------------------------------------------------------------- |
## | rounding chain | rstd = rsqrt(mean(x²) + eps), out = bf16(bf16(w · bf16(x · rstd)) · silu(g)) per row   |
## | naive side     | rstd and silu in fp64 over the exact widenings, the chain on the fp32-rounded operands |
## | pair split     | rmsWeightTile + siluMulTile bit-exact vs the fused entry, the bf16 weighted split      |

##
## Shapes (Dv = 128 = the tile width, TileR = 8, grid (1, ceil(M/8)), 32 lanes):
##
## | shape  | M  | grid tail              | cases |
## | ------ | --- | ---------------------- | ----- |
## | exact  | 8  | one full tile          | 32    |
## | tail   | 13 | 3 rows zero-filled     | 32    |
## | multi  | 29 | 3 blocks, 2 tail       | 16    |
## | single | 1  | one row, 7 zero-filled | 16    |
##
## The single row is the decode regime's M = 1 launch, the tile loads 7
## zero-filled rows and stores exactly one.

##
## | check       | content                                                                    |
## | ----------- | -------------------------------------------------------------------------- |
## | band        | the closed-form band, 32 seeded random cases per shape                     |
## | pair split  | fused entry and composed pair bit-identical per element, every case        |
## | determinism | case 0 relaunched bit-identical per shape                                  |
## | sentinels   | out stays inside its extent every launch, x, gate and w stay bit-identical |

##
## Band model, stated before measurement and judged per element, the fp32 unit
## roundoff u32 = 2⁻²⁴ and the bf16 unit roundoff u_bf = 2⁻⁸:
##
## | term        | bound                                            |
## | ----------- | ------------------------------------------------ |
## | relRstd     | (Dv*2^-24*sum x^2) / (mean(x^2) + eps) + 2*2^-21 |
## | relNormed   | relRstd + 2*u_bf                                 |
## | relWeighted | relNormed + 2*u_bf + 2*u32                       |
## | relSilu     | 8*u32                                            |
## | relOut      | relWeighted + relSilu + 2*u_bf                   |
##
## | bar        | bound                    | covers                                                     |
## | ---------- | ------------------------ | ---------------------------------------------------------- |
## | out (m, c) | abs(out)*relOut + 2^-126 | every term above, one bf16 RNE per side per rounding stage |
##
## - relRstd covers the row-sum order, both sides accumulate the squares in fp32
##   within Dv·2⁻²⁴·Σc x² of the exact mean, and the kernel's rsqrt sits
##   2·2⁻²¹ off the correctly-rounded 1/sqrt (the GDN q̃ class)
## - 2·u_bf per bf16 RNE covers the operand difference, the two sides round slightly
##   different fp32 values, each RNE within u_bf of its operand
## - relSilu covers the exp2-form kernel silu vs the fp64 naive exp form,
##   the 1-ulp-class exponential difference carried by the silu_and_mul band
##
##   x → x·rstd → bf16 → ·w → bf16 → ·silu(g) → bf16, the 3-round chain the band walks
## - the 2⁻¹²⁶ floor covers the bf16 subnormal output grid
## - the measured divergence justifies the model, never sets the bar
## - adjudicated on Apple M4 Max with fresh seeded xorshift64 inputs

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/o_norm_gated
import ../naive/naive_rng
import ../naive/naive_tensors
import ceramic_pagebuf

# ─── Device entry ─────────────────────────────────────────────────────

const ONormMsl = metal:
  proc cer_o_norm_gated(
      outp, x, gate, w: ptr UncheckedArray[bfloat16],
      M: int32, eps: float32) {.global.} =
    rmsNormGatedTile(outp, x, gate, w, M, eps, 128, 8)

  proc cer_o_norm_pair_w(
      midp, x, w: ptr UncheckedArray[bfloat16],
      M: int32, eps: float32) {.global.} =
    rmsWeightTile(midp, x, w, M, eps, 128, 8)

  proc cer_o_norm_pair_silu(
      outp, mid, gate: ptr UncheckedArray[bfloat16],
      M: int32) {.global.} =
    siluMulTile(outp, mid, gate, M, 128, 8)

# ─── Host, the closed-form naive reference and the band ───────────────

const
  Dv = 128
  U32 = 5.9604644775390625e-8        # 2⁻²⁴, the fp32 unit roundoff
  UBf = 3.90625e-3                   # 2⁻⁸, the bf16 unit roundoff
  RelRsqrt = 2.0 * 9.5367431640625e-7  # 2·2⁻²¹, rsqrt vs correctly-rounded 1/sqrt
  RelSilu = 8.0 * U32                # exp2-form vs exp-form silu, the 1-ulp class
  FloorBf = 7.346879709099078e-39    # 2⁻¹²⁶, the bf16 subnormal output grid floor

proc naiveONorm(x, gate, w: seq[uint16]; eps: float32; M: int): seq[float64] =
  ## Closed-form reference over the exact bf16 widenings, fp64 row math,
  ## then the recorded rounding chain applied on the fp32-rounded operands.
  result = newSeq[float64](M * Dv)
  for m in 0 ..< M:
    var sumSq = 0.0'f64
    for c in 0 ..< Dv:
      let xv = bf16ToF32(x[m * Dv + c]).float64
      sumSq += xv * xv
    let rstd = 1.0 / sqrt(sumSq / Dv.float64 + eps.float64)
    for c in 0 ..< Dv:
      let xv = bf16ToF32(x[m * Dv + c])
      let wv = bf16ToF32(w[c])
      let gv = bf16ToF32(gate[m * Dv + c])
      let normed = f32ToBf16(xv * rstd.float32)
      let weighted = f32ToBf16(wv * bf16ToF32(normed))
      let silu = gv / (1.0'f32 + exp(-gv))
      result[m * Dv + c] =
        bf16ToF32(f32ToBf16(bf16ToF32(weighted) * silu)).float64

type
  CaseInputs = object
    ## One case's seeded inputs, bf16 bits shared by the kernel and the naive
    ## sides through their exact fp32 widenings.
    xBits: seq[uint16]
    gateBits: seq[uint16]
    wBits: seq[uint16]

proc relRstdOf(xBits: seq[uint16]; eps: float32; m: int): float64 =
  ## rstd relative-error term of the band, from the widened row's squares.
  var sumSq = 0.0'f64
  var sumSqAbs = 0.0'f64
  for c in 0 ..< Dv:
    let xv = bf16ToF32(xBits[m * Dv + c]).float64
    sumSq += xv * xv
    sumSqAbs += abs(xv * xv)
  let denom = sumSq / Dv.float64 + eps.float64
  result = (Dv.float64 * U32 * sumSqAbs) / denom + RelRsqrt

proc runCombo(engine: HwEngine; M, cases: int; seed: uint64; label: string) =
  ## One shape over `cases` independent seeded runs, judged per element against
  ## the closed-form reference under the band, case 0 relaunched bit-identical
  ## and the composed pair bit-compared against the fused entry every case.
  let nElems = M * Dv
  var outB = allocPageBuf[uint16](nElems)
  var midB = allocPageBuf[uint16](nElems)
  var xB = allocPageBuf[uint16](nElems)
  var gateB = allocPageBuf[uint16](nElems)
  var wB = allocPageBuf[uint16](Dv)
  defer:
    freePageBuf(outB); freePageBuf(midB); freePageBuf(xB)
    freePageBuf(gateB); freePageBuf(wB)
  var outPA = outB.pa()
  var midPA = midB.pa()
  var xPA = xB.pa()
  var gatePA = gateB.pa()
  var wPA = wB.pa()
  let eps = 1.0e-6'f32
  let gridY = int32((M + 7) div 8)

  var worstUse = 0.0'f64
  var exact = 0
  var total = 0
  var launches = 0

  proc takeInputs(rng: var NaiveRng): CaseInputs =
    ## Seeded inputs, bf16 bits for the gated norm operands x, gate and w.
    var xBits = newSeq[uint16](nElems)
    var gateBits = newSeq[uint16](nElems)
    var wBits = newSeq[uint16](Dv)
    for i in 0 ..< nElems:
      xBits[i] = f32ToBf16(rng.nextF32(-2.0'f32, 2.0'f32))
      gateBits[i] = f32ToBf16(rng.nextF32(-2.0'f32, 2.0'f32))
    for c in 0 ..< Dv:
      wBits[c] = f32ToBf16(rng.nextF32(0.5'f32, 1.5'f32))
    result = CaseInputs(xBits: xBits, gateBits: gateBits, wBits: wBits)

  proc load(ci: CaseInputs) =
    for i in 0 ..< nElems:
      xB.hostPtr[i] = ci.xBits[i]
      gateB.hostPtr[i] = ci.gateBits[i]
    for c in 0 ..< Dv:
      wB.hostPtr[c] = ci.wBits[c]

  proc launchFused(ci: CaseInputs) =
    engine.run << (grid: (1, int(gridY), 1), blk: (32, 1, 1)) >>
      ("cer_o_norm_gated", outPA, (xPA, gatePA, wPA, int32(M), eps))
    inc launches

  proc sentinels(ci: CaseInputs) =
    assertTailZero(outB, nElems)
    assertReadUnchanged(xB, ci.xBits)
    assertReadUnchanged(gateB, ci.gateBits)
    assertReadUnchanged(wB, ci.wBits)

  proc snapOut(): seq[uint16] =
    result = newSeq[uint16](nElems)
    for i in 0 ..< nElems:
      result[i] = outB.hostPtr[i]

  var case0Snap: seq[uint16]
  var rng = initNaiveRng(seed)
  for caseId in 0 ..< cases:
    let ci = takeInputs(rng)
    let want = naiveONorm(ci.xBits, ci.gateBits, ci.wBits, eps, M)
    load(ci)
    launchFused(ci)
    sentinels(ci)
    let fused = snapOut()

    for m in 0 ..< M:
      let relRstd = relRstdOf(ci.xBits, eps, m)
      let relOut = relRstd + 4.0 * UBf + 2.0 * U32 + RelSilu
      for c in 0 ..< Dv:
        let idx = m * Dv + c
        let got = bf16ToF32(fused[idx]).float64
        let bar = abs(want[idx]) * relOut + FloorBf
        let diff = abs(got - want[idx])
        doAssert diff <= bar,
          &"out outside the bar at (m {m}, c {c}, case {caseId}): " &
          &"{diff:.3e} > {bar:.3e}"
        worstUse = max(worstUse, diff / bar)
        if got == want[idx]:
          inc exact
        inc total

    # the composed pair, bit-identical to the fused entry on the same inputs
    engine.run << (grid: (1, int(gridY), 1), blk: (32, 1, 1)) >>
      ("cer_o_norm_pair_w", midPA, (xPA, wPA, int32(M), eps))
    engine.run << (grid: (1, int(gridY), 1), blk: (32, 1, 1)) >>
      ("cer_o_norm_pair_silu", outPA, (midPA, gatePA, int32(M)))
    inc launches, 2
    sentinels(ci)
    for i in 0 ..< nElems:
      doAssert outB.hostPtr[i] == fused[i],
        "composed pair not bit-identical to the fused entry"

    if caseId == 0:
      case0Snap = fused

  block determinism:
    var rng0 = initNaiveRng(seed)
    let ci0 = takeInputs(rng0)
    load(ci0)
    launchFused(ci0)
    let again = snapOut()
    for i in 0 ..< nElems:
      doAssert again[i] == case0Snap[i], "out differs run to run"

  echo &"[{label} M={M}] cases={cases} launches={launches} " &
    &"worst bar usage {worstUse:.3f}, bit-exact {exact}/{total}"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(ONormMsl)
  runCombo(engine, 8, 32, 0xC04D0501'u64, "exact one tile")
  runCombo(engine, 13, 32, 0xC04D0502'u64, "tail rows")
  runCombo(engine, 29, 16, 0xC04D0503'u64, "multi block tail")
  runCombo(engine, 1, 16, 0xC04D0504'u64, "single row")
  echo "CERAMIC O_NORM_GATED VERDICT: all cases inside the stated per-element bars"

main()
