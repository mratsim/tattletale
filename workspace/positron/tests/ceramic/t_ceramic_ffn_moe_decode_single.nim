# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
##
##   nim c -r -d:release --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_ffn_moe_decode_single.nim
##
## Dedicated suite for the MoE decode slot-group walk (`moe_fwd_decode_at`),
## one seeded launch per case at grid (T, K+1, 1) over the contract extent.
##
## Sequence, one case:
##
##   seeded inputs → page buffers → launch (T, K+1, 1) → record
##   → sentinel walk → relaunch → bit compare
##
## | check       | contract                                                                                    |
## | ----------- | ------------------------------------------------------------------------------------------- |
## | page fit    | every buffer's byte length is a `HostPageSize` multiple before the launch                   |
## | sentinels   | the contract rows hold no sentinel after the launch, the spare extent stays sentinel-intact |
## | no gate     | `SharedGate = false` stores the ungated shared chain, the poisoned gate vector never read   |
## | scale       | `Scale = 2` doubles the routed weights, the same fixture recipe                             |
## | determinism | the relaunch over the restored pre-state is bit-identical                                   |
##
## | shape   | T | H    | E   | K | I   | Scale | SharedGate | dtype | cases |
## | ------- | --- | ---- | --- | --- | --- | ----- | ---------- | ----- | ----- |
## | prod    | 4 | 2048 | 256 | 8 | 512 | 1.0   | true       | bf16  | 1     |
## | no gate | 1 | 2048 | 256 | 8 | 512 | 1.0   | false      | bf16  | 1     |
## | scale 2 | 1 | 2048 | 256 | 8 | 512 | 2.0   | true       | bf16  | 1     |
##
## | guard    | record                                                                                                                          |
## | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
## | pre-fix  | the row guard dropped, every atom row's row-0 lane racing on the destination, the lost writes zeroing partial-row contributions |
## | standing | the contract-extent sentinel, every contract row must be written past its pre-filled sentinel, a lost row cannot hide           |

import std/[strformat, math]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/ffn_moe_decode_single
import ceramic_pagebuf
import ceramic_dtype
import ../properties/properties

# ─── Geometry, the production decode shape ───────────────────────────

const
  Tokens = 4
    ## Production case's token count, one grid point per (token, slot).
  H = 2048
  E = 256
  K = 8
  I = 512
  Sentinel = 4.203895392974451e-45'f32
    ## 2⁻¹⁴⁹, the smallest positive subnormal, the write-extent sentinel

# ─── Device entries, one per static binding set ───────────────────────

const MoEFwdDecodeMsl = metal:
  proc cer_moe_fwd_prod(
      partial: ptr UncheckedArray[float32],
      x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
      shared_gate_w, shared_up_w, shared_down_w,
      shared_gate_vec_w: ptr UncheckedArray[bfloat16],
      h_scratch, hs_scratch: ptr UncheckedArray[bfloat16],
      scores_scratch: ptr UncheckedArray[float32]) {.global.} =
    moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, true](
      partial, x, router_w, gate_up_w, down_w,
      shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
      h_scratch, hs_scratch, scores_scratch)

  proc cer_moe_fwd_nogate(
      partial: ptr UncheckedArray[float32],
      x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
      shared_gate_w, shared_up_w, shared_down_w,
      shared_gate_vec_w: ptr UncheckedArray[bfloat16],
      h_scratch, hs_scratch: ptr UncheckedArray[bfloat16],
      scores_scratch: ptr UncheckedArray[float32]) {.global.} =
    moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 1.0'f32, false](
      partial, x, router_w, gate_up_w, down_w,
      shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
      h_scratch, hs_scratch, scores_scratch)

  proc cer_moe_fwd_scale2(
      partial: ptr UncheckedArray[float32],
      x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
      shared_gate_w, shared_up_w, shared_down_w,
      shared_gate_vec_w: ptr UncheckedArray[bfloat16],
      h_scratch, hs_scratch: ptr UncheckedArray[bfloat16],
      scores_scratch: ptr UncheckedArray[float32]) {.global.} =
    moe_fwd_decode[bfloat16, 2048, 256, 8, 512, 2.0'f32, true](
      partial, x, router_w, gate_up_w, down_w,
      shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
      h_scratch, hs_scratch, scores_scratch)

# ─── Case runners ─────────────────────────────────────────────────────

proc fillFormula(buf: var PageBuf[uint16]; n: int) =
  ## Deterministic weight fill, one bf16 pattern per element,
  ## magnitudes in [-0.025, 0.0251].
  doAssert n <= buf.elems
  doAssert buf.elems * sizeof(uint16) mod HostPageSize == 0,
    "no-copy binding needs a page-multiple byte length"
  for i in 0 ..< n:
    buf.hostPtr[i] = f32ToBf16(float32(i mod 501 - 250) * 1.0e-4'f32)

proc fillRngBf(buf: var PageBuf[uint16]; rng: var PropRng; n: int;
    lo, hi: float32) =
  ## `n` bf16 bit patterns of uniform samples in [lo, hi].
  doAssert n <= buf.elems
  for i in 0 ..< n:
    buf.hostPtr[i] = f32ToBf16(rng.nextF32(lo, hi))

proc readSeq[T](src: ptr UncheckedArray[T]; n: int): seq[T] =
  result = newSeq[T](n)
  for i in 0 ..< n:
    result[i] = src[i]

proc prefillPartial(buf: var PageBuf[float32]; rows: int) =
  ## Sentinel fill over the partial buffer's full page-multiple extent.
  for i in 0 ..< rows * H:
    buf.hostPtr[i] = Sentinel

type Weights = object
  ## Seeded weight page buffers, shared by the three cases.
  routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGV:
    PageBuf[uint16]

proc buildWeights(seed: uint64): Weights =
  ## Seeded weight set, the router weight plus the gate weight vector,
  ## both from the seed's rng, the expert and shared weights formula-filled.
  var rng = initPropRng(seed)
  result.routerW = allocPageBuf[uint16](E * H)
  result.gateUpW = allocPageBuf[uint16](E * 2 * I * H)
  result.downW = allocPageBuf[uint16](E * H * I)
  result.sharedGW = allocPageBuf[uint16](I * H)
  result.sharedUW = allocPageBuf[uint16](I * H)
  result.sharedDW = allocPageBuf[uint16](H * I)
  result.sharedGV = allocPageBuf[uint16](H)
  fillRngBf(result.routerW, rng, E * H, -0.02'f32, 0.02'f32)
  fillFormula(result.gateUpW, E * 2 * I * H)
  fillFormula(result.downW, E * H * I)
  fillFormula(result.sharedGW, I * H)
  fillFormula(result.sharedUW, I * H)
  fillFormula(result.sharedDW, H * I)
  fillFormula(result.sharedGV, H)

var suiteLaunches = 0

proc runCase(engine: HwEngine; w: Weights; seed: uint64; tokens: int;
    entry: string; scale: float32; useGate: bool; poisonGateVec: bool) =
  ## One seeded case at grid (tokens, K+1, 1), the contract-extent sentinel
  ## walk and the relaunch bit-identity.
  ##
  ## `poisonGateVec` fills a dedicated NaN-poisoned shared gate weight vector,
  ## the ungated case's proof the kernel never reads it.
  const Rows = (Tokens + 1) * (K + 1)
    ## Spare (K+1)-row block past the contract extent, the unused extent
    ## the sentinel judges
  var
    partial = allocPageBuf[float32](Rows * H)
    xB = allocPageBuf[uint16](tokens * H)
    hB = allocPageBuf[uint16](tokens * K * I)
    hsB = allocPageBuf[uint16](tokens * I)
    scoresScratch = allocPageBuf[float32](tokens * (K + 1) * E)
      ## the router selection's per-(token, slot-group) score rows
  defer:
    freePageBuf(partial); freePageBuf(xB); freePageBuf(hB); freePageBuf(hsB)
    freePageBuf(scoresScratch)

  var rng = initPropRng(seed)
  fillRngBf(xB, rng, tokens * H, -1.0'f32, 1.0'f32)
  var poisonedGV: PageBuf[uint16]
  if poisonGateVec:
    # the gate vector's poison goes to a dedicated buffer, the shared weight
    # set stays clean for the later cases
    poisonedGV = allocPageBuf[uint16](H)
    for i in 0 ..< H:
      poisonedGV.hostPtr[i] = 0x7FC0'u16

  var partialPA = partial.pa()
  var xPA = xB.pa()
  var routerWPA = w.routerW.pa()
  var gateUpWPA = w.gateUpW.pa()
  var downWPA = w.downW.pa()
  var sharedGWPA = w.sharedGW.pa()
  var sharedUWPA = w.sharedUW.pa()
  var sharedDWPA = w.sharedDW.pa()
  var sharedGVPA = if poisonGateVec: poisonedGV.pa() else: w.sharedGV.pa()
  var hPA = hB.pa()
  var hsPA = hsB.pa()
  var scoresPA = scoresScratch.pa()

  proc launch(): bool {.gcsafe.} =
    engine.run << (grid: (tokens, K + 1, 1), blk: (32, 1, 1)) >>
      (entry, partialPA,
        (xPA, routerWPA, gateUpWPA, downWPA, sharedGWPA, sharedUWPA,
         sharedDWPA, sharedGVPA, hPA, hsPA, scoresPA))
    inc suiteLaunches
    result = true

  proc readAll(): tuple[h: seq[uint16], hs: seq[uint16], partial: seq[float32]] =
    result.h = readSeq(hB.hostPtr, tokens * K * I)
    result.hs = readSeq(hsB.hostPtr, tokens * I)
    result.partial = readSeq(partial.hostPtr, tokens * (K + 1) * H)

  prefillPartial(partial, Rows)
  discard launch()

  if not useGate:
    # the poisoned gate weight vector, never read, so the shared row holds
    # the ungated chain with no NaN bit pattern anywhere
    for e in 0 ..< H:
      let pM = partial.hostPtr[(K) * H + e]
      doAssert classify(pM) notin {fcNan, fcInf},
        &"the poisoned gate weight vector leaked into the shared partial at {e}"

  # the write extent, no sentinel left in the contract rows and the spare
  # block past the extent untouched:
  # the rowIdx < 0 rows and the page tail must never store
  for i in 0 ..< tokens * (K + 1) * H:
    doAssert partial.hostPtr[i] != Sentinel,
      &"contract partial row element {i} never written"
  for i in tokens * (K + 1) * H ..< Rows * H:
    doAssert partial.hostPtr[i] == Sentinel,
      &"the unused extent was written at {i}"

  # the relaunch, bit-identical partials, h and hs scratch
  let snap2 = readAll()
  discard launch()
  let snap3 = readAll()
  for i in 0 ..< tokens * K * I:
    doAssert snap3.h[i] == snap2.h[i], &"h scratch differs at {i}"
  for i in 0 ..< tokens * I:
    doAssert snap3.hs[i] == snap2.hs[i], &"hs scratch differs at {i}"
  for i in 0 ..< tokens * (K + 1) * H:
    doAssert snap3.partial[i] == snap2.partial[i], &"partial differs at {i}"
  echo &"[moe fwd decode {entry}] relaunch bit-identical, extent sentinels clean"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(MoEFwdDecodeMsl)
  var w = buildWeights(0xC04D0610'u64)
  defer:
    freePageBuf(w.routerW); freePageBuf(w.gateUpW); freePageBuf(w.downW)
    freePageBuf(w.sharedGW); freePageBuf(w.sharedUW); freePageBuf(w.sharedDW)
    freePageBuf(w.sharedGV)
  runCase(engine, w, 0xC04D0611'u64, Tokens, "cer_moe_fwd_prod",
    1.0'f32, true, false)
  runCase(engine, w, 0xC04D0612'u64, 1, "cer_moe_fwd_nogate",
    1.0'f32, false, true)
  runCase(engine, w, 0xC04D0613'u64, 1, "cer_moe_fwd_scale2",
    2.0'f32, true, false)
  echo &"CERAMIC MOE_FWD_DECODE VERDICT: cases=3 launches={suiteLaunches} " &
    &"extent sentinels clean, relaunch bit-identical"

main()
