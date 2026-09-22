# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim c -r -d:release --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_moe_fwd_decode.nim
## - store-race sabotage build, judgment contract below:
##   nim c -r -d:release -d:StoreRaceSabotage --outdir:build/tests --nimcache:nimcache/red tests/ceramic/t_ceramic_moe_fwd_decode.nim
##
## Dedicated suite for the MoE decode slot-group walk (`moe_fwd_decode_at`),
## one seeded launch per case at grid (T, K+1, 1), the partial rows judged
## per element against a naive walk of the same bits.
##
## Sequence, one case:
##
##   seeded inputs → page buffers → launch (T, K+1, 1) → snapshot
##   → naive walk → per-element judgment → sentinel walk → relaunch → bit compare
##
## | check       | contract                                                                                   |
## | ----------- | ------------------------------------------------------------------------------------------ |
## | page fit    | every buffer's byte length is a `HostPageSize` multiple before the launch                  |
## | selection   | the walk exposes no ids or weights buffers, both judged through the partial rows           |
## | partials    | every fp32 partial row inside its stated per-element bar, two-sided against the naive walk |
## | sentinels   | kernel-written buffers hold no sentinel, the contract-extent tail stays sentinel-intact    |
## | no gate     | `SharedGate = false` stores the ungated shared chain, the poisoned gate vector never read  |
## | scale       | `Scale = 2` doubles the routed weights, the partial bars carried through the same walk     |
## | determinism | the relaunch over the restored pre-state is bit-identical                                  |
##
## | shape   | T | H    | E   | K | I   | Scale | SharedGate | family | cases |
## | ------- | --- | ---- | --- | --- | --- | ----- | ---------- | ------ | ----- |
## | prod    | 4 | 2048 | 256 | 8 | 512 | 1.0   | true       | bf16   | 1     |
## | no gate | 1 | 2048 | 256 | 8 | 512 | 1.0   | false      | bf16   | 1     |
## | scale 2 | 1 | 2048 | 256 | 8 | 512 | 2.0   | true       | bf16   | 1     |
##
## Band model, stated before measurement, u32 = 2⁻²⁴ fp32, UBf = 2⁻⁸ bf16,
## the same classes the composition tier states
##
## | link         | bar                                                                                            |
## | ------------ | ---------------------------------------------------------------------------------------------- |
## | logit        | 2·H·u32·Σ_k abs(x_k·w_k) + 2·UBf·abs(logit) + floor, the 16-wide-chunk mma vs sequential sum   |
## | gate/up sums | 2·H·u32·Σ_k abs(x_k·w_k) + 2·u32·abs(sum)                                                      |
## | h (silu·mul) | (RelSilu + 4·u32)·max(abs h) + 2·UBf·(abs h sum) + 1.1·abs(u)·gBar + abs(silu(g))·uBar + floor |
## | down sums    | 2·I·u32·Σ_i abs(h_i·w_i) + h-bar flow + 2·u32·abs(sum) + floor                                 |
## | weight       | abs(w)·(2·logitBar + 2·UBf) + floor, the softmax's relative logit-error class                  |
## | partial row  | w·downBar + abs(down)·weightBar + 2·u32·abs(partial) + floor + the two centers' own deviation  |
##
## Store-race sabotage (`-d:StoreRaceSabotage`):
## - restores the pre-fix store spelling in `storeRowsScaledF32`, the row
##   guard dropped, every atom row's row-0 lane racing on the destination
## - the callers load with `rowLimit = 1`, so the racing lanes carry the exact
##   zeros of the rows above row 0 and the lost writes zero the contribution
## - the prod case's per-element judgment is two-sided, a zeroed partial row
##   leaves the naive center's magnitude and exits the bar
##
## | outcome        | record                                                                                         |
## | -------------- | ---------------------------------------------------------------------------------------------- |
## | sabotage       | the judgment stays green on the recorded driver, same-address stores resolve by lowest lane id |
## | outcome        | the row-0 value lane won 4096 of 4096 repeat trials                                            |
## | guard standing | single-writer in every default build, the Metal API defines no same-address store winner       |

import std/[strformat, math]
import workspace/crucible
import workspace/ceramic
import ../../src/kernels/ceramic/moe_fwd_decode
import ../naive/naive_rng
import ../naive/naive_tensors
from ../naive/naive_layer_ops import naiveSoftmaxTopKRouter, naiveSiluMulEl,
    naiveSharedGate, bf16Round
from ../naive/naive_grouped_mm import naiveGroupedMmSums, gmmBf16
import ceramic_pagebuf

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

# ─── Band-model constants ─────────────────────────────────────────────

const
  U32 = 5.9604644775390625e-8'f64        # 2⁻²⁴, the fp32 unit roundoff
  UBf = 3.90625e-3'f64                   # 2⁻⁸, the bf16 unit roundoff
  RelSilu = 4.76837158203125e-7'f64      # 8·u32, the exp2-form transcendental class
  FloorBf = 7.346879709099078e-39'f64    # 2⁻¹²⁶, the bf16 subnormal grid floor
  FloorSub = 2.9802322387695312e-8'f64   # 2⁻²⁵, the fp32 partial-row output grid

func widen(s: seq[uint16]): seq[float64] =
  ## Exact widenings of bf16 bit patterns, the band walk's magnitudes.
  result = newSeq[float64](s.len)
  for i in 0 ..< s.len:
    result[i] = bf16ToF32(s[i]).float64

func widenF32(s: seq[float32]): seq[float64] =
  ## Exact fp64 widening of an fp32 row, the band walk's magnitudes.
  result = newSeq[float64](s.len)
  for i in 0 ..< s.len:
    result[i] = s[i].float64

func silu64(x: float64): float64 =
  ## silu in fp64, the h-link band's magnitude reference.
  x / (1.0 + exp(-x))

# ─── Device entries, one per static binding set ───────────────────────

const MoEFwdDecodeMsl = metal:
  proc cer_moe_fwd_prod(
      partial: ptr UncheckedArray[float32],
      x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
      shared_gate_w, shared_up_w, shared_down_w,
      shared_gate_vec_w: ptr UncheckedArray[bfloat16],
      h_scratch, hs_scratch: ptr UncheckedArray[bfloat16]) {.global.} =
    moe_fwd_decode[2048, 256, 8, 512, 1.0'f32, true](
      partial, x, router_w, gate_up_w, down_w,
      shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
      h_scratch, hs_scratch)

  proc cer_moe_fwd_nogate(
      partial: ptr UncheckedArray[float32],
      x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
      shared_gate_w, shared_up_w, shared_down_w,
      shared_gate_vec_w: ptr UncheckedArray[bfloat16],
      h_scratch, hs_scratch: ptr UncheckedArray[bfloat16]) {.global.} =
    moe_fwd_decode[2048, 256, 8, 512, 1.0'f32, false](
      partial, x, router_w, gate_up_w, down_w,
      shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
      h_scratch, hs_scratch)

  proc cer_moe_fwd_scale2(
      partial: ptr UncheckedArray[float32],
      x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
      shared_gate_w, shared_up_w, shared_down_w,
      shared_gate_vec_w: ptr UncheckedArray[bfloat16],
      h_scratch, hs_scratch: ptr UncheckedArray[bfloat16]) {.global.} =
    moe_fwd_decode[2048, 256, 8, 512, 2.0'f32, true](
      partial, x, router_w, gate_up_w, down_w,
      shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
      h_scratch, hs_scratch)

# ─── Host, the naive walk and its band ────────────────────────────────

type NaiveWalk = object
  ## One token's naive decode walk, the judgment's centers
  ##
  ## | field    | value                                                  |
  ## | -------- | ------------------------------------------------------ |
  ## | ids, w   | the naive router's top-K ids and bf16-rounded weights  |
  ## | partial  | the (K+1, H) fp32 partial rows, w·down and gate·shared |
  ## | gP, uP   | per-slot gate/up fp32 sums (I), the h-band's operands  |
  ## | hP       | per-slot h bit patterns (I)                            |
  ## | sgP, suP | the shared gate/up fp32 sums (I)                       |
  ## | hsP      | the shared h bit patterns (I)                          |
  ## | sdP      | the shared down fp32 sums (H)                          |
  ## | gvP      | the shared gate scalar, 1.0 ungated                    |
  ids: seq[int32]
  w: seq[float32]
  partial: seq[float32]
  gP: array[K, seq[float32]]
  uP: array[K, seq[float32]]
  hP: array[K, seq[uint16]]
  sgP: seq[float32]
  suP: seq[float32]
  hsP: seq[uint16]
  sdP: seq[float32]
  gvP: float32

proc naiveWalk(x, routerW, gateUpW, downW, sgW, suW, sdW, gvW: seq[uint16];
    scale: float32; useGate: bool): NaiveWalk =
  ## One token's naive decode walk, the walk `moe_fwd_decode_at` composes,
  ## the gate weight scalar optional.
  ##
  ## | part        | contract                                                       |
  ## | ----------- | -------------------------------------------------------------- |
  ## | router      | `naiveSoftmaxTopKRouter`, bf16-rounded logits, fp32 softmax    |
  ## | projections | `naiveGroupedMmSums` fp32 accumulations, one-expert cubes      |
  ## | activation  | h = `naiveSiluMulEl`(g, u) per element                         |
  ## | partial     | row slot·H + e = w[slot]·down_e, row K·H + e = gate·sharedDown |
  doAssert x.len == H and routerW.len == E * H
  let (ids, w) = naiveSoftmaxTopKRouter(x, routerW, E, H, K, scale)
  result.ids = ids
  result.w = w
  result.partial = newSeq[float32]((K + 1) * H)
  let xMat = NaiveMat[uint16](rows: 1, cols: H, data: x)
  for slot in 0 ..< K:
    let id = ids[slot].int
    let gu = naiveGroupedMmSums(gmmBf16, xMat,
      NaiveCube[uint16](planes: 1, rows: 2 * I, cols: H,
        data: gateUpW[(id * 2 * I) * H ..< ((id + 1) * 2 * I) * H]),
      @[1'i32])
    var hBits = newSeq[uint16](I)
    for i in 0 ..< I:
      hBits[i] = naiveSiluMulEl(gu.data[i], gu.data[I + i])
    result.gP[slot] = gu.data[0 ..< I]
    result.uP[slot] = gu.data[I ..< 2 * I]
    result.hP[slot] = hBits
    let dn = naiveGroupedMmSums(gmmBf16,
      NaiveMat[uint16](rows: 1, cols: I, data: hBits),
      NaiveCube[uint16](planes: 1, rows: H, cols: I,
        data: downW[id * H * I ..< (id + 1) * H * I]),
      @[1'i32])
    for e in 0 ..< H:
      result.partial[slot * H + e] = w[slot] * dn.data[e]
  # the shared expert, the scalar gate weight only when the walk uses it
  result.gvP = if useGate: naiveSharedGate(x, gvW, H) else: 1.0'f32
  let sg = naiveGroupedMmSums(gmmBf16, xMat,
    NaiveCube[uint16](planes: 1, rows: I, cols: H, data: sgW), @[1'i32])
  let su = naiveGroupedMmSums(gmmBf16, xMat,
    NaiveCube[uint16](planes: 1, rows: I, cols: H, data: suW), @[1'i32])
  var hsBits = newSeq[uint16](I)
  for i in 0 ..< I:
    hsBits[i] = naiveSiluMulEl(sg.data[i], su.data[i])
  result.sgP = sg.data
  result.suP = su.data
  result.hsP = hsBits
  let sd = naiveGroupedMmSums(gmmBf16,
    NaiveMat[uint16](rows: 1, cols: I, data: hsBits),
    NaiveCube[uint16](planes: 1, rows: H, cols: I, data: sdW), @[1'i32])
  result.sdP = sd.data
  for e in 0 ..< H:
    result.partial[K * H + e] = result.gvP * sd.data[e]

proc logitOf(x, routerW: seq[uint16]; id: int): float32 =
  ## One expert's naive bf16-rounded logit, the sequential fp32 dot.
  var acc = 0.0'f32
  for k in 0 ..< H:
    acc += bf16ToF32(x[k]) * bf16ToF32(routerW[id * H + k])
  result = bf16ToF32(bf16Round(acc))

func sumAbsLinks(xw: seq[float64]; w: seq[uint16]; off, n: int): float64 =
  ## Σ over one weight row's n elements of abs(xw·w), the logit-band summand.
  for k in 0 ..< n:
    result += abs(xw[k] * bf16ToF32(w[off + k]).float64)

proc judgeSlotPartials(token, slot: int; id: int;
    nw: NaiveWalk; xw: seq[float64];
    gateUpW, downW: seq[uint16];
    hM: seq[uint16]; partialM: seq[float32]; logitBar: float64): float64 =
  ## One routed slot's partial row, per-element judgment against
  ## the naive walk, the h-link band flowed through the down projection.
  ##
  ## - center, the naive partial row's element w[slot]·down_e
  ## - bar, w·downBar + abs(down)·weightBar + 2·u32·abs(pM) + floor
  ##   plus the two centers' own deviation
  let hMrow = widen(hM[(token * K + slot) * I ..< (token * K + slot + 1) * I])
  let hProw = widen(nw.hP[slot])
  let guBase = id * 2 * I * H
  let dnBase = id * H * I
  let wPrime = abs(nw.w[slot].float64)
  # the gate/up link's per-element band, one H-length weight-row sum each
  var hBand = newSeq[float64](I)
  for i in 0 ..< I:
    let gBar = 2.0 * float64(H) * U32 *
      sumAbsLinks(xw, gateUpW, guBase + i * H, H) +
      2.0 * U32 * abs(nw.gP[slot][i].float64)
    let uBar = 2.0 * float64(H) * U32 *
      sumAbsLinks(xw, gateUpW, guBase + (I + i) * H, H) +
      2.0 * U32 * abs(nw.uP[slot][i].float64)
    hBand[i] = (RelSilu + 4.0 * U32) *
      max(abs(hMrow[i]), abs(hProw[i])) +
      2.0 * UBf * (abs(hMrow[i]) + abs(hProw[i])) +
      1.1 * abs(nw.uP[slot][i].float64) * gBar +
      abs(silu64(nw.gP[slot][i].float64)) * uBar + FloorBf
  let dnP = widenF32(naiveGroupedMmSums(gmmBf16,
    NaiveMat[uint16](rows: 1, cols: I, data: nw.hP[slot]),
    NaiveCube[uint16](planes: 1, rows: H, cols: I,
      data: downW[dnBase ..< (id + 1) * H * I]), @[1'i32]).data)
  for e in 0 ..< H:
    var downLocal = 0.0'f64
    let dwBase = dnBase + e * I
    for i in 0 ..< I:
      let dwAbs = abs(bf16ToF32(downW[dwBase + i]).float64)
      downLocal += max(abs(hMrow[i]), abs(hProw[i])) * dwAbs +
        hBand[i] * dwAbs
    downLocal = 2.0 * float64(I) * U32 * downLocal +
      2.0 * U32 * abs(dnP[e]) + FloorSub
    let wBand = abs(dnP[e]) * wPrime * (2.0 * logitBar + 2.0 * UBf)
    let pM = partialM[(token * (K + 1) + slot) * H + e].float64
    let pN = nw.partial[slot * H + e].float64
    let bar = wPrime * downLocal + wBand + 2.0 * U32 * abs(pM) +
      FloorSub + abs(pM - pN)
    doAssert abs(pM - pN) <= bar,
      &"partial row outside the bar at (token {token}, slot {slot}, col {e}): " &
      &"{abs(pM - pN):.3e} > {bar:.3e}"
    result = max(result, abs(pM - pN) / bar)

proc judgeSharedPartial(token: int;
    nw: NaiveWalk; xw: seq[float64];
    sharedGW, sharedUW, sharedDW: seq[uint16];
    hsM: seq[uint16]; partialM: seq[float32]; gateBar: float64): float64 =
  ## One token's shared partial row, per-element judgment against
  ## the naive walk, the hs-link band flowed through the shared down weights.
  let hsMrow = widen(hsM[token * I ..< (token + 1) * I])
  let hsProw = widen(nw.hsP)
  var hsBand = newSeq[float64](I)
  for i in 0 ..< I:
    let gBar = 2.0 * float64(H) * U32 *
      sumAbsLinks(xw, sharedGW, i * H, H) +
      2.0 * U32 * abs(nw.sgP[i].float64)
    let uBar = 2.0 * float64(H) * U32 *
      sumAbsLinks(xw, sharedUW, i * H, H) +
      2.0 * U32 * abs(nw.suP[i].float64)
    hsBand[i] = (RelSilu + 4.0 * U32) *
      max(abs(hsMrow[i]), abs(hsProw[i])) +
      2.0 * UBf * (abs(hsMrow[i]) + abs(hsProw[i])) +
      1.1 * abs(nw.suP[i].float64) * gBar +
      abs(silu64(nw.sgP[i].float64)) * uBar + FloorBf
  for e in 0 ..< H:
    var downLocal = 0.0'f64
    let dwBase = e * I
    for i in 0 ..< I:
      let dwAbs = abs(bf16ToF32(sharedDW[dwBase + i]).float64)
      downLocal += max(abs(hsMrow[i]), abs(hsProw[i])) * dwAbs +
        hsBand[i] * dwAbs
    downLocal = 2.0 * float64(I) * U32 * downLocal +
      2.0 * U32 * abs(nw.sdP[e].float64) + FloorSub
    let gate = abs(nw.gvP.float64)
    let pM = partialM[(token * (K + 1) + K) * H + e].float64
    let pN = nw.partial[K * H + e].float64
    let bar = gate * downLocal + abs(nw.sdP[e].float64) * gateBar +
      2.0 * U32 * abs(pM) + FloorSub + abs(pM - pN)
    doAssert abs(pM - pN) <= bar,
      &"shared partial outside the bar at (token {token}, col {e}): " &
      &"{abs(pM - pN):.3e} > {bar:.3e}"
    result = max(result, abs(pM - pN) / bar)

# ─── Case runners ─────────────────────────────────────────────────────

proc fillFormula(buf: var PageBuf[uint16]; n: int) =
  ## Deterministic weight fill, one bf16 pattern per element, the kernel
  ## and the naive walk reading the same bits, magnitudes in [-0.025, 0.0251].
  doAssert n <= buf.elems
  doAssert buf.elems * sizeof(uint16) mod HostPageSize == 0,
    "no-copy binding needs a page-multiple byte length"
  for i in 0 ..< n:
    buf.hostPtr[i] = f32ToBf16(float32(i mod 501 - 250) * 1.0e-4'f32)

proc fillRngBf(buf: var PageBuf[uint16]; rng: var NaiveRng; n: int;
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
  ## Seeded weight page buffers, shared by the three cases,
  ## the kernel and the naive walk reading the same bits.
  routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGV:
    PageBuf[uint16]

proc buildWeights(seed: uint64): Weights =
  ## Seeded weight set, the router weight plus the gate weight vector,
  ## both from the seed's rng, the expert and shared weights formula-filled.
  var rng = initNaiveRng(seed)
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

proc runCase(engine: HwEngine; w: Weights; seed: uint64; tokens: int;
    entry: string; scale: float32; useGate: bool; poisonGateVec: bool) =
  ## One seeded case at grid (tokens, K+1, 1), the naive walk centering
  ## the judgment, the contract-extent sentinel, the relaunch bit-identity.
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
  defer:
    freePageBuf(partial); freePageBuf(xB); freePageBuf(hB); freePageBuf(hsB)

  var rng = initNaiveRng(seed)
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

  proc launch(): bool {.gcsafe.} =
    engine.run << (grid: (tokens, K + 1, 1), blk: (32, 1, 1)) >>
      (entry, partialPA,
        (xPA, routerWPA, gateUpWPA, downWPA, sharedGWPA, sharedUWPA,
         sharedDWPA, sharedGVPA, hPA, hsPA))
    result = true

  proc readAll(): tuple[h: seq[uint16], hs: seq[uint16], partial: seq[float32]] =
    result.h = readSeq(hB.hostPtr, tokens * K * I)
    result.hs = readSeq(hsB.hostPtr, tokens * I)
    result.partial = readSeq(partial.hostPtr, tokens * (K + 1) * H)

  prefillPartial(partial, Rows)
  discard launch()
  let snap = readAll()

  # the naive walk and the per-element judgment, one token at a time
  let xSeq = readSeq(xB.hostPtr, tokens * H)
  let routerWSeq = readSeq(w.routerW.hostPtr, E * H)
  let gateUpWSeq = readSeq(w.gateUpW.hostPtr, E * 2 * I * H)
  let downWSeq = readSeq(w.downW.hostPtr, E * H * I)
  let sharedGWSeq = readSeq(w.sharedGW.hostPtr, I * H)
  let sharedUWSeq = readSeq(w.sharedUW.hostPtr, I * H)
  let sharedDWSeq = readSeq(w.sharedDW.hostPtr, H * I)
  let sharedGVSeq = readSeq(w.sharedGV.hostPtr, H)
  var worst = 0.0'f64
  for t in 0 ..< tokens:
    let xTok = xSeq[t * H ..< (t + 1) * H]
    let nw = naiveWalk(xTok, routerWSeq, gateUpWSeq, downWSeq,
      sharedGWSeq, sharedUWSeq, sharedDWSeq, sharedGVSeq, scale, useGate)
    let xw = widen(xTok)
    # the walk exposes no ids or weight buffers, the expert selection
    # and the routing weight are judged through the partial rows they
    # produce (a diverging id or weight moves the row off the band)
    for slot in 0 ..< K:
      let id = nw.ids[slot].int
      let logitBar = 2.0 * float64(H) * U32 *
        sumAbsLinks(xw, routerWSeq, id * H, H) +
        2.0 * UBf * abs(logitOf(xTok, routerWSeq, id).float64) + FloorBf
      worst = max(worst, judgeSlotPartials(t, slot, id, nw, xw,
        gateUpWSeq, downWSeq, snap.h, snap.partial, logitBar))
    # the shared row's gate-weight band, the scalar logit's reduction class
    let gvP = nw.gvP.float64
    var gvAbs = 0.0'f64
    for k in 0 ..< H:
      gvAbs += abs(xw[k] * bf16ToF32(sharedGVSeq[k]).float64)
    let gateBar = if useGate:
      0.25 * (2.0 * float64(H) * U32 * gvAbs) +
      abs(gvP) * (RelSilu + 2.0 * U32 + 2.0 * UBf) + FloorBf
    else:
      0.0'f64
    worst = max(worst, judgeSharedPartial(t, nw, xw, sharedGWSeq,
      sharedUWSeq, sharedDWSeq, snap.hs, snap.partial, gateBar))

  if not useGate:
    # the poisoned gate weight vector, never read, so the shared row holds
    # the ungated chain with no NaN bit pattern anywhere
    for e in 0 ..< H:
      let pM = partial.hostPtr[(K) * H + e]
      doAssert classify(pM) notin {fcNan, fcInf},
        &"the poisoned gate weight vector leaked into the shared partial at {e}"
  echo &"[moe fwd decode {entry}] every partial row inside its bar, " &
    &"worst bar usage {worst:.3f}"

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
  echo "[moe fwd decode prod] relaunch bit-identical"

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
  echo "CERAMIC MOE_FWD_DECODE VERDICT: prod, no-gate and scale-2 walks " &
    "inside their bars, extent sentinel clean, relaunch bit-identical"

main()
