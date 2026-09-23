# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/ceramic_mega_gdn_compare.nim
##
## Composition tier for the `qwen35_moe` fused GDN decoder layer:
## - the band-model home, the composition MSL, the seeded inputs
## - the segment runners (norm check, comparison, chain) live in `ceramic_mega_gdn_compare.nim`
##
## Band model, stated before any measurement. Every judged field's bound decomposes
## exactly into the two-term bar below, from the triangle inequality.
##
## | term  | content                                                                                                                                                     |
## | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | local | the stage's landed single-op band (`t_ceramic_dense_linear`, `t_ceramic_gdn_decode`, `t_ceramic_o_norm_gated`) at the mega's observed operands              |
## | sens  | the exact deviation of the naive reference evaluated at the mega's operands vs at the naive's operands, the same naive op run on both observed operand sets |
## | bar   | `bar = local + sens`, the exact decomposition `abs(m − n(x)) ≤ abs(m − n(x')) + abs(n(x') − n(x))`, no model content of its own                             |
##
## | symbol   | value                | meaning                                                 |
## | -------- | -------------------- | ------------------------------------------------------- |
## | u32      | 2⁻²⁴                 | the fp32 unit roundoff                                  |
## | u_bf     | 2⁻⁸                  | the bf16 unit roundoff, one RNE round's relative bound  |
## | relRsqrt | 2·2⁻²¹               | the MSL approximate rsqrt vs correctly-rounded 1/sqrt   |
## | relSilu  | 8·u32                | the exp2-form vs exp-form transcendental class          |
## | 2⁻¹²⁶    | bf16 subnormal floor | the smallest bf16 relative grid step                    |
## | 2⁻²⁵     | fp16 grid floor      | also covering the bf16 output grid, per the landed band |
##
## | fact       | content                                                                                                                                                           |
## | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | operands   | the mega's stage operands are host-readable, so the operand set x' is observed data, not a bound (the arena sections, the fp32 state, the bf16 ring)              |
## | router ids | the router's expert ids are internal to the mega kernel, the tie-region band guaranteeing the ids when every adjacent top-K gap exceeds the pair's combined terms |
## | stream     | the stream is bit-exact by construction (both sides round the same fp32 add), asserted. The measured divergence justifies the model, never sets a bound           |
##
## | binding     | value                                                                                                                                |
## | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
## | shape       | Dk = Dv = 128 static at the mega's GDN call site, checked against the geometry constants, exercised by the (32, 128, 128) state cube |
## | loop bounds | 2 shapes × 16 samples (check), 3 cases (comparison), 8 steps (chain), every host walk a fixed bounded loop, none unbounded           |

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/gdn_moe_decode_megakernel
import ../naive/naive_rng
import ../naive/naive_tensors
import ../naive/naive_gdn
import ../naive/naive_grouped_mm
import ../naive/naive_layer_ops
from ../naive/naive_qwen35_layer import naiveQwen35GdnLayer, LayerOut,
    moeDecodeBody
import ceramic_pagebuf
import mega_bounded_wait
import ceramic_fam

const TTT_DEBUG_OROW* {.booldefine.} = false
const TTT_DEBUG_TIE* {.booldefine.} = false

# ─── Band-model constants ─────────────────────────────────────────────

const
  UBf* = 3.90625e-3'f64                   # 2⁻⁸, the bf16 unit roundoff
  RelRsqrt* = 9.5367431640625e-7'f64      # 2·2⁻²¹, the approximate rsqrt class
  RelSilu* = 4.76837158203125e-7'f64      # 8·u32, the exp2-form transcendental class
  # silu-derivative bound for the 1.1 factor below:
  #   |silu'(x)| <= 1.0851, peak at x ~ 1.28 (tails e^-x / sigmoid),
  # so the factor dominates the true maximum.
  FloorBf* = 7.346879709099078e-39'f64    # 2⁻¹²⁶, the bf16 subnormal grid floor
  FloorSub* = 2.9802322387695312e-8'f64   # 2⁻²⁵, the fp16 grid floor, also the bf16 output grid

func widen*(s: seq[uint16]): seq[float64] =
  ## Exact widenings of bf16 bit patterns, the band walk's magnitudes.
  result = newSeq[float64](s.len)
  for i in 0 ..< s.len:
    result[i] = bf16ToF32(s[i]).float64

func silu64*(x: float64): float64 =
  ## silu in fp64, the h-link band's magnitude reference.
  x / (1.0 + exp(-x))

# ─── Device entry ─────────────────────────────────────────────────────

proc normRowCheck[H, Lanes, Span: static int](
    x, y, normW, stream, outp: ptr UncheckedArray[bfloat16], eps: float32) {.device.} =
  ## Mega norm stage's fused spelling, the lane geometry parametrized.
  ##
  ##   serial per-lane fp32 sum of squares → butterfly reduction →
  ##   broadcast → approximate rsqrt → rstd-first multiply → one bf16
  ##   round at the store
  ##
  ## At (H 2048, Lanes 32, Span 64) this is the mega `normRow`'s exact arithmetic order.
  let lane = int32(thread_index_in_threadgroup)
  let base = lane * int32(Span)
  var acc = 0.0'f32
  for e in base ..< base + int32(Span):
    let s = (x[e].float32 + y[e].float32).bfloat16
    stream[e] = s
    acc += s.float32 * s.float32
  var total = acc
  var off = Lanes div 2
  while off > 0:
    total += simdShuffleDown(total, uint32(off))
    off = off div 2
  total = simdShuffle(total, 0'u32)
  let rstd = rsqrt(total / float32(H) + eps)
  for e in base ..< base + int32(Span):
    outp[e] = (stream[e].float32 * rstd * (normW[e].float32 + 1.0'f32)).bfloat16

const CompositionMsl* = metal:
  proc qwen35_gdn_layer_bf16(
      counters: ptr UncheckedArray[uint32],
      bfA: ptr UncheckedArray[bfloat16],
      f32A: ptr UncheckedArray[float32],
      xPrev, rPrev: ptr UncheckedArray[bfloat16],
      state: ptr UncheckedArray[float32],
      ring: ptr UncheckedArray[bfloat16],
      norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W,
      routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW:
        ptr UncheckedArray[bfloat16],
      aLog: ptr UncheckedArray[float32],
      dtBias: ptr UncheckedArray[bfloat16],
      eps: float32) {.global.} =
    gdnMoeLayerWalk[bfloat16, true](counters, bfA, f32A, xPrev, rPrev, state, ring,
      norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W,
      routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW,
      aLog, dtBias, eps)

  proc norm_check_2048(
      x, y, normW, stream, outp: ptr UncheckedArray[bfloat16], eps: float32) {.global.} =
    normRowCheck[2048, 32, 64](x, y, normW, stream, outp, eps)

  proc norm_check_512(
      x, y, normW, stream, outp: ptr UncheckedArray[bfloat16], eps: float32) {.global.} =
    normRowCheck[512, 8, 64](x, y, normW, stream, outp, eps)

# ─── Host, the seeded inputs ──────────────────────────────────────────

const H* = Hidden
  ## Layer width, the naive composition's spelling of the mega's Hidden.
const Seed* = 0xC04D0601'u64
const Eps* = 1.0e-6'f32
const NumCounters* = 13
const NumSteps* = 8
const NumCases* = 3
const CaseSeedStep* = 0x9E3779B9'u64
const ChainSeed* = 0x5EEDC0DE'u64

type Weights* = object
  ## Seeded weights and recurrence constants, bf16 bit patterns shared
  ## by the mega and the naive sides through their exact fp32 widenings.
  norm1W*, qkvW*, zW*, aW*, bW*, convW*, onormW*, outprojW*, norm2W*: seq[uint16]
  routerW*, gateUpW*, downW*, sharedGW*, sharedUW*, sharedDW*, sharedGVW*: seq[uint16]
  aLog*: seq[float32]
  dtBias*: seq[uint16]

type Token* = object
  ## One token's inputs:
  ##   the residual-stream addends.
  x*, r*: seq[uint16]

type Carry* = object
  ## One walk's carried state and ring, fp32 and bf16 bit patterns.
  state*: seq[float32]
  ring*: seq[uint16]

proc randBits*(rng: var NaiveRng; n: int; lo, hi: float32): seq[uint16] =
  ## `n` bf16 bit patterns of uniform samples in [lo, hi].
  result = newSeq[uint16](n)
  for i in 0 ..< n:
    result[i] = f32ToBf16(rng.nextF32(lo, hi))

proc buildWeights*(rng: var NaiveRng): Weights =
  ## Seeded weights at the Qwen bf16 class geometry, the same generation
  ## recipe as the naive composition suite and the mega smoke, modest
  ## magnitudes so no stage saturates.
  result.norm1W = randBits(rng, Hidden, -0.05'f32, 0.05'f32)
  result.qkvW = randBits(rng, ConvDim * Hidden, -0.02'f32, 0.02'f32)
  result.zW = randBits(rng, NumVHeads * HeadVDim * Hidden, -0.02'f32, 0.02'f32)
  result.aW = randBits(rng, NumVHeads * Hidden, -0.02'f32, 0.02'f32)
  result.bW = randBits(rng, NumVHeads * Hidden, -0.02'f32, 0.02'f32)
  result.convW = randBits(rng, ConvDim * ConvKernel, -0.1'f32, 0.1'f32)
  result.onormW = randBits(rng, NumVHeads * HeadVDim, -0.05'f32, 0.05'f32)
  result.outprojW = randBits(rng, Hidden * NumVHeads * HeadVDim, -0.02'f32, 0.02'f32)
  result.norm2W = randBits(rng, Hidden, -0.05'f32, 0.05'f32)
  result.routerW = randBits(rng, NumExperts * Hidden, -0.02'f32, 0.02'f32)
  result.gateUpW = randBits(rng, NumExperts * 2 * Inter * Hidden, -0.02'f32, 0.02'f32)
  result.downW = randBits(rng, NumExperts * Hidden * Inter, -0.02'f32, 0.02'f32)
  result.sharedGW = randBits(rng, Inter * Hidden, -0.02'f32, 0.02'f32)
  result.sharedUW = randBits(rng, Inter * Hidden, -0.02'f32, 0.02'f32)
  result.sharedDW = randBits(rng, Hidden * Inter, -0.02'f32, 0.02'f32)
  result.sharedGVW = randBits(rng, Hidden, -0.02'f32, 0.02'f32)
  result.aLog = newSeq[float32](NumVHeads)
  for h in 0 ..< NumVHeads:
    result.aLog[h] = rng.nextF32(-2.0'f32, -0.1'f32)
  result.dtBias = randBits(rng, NumVHeads, -0.1'f32, 0.1'f32)

proc newToken*(rng: var NaiveRng): Token =
  ## One token's residual addends over the fixed weights.
  result.x = randBits(rng, Hidden, -1.0'f32, 1.0'f32)
  result.r = randBits(rng, Hidden, -1.0'f32, 1.0'f32)

proc newCarry*(rng: var NaiveRng): Carry =
  ## One walk's initial state and ring.
  result.state = newSeq[float32](NumVHeads * HeadVDim * HeadKDim)
  for i in 0 ..< result.state.len:
    result.state[i] = rng.nextF32(-0.5'f32, 0.5'f32)
  result.ring = randBits(rng, ConvDim * RingWidth, -1.0'f32, 1.0'f32)

proc fillBf*(buf: var PageBuf[uint16], src: seq[uint16]) =
  ## Copies bf16 bit patterns into a page buffer.
  doAssert buf.elems * sizeof(uint16) mod HostPageSize == 0,
    "no-copy binding needs a page-multiple byte length"
  for i in 0 ..< src.len:
    buf.hostPtr[i] = src[i]

proc fillF32*(buf: var PageBuf[float32], src: seq[float32]) =
  ## Copies fp32 values into a page buffer.
  doAssert buf.elems * sizeof(float32) mod HostPageSize == 0,
    "no-copy binding needs a page-multiple byte length"
  for i in 0 ..< src.len:
    buf.hostPtr[i] = src[i]

proc naiveWalk*(w: Weights; tok: Token; carryIn: Carry):
    tuple[lo: LayerOut, carry: Carry] =
  ## One naive walk over copies of the carried state and ring, the walk
  ## mutating the copies, the post-walk carry returned with the outputs.
  var stateN = NaiveCube[float32](planes: NumVHeads, rows: HeadVDim,
    cols: HeadKDim)
  stateN.data = carryIn.state
  var ringN = carryIn.ring
  result.lo = naiveQwen35GdnLayer(stateN, ringN, tok.x, tok.r,
    w.norm1W, w.qkvW, w.zW, w.aW, w.bW, w.convW, w.onormW,
    w.outprojW, w.norm2W, w.routerW, w.gateUpW, w.downW,
    w.sharedGW, w.sharedUW, w.sharedDW, w.sharedGVW, w.aLog,
    w.dtBias, Eps)
  result.carry.state = stateN.data
  result.carry.ring = ringN

# ─── Host replays of the naive spellings ──────────────────────────────

proc logits32Of*(x: seq[uint16]; routerW: seq[uint16]): seq[float64] =
  ## Router's bf16-rounded logits, the naive spelling replayed.
  ##
  ##   fp32 serial dot, one bf16 round
  result = newSeq[float64](NumExperts)
  for e in 0 ..< NumExperts:
    var acc = 0.0'f32
    for k in 0 ..< H:
      acc += bf16ToF32(x[k]) * bf16ToF32(routerW[e * H + k])
    result[e] = bf16ToF32(f32ToBf16(acc)).float64

proc topKSlots*(logits: seq[float64]): tuple[ids: seq[int], runner: int] =
  ## Top-K by score, the lowest index on a tie, the naive router's order.
  var used = newSeq[bool](NumExperts)
  result.ids = newSeq[int](TopK)
  for s in 0 ..< TopK:
    var best = -1
    for e in 0 ..< NumExperts:
      if not used[e] and (best < 0 or logits[e] > logits[best]):
        best = e
    used[best] = true
    result.ids[s] = best
  result.runner = -1
  for e in 0 ..< NumExperts:
    if not used[e] and (result.runner < 0 or logits[e] > logits[result.runner]):
      result.runner = e

proc l2AccOf*(row: seq[uint16]): float32 =
  ## L2 row's fp32 serial sum of bf16-rounded squares, the naive spelling replayed.
  for c in 0 ..< row.len:
    let xi = bf16ToF32(row[c])
    result += bf16ToF32(f32ToBf16(xi * xi))

proc l2InvOf*(row: seq[uint16]): float64 =
  ## L2 row's bf16-rounded reciprocal, the recorded chain's spelling replayed.
  ##
  ##   fp32 sum rounds to bf16 first (the model's `.sum` output dtype)
  ##   → the eps add rounds again
  let acc = l2AccOf(row)
  result = bf16ToF32(f32ToBf16(1.0'f32 / sqrt(bf16ToF32(
    f32ToBf16(bf16ToF32(f32ToBf16(acc)) + 1.0e-6'f32))))).float64

proc normRelRstd*(accAbs: float64; n, span, lanes: int; eps: float32): float64 =
  ## One bias-one norm's relative rstd deviation class.
  ##
  ##   serial side ─── (n − 1)-term fp32 sum
  ##   mega side   ─── (span − 1)-term lane sums + butterfly reduction
  ##
  ## Each side within its own order count of the exact sum, through the variance division
  ## and the approximate rsqrt.
  let reassoc = float64(n - 1 + span - 1 + log2(lanes.float64).int + 1)
  result = 0.5 * reassoc * U32 * accAbs /
    (accAbs / float64(n) + eps.float64) + RelRsqrt

proc rmsAccOf*(row: seq[uint16]): float64 =
  ## Sum of squares over a widened bf16 row, fp64.
  let w = widen(row)
  for e in 0 ..< row.len:
    result += w[e] * w[e]

proc widenF32*(s: seq[float32]): seq[float64] =
  ## Exact fp64 widening of an fp32 section, the band walk's magnitudes.
  result = newSeq[float64](s.len)
  for i in 0 ..< s.len:
    result[i] = s[i].float64

proc readBf*(snap: seq[uint16]; off, len: int): seq[uint16] =
  ## One bf16 arena section's snapshot.
  doAssert off + len <= snap.len
  result = snap[off ..< off + len]

proc readF32Sec*(snap: seq[float32]; off, len: int): seq[float32] =
  ## One fp32 arena section's snapshot.
  doAssert off + len <= snap.len
  result = snap[off ..< off + len]

type Snap* = object
  ## One launch's arena and carry snapshots, observed operand values:
  ##
  ##   full bf16 arena, full fp32 arena, post-launch state and ring
  bf*: seq[uint16]
  f32*: seq[float32]
  state*: seq[float32]
  ring*: seq[uint16]

proc secBf*(s: Snap; off, len: int): seq[uint16] {.inline.} =
  ## One bf16 section of the snapshot.
  readBf(s.bf, off, len)

proc secF32*(s: Snap; off, len: int): seq[float32] {.inline.} =
  ## One fp32 section of the snapshot.
  readF32Sec(s.f32, off, len)

proc toF32Mat*(bits: seq[uint16]; rows, cols: int): NaiveMat[float32] =
  ## A widened fp32 matrix over bf16 bits, the naive step's operand form.
  doAssert bits.len == rows * cols
  result = NaiveMat[float32](rows: rows, cols: cols)
  result.data = newSeq[float32](rows * cols)
  for i in 0 ..< bits.len:
    result.data[i] = bf16ToF32(bits[i])

proc toF32Vec*(bits: seq[uint16]; n: int): seq[float32] =
  ## A widened fp32 vector over bf16 bits.
  doAssert bits.len == n
  result = newSeq[float32](n)
  for i in 0 ..< n:
    result[i] = bf16ToF32(bits[i])

proc compositionInit*(): HwEngine =
  ## Device setup for every composition-segment test:
  ## - the geometry asserts, the Metal engine, the composition MSL ingested
  ## - each test file calls the proc once, then its segment's runner
  doAssert HeadKDim == 128 and HeadVDim == 128,
    "the composition's GDN binding is Dk = Dv = 128"
  doAssert TopK == 8 and NumExperts == 256 and Inter == 512,
    "the composition's MoE geometry is the Qwen decode class"
  echo "device: ", bkMetal.init().deviceName()
  result = bkMetal.init()
  result.ingest(CompositionMsl)
