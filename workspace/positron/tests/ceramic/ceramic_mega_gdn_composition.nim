# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_compare.nim
## Composition tier for the `qwen35_moe` fused GDN decoder layer, the shared
## driver behind the three t_-prefixed segment tests. The one-launch mega kernel
## runs against the composed naive 13-stage reference, every arena section judged per element.
##
## Segments, split so each fits the umbrella's per-test cap:
##   `norm_probe.nim` → `compare.nim` → `chain.nim`
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
import ../../src/mega_kernels/decode_layers/qwen35_moe/qwen35_moe_decode_gdn_bf16
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

const debugOrow {.booldefine.} = false
const debugTie {.booldefine.} = false

# ─── Band-model constants ─────────────────────────────────────────────

const
  UBf = 3.90625e-3'f64                   # 2⁻⁸, the bf16 unit roundoff
  RelRsqrt = 9.5367431640625e-7'f64      # 2·2⁻²¹, the approximate rsqrt class
  RelSilu = 4.76837158203125e-7'f64      # 8·u32, the exp2-form transcendental class
  # silu-derivative bound for the 1.1 factor below:
  #   |silu'(x)| <= 1.0851, peak at x ~ 1.28 (tails e^-x / sigmoid),
  # so the factor dominates the true maximum.
  FloorBf = 7.346879709099078e-39'f64    # 2⁻¹²⁶, the bf16 subnormal grid floor
  FloorSub = 2.9802322387695312e-8'f64   # 2⁻²⁵, the fp16 grid floor, also the bf16 output grid

func widen(s: seq[uint16]): seq[float64] =
  ## Exact widenings of bf16 bit patterns, the band walk's magnitudes.
  result = newSeq[float64](s.len)
  for i in 0 ..< s.len:
    result[i] = bf16ToF32(s[i]).float64

func silu64(x: float64): float64 =
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
    qwen35GdnLayerWalk[true](counters, bfA, f32A, xPrev, rPrev, state, ring,
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

const H = Hidden
  ## Layer width, the naive composition's spelling of the mega's Hidden.
const Seed = 0xC04D0601'u64
const Eps = 1.0e-6'f32
const NumCounters = 13
const NumSteps = 8
const NumCases = 3
const CaseSeedStep = 0x9E3779B9'u64
const ChainSeed = 0x5EEDC0DE'u64

type Weights = object
  ## Seeded weights and recurrence constants, bf16 bit patterns shared
  ## by the mega and the naive sides through their exact fp32 widenings.
  norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W: seq[uint16]
  routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW: seq[uint16]
  aLog: seq[float32]
  dtBias: seq[uint16]

type Token = object
  ## One token's inputs:
  ##   the residual-stream addends.
  x, r: seq[uint16]

type Carry = object
  ## One walk's carried state and ring, fp32 and bf16 bit patterns.
  state: seq[float32]
  ring: seq[uint16]

proc randBits(rng: var NaiveRng; n: int; lo, hi: float32): seq[uint16] =
  ## `n` bf16 bit patterns of uniform samples in [lo, hi].
  result = newSeq[uint16](n)
  for i in 0 ..< n:
    result[i] = f32ToBf16(rng.nextF32(lo, hi))

proc buildWeights(rng: var NaiveRng): Weights =
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

proc newToken(rng: var NaiveRng): Token =
  ## One token's residual addends over the fixed weights.
  result.x = randBits(rng, Hidden, -1.0'f32, 1.0'f32)
  result.r = randBits(rng, Hidden, -1.0'f32, 1.0'f32)

proc newCarry(rng: var NaiveRng): Carry =
  ## One walk's initial state and ring.
  result.state = newSeq[float32](NumVHeads * HeadVDim * HeadKDim)
  for i in 0 ..< result.state.len:
    result.state[i] = rng.nextF32(-0.5'f32, 0.5'f32)
  result.ring = randBits(rng, ConvDim * RingWidth, -1.0'f32, 1.0'f32)

proc fillBf(buf: var PageBuf[uint16], src: seq[uint16]) =
  ## Copies bf16 bit patterns into a page buffer.
  doAssert buf.elems * sizeof(uint16) mod HostPageSize == 0,
    "no-copy binding needs a page-multiple byte length"
  for i in 0 ..< src.len:
    buf.hostPtr[i] = src[i]

proc fillF32(buf: var PageBuf[float32], src: seq[float32]) =
  ## Copies fp32 values into a page buffer.
  doAssert buf.elems * sizeof(float32) mod HostPageSize == 0,
    "no-copy binding needs a page-multiple byte length"
  for i in 0 ..< src.len:
    buf.hostPtr[i] = src[i]

proc naiveWalk(w: Weights; tok: Token; carryIn: Carry):
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

proc logits32Of(x: seq[uint16]; routerW: seq[uint16]): seq[float64] =
  ## Router's bf16-rounded logits, the naive spelling replayed.
  ##
  ##   fp32 serial dot, one bf16 round
  result = newSeq[float64](NumExperts)
  for e in 0 ..< NumExperts:
    var acc = 0.0'f32
    for k in 0 ..< H:
      acc += bf16ToF32(x[k]) * bf16ToF32(routerW[e * H + k])
    result[e] = bf16ToF32(f32ToBf16(acc)).float64

proc topKSlots(logits: seq[float64]): tuple[ids: seq[int], runner: int] =
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
  for e in 0 ..< NumExperts:
    if not used[e] and (result.runner < 0 or logits[e] > logits[result.runner]):
      result.runner = e

proc l2AccOf(row: seq[uint16]): float32 =
  ## L2 row's fp32 serial sum of bf16-rounded squares, the naive spelling replayed.
  for c in 0 ..< row.len:
    let xi = bf16ToF32(row[c])
    result += bf16ToF32(f32ToBf16(xi * xi))

proc l2InvOf(row: seq[uint16]): float64 =
  ## L2 row's bf16-rounded reciprocal, the recorded chain's spelling replayed.
  ##
  ##   fp32 sum rounds to bf16 first (the model's `.sum` output dtype)
  ##   → the eps add rounds again
  let acc = l2AccOf(row)
  result = bf16ToF32(f32ToBf16(1.0'f32 / sqrt(bf16ToF32(
    f32ToBf16(bf16ToF32(f32ToBf16(acc)) + 1.0e-6'f32))))).float64

proc normRelRstd(accAbs: float64; n, span, lanes: int; eps: float32): float64 =
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

proc rmsAccOf(row: seq[uint16]): float64 =
  ## Sum of squares over a widened bf16 row, fp64.
  let w = widen(row)
  for e in 0 ..< row.len:
    result += w[e] * w[e]

proc widenF32(s: seq[float32]): seq[float64] =
  ## Exact fp64 widening of an fp32 section, the band walk's magnitudes.
  result = newSeq[float64](s.len)
  for i in 0 ..< s.len:
    result[i] = s[i].float64

proc readBf(snap: seq[uint16]; off, len: int): seq[uint16] =
  ## One bf16 arena section's snapshot.
  doAssert off + len <= snap.len
  result = snap[off ..< off + len]

proc readF32Sec(snap: seq[float32]; off, len: int): seq[float32] =
  ## One fp32 arena section's snapshot.
  doAssert off + len <= snap.len
  result = snap[off ..< off + len]

type Snap = object
  ## One launch's arena and carry snapshots, observed operand values:
  ##
  ##   full bf16 arena, full fp32 arena, post-launch state and ring
  bf: seq[uint16]
  f32: seq[float32]
  state: seq[float32]
  ring: seq[uint16]

proc secBf(s: Snap; off, len: int): seq[uint16] {.inline.} =
  ## One bf16 section of the snapshot.
  readBf(s.bf, off, len)

proc secF32(s: Snap; off, len: int): seq[float32] {.inline.} =
  ## One fp32 section of the snapshot.
  readF32Sec(s.f32, off, len)

proc toF32Mat(bits: seq[uint16]; rows, cols: int): NaiveMat[float32] =
  ## A widened fp32 matrix over bf16 bits, the naive step's operand form.
  doAssert bits.len == rows * cols
  result = NaiveMat[float32](rows: rows, cols: cols)
  result.data = newSeq[float32](rows * cols)
  for i in 0 ..< bits.len:
    result.data[i] = bf16ToF32(bits[i])

proc toF32Vec(bits: seq[uint16]; n: int): seq[float32] =
  ## A widened fp32 vector over bf16 bits.
  doAssert bits.len == n
  result = newSeq[float32](n)
  for i in 0 ..< n:
    result[i] = bf16ToF32(bits[i])

# ─── The judge ────────────────────────────────────────────────────────

const FieldNames = [
  "stream", "norm1", "qkv", "z", "a", "b", "conv", "qn", "kn", "g",
  "beta", "y", "normed", "blockOut", "h1", "normed2", "moeOut",
  "partial", "state", "ring"]
const NumFields = FieldNames.len

type Usage = object
  ## Worst per-element bar usage and bit-exact counts per field.
  worst: array[NumFields, float64]
  exact: array[NumFields, int]
  total: array[NumFields, int]

proc judgeBf(field: int; megaBits: seq[uint16]; naive: seq[uint16];
    bar: seq[float64]; u: var Usage) =
  ## One bf16 field's per-element judgment, widened exactly on both sides.
  doAssert megaBits.len == naive.len and bar.len == naive.len
  for i in 0 ..< naive.len:
    let got = bf16ToF32(megaBits[i]).float64
    let want = bf16ToF32(naive[i]).float64
    let diff = abs(got - want)
    doAssert diff <= bar[i],
      &"{FieldNames[field]} outside the bar at {i}: {diff:.3e} > {bar[i]:.3e}"
    u.worst[field] = max(u.worst[field], diff / bar[i])
    if megaBits[i] == naive[i]:
      inc u.exact[field]
    inc u.total[field]

proc judgeF32(field: int; mega: seq[float32]; naive: seq[float32];
    bar: seq[float64]; u: var Usage) =
  ## One fp32 field's per-element judgment.
  doAssert mega.len == naive.len and bar.len == naive.len
  for i in 0 ..< naive.len:
    let diff = abs(mega[i].float64 - naive[i].float64)
    doAssert diff <= bar[i],
      &"{FieldNames[field]} outside the bar at {i}: {diff:.3e} > {bar[i]:.3e}"
    u.worst[field] = max(u.worst[field], diff / bar[i])
    if mega[i] == naive[i]:
      inc u.exact[field]
    inc u.total[field]

proc judgeExactBf(field: int; megaBits: seq[uint16]; naive: seq[uint16];
    u: var Usage) =
  ## One bit-exact field's judgment, bar 0:
  ##   both sides round the same
  ## fp32 value, any bit difference is a model violation.
  doAssert megaBits.len == naive.len
  for i in 0 ..< naive.len:
    doAssert megaBits[i] == naive[i],
      &"{FieldNames[field]} not bit-exact at {i}"
    inc u.exact[field]
    inc u.total[field]

proc printUsage(u: Usage; label: string) =
  ## Per-field usage table, worst bar usage and bit-exact counts.
  echo &"[{label}] worst bar usage per field:"
  for f in 0 ..< NumFields:
    if u.total[f] > 0:
      echo &"  {FieldNames[f]:<9} worst {u.worst[f]:.3f} " &
        &"bit-exact {u.exact[f]}/{u.total[f]}"

# ─── Case plumbing ────────────────────────────────────────────────────

type MegaBuffers = object
  ## One case's page buffers and launch pointers, allocated once.
  counters: PageBuf[uint32]
  bfA: PageBuf[uint16]
  f32A: PageBuf[float32]
  xPrev, rPrev: PageBuf[uint16]
  state: PageBuf[float32]
  ring: PageBuf[uint16]
  norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W: PageBuf[uint16]
  routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW:
    PageBuf[uint16]
  aLog: PageBuf[float32]
  dtBias: PageBuf[uint16]

proc allocMega(): MegaBuffers =
  ## Full buffer set, the byte lengths page-multiple asserted in the fills.
  result.counters = allocPageBuf[uint32](NumCounters)
  result.bfA = allocPageBuf[uint16](BfArenaLen)
  result.f32A = allocPageBuf[float32](F32ArenaLen)
  result.xPrev = allocPageBuf[uint16](Hidden)
  result.rPrev = allocPageBuf[uint16](Hidden)
  result.state = allocPageBuf[float32](NumVHeads * HeadVDim * HeadKDim)
  result.ring = allocPageBuf[uint16](ConvDim * RingWidth)
  result.norm1W = allocPageBuf[uint16](Hidden)
  result.qkvW = allocPageBuf[uint16](ConvDim * Hidden)
  result.zW = allocPageBuf[uint16](NumVHeads * HeadVDim * Hidden)
  result.aW = allocPageBuf[uint16](NumVHeads * Hidden)
  result.bW = allocPageBuf[uint16](NumVHeads * Hidden)
  result.convW = allocPageBuf[uint16](ConvDim * ConvKernel)
  result.onormW = allocPageBuf[uint16](NumVHeads * HeadVDim)
  result.outprojW = allocPageBuf[uint16](Hidden * NumVHeads * HeadVDim)
  result.norm2W = allocPageBuf[uint16](Hidden)
  result.routerW = allocPageBuf[uint16](NumExperts * Hidden)
  result.gateUpW = allocPageBuf[uint16](NumExperts * 2 * Inter * Hidden)
  result.downW = allocPageBuf[uint16](NumExperts * Hidden * Inter)
  result.sharedGW = allocPageBuf[uint16](Inter * Hidden)
  result.sharedUW = allocPageBuf[uint16](Inter * Hidden)
  result.sharedDW = allocPageBuf[uint16](Hidden * Inter)
  result.sharedGVW = allocPageBuf[uint16](Hidden)
  result.aLog = allocPageBuf[float32](NumVHeads)
  result.dtBias = allocPageBuf[uint16](NumVHeads)

proc freeMega(m: var MegaBuffers) =
  ## Buffer set's release, the defer call.
  freePageBuf(m.counters)
  freePageBuf(m.bfA)
  freePageBuf(m.f32A)
  freePageBuf(m.xPrev)
  freePageBuf(m.rPrev)
  freePageBuf(m.state)
  freePageBuf(m.ring)
  freePageBuf(m.norm1W)
  freePageBuf(m.qkvW)
  freePageBuf(m.zW)
  freePageBuf(m.aW)
  freePageBuf(m.bW)
  freePageBuf(m.convW)
  freePageBuf(m.onormW)
  freePageBuf(m.outprojW)
  freePageBuf(m.norm2W)
  freePageBuf(m.routerW)
  freePageBuf(m.gateUpW)
  freePageBuf(m.downW)
  freePageBuf(m.sharedGW)
  freePageBuf(m.sharedUW)
  freePageBuf(m.sharedDW)
  freePageBuf(m.sharedGVW)
  freePageBuf(m.aLog)
  freePageBuf(m.dtBias)

proc fillWeights(m: var MegaBuffers; w: Weights) =
  ## Weights and recurrence constants, written once.
  fillBf(m.norm1W, w.norm1W)
  fillBf(m.qkvW, w.qkvW)
  fillBf(m.zW, w.zW)
  fillBf(m.aW, w.aW)
  fillBf(m.bW, w.bW)
  fillBf(m.convW, w.convW)
  fillBf(m.onormW, w.onormW)
  fillBf(m.outprojW, w.outprojW)
  fillBf(m.norm2W, w.norm2W)
  fillBf(m.routerW, w.routerW)
  fillBf(m.gateUpW, w.gateUpW)
  fillBf(m.downW, w.downW)
  fillBf(m.sharedGW, w.sharedGW)
  fillBf(m.sharedUW, w.sharedUW)
  fillBf(m.sharedDW, w.sharedDW)
  fillBf(m.sharedGVW, w.sharedGVW)
  fillBf(m.dtBias, w.dtBias)
  fillF32(m.aLog, w.aLog)

proc launchMega(engine: HwEngine; m: var MegaBuffers) =
  ## One launch, no host-side counter zeroing. The kernel re-zeroes
  ## counters at the launch's end and the page allocator's zero fill covers
  ## the first launch, so the post-launch counters read zero exactly.
  ##
  ## - the per-stage threadgroup totals stay statically recorded in `WaveCounts`
  var cPA = m.counters.pa()
  var bfAPA = m.bfA.pa()
  var f32APA = m.f32A.pa()
  var xPA = m.xPrev.pa()
  var rPA = m.rPrev.pa()
  var stPA = m.state.pa()
  var rgPA = m.ring.pa()
  var n1PA = m.norm1W.pa()
  var qkvPA = m.qkvW.pa()
  var zPA = m.zW.pa()
  var aPA = m.aW.pa()
  var bPA = m.bW.pa()
  var cvPA = m.convW.pa()
  var onPA = m.onormW.pa()
  var opPA = m.outprojW.pa()
  var n2PA = m.norm2W.pa()
  var rtPA = m.routerW.pa()
  var guPA = m.gateUpW.pa()
  var dnPA = m.downW.pa()
  var sgPA = m.sharedGW.pa()
  var suPA = m.sharedUW.pa()
  var sdPA = m.sharedDW.pa()
  var gvPA = m.sharedGVW.pa()
  var alPA = m.aLog.pa()
  var dbPA = m.dtBias.pa()
  proc dispatch(): bool {.gcsafe.} =
    engine.run << (grid: (950, 1, 1), blk: (32, 1, 1)) >>
      ("qwen35_gdn_layer_bf16", cPA,
        (bfAPA, f32APA, xPA, rPA, stPA, rgPA, n1PA, qkvPA, zPA, aPA,
         bPA, cvPA, onPA, opPA, n2PA, rtPA, guPA, dnPA,
         sgPA, suPA, sdPA, gvPA, alPA, dbPA, Eps))
    result = true
  runMegaBounded(dispatch, m.counters.hostPtr, StageNames)
  for i in 0 ..< NumCounters:
    doAssert m.counters.hostPtr[i] == 0'u32,
      &"stage counter {i} {m.counters.hostPtr[i]} want 0 (the launch-end reset)"
proc snapMega(m: MegaBuffers): Snap =
  ## One launch's full snapshot.
  result.bf = readInto(m.bfA.hostPtr, BfArenaLen)
  result.f32 = readInto(m.f32A.hostPtr, F32ArenaLen)
  result.state = readInto(m.state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  result.ring = readInto(m.ring.hostPtr, ConvDim * RingWidth)


# ─── Per-stage bands ──────────────────────────────────────────────────

func zipAdd(a, b: seq[float64]): seq[float64] =
  ## Elementwise sum of two bar vectors, the local band plus the sensitivity.
  doAssert a.len == b.len
  result = newSeq[float64](a.len)
  for i in 0 ..< a.len:
    result[i] = a[i] + b[i]

func sensOf(nPrime, n: seq[uint16]): seq[float64] =
  ## Exact sensitivity of a naive bf16 output to the deviation of its observed operands,
  ## widened difference of the two naive evaluations.
  doAssert nPrime.len == n.len
  result = newSeq[float64](n.len)
  let a = widen(nPrime)
  let b = widen(n)
  for i in 0 ..< n.len:
    result[i] = abs(a[i] - b[i])

proc denseLocal(x: seq[uint16]; w: seq[uint16]; nPrime: seq[uint16];
    N, K: int): seq[float64] =
  ## Landed dense-linear band per output element at the mega's operand magnitudes.
  ##
  ##   terms ─── accumulator reassociation, store round, grid floor
  ##
  ## `x` the mega's input bits, `nPrime` the naive reference at them.
  doAssert x.len == K and nPrime.len == N and w.len == N * K
  result = newSeq[float64](N)
  let xw = widen(x)
  for n in 0 ..< N:
    var sumAbs = 0.0'f64
    for k in 0 ..< K:
      sumAbs += abs(xw[k] * bf16ToF32(w[n * K + k]).float64)
    result[n] = 2.0 * float64(K) * U32 * sumAbs +
      2.0 * UBf * abs(bf16ToF32(nPrime[n]).float64) + FloorSub

proc gemvBars(xPrime, w, nPrime, n: seq[uint16]; N, K: int): seq[float64] =
  ## One GEMV field's bars.
  ##
  ##   landed dense band at the mega's operands
  ##   + exact sensitivity of the naive dot to the observed input deviation
  zipAdd(denseLocal(xPrime, w, nPrime, N, K), sensOf(nPrime, n))

proc normLocal(row: seq[uint16]; nPrime: seq[uint16]; span, lanes: int):
    seq[float64] =
  ## One bias-one norm's local band at the mega's observed stream.
  ##
  ##   sum-of-squares reassociation through the variance
  ##   + approximate rsqrt, fp32 product class, store round, grid floor
  let acc = rmsAccOf(row)
  let rel = normRelRstd(acc, row.len, span, lanes, Eps)
  result = newSeq[float64](row.len)
  for e in 0 ..< row.len:
    let nAbs = abs(bf16ToF32(nPrime[e]).float64)
    result[e] = nAbs * (rel + 4.0 * U32 + 2.0 * UBf) + FloorBf

type Bars = object
  ## Per-element bars for the 20 judged fields, each the stage's landed
  ## local band at the mega's observed operands plus the exact sensitivity
  ## of the naive op to the observed operand deviation.
  norm1, qkv, z, a, b, conv, qn, kn, beta: seq[float64]
  g, y, normed, blockOut, h1, normed2, moeOut: seq[float64]
  partial, state, ring: seq[float64]
  partialN: seq[float32]
  ids: seq[int]
  runner: int
  gateClear: bool
  minGap, gapBar: float64

proc walkBars(w: Weights; preM, preN, postN: Carry; lo: LayerOut;
    snap: Snap): Bars =
  ## One walk's per-element bars from the observed operands. Each stage's landed local band
  ## at the mega's section values plus the exact sensitivity of the naive op
  ## to the observed operand deviation.
  ##
  ## Returns with `gateClear` false when the router's tie region is hit.
  ## Caller regenerates the token, the bars discarded.
  let
    streamM = snap.secBf(sStream, H)
    norm1M = snap.secBf(sNorm1, H)
    qkvM = snap.secBf(sQkvCol, ConvDim)
    zM = snap.secBf(sZ, NumVHeads * HeadVDim)
    aM = snap.secBf(sA, NumVHeads)
    bM = snap.secBf(sB, NumVHeads)
    convM = snap.secBf(sConv, ConvDim)
    qnM = snap.secBf(sQN, NumKHeads * HeadKDim)
    knM = snap.secBf(sKN, NumKHeads * HeadKDim)
    gM = snap.secF32(sG, NumVHeads)
    betaM = snap.secBf(sBeta, NumVHeads)
    yM = snap.secBf(sY, NumVHeads * HeadVDim)
    normedM = snap.secBf(sNormed, NumVHeads * HeadVDim)
    blockOutM = snap.secBf(sBlockOut, H)
    h1M = snap.secBf(sH1, H)
    normed2M = snap.secBf(sNormed2, H)
    moeOutM = snap.secBf(sMoeOut, H)
    partialM = snap.secF32(sPartial, (TopK + 1) * H)

  # Stage 1:
  #   the stream rounds the same fp32 add on both sides, asserted bit-exact.
  #   The norm1 bar is the local band at the shared stream, the sensitivity
  #   zero at the identical input
  doAssert streamM == lo.stream, "the residual stream is not bit-exact"
  result.norm1 = normLocal(streamM, lo.norm1, 64, 32)

  # Stages 2 to 4:
  #   the projection GEMVs. The mega's norm1 is the observed input, the naive dot replayed
  #   at both operand sets
  let qkvPrime = naiveDenseLinear(norm1M, w.qkvW, ConvDim, H)
  result.qkv = gemvBars(norm1M, w.qkvW, qkvPrime, lo.qkvCol, ConvDim, H)
  let zPrime = naiveDenseLinear(norm1M, w.zW, NumVHeads * HeadVDim, H)
  result.z = gemvBars(norm1M, w.zW, zPrime, lo.z, NumVHeads * HeadVDim, H)
  let aPrime = naiveDenseLinear(norm1M, w.aW, NumVHeads, H)
  result.a = gemvBars(norm1M, w.aW, aPrime, lo.a, NumVHeads, H)
  let bPrime = naiveDenseLinear(norm1M, w.bW, NumVHeads, H)
  result.b = gemvBars(norm1M, w.bW, bPrime, lo.b, NumVHeads, H)

  # Stage 5:
  #   the conv + silu. The naive step is replayed at the mega's ring and qkv column.
  #   The tap dot is spelling-identical serial arithmetic over exact bf16 products,
  #   the local band the silu class and the stores
  var ringPrime = preM.ring
  let convPrime = naiveCausalConvSiluStep(w.convW, ringPrime, qkvM,
    ConvDim, ConvKernel)
  let convSens = sensOf(convPrime, lo.conv)
  result.conv = newSeq[float64](ConvDim)
  let convPrimeW = widen(convPrime)
  let convMW = widen(convM)
  for c in 0 ..< ConvDim:
    let local = (RelSilu + 4.0 * U32) * max(abs(convPrimeW[c]),
      abs(convMW[c])) + 2.0 * UBf * (abs(convMW[c]) + abs(convPrimeW[c])) +
      FloorBf
    result.conv[c] = local + convSens[c]

  # Ring roll:
  #   the history shifts down one slot, the step column lands in the newest slot, bit copies
  #   on both sides
  result.ring = newSeq[float64](ConvDim * RingWidth)
  let preMW = widen(preM.ring)
  let preNW = widen(preN.ring)
  let qkvMW = widen(qkvM)
  let qkvNW = widen(lo.qkvCol)
  for c in 0 ..< ConvDim:
    for j in 0 ..< RingWidth - 1:
      doAssert snap.ring[c * RingWidth + j] ==
        preM.ring[c * RingWidth + j + 1], "the mega's ring roll is not a copy"
      doAssert postN.ring[c * RingWidth + j] ==
        preN.ring[c * RingWidth + j + 1],
        "the naive's ring roll is not a copy"
      result.ring[c * RingWidth + j] =
        abs(preMW[c * RingWidth + j + 1] - preNW[c * RingWidth + j + 1])
    doAssert snap.ring[c * RingWidth + RingWidth - 1] == qkvM[c],
      "the mega's ring tail is not the step column"
    doAssert postN.ring[c * RingWidth + RingWidth - 1] == lo.qkvCol[c],
      "the naive's ring tail is not the step column"
    result.ring[c * RingWidth + RingWidth - 1] = abs(qkvMW[c] - qkvNW[c])

  # Stage 6:
  #   the q/k l2 normalization. The two sides are spelling-identical arithmetic, asserted
  #   bit-exact at the mega's own conv rows. The bar is the reciprocal chain's sensitivity
  #   to the observed conv deviation
  result.qn = newSeq[float64](NumKHeads * HeadKDim)
  result.kn = newSeq[float64](NumKHeads * HeadKDim)
  let eps64 = Eps.float64
  for half in 0 ..< 2:
    let srcBase = half * (NumKHeads * HeadKDim)
    for h in 0 ..< NumKHeads:
      let rowM = convM[srcBase + h * HeadKDim ..<
        srcBase + (h + 1) * HeadKDim]
      let rowN = lo.conv[srcBase + h * HeadKDim ..<
        srcBase + (h + 1) * HeadKDim]
      let outM = (if half == 0: qnM else: knM)[
        h * HeadKDim ..< (h + 1) * HeadKDim]
      let outN = (if half == 0: lo.qn else: lo.kn)[
        h * HeadKDim ..< (h + 1) * HeadKDim]
      let outPrime = naiveL2NormRow(rowM, HeadKDim)
      doAssert outM == outPrime,
        "the mega's l2norm spelling diverges from the naive replay"
      let invM = l2InvOf(rowM)
      let invN = l2InvOf(rowN)
      let accM = l2AccOf(rowM).float64
      let accN = l2AccOf(rowN).float64
      let rowMW = widen(rowM)
      let rowNW = widen(rowN)
      var dA = 0.0'f64
      for c in 0 ..< HeadKDim:
        let dx = abs(rowMW[c] - rowNW[c])
        dA += 2.0 * abs(rowMW[c]) * dx + dx * dx +
          2.0 * UBf * (rowMW[c] * rowMW[c] + rowNW[c] * rowNW[c])
      let dS = dA + float64(HeadKDim - 1) * U32 * (accM + accN)
      let dE = dS + 2.0 * UBf * (abs(accM + eps64) + abs(accN + eps64))
      let dInv = 0.5 * invM * invN * dE + 4.0 * U32 * invM +
        2.0 * UBf * (invM + invN)
      for c in 0 ..< HeadKDim:
        let i = h * HeadKDim + c
        let bar = abs(rowMW[c] - rowNW[c]) * invN + abs(rowNW[c]) * dInv +
          2.0 * UBf * (abs(bf16ToF32(outM[c]).float64) +
          abs(bf16ToF32(outN[c]).float64)) + FloorBf
        if half == 0:
          result.qn[i] = bar
        else:
          result.kn[i] = bar

  # Stage 7:
  #   the recurrence values, g and beta. The naive spellings are replayed
  #   at the mega's a/b rows. G stays fp32 end to end, beta one bf16 round
  let gatesPrime = naiveGdnGates(aM, bM, w.dtBias, w.aLog, NumVHeads)
  result.g = newSeq[float64](NumVHeads)
  result.beta = newSeq[float64](NumVHeads)
  let betaPrimeW = widen(gatesPrime.beta)
  let betaNW = widen(lo.beta)
  let betaMW = widen(betaM)
  for h in 0 ..< NumVHeads:
    result.g[h] = abs(gatesPrime.g[h].float64 - lo.g[h].float64) +
      (RelSilu + 4.0 * U32) * max(abs(gatesPrime.g[h].float64),
      abs(lo.g[h].float64))
    result.beta[h] = abs(betaPrimeW[h] - betaNW[h]) +
      (RelSilu + 4.0 * U32) * max(abs(betaPrimeW[h]), abs(betaMW[h])) +
      2.0 * UBf * (abs(betaPrimeW[h]) + abs(betaMW[h])) + FloorBf

  # Stage 8:
  #   the GDN step. The naive fp32 step is replayed at both operand sets
  #   for the exact sensitivity. The landed single-step closed form is
  #   the local band at the mega's operands
  let vBase = 2 * NumKHeads * HeadKDim
  let vM = convM[vBase ..< vBase + NumVHeads * HeadVDim]
  let vN = lo.conv[vBase ..< vBase + NumVHeads * HeadVDim]
  var stM32 = NaiveCube[float32](planes: NumVHeads, rows: HeadVDim,
    cols: HeadKDim)
  stM32.data = preM.state
  var yM32 = NaiveMat[float32](rows: NumVHeads, cols: HeadVDim)
  yM32.data = newSeq[float32](NumVHeads * HeadVDim)
  gdnDecodeStep(stM32, yM32, toF32Mat(qnM, NumKHeads, HeadKDim),
    toF32Mat(knM, NumKHeads, HeadKDim), toF32Mat(vM, NumVHeads, HeadVDim),
    toF32Vec(betaM, NumVHeads), gM, NumVHeads, NumKHeads, HkRatio)
  var stN32 = NaiveCube[float32](planes: NumVHeads, rows: HeadVDim,
    cols: HeadKDim)
  stN32.data = preN.state
  var yN32 = NaiveMat[float32](rows: NumVHeads, cols: HeadVDim)
  yN32.data = newSeq[float32](NumVHeads * HeadVDim)
  gdnDecodeStep(stN32, yN32, toF32Mat(lo.qn, NumKHeads, HeadKDim),
    toF32Mat(lo.kn, NumKHeads, HeadKDim),
    toF32Mat(vN, NumVHeads, HeadVDim), toF32Vec(lo.beta, NumVHeads), lo.g,
    NumVHeads, NumKHeads, HkRatio)
  let st64M = widenF32(preM.state)
  let k64M = widen(knM)
  let v64M = widen(vM)
  let beta64M = widen(betaM)
  let g64M = widenF32(gM)
  let invSqrtDk = 1.0 / sqrt(float64(HeadKDim))
  result.state = newSeq[float64](NumVHeads * HeadVDim * HeadKDim)
  result.y = newSeq[float64](NumVHeads * HeadVDim)
  for bh in 0 ..< NumVHeads:
    # the kernel's head mapping, both terms kept live even at batch 1,
    # where the sequence-offset term is zero, so the bar stays correct
    # under GQA generalization
    let kh = ((bh mod NumVHeads) div HkRatio) + ((bh div NumVHeads) * NumKHeads)
    let gamma = exp(g64M[bh])
    for r in 0 ..< HeadVDim:
      let rowBase = (bh * HeadVDim + r) * HeadKDim
      var kv = 0.0'f64
      var kvAbs = 0.0'f64
      for c in 0 ..< HeadKDim:
        let term = gamma * st64M[rowBase + c] * k64M[kh * HeadKDim + c]
        kv += term
        kvAbs += abs(term)
      let delta = beta64M[bh] * (v64M[bh * HeadVDim + r] - kv)
      for c in 0 ..< HeadKDim:
        let a = abs(gamma * st64M[rowBase + c])
        let b = abs(k64M[kh * HeadKDim + c] * delta)
        let kAbs = abs(k64M[kh * HeadKDim + c])
        let bar = 4.0 * U32 * a + kAbs * (beta64M[bh] * 2.0 *
          float64(HeadKDim) * U32 * kvAbs + 2.0 * U32 * beta64M[bh] *
          (abs(v64M[bh * HeadVDim + r]) + abs(kv)) + 2.0 * U32 * abs(delta)) +
          4.0 * U32 * (a + b) +
          abs(stM32.data[rowBase + c].float64 -
          postN.state[rowBase + c].float64)
        result.state[rowBase + c] = bar
      var yAbs = 0.0'f64
      for c in 0 ..< HeadKDim:
        yAbs += abs(stM32.data[rowBase + c].float64 *
          (bf16ToF32(qnM[kh * HeadKDim + c]).float64 * invSqrtDk))
      let yPrimeAbs = abs(yM32.data[bh * HeadVDim + r].float64)
      let yXAbs = abs(yN32.data[bh * HeadVDim + r].float64)
      result.y[bh * HeadVDim + r] =
        2.0 * UBf * yPrimeAbs +
        (2.0 * float64(HeadKDim) * U32 + 2.0 * RelRsqrt) * yAbs +
        2.0 * U32 * yPrimeAbs + FloorSub +
        abs(yM32.data[bh * HeadVDim + r].float64 -
        yN32.data[bh * HeadVDim + r].float64) +
        2.0 * UBf * (yPrimeAbs + yXAbs)

  # Stage 9:
  #   the o_norm epilogue, the landed gated-norm band at the mega's
  # y and z rows, plus the exact sensitivity of the naive row op
  result.normed = newSeq[float64](NumVHeads * HeadVDim)
  for bh in 0 ..< NumVHeads:
    let yRow = yM[bh * HeadVDim ..< (bh + 1) * HeadVDim]
    let zRow = zM[bh * HeadVDim ..< (bh + 1) * HeadVDim]
    let wRow = w.onormW[bh * HeadVDim ..< (bh + 1) * HeadVDim]
    let normedPrime = naiveRmsNormGated(yRow, zRow, wRow, HeadVDim, Eps)
    let normedSens = sensOf(normedPrime,
      lo.normed[bh * HeadVDim ..< (bh + 1) * HeadVDim])
    let yW = widen(yRow)
    var sumSq = 0.0'f64
    var sumSqAbs = 0.0'f64
    for c in 0 ..< HeadVDim:
      sumSq += yW[c] * yW[c]
      sumSqAbs += abs(yW[c] * yW[c])
    let denom = sumSq / float64(HeadVDim) + eps64
    let relRstd = float64(HeadVDim) * U32 * sumSqAbs / denom + RelRsqrt
    let relOut = relRstd + 4.0 * UBf + 2.0 * U32 + RelSilu
    when debugOrow:
      if bh < 2:
        let normedMSec = snap.secBf(sNormed, NumVHeads * HeadVDim)
        var lineY = "[dbg] y  :"
        var lineZ = "[dbg] z  :"
        var lineM = "[dbg] nmM:"
        var lineN = "[dbg] nmN:"
        for cc in 0 ..< HeadVDim:
          lineY.add &" {bf16ToF32(yRow[cc]):.5f}"
          lineZ.add &" {bf16ToF32(zRow[cc]):.5f}"
          lineM.add &" {bf16ToF32(normedMSec[bh * HeadVDim + cc]):.5f}"
          lineN.add &" {bf16ToF32(lo.normed[bh * HeadVDim + cc]):.5f}"
        echo lineY
        echo lineZ
        echo lineM
        echo lineN

    for c in 0 ..< HeadVDim:
      result.normed[bh * HeadVDim + c] =
        abs(bf16ToF32(normedPrime[c]).float64) * relOut + FloorBf +
        normedSens[c]

  # Stage 10:
  #   the out projection GEMV
  let blockPrime = naiveDenseLinear(normedM, w.outprojW, H,
    NumVHeads * HeadVDim)
  result.blockOut = gemvBars(normedM, w.outprojW, blockPrime, lo.blockOut,
    H, NumVHeads * HeadVDim)

  # Stage 11:
  #   the fold and the second norm. The fold is an exact fp32 add of the observed
  #   stream and blockOut, one bf16 round each side, asserted bit-exact against
  #   its replay. The norm reuses the stage-1 band, the sensitivity via the second pass
  let streamMW = widen(streamM)
  let blockOutMW = widen(blockOutM)
  let blockOutNW = widen(lo.blockOut)
  result.h1 = newSeq[float64](H)
  for e in 0 ..< H:
    let expected = f32ToBf16(
      streamMW[e].float32 + blockOutMW[e].float32)
    doAssert h1M[e] == expected, "the mega's fold is not the exact add"
    result.h1[e] = abs(blockOutMW[e] - blockOutNW[e]) +
      2.0 * UBf * (abs(bf16ToF32(h1M[e]).float64) +
      abs(bf16ToF32(lo.h1[e]).float64)) + FloorBf
  let n2Prime = naiveRmsNormRes(streamM, blockOutM, w.norm2W, H, Eps)
  let n2Sens = sensOf(n2Prime.normed, lo.normed2)
  result.normed2 = zipAdd(normLocal(n2Prime.stream, lo.normed2, 64, 32),
    n2Sens)

  # Stage 12:
  #   the router. The host replays the bf16-rounded logits at both operand sets.
  #   When every adjacent top-K gap exceeds the pair's combined terms, the mega's
  #   local band vs the replays' exact sensitivity, all three logit orders coincide
  let logitsX = logits32Of(lo.normed2, w.routerW)
  let slots = topKSlots(logitsX)
  let logitsP = logits32Of(normed2M, w.routerW)
  var localP = newSeq[float64](NumExperts)
  var sensE = newSeq[float64](NumExperts)
  let n2MW = widen(normed2M)
  for e in 0 ..< NumExperts:
    var sumAbs = 0.0'f64
    for k in 0 ..< H:
      sumAbs += abs(n2MW[k] * bf16ToF32(w.routerW[e * H + k]).float64)
    localP[e] = 2.0 * float64(H) * U32 * sumAbs + 2.0 * UBf * abs(logitsP[e])
    sensE[e] = abs(logitsP[e] - logitsX[e])
  result.ids = slots.ids
  result.runner = slots.runner
  result.gateClear = true
  result.minGap = 1.0e300
  result.gapBar = 0.0'f64
  for s in 0 ..< TopK - 1:
    let i = slots.ids[s]
    let j = slots.ids[s + 1]
    let gapP = logitsP[i] - logitsP[j]
    let gapX = logitsX[i] - logitsX[j]
    let pairBar = max(localP[i] + localP[j], sensE[i] + sensE[j])
    result.minGap = min(result.minGap, min(gapP, gapX))
    result.gapBar = max(result.gapBar, pairBar)
    if gapP <= localP[i] + localP[j] or gapX <= sensE[i] + sensE[j]:
      result.gateClear = false
      when debugTie:
        echo &"[tie] slot {s}: gapP {gapP:.3e} vs localP " &
          &"{localP[i]+localP[j]:.3e}, gapX {gapX:.3e} vs sensE " &
          &"{sensE[i]+sensE[j]:.3e}"
  let iR = slots.ids[TopK - 1]
  let jR = slots.runner
  let gapPR = logitsP[iR] - logitsP[jR]
  let gapXR = logitsX[iR] - logitsX[jR]
  let pairBarR = max(localP[iR] + localP[jR], sensE[iR] + sensE[jR])
  result.minGap = min(result.minGap, min(gapPR, gapXR))
  result.gapBar = max(result.gapBar, pairBarR)
  if gapPR <= localP[iR] + localP[jR] or gapXR <= sensE[iR] + sensE[jR]:
    result.gateClear = false
    when debugTie:
      echo &"[tie] runner: gapP {gapPR:.3e} vs localP " &
        &"{localP[iR]+localP[jR]:.3e}, gapX {gapXR:.3e} vs sensE " &
        &"{sensE[iR]+sensE[jR]:.3e}"
  if not result.gateClear:
    return

  # MoE body:
  #   the naive body is replayed at both operand sets, the tie-region check
  #   guaranteeing identical expert id order. Each slot's local band composes
  #   the terms below, the sensitivity the two body evaluations' exact deviation
  #   - projection reassociation, down reassociation, routing-weight band
  #   - silu-mul class over the mega's own h scratch
  let bodyN = moeDecodeBody(lo.normed2, w.routerW, w.gateUpW, w.downW,
    w.sharedGW, w.sharedUW, w.sharedDW, w.sharedGVW)
  for s in 0 ..< TopK:
    doAssert int(bodyN.ids[s]) == slots.ids[s],
      "the naive body's ids diverge from the host replay"
  let bodyP = moeDecodeBody(normed2M, w.routerW, w.gateUpW, w.downW,
    w.sharedGW, w.sharedUW, w.sharedDW, w.sharedGVW)
  for s in 0 ..< TopK:
    doAssert int(bodyP.ids[s]) == int(bodyN.ids[s]),
      "the expert id order diverges inside the tie region"
  result.partialN = bodyN.partial
  result.partial = newSeq[float64]((TopK + 1) * H)
  let hM = snap.secBf(sH, TopK * Inter)
  let hsM = snap.secBf(sHs, Inter)
  let hMW = widen(hM)
  let hsMW = widen(hsM)
  var logitBarLocal = newSeq[float64](NumExperts)
  for e in 0 ..< NumExperts:
    var sumAbs = 0.0'f64
    for k in 0 ..< H:
      sumAbs += abs(n2MW[k] * bf16ToF32(w.routerW[e * H + k]).float64)
    logitBarLocal[e] = 2.0 * float64(H) * U32 * sumAbs +
      2.0 * UBf * abs(logitsP[e]) + FloorBf
  for slot in 0 ..< TopK:
    let id = int(bodyN.ids[slot])
    # the naive slot chain at the mega's normed2, the magnitudes' source
    let guPrime = naiveGroupedMmSums(gmmBf16,
      NaiveMat[uint16](rows: 1, cols: H, data: normed2M),
      NaiveCube[uint16](planes: 1, rows: 2 * Inter, cols: H,
        data: w.gateUpW[(id * 2 * Inter) * H ..< ((id + 1) * 2 * Inter) * H]),
      @[1'i32])
    var hPrime = newSeq[uint16](Inter)
    for i in 0 ..< Inter:
      hPrime[i] = naiveSiluMulEl(guPrime.data[i], guPrime.data[Inter + i])
    let dnPrime = naiveGroupedMmSums(gmmBf16,
      NaiveMat[uint16](rows: 1, cols: Inter, data: hPrime),
      NaiveCube[uint16](planes: 1, rows: H, cols: Inter,
        data: w.downW[id * H * Inter ..< (id + 1) * H * Inter]),
      @[1'i32])
    let hPrimeW = widen(hPrime)
    let dnPrimeW = widenF32(dnPrime.data)
    let guPrimeW = widenF32(guPrime.data)
    # H link's band. The silu class and the two bf16 rounds live at the mega's own h scratch,
    # the two projection outputs' bands flowing through the silu-mul sensitivities
    var hBand = newSeq[float64](Inter)
    for i in 0 ..< Inter:
      var guBandG = 0.0'f64
      var guBandU = 0.0'f64
      for k in 0 ..< H:
        guBandG += abs(n2MW[k] * bf16ToF32(
          w.gateUpW[(id * 2 * Inter) * H + i * H + k]).float64)
        guBandU += abs(n2MW[k] * bf16ToF32(
          w.gateUpW[(id * 2 * Inter) * H + (Inter + i) * H + k]).float64)
      guBandG = 2.0 * float64(H) * U32 * guBandG + 2.0 * U32 *
        abs(guPrimeW[i])
      guBandU = 2.0 * float64(H) * U32 * guBandU + 2.0 * U32 *
        abs(guPrimeW[Inter + i])
      hBand[i] = (RelSilu + 4.0 * U32) *
        max(abs(hMW[slot * Inter + i]), abs(hPrimeW[i])) +
        2.0 * UBf * (abs(hMW[slot * Inter + i]) + abs(hPrimeW[i])) +
        1.1 * abs(guPrimeW[Inter + i]) * guBandG +
        abs(silu64(guPrimeW[i])) * guBandU + FloorBf
    let wPrime = abs(bodyP.w[slot].float64)
    for e in 0 ..< H:
      var downLocal = 0.0'f64
      for i in 0 ..< Inter:
        let dwAbs = abs(bf16ToF32(
          w.downW[id * H * Inter + e * Inter + i]).float64)
        downLocal += max(abs(hMW[slot * Inter + i]), abs(hPrimeW[i])) *
          dwAbs + hBand[i] * dwAbs
      downLocal = 2.0 * float64(Inter) * U32 * downLocal +
        2.0 * U32 * abs(dnPrimeW[e]) + FloorSub
      let wBand = abs(dnPrimeW[e]) * wPrime *
        (2.0 * logitBarLocal[id] + 2.0 * UBf)
      let pPrime = abs(bodyP.partial[slot * H + e].float64)
      result.partial[slot * H + e] = wPrime * downLocal + wBand +
        2.0 * U32 * pPrime + FloorSub +
        abs(bodyP.partial[slot * H + e].float64 -
        bodyN.partial[slot * H + e].float64)
  # Shared expert's slot. The scalar routing weight's band is the sigmoid
  # class over the scalar logit's dense band, the shared chain evaluated
  # at the mega's normed2 with its own hs scratch
  block shared:
    let gvPrime = naiveSharedGate(normed2M, w.sharedGVW, H)
    let sgPrime = naiveGroupedMmSums(gmmBf16,
      NaiveMat[uint16](rows: 1, cols: H, data: normed2M),
      NaiveCube[uint16](planes: 1, rows: Inter, cols: H, data: w.sharedGW),
      @[1'i32])
    let suPrime = naiveGroupedMmSums(gmmBf16,
      NaiveMat[uint16](rows: 1, cols: H, data: normed2M),
      NaiveCube[uint16](planes: 1, rows: Inter, cols: H, data: w.sharedUW),
      @[1'i32])
    var hsPrime = newSeq[uint16](Inter)
    for i in 0 ..< Inter:
      hsPrime[i] = naiveSiluMulEl(sgPrime.data[i], suPrime.data[i])
    let sdPrime = naiveGroupedMmSums(gmmBf16,
      NaiveMat[uint16](rows: 1, cols: Inter, data: hsPrime),
      NaiveCube[uint16](planes: 1, rows: H, cols: Inter, data: w.sharedDW),
      @[1'i32])
    let hsPrimeW = widen(hsPrime)
    let sgPrimeW = widenF32(sgPrime.data)
    let suPrimeW = widenF32(suPrime.data)
    var hsBand = newSeq[float64](Inter)
    for i in 0 ..< Inter:
      var guBandG = 0.0'f64
      var guBandU = 0.0'f64
      for k in 0 ..< H:
        guBandG += abs(n2MW[k] * bf16ToF32(w.sharedGW[i * H + k]).float64)
        guBandU += abs(n2MW[k] * bf16ToF32(w.sharedUW[i * H + k]).float64)
      guBandG = 2.0 * float64(H) * U32 * guBandG + 2.0 * U32 *
        abs(sgPrimeW[i])
      guBandU = 2.0 * float64(H) * U32 * guBandU + 2.0 * U32 *
        abs(suPrimeW[i])
      hsBand[i] = (RelSilu + 4.0 * U32) *
        max(abs(hsMW[i]), abs(hsPrimeW[i])) +
        2.0 * UBf * (abs(hsMW[i]) + abs(hsPrimeW[i])) +
        1.1 * abs(suPrimeW[i]) * guBandG +
        abs(silu64(sgPrimeW[i])) * guBandU + FloorBf
    var sumAbsG = 0.0'f64
    for k in 0 ..< H:
      sumAbsG += abs(n2MW[k] * bf16ToF32(w.sharedGVW[k]).float64)
    let gateBand = 0.25 * (2.0 * float64(H) * U32 * sumAbsG) +
      abs(gvPrime.float64) * (RelSilu + 2.0 * U32 + 2.0 * UBf) + FloorBf
    for e in 0 ..< H:
      var downLocal = 0.0'f64
      for i in 0 ..< Inter:
        let dwAbs = abs(bf16ToF32(w.sharedDW[e * Inter + i]).float64)
        downLocal += max(abs(hsMW[i]), abs(hsPrimeW[i])) * dwAbs +
          hsBand[i] * dwAbs
      downLocal = 2.0 * float64(Inter) * U32 * downLocal +
        2.0 * U32 * abs(sdPrime.data[e].float64) + FloorSub
      let pPrime = abs(bodyP.partial[TopK * H + e].float64)
      result.partial[TopK * H + e] =
        abs(gvPrime.float64) * downLocal + abs(sdPrime.data[e].float64) *
        gateBand + 2.0 * U32 * pPrime + FloorSub +
        abs(bodyP.partial[TopK * H + e].float64 -
        bodyN.partial[TopK * H + e].float64)

  # Stage 13:
  #   the merge. The mega's merge is spelling-identical to the naive slot-order fp32 sum,
  #   asserted bit-exact at the mega's own partials. The bar adds the exact partial
  #   deviations and the two sums' fp32 classes
  doAssert moeOutM == naiveMoeMerge(partialM, TopK, H),
    "the mega's merge spelling diverges from the naive replay"
  let partialMW = widenF32(partialM)
  let partialNW = widenF32(bodyN.partial)
  result.moeOut = newSeq[float64](H)
  for e in 0 ..< H:
    var dPartial = 0.0'f64
    var pSum = 0.0'f64
    for slot in 0 .. TopK:
      dPartial += abs(partialMW[slot * H + e] - partialNW[slot * H + e])
      pSum += max(abs(partialMW[slot * H + e]),
        abs(partialNW[slot * H + e]))
    result.moeOut[e] = dPartial + 16.0 * U32 * pSum +
      2.0 * UBf * (abs(bf16ToF32(moeOutM[e]).float64) +
      abs(bf16ToF32(lo.moeOut[e]).float64)) + FloorBf

proc judgeAll(bars: Bars; snap: Snap; postN: Carry; lo: LayerOut;
    u: var Usage) =
  ## One walk's per-element judgment, the 20 fields against their bars.
  judgeExactBf(0, snap.secBf(sStream, H), lo.stream, u)
  judgeBf(1, snap.secBf(sNorm1, H), lo.norm1, bars.norm1, u)
  judgeBf(2, snap.secBf(sQkvCol, ConvDim), lo.qkvCol, bars.qkv, u)
  judgeBf(3, snap.secBf(sZ, NumVHeads * HeadVDim), lo.z, bars.z, u)
  judgeBf(4, snap.secBf(sA, NumVHeads), lo.a, bars.a, u)
  judgeBf(5, snap.secBf(sB, NumVHeads), lo.b, bars.b, u)
  judgeBf(6, snap.secBf(sConv, ConvDim), lo.conv, bars.conv, u)
  judgeBf(7, snap.secBf(sQN, NumKHeads * HeadKDim), lo.qn, bars.qn, u)
  judgeBf(8, snap.secBf(sKN, NumKHeads * HeadKDim), lo.kn, bars.kn, u)
  judgeF32(9, snap.secF32(sG, NumVHeads), lo.g, bars.g, u)
  judgeBf(10, snap.secBf(sBeta, NumVHeads), lo.beta, bars.beta, u)
  judgeBf(11, snap.secBf(sY, NumVHeads * HeadVDim), lo.y, bars.y, u)
  judgeBf(12, snap.secBf(sNormed, NumVHeads * HeadVDim), lo.normed,
    bars.normed, u)
  judgeBf(13, snap.secBf(sBlockOut, H), lo.blockOut, bars.blockOut, u)
  judgeBf(14, snap.secBf(sH1, H), lo.h1, bars.h1, u)
  judgeBf(15, snap.secBf(sNormed2, H), lo.normed2, bars.normed2, u)
  judgeBf(16, snap.secBf(sMoeOut, H), lo.moeOut, bars.moeOut, u)
  judgeF32(17, snap.secF32(sPartial, (TopK + 1) * H), bars.partialN,
    bars.partial, u)
  judgeF32(18, snap.state, postN.state, bars.state, u)
  judgeBf(19, snap.ring, postN.ring, bars.ring, u)

type Comparison = object
  ## One walk's full comparison result:
  ##   the bars, the naive post-carry,
  ## the snapshot and the naive outputs.
  bars: Bars
  postN: Carry
  snap: Snap
  lo: LayerOut

proc compareWalk(engine: HwEngine; m: var MegaBuffers; w: Weights;
    tok: Token; preM, preN: Carry): Comparison =
  ## One launch plus one naive walk over the same bits, the mega's operand
  ## values snapshotted, the bars walked over the observed operands.
  fillBf(m.xPrev, tok.x)
  fillBf(m.rPrev, tok.r)
  fillF32(m.state, preM.state)
  fillBf(m.ring, preM.ring)
  launchMega(engine, m)
  result.snap = snapMega(m)
  let nw = naiveWalk(w, tok, preN)
  result.postN = nw.carry
  result.lo = nw.lo
  result.bars = walkBars(w, preM, preN, nw.carry, nw.lo, result.snap)

proc runNormCheck*(engine: HwEngine) =
  ## Fused add+norm against the composed naive norm:
  ## - 16 samples per shape over (2048, 32×64) and (512, 8×64)
  ## - judged per element under the norm stage's band
  ## - the mega spelling is a test-local replica, the lane geometry parametrized
  const Samples = 16
  var worstUse = 0.0'f64
  var exact = 0
  var total = 0
  var bitExactStream = 0
  var rng = initNaiveRng(Seed xor 0xB0D'u64)
  for shape in 0 ..< 2:
    let geom = if shape == 0:
      (width: 2048, kernel: "norm_check_2048", lanes: 32, span: 64)
    else:
      (width: 512, kernel: "norm_check_512", lanes: 8, span: 64)
    for s in 0 ..< Samples:
      var xb = allocPageBuf[uint16](geom.width)
      var rb = allocPageBuf[uint16](geom.width)
      var wb = allocPageBuf[uint16](geom.width)
      var sb = allocPageBuf[uint16](geom.width)
      var nb = allocPageBuf[uint16](geom.width)
      defer:
        freePageBuf(xb)
        freePageBuf(rb)
        freePageBuf(wb)
        freePageBuf(sb)
        freePageBuf(nb)
      let x = randBits(rng, geom.width, -1.0'f32, 1.0'f32)
      let r = randBits(rng, geom.width, -1.0'f32, 1.0'f32)
      let wv = randBits(rng, geom.width, -0.05'f32, 0.05'f32)
      fillBf(xb, x)
      fillBf(rb, r)
      fillBf(wb, wv)
      var xPA = xb.pa()
      var rPA = rb.pa()
      var wPA = wb.pa()
      var sPA = sb.pa()
      var nPA = nb.pa()
      engine.run << (grid: (1, 1, 1), blk: (geom.lanes, 1, 1)) >>
        (geom.kernel, xPA, (rPA, wPA, sPA, nPA, Eps))
      let refOut = naiveRmsNormRes(x, r, wv, geom.width, Eps)
      let acc = rmsAccOf(refOut.stream)
      let rel = normRelRstd(acc, geom.width, geom.span, geom.lanes, Eps)
      for e in 0 ..< geom.width:
        doAssert sb.hostPtr[e] == refOut.stream[e], "stream not bit-exact"
        inc bitExactStream
        let want = bf16ToF32(refOut.normed[e]).float64
        let got = bf16ToF32(nb.hostPtr[e]).float64
        let bar = abs(want) * (rel + 4.0 * U32 + 2.0 * UBf) + FloorBf
        let diff = abs(got - want)
        doAssert diff <= bar,
          &"normed outside the bar at (shape {shape}, sample {s}, e {e}): " &
          &"{diff:.3e} > {bar:.3e}"
        worstUse = max(worstUse, diff / bar)
        if nb.hostPtr[e] == refOut.normed[e]:
          inc exact
        inc total
  echo &"[check] fused-vs-separate norm, 32 samples: stream bit-exact " &
    &"{bitExactStream}/{total}, normed bit-exact {exact}/{total}, " &
    &"worst bar usage {worstUse:.3f}"

proc runComparison*(engine: HwEngine) =
  ## 3 seeded cases, one mega launch and one naive walk each, judged per stage under
  ## the observed-operand bars, the usage table printed per field. The router's tie region
  ## clears by regenerating the token over the fixed weights
  var usage = Usage()
  for caseId in 0 ..< NumCases:
    var rng = initNaiveRng(Seed + uint64(caseId) * CaseSeedStep)
    let w = buildWeights(rng)
    let carry0 = newCarry(rng)
    var m = allocMega()
    defer: freeMega(m)
    fillWeights(m, w)
    var tok = newToken(rng)
    var regen = 0
    block found:
      while true:
        let cw = compareWalk(engine, m, w, tok, carry0, carry0)
        if cw.bars.gateClear:
          judgeAll(cw.bars, cw.snap, cw.postN, cw.lo, usage)
          echo &"[comparison] case {caseId}: router gap " &
            &"{cw.bars.minGap:.3e} > tie-region bar " &
            &"{cw.bars.gapBar:.3e}, {regen} samples regenerated"
          break found
        inc regen
        doAssert regen < 4096, "router tie region never clears"
        tok = newToken(rng)
  printUsage(usage, "comparison")
  var worstAll = 0.0'f64
  for f in 0 ..< NumFields:
    worstAll = max(worstAll, usage.worst[f])
  echo &"[comparison] worst usage over all fields {worstAll:.3f}"

proc runChain*(engine: HwEngine) =
  ## 8 decode steps over carried state and ring:
  ## - the mega side launching per step, the naive side walking per step
  ## - every field judged per step under its bars
  ## - the carries advancing from the observed outputs
  var usage = Usage()
  var rng = initNaiveRng(ChainSeed)
  let w = buildWeights(rng)
  let carry0 = newCarry(rng)
  var carryM = Carry(state: carry0.state, ring: carry0.ring)
  var carryN = Carry(state: carry0.state, ring: carry0.ring)
  var m = allocMega()
  defer: freeMega(m)
  fillWeights(m, w)
  for step in 0 ..< NumSteps:
    var tok = newToken(rng)
    var regen = 0
    block found:
      while true:
        let cw = compareWalk(engine, m, w, tok, carryM, carryN)
        if cw.bars.gateClear:
          judgeAll(cw.bars, cw.snap, cw.postN, cw.lo, usage)
          echo &"[chain] step {step}: y worst {usage.worst[11]:.3f}, " &
            &"state worst {usage.worst[18]:.3f}, moeOut worst " &
            &"{usage.worst[16]:.3f}, {regen} samples regenerated"
          carryM = Carry(state: cw.snap.state, ring: cw.snap.ring)
          carryN = cw.postN
          break found
        inc regen
        doAssert regen < 4096, &"chain step {step}: tie region never clears"
        tok = newToken(rng)
  printUsage(usage, "chain")

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
