# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_mixer.nim
## - sabotage build, the landed stage counters observable, the launch-end reset compiled out,
##   `--warnings:off` as in the default run:
##   nim c -r -d:release -d:WaveResetSabotage --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_mixer.nim
##
## Mixer-internals tier for the `qwen35_moe` fused GDN decoder layer.
##
## One launch of the static `HaveNorm = false` entry at grid
## (876, 1, 1) = `StageEnds[9]`, stages 0..9 with the norm bookends
## and the MoE tail compiled out.
##
## The projection stages read the host-preloaded normed row, the walk
## stopping after the out_proj row.
##
## The conv, split, l2norm, g/beta, recurrence and out_proj anchors are
## judged without the norm's reduction band compounding through them.
##
## | check       | contract                                                                    |
## | ----------- | --------------------------------------------------------------------------- |
## | page fit    | every buffer's byte length is a `HostPageSize` multiple before the launch   |
## | preload     | the norm1 row is the shared operand of both sides, judged bit-identical     |
## | sentinels   | kernel-unread sections keep their poison bits through the launch            |
## | bands       | each field's bar is the stage's landed band at the mega's observed operands |
## | counters    | 0..9 = `WaveCounts` landed (sabotage) or all zero post-launch (default)     |
## | determinism | the relaunch over restored state and ring is bit-identical                  |
##
## No-copy binding:
##
## - page-aligned pointers with page-multiple byte lengths
## - anything else copy-ins, in-place writes are lost
##
import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/qwen35_moe/qwen35_moe_decode_gdn_bf16
import ../naive/naive_rng
import ../naive/naive_tensors
import ../naive/naive_gdn
import ../naive/naive_layer_ops
import ceramic_pagebuf
import mega_bounded_wait
import ../../src/kernels/ceramic/launch_contract
import ceramic_fam

# ─── Band-model constants (the composition tier's two-term model) ─────

const
  UBf = 3.90625e-3'f64                   # 2⁻⁸, the bf16 unit roundoff
  RelRsqrt = 9.5367431640625e-7'f64      # 2·2⁻²¹, the approximate rsqrt class
  RelSilu = 4.76837158203125e-7'f64      # 8·u32, the exp2-form transcendental class
  FloorBf = 7.346879709099078e-39'f64    # 2⁻¹²⁶, the bf16 subnormal grid floor
  FloorSub = 2.9802322387695312e-8'f64   # 2⁻²⁵, the fp16 grid floor, also the bf16 output grid

# ─── Device entry, one launch of the mega's stages 0..9 ───────────────

const MixerMsl = metal:
  proc qwen35_gdn_mixer_bf16(
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
    qwen35GdnLayerWalk[false](counters, bfA, f32A, xPrev, rPrev, state, ring,
      norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W,
      routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW,
      aLog, dtBias, eps)

# ─── Host, the seeded inputs ──────────────────────────────────────────

const H = Hidden
const Seed = 0xC04D0603'u64
const Eps = 1.0e-6'f32
const NumCounters = 13
const NumCases = 2
const CaseSeedStep = 0x9E3779B9'u64

type Weights = object
  ## Seeded weights and recurrence constants over the mixer's stages,
  ## bf16 bit patterns shared by the mega and the naive sides through
  ## their exact fp32 widenings.
  norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW: seq[uint16]
  aLog: seq[float32]
  dtBias: seq[uint16]

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
  ## Seeded weights at the Qwen bf16 class geometry, the composition
  ## tier's generation recipe, modest magnitudes so no stage saturates.
  result.norm1W = randBits(rng, Hidden, -0.05'f32, 0.05'f32)
  result.qkvW = randBits(rng, ConvDim * Hidden, -0.02'f32, 0.02'f32)
  result.zW = randBits(rng, NumVHeads * HeadVDim * Hidden, -0.02'f32, 0.02'f32)
  result.aW = randBits(rng, NumVHeads * Hidden, -0.02'f32, 0.02'f32)
  result.bW = randBits(rng, NumVHeads * Hidden, -0.02'f32, 0.02'f32)
  result.convW = randBits(rng, ConvDim * ConvKernel, -0.1'f32, 0.1'f32)
  result.onormW = randBits(rng, NumVHeads * HeadVDim, -0.05'f32, 0.05'f32)
  result.outprojW = randBits(rng, Hidden * NumVHeads * HeadVDim, -0.02'f32, 0.02'f32)
  result.aLog = newSeq[float32](NumVHeads)
  for h in 0 ..< NumVHeads:
    result.aLog[h] = rng.nextF32(-2.0'f32, -0.1'f32)
  result.dtBias = randBits(rng, NumVHeads, -0.1'f32, 0.1'f32)

# ─── Host replays and band helpers (ported from the composition tier) ─

func widen(s: seq[uint16]): seq[float64] =
  ## Exact widenings of bf16 bit patterns, the band walk's magnitudes.
  result = newSeq[float64](s.len)
  for i in 0 ..< s.len:
    result[i] = bf16ToF32(s[i]).float64

func widenF32(s: seq[float32]): seq[float64] =
  ## Exact fp64 widening of an fp32 section, the band walk's magnitudes.
  result = newSeq[float64](s.len)
  for i in 0 ..< s.len:
    result[i] = s[i].float64

func silu64(x: float64): float64 =
  ## silu in fp64, the h-link band's magnitude reference.
  x / (1.0 + exp(-x))

func l2AccOf(row: seq[uint16]): float32 =
  ## L2 row's fp32 serial sum of bf16-rounded squares, the naive spelling replayed.
  for c in 0 ..< row.len:
    let xi = bf16ToF32(row[c])
    result += bf16ToF32(f32ToBf16(xi * xi))

func l2InvOf(row: seq[uint16]): float64 =
  ## L2 row's bf16-rounded reciprocal, the recorded chain's spelling replayed.
  ##
  ##   fp32 sum rounds to bf16 first (the model's `.sum` output dtype)
  ##   → the eps add rounds again
  let acc = l2AccOf(row)
  result = bf16ToF32(f32ToBf16(1.0'f32 / sqrt(bf16ToF32(
    f32ToBf16(bf16ToF32(f32ToBf16(acc)) + 1.0e-6'f32))))).float64

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

func sensF32Of(nPrime, n: seq[float32]): seq[float64] =
  ## Exact sensitivity of an fp32 naive output to its operands' deviation.
  doAssert nPrime.len == n.len
  result = newSeq[float64](n.len)
  for i in 0 ..< n.len:
    result[i] = abs(nPrime[i].float64 - n[i].float64)

func denseLocal(x: seq[uint16]; w: seq[uint16]; nPrime: seq[uint16];
    N, K: int): seq[float64] =
  ## Landed dense-linear band per output element at the mega's operand magnitudes.
  ##
  ##   terms ─── accumulator reassociation, store round, grid floor
  doAssert x.len == K and nPrime.len == N and w.len == N * K
  result = newSeq[float64](N)
  let xw = widen(x)
  for n in 0 ..< N:
    var sumAbs = 0.0'f64
    for k in 0 ..< K:
      sumAbs += abs(xw[k] * bf16ToF32(w[n * K + k]).float64)
    result[n] = 2.0 * float64(K) * U32 * sumAbs +
      2.0 * UBf * abs(bf16ToF32(nPrime[n]).float64) + FloorSub

func gemvBars(xPrime, w, nPrime, n: seq[uint16]; N, K: int): seq[float64] =
  ## One GEMV field's bars, the landed dense band at the mega's operands
  ## plus the exact sensitivity of the naive dot to the observed input deviation.
  zipAdd(denseLocal(xPrime, w, nPrime, N, K), sensOf(nPrime, n))

func toF32Mat(bits: seq[uint16]; rows, cols: int): NaiveMat[float32] =
  ## A widened fp32 matrix over bf16 bits, the naive step's operand form.
  doAssert bits.len == rows * cols
  result = NaiveMat[float32](rows: rows, cols: cols)
  result.data = newSeq[float32](rows * cols)
  for i in 0 ..< bits.len:
    result.data[i] = bf16ToF32(bits[i])

func toF32Vec(bits: seq[uint16]; n: int): seq[float32] =
  ## A widened fp32 vector over bf16 bits.
  doAssert bits.len == n
  result = newSeq[float32](n)
  for i in 0 ..< n:
    result[i] = bf16ToF32(bits[i])

# ─── The naive chain, stages 2..10 composed at the preloaded row ──────

type NaiveLo = object
  ## Naive chain stages 2..10's outputs at the shared preloaded
  ## norm1 row, the band walk's reference side.
  qkvCol, z, a, b, conv, qn, kn, beta, y, normed, blockOut: seq[uint16]
  g: seq[float32]
  postState: seq[float32]
  postRing: seq[uint16]

proc naiveChain(w: Weights; norm1: seq[uint16]; carryIn: Carry): NaiveLo =
  ## Naive stages 2..10 over copies of the carried state and ring,
  ## the walk mutating the copies.
  result.qkvCol = naiveDenseLinear(norm1, w.qkvW, ConvDim, H)
  result.z = naiveDenseLinear(norm1, w.zW, NumVHeads * HeadVDim, H)
  result.a = naiveDenseLinear(norm1, w.aW, NumVHeads, H)
  result.b = naiveDenseLinear(norm1, w.bW, NumVHeads, H)
  var ringN = carryIn.ring
  result.conv = naiveCausalConvSiluStep(w.convW, ringN, result.qkvCol,
    ConvDim, ConvKernel)
  result.qn = newSeq[uint16](NumKHeads * HeadKDim)
  result.kn = newSeq[uint16](NumKHeads * HeadKDim)
  for half in 0 ..< 2:
    let srcBase = half * (NumKHeads * HeadKDim)
    for h in 0 ..< NumKHeads:
      let row = result.conv[srcBase + h * HeadKDim ..<
        srcBase + (h + 1) * HeadKDim]
      let outp = naiveL2NormRow(row, HeadKDim)
      for c in 0 ..< HeadKDim:
        if half == 0:
          result.qn[h * HeadKDim + c] = outp[c]
        else:
          result.kn[h * HeadKDim + c] = outp[c]
  let gates = naiveGdnGates(result.a, result.b, w.dtBias, w.aLog, NumVHeads)
  result.g = gates.g
  result.beta = gates.beta
  let vBase = 2 * NumKHeads * HeadKDim
  var stateN = NaiveCube[float32](planes: NumVHeads, rows: HeadVDim,
    cols: HeadKDim)
  stateN.data = carryIn.state
  var yN = NaiveMat[float32](rows: NumVHeads, cols: HeadVDim)
  yN.data = newSeq[float32](NumVHeads * HeadVDim)
  gdnDecodeStep(stateN, yN, toF32Mat(result.qn, NumKHeads, HeadKDim),
    toF32Mat(result.kn, NumKHeads, HeadKDim),
    toF32Mat(result.conv[vBase ..< vBase + NumVHeads * HeadVDim],
      NumVHeads, HeadVDim),
    toF32Vec(result.beta, NumVHeads), result.g, NumVHeads, NumKHeads, HkRatio)
  result.postState = stateN.data
  result.y = newSeq[uint16](NumVHeads * HeadVDim)
  for bh in 0 ..< NumVHeads:
    for c in 0 ..< HeadVDim:
      result.y[bh * HeadVDim + c] = f32ToBf16(yN.data[bh * HeadVDim + c])
  result.normed = newSeq[uint16](NumVHeads * HeadVDim)
  for bh in 0 ..< NumVHeads:
    let yRow = result.y[bh * HeadVDim ..< (bh + 1) * HeadVDim]
    let zRow = result.z[bh * HeadVDim ..< (bh + 1) * HeadVDim]
    let wRow = w.onormW[bh * HeadVDim ..< (bh + 1) * HeadVDim]
    let outp = naiveRmsNormGated(yRow, zRow, wRow, HeadVDim, Eps)
    for c in 0 ..< HeadVDim:
      result.normed[bh * HeadVDim + c] = outp[c]
  result.blockOut = naiveDenseLinear(result.normed, w.outprojW, H,
    NumVHeads * HeadVDim)
  result.postRing = ringN

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

proc fillWeights(m: var MegaBuffers; w: Weights) =
  ## Mixer weights and recurrence constants, written once.
  ##
  ## The norm bookends' and MoE tail's weights stay zero-filled,
  ## the `HaveNorm = false` dispatcher never reads them.
  fillBf(m.norm1W, w.norm1W)
  fillBf(m.qkvW, w.qkvW)
  fillBf(m.zW, w.zW)
  fillBf(m.aW, w.aW)
  fillBf(m.bW, w.bW)
  fillBf(m.convW, w.convW)
  fillBf(m.onormW, w.onormW)
  fillBf(m.outprojW, w.outprojW)
  fillBf(m.dtBias, w.dtBias)
  fillF32(m.aLog, w.aLog)

const PoisonBf = 0x7B7B'u16
  ## Untouched-section poison bits, a plausible nonzero bf16 payload any
  ## stray write would overwrite.

proc poisonBf(buf: var PageBuf[uint16]; off, len: int) =
  ## Poisons one bf16 arena section.
  for i in off ..< off + len:
    buf.hostPtr[i] = PoisonBf

proc poisonF32(buf: var PageBuf[float32]; off, len: int) =
  ## Poisons one fp32 arena section.
  for i in off ..< off + len:
    buf.hostPtr[i] = 7.0e30'f32

proc poisonUntouched(m: var MegaBuffers) =
  ## Poisons every section the `HaveNorm = false` dispatcher never reads
  ## or writes, the residual addends included (stage 0 runs empty).
  poisonBf(m.xPrev, 0, Hidden)
  poisonBf(m.rPrev, 0, Hidden)
  poisonBf(m.bfA, sStream, Hidden)
  poisonBf(m.bfA, sH1, Hidden)
  poisonBf(m.bfA, sNormed2, Hidden)
  poisonBf(m.bfA, sMoeOut, Hidden)
  poisonBf(m.bfA, sH, TopK * Inter)
  poisonBf(m.bfA, sHs, Inter)
  poisonF32(m.f32A, sPartial, (TopK + 1) * Hidden)

proc assertUntouched(m: var MegaBuffers) =
  ## Poison bits survive the launch bit-identical.
  for i in 0 ..< Hidden:
    doAssert m.xPrev.hostPtr[i] == PoisonBf and m.rPrev.hostPtr[i] == PoisonBf,
      &"the residual addends were touched at {i}"
  for i in sStream ..< sStream + Hidden:
    doAssert m.bfA.hostPtr[i] == PoisonBf, &"sStream was touched at {i}"
  for i in sH1 ..< sH1 + Hidden:
    doAssert m.bfA.hostPtr[i] == PoisonBf, &"sH1 was touched at {i}"
  for i in sNormed2 ..< sNormed2 + Hidden:
    doAssert m.bfA.hostPtr[i] == PoisonBf, &"sNormed2 was touched at {i}"
  for i in sMoeOut ..< sMoeOut + Hidden:
    doAssert m.bfA.hostPtr[i] == PoisonBf, &"sMoeOut was touched at {i}"
  for i in sH ..< sH + TopK * Inter:
    doAssert m.bfA.hostPtr[i] == PoisonBf, &"sH was touched at {i}"
  for i in sHs ..< sHs + Inter:
    doAssert m.bfA.hostPtr[i] == PoisonBf, &"sHs was touched at {i}"
  for i in sPartial ..< sPartial + (TopK + 1) * Hidden:
    doAssert m.f32A.hostPtr[i] == 7.0e30'f32, &"sPartial was touched at {i}"

type Snap = object
  ## One launch's judged sections and carry, the bit-identity
  ## compare's reference and the band walk's observed operands.
  qkv, z, a, b, conv, qn, kn, beta, y, normed, blockOut: seq[uint16]
  g: seq[float32]
  state: seq[float32]
  ring: seq[uint16]
  norm1: seq[uint16]

proc snapMega(m: MegaBuffers): Snap =
  ## One launch's full snapshot of the judged sections.
  result.norm1 = readInto(m.bfA.hostPtr +% sNorm1, H)
  result.qkv = readInto(m.bfA.hostPtr +% sQkvCol, ConvDim)
  result.z = readInto(m.bfA.hostPtr +% sZ, NumVHeads * HeadVDim)
  result.a = readInto(m.bfA.hostPtr +% sA, NumVHeads)
  result.b = readInto(m.bfA.hostPtr +% sB, NumVHeads)
  result.conv = readInto(m.bfA.hostPtr +% sConv, ConvDim)
  result.qn = readInto(m.bfA.hostPtr +% sQN, NumKHeads * HeadKDim)
  result.kn = readInto(m.bfA.hostPtr +% sKN, NumKHeads * HeadKDim)
  result.g = readInto(m.f32A.hostPtr +% sG, NumVHeads)
  result.beta = readInto(m.bfA.hostPtr +% sBeta, NumVHeads)
  result.y = readInto(m.bfA.hostPtr +% sY, NumVHeads * HeadVDim)
  result.normed = readInto(m.bfA.hostPtr +% sNormed, NumVHeads * HeadVDim)
  result.blockOut = readInto(m.bfA.hostPtr +% sBlockOut, H)
  result.state = readInto(m.state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  result.ring = readInto(m.ring.hostPtr, ConvDim * RingWidth)

proc launchMixer(engine: HwEngine; m: var MegaBuffers) =
  ## One launch at the mixer's grid, no host-side counter zeroing.
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
    assertMegaGrid(int(StageEnds[9]), 32, StageEnds, false)
    engine.run << (grid: (int(StageEnds[9]), 1, 1), blk: (32, 1, 1)) >>
      ("qwen35_gdn_mixer_bf16", cPA,
        (bfAPA, f32APA, xPA, rPA, stPA, rgPA, n1PA, qkvPA, zPA, aPA,
         bPA, cvPA, onPA, opPA, n2PA, rtPA, guPA, dnPA,
         sgPA, suPA, sdPA, gvPA, alPA, dbPA, Eps))
    result = true
  runMegaBounded(dispatch, m.counters.hostPtr, StageNames)

proc assertCounters(m: var MegaBuffers) =
  ## Stage counters, per the build:
  ##
  ## - the default build asserts the launch-end reset's zero state, all
  ##   13 counters read zero exactly after the launch
  ## - the WaveResetSabotage build asserts the landed per-stage
  ##   threadgroup totals at stages 0..9, the stages 10..12 without
  ##   threadgroups stay zero
  when WaveResetSabotage:
    for i in 0 ..< 10:
      doAssert m.counters.hostPtr[i] == WaveCounts[i],
        &"stage counter {i} landed {m.counters.hostPtr[i]} " &
        &"want {WaveCounts[i]} (the sabotage build's landed total)"
    for i in 10 ..< NumCounters:
      doAssert m.counters.hostPtr[i] == 0'u32,
        &"stage counter {i} landed {m.counters.hostPtr[i]} want 0"
  else:
    for i in 0 ..< NumCounters:
      doAssert m.counters.hostPtr[i] == 0'u32,
        &"stage counter {i} is {m.counters.hostPtr[i]} want 0 " &
        &"(the launch-end reset)"

# ─── The judge ────────────────────────────────────────────────────────

const FieldNames = [
  "qkv", "z", "a", "b", "conv", "qn", "kn", "g",
  "beta", "y", "normed", "blockOut", "state", "ring"]
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

proc printUsage(u: Usage; label: string) =
  ## Per-field usage table, worst bar usage and bit-exact counts.
  echo &"[{label}] worst bar usage per field:"
  for f in 0 ..< NumFields:
    if u.total[f] > 0:
      echo &"  {FieldNames[f]:<9} worst {u.worst[f]:.3f} " &
        &"bit-exact {u.exact[f]}/{u.total[f]}"

type Bars = object
  ## Per-element bars for the 12 judged fields, each the stage's landed
  ## local band at the mega's observed operands plus the exact sensitivity
  ## of the naive op to the observed operand deviation.
  qkv, z, a, b, conv, qn, kn, beta: seq[float64]
  g, y, normed, blockOut, state, ring: seq[float64]

proc walkBars(w: Weights; norm1M: seq[uint16]; preM, preN: Carry;
    lo: NaiveLo; snap: Snap): Bars =
  ## One walk's per-element bars from the observed operands.
  ##
  ## Both sides share the preloaded norm1 row bit-exact,
  ## so the projection stages' sensitivity terms reduce to the two
  ## naive evaluations' deviation through the downstream operands.
  let
    qkvM = snap.qkv
    zM = snap.z
    aM = snap.a
    bM = snap.b
    convM = snap.conv
    qnM = snap.qn
    knM = snap.kn
    gM = snap.g
    betaM = snap.beta
    yM = snap.y
    normedM = snap.normed

  # Stages 2 to 4:
  #   the projection GEMVs. The preloaded norm1 row is the observed input,
  #   the naive dot replayed at both operand sets
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
      doAssert lo.postRing[c * RingWidth + j] ==
        preN.ring[c * RingWidth + j + 1],
        "the naive's ring roll is not a copy"
      result.ring[c * RingWidth + j] =
        abs(preMW[c * RingWidth + j + 1] - preNW[c * RingWidth + j + 1])
    doAssert snap.ring[c * RingWidth + RingWidth - 1] == qkvM[c],
      "the mega's ring tail is not the step column"
    doAssert lo.postRing[c * RingWidth + RingWidth - 1] == lo.qkvCol[c],
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
          lo.postState[rowBase + c].float64)
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
  #   y and z rows, plus the exact sensitivity of the naive row op
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

proc judgeAll(bars: Bars; snap: Snap; lo: NaiveLo; u: var Usage) =
  ## One walk's per-element judgment, the 12 fields against their bars.
  judgeBf(0, snap.qkv, lo.qkvCol, bars.qkv, u)
  judgeBf(1, snap.z, lo.z, bars.z, u)
  judgeBf(2, snap.a, lo.a, bars.a, u)
  judgeBf(3, snap.b, lo.b, bars.b, u)
  judgeBf(4, snap.conv, lo.conv, bars.conv, u)
  judgeBf(5, snap.qn, lo.qn, bars.qn, u)
  judgeBf(6, snap.kn, lo.kn, bars.kn, u)
  judgeF32(7, snap.g, lo.g, bars.g, u)
  judgeBf(8, snap.beta, lo.beta, bars.beta, u)
  judgeBf(9, snap.y, lo.y, bars.y, u)
  judgeBf(10, snap.normed, lo.normed, bars.normed, u)
  judgeBf(11, snap.blockOut, lo.blockOut, bars.blockOut, u)
  judgeF32(12, snap.state, lo.postState, bars.state, u)
  judgeBf(13, snap.ring, lo.postRing, bars.ring, u)

# ─── The runner ───────────────────────────────────────────────────────

proc mixerInit*(): HwEngine =
  ## Device setup for the mixer tier:
  ## - the geometry asserts, the Metal engine, the mixer MSL ingested
  doAssert HeadKDim == 128 and HeadVDim == 128,
    "the mixer's GDN binding is Dk = Dv = 128"
  doAssert TopK == 8 and NumExperts == 256 and Inter == 512,
    "the mixer's MoE geometry is the Qwen decode class"
  echo "device: ", bkMetal.init().deviceName()
  result = bkMetal.init()
  result.ingest(MixerMsl)

proc fillNorm1(m: var MegaBuffers; norm1: seq[uint16]) =
  ## Norm1 row preloaded into its arena section, the projection stages'
  ## shared operand. The host-computed row is the shared operand,
  ## the mega entry's stage 1 compiled out, no device op recomputes it.
  doAssert norm1.len == H
  for i in 0 ..< H:
    m.bfA.hostPtr[sNorm1 + i] = norm1[i]

proc assertNorm1Unchanged(m: MegaBuffers; norm1: seq[uint16]) =
  ## Preloaded norm1 row survives the launch bit-identical.
  for i in 0 ..< H:
    doAssert m.bfA.hostPtr[sNorm1 + i] == norm1[i],
      &"the preloaded norm1 row was touched at {i}"

proc runMixerWalk(engine: HwEngine; w: Weights; carry0: Carry;
    norm1: seq[uint16]; caseId: int; usage: var Usage) =
  ## One case, fill → poison → launch → counters → sentinels → preload
  ## bit-identity → naive chain → per-element judgment, then the fresh
  ## relaunch over the restored carry judged bit-identical
  ## (default build only, the sabotage build leaves stale counters).
  var m = allocMega()
  defer: freeMega(m)
  fillWeights(m, w)
  fillNorm1(m, norm1)
  fillF32(m.state, carry0.state)
  fillBf(m.ring, carry0.ring)
  poisonUntouched(m)
  launchMixer(engine, m)
  assertCounters(m)
  assertUntouched(m)
  assertNorm1Unchanged(m, norm1)

  let snap = snapMega(m)
  let lo = naiveChain(w, norm1, carry0)
  let bars = walkBars(w, norm1, carry0, carry0, lo, snap)
  judgeAll(bars, snap, lo, usage)

  when not WaveResetSabotage:
    # the relaunch, the carry restored to its pre-image, the judged
    # sections rewritten over the first launch's own outputs
    for i in 0 ..< NumVHeads * HeadVDim * HeadKDim:
      m.state.hostPtr[i] = carry0.state[i]
    for i in 0 ..< ConvDim * RingWidth:
      m.ring.hostPtr[i] = carry0.ring[i]
    launchMixer(engine, m)
    assertCounters(m)
    assertUntouched(m)
    assertNorm1Unchanged(m, norm1)
    let snapRel = snapMega(m)
    doAssert snapRel.state == snap.state,
      "the relaunch's state differs from the first launch's"
    doAssert snapRel.ring == snap.ring,
      "the relaunch's ring differs from the first launch's"
    doAssert snapRel.qkv == snap.qkv and snapRel.z == snap.z and
      snapRel.a == snap.a and snapRel.b == snap.b and
      snapRel.conv == snap.conv and snapRel.qn == snap.qn and
      snapRel.kn == snap.kn and snapRel.beta == snap.beta and
      snapRel.y == snap.y and snapRel.normed == snap.normed and
      snapRel.blockOut == snap.blockOut and snapRel.g == snap.g,
      "the relaunch's judged sections differ from the first launch's"
    echo &"[mixer case {caseId}] relaunch bit-identical"

proc runMixer(engine: HwEngine) =
  ## 2 seeded cases, one mega launch and one naive chain each, judged
  ## per stage under the observed-operand bars, then a fresh relaunch
  ## over the restored carry judged bit-identical (default build).
  var usage = Usage()
  for caseId in 0 ..< NumCases:
    var rng = initNaiveRng(Seed + uint64(caseId) * CaseSeedStep)
    let w = buildWeights(rng)
    let carry0 = Carry(
      state: (proc(): seq[float32] =
        result = newSeq[float32](NumVHeads * HeadVDim * HeadKDim)
        for i in 0 ..< result.len:
          result[i] = rng.nextF32(-0.5'f32, 0.5'f32))(),
      ring: randBits(rng, ConvDim * RingWidth, -1.0'f32, 1.0'f32))
    let x = randBits(rng, Hidden, -1.0'f32, 1.0'f32)
    let r = randBits(rng, Hidden, -1.0'f32, 1.0'f32)
    let norm1 = naiveRmsNormRes(x, r, w.norm1W, H, Eps).normed
    runMixerWalk(engine, w, carry0, norm1, caseId, usage)
  printUsage(usage, "mixer")

proc main =
  var engine = mixerInit()
  let t0 = epochTime()
  runMixer(engine)
  echo &"[mixer] wall clock {epochTime() - t0:.2f} s"
  echo "CERAMIC MEGA GDN MIXER VERDICT: preload, sentinels, bands, counters, determinism"

main()
