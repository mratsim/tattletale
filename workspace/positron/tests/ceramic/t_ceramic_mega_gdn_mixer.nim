# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_properties
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_mixer.nim
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
## | check       | contract                                                                                           |
## | ----------- | -------------------------------------------------------------------------------------------------- |
## | page fit    | every buffer's byte length is a `HostPageSize` multiple before the launch                          |
## | preload     | the norm1 row is the shared operand of the projection stages, it survives the launch bit-identical |
## | sentinels   | kernel-unread sections keep their poison bits through the launch                                   |
## | ring roll   | the history shifts down one slot as bit copies, the step column lands in the newest slot           |
## | counters    | all 13 counters zero post-launch, the launch-end reset                                             |
## | determinism | the relaunch over restored state and ring is bit-identical                                         |
##
## No-copy binding:
##
## - page-aligned pointers with page-multiple byte lengths
## - anything else copy-ins, in-place writes are lost
##
import std/[strformat, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/gdn_moe_decode_megakernel
import ceramic_pagebuf
import mega_bounded_wait
import ceramic_dtype
import mega_gdn_harness

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
    gdnMoeLayerWalk[bfloat16, false](counters, bfA, f32A, xPrev, rPrev, state, ring,
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
  ## Seeded weights and recurrence constants over the mixer's stages, bf16 bits.
  norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW: seq[uint16]
  aLog: seq[float32]
  dtBias: seq[uint16]

type Carry = object
  ## One walk's carried state and ring, fp32 and bf16 bit patterns.
  state: seq[float32]
  ring: seq[uint16]

proc randBits(rng: var PropRng; n: int; lo, hi: float32): seq[uint16] =
  ## `n` bf16 bit patterns of uniform samples in [lo, hi].
  result = newSeq[uint16](n)
  for i in 0 ..< n:
    result[i] = f32ToBf16(rng.nextF32(lo, hi))

proc buildWeights(rng: var PropRng): Weights =
  ## Seeded weights at the Qwen bf16 class geometry, modest magnitudes
  ## so no stage saturates.
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

# ─── Case plumbing ────────────────────────────────────────────────────

proc fillWeights(m: var MegaGdnBufs; w: Weights) =
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

proc poisonUntouched(m: var MegaGdnBufs) =
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

proc assertUntouched(m: var MegaGdnBufs) =
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

type Record = object
  ## One launch's judged sections and carry.
  qkv, z, a, b, conv, qn, kn, beta, y, normed, blockOut: seq[uint16]
  g: seq[float32]
  state: seq[float32]
  ring: seq[uint16]
  norm1: seq[uint16]

proc recordMega(m: MegaGdnBufs): Record =
  ## One launch's full record of the judged sections.
  result.norm1 = readRecord(m.bfA.hostPtr +% sNorm1, H)
  result.qkv = readRecord(m.bfA.hostPtr +% sQkvCol, ConvDim)
  result.z = readRecord(m.bfA.hostPtr +% sZ, NumVHeads * HeadVDim)
  result.a = readRecord(m.bfA.hostPtr +% sA, NumVHeads)
  result.b = readRecord(m.bfA.hostPtr +% sB, NumVHeads)
  result.conv = readRecord(m.bfA.hostPtr +% sConv, ConvDim)
  result.qn = readRecord(m.bfA.hostPtr +% sQN, NumKHeads * HeadKDim)
  result.kn = readRecord(m.bfA.hostPtr +% sKN, NumKHeads * HeadKDim)
  result.g = readRecord(m.f32A.hostPtr +% sG, NumVHeads)
  result.beta = readRecord(m.bfA.hostPtr +% sBeta, NumVHeads)
  result.y = readRecord(m.bfA.hostPtr +% sY, NumVHeads * HeadVDim)
  result.normed = readRecord(m.bfA.hostPtr +% sNormed, NumVHeads * HeadVDim)
  result.blockOut = readRecord(m.bfA.hostPtr +% sBlockOut, H)
  result.state = readRecord(m.state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  result.ring = readRecord(m.ring.hostPtr, ConvDim * RingWidth)

proc launchMixer(engine: HwEngine; m: var MegaGdnBufs) =
  ## One launch at the mixer's grid, no host-side counter zeroing.
  var mp = addr m
  proc dispatch() {.gcsafe.} =
    runMegaGdn(engine, mp[], "qwen35_gdn_mixer_bf16", int(StageEnds[9]), Eps)
  runMegaBounded(dispatch, m.counters.hostPtr, stageNames)

proc assertCounters(m: var MegaGdnBufs) =
  ## All 13 counters read zero exactly after the launch,
  ## the launch-end reset's zero state.
  for i in 0 ..< NumCounters:
      doAssert m.counters.hostPtr[i] == 0'u32,
        &"stage counter {i} is {m.counters.hostPtr[i]} want 0 " &
        &"(the launch-end reset)"

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

proc fillNorm1(m: var MegaGdnBufs; norm1: seq[uint16]) =
  ## Norm1 row preloaded into its arena section, the projection stages'
  ## shared operand. The seeded row is the shared operand, the mega
  ## entry's stage 1 compiled out, no device op recomputes it.
  doAssert norm1.len == H
  for i in 0 ..< H:
    m.bfA.hostPtr[sNorm1 + i] = norm1[i]

proc assertNorm1Unchanged(m: MegaGdnBufs; norm1: seq[uint16]) =
  ## Preloaded norm1 row survives the launch bit-identical.
  for i in 0 ..< H:
    doAssert m.bfA.hostPtr[sNorm1 + i] == norm1[i],
      &"the preloaded norm1 row was touched at {i}"

proc assertRingRoll(ringPost, ringPre, qkvCol: seq[uint16]) =
  ## Conv step's ring contract over the launch's own sections:
  ##
  ## - the history shifts down one slot as bit copies, the newest slot
  ##   reading the step column before the update
  ## - the step column lands in the newest slot, bit-identical to the qkv column
  for c in 0 ..< ConvDim:
    for j in 0 ..< RingWidth - 1:
      doAssert ringPost[c * RingWidth + j] == ringPre[c * RingWidth + j + 1],
        "the ring roll is not a copy"
    doAssert ringPost[c * RingWidth + RingWidth - 1] == qkvCol[c],
      "the ring tail is not the step column"

proc runMixerWalk(engine: HwEngine; w: Weights; carry0: Carry;
    norm1: seq[uint16]; caseId: int) =
  ## One case, fill → poison → launch → counters → sentinels → preload
  ## bit-identity → ring-roll contract → fresh relaunch, bit-identical.
  var m = allocMegaGdn()
  defer: freeMegaGdn(m)
  fillWeights(m, w)
  fillNorm1(m, norm1)
  fillF32(m.state, carry0.state)
  fillBf(m.ring, carry0.ring)
  poisonUntouched(m)
  launchMixer(engine, m)
  assertCounters(m)
  assertUntouched(m)
  assertNorm1Unchanged(m, norm1)

  let record = recordMega(m)
  # the ring's pre-image is the carry's ring, the ring roll shifts it down
  # one slot with the qkv column landing in the newest slot
  assertRingRoll(record.ring, carry0.ring, record.qkv)

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
  let snapRel = recordMega(m)
  doAssert snapRel.state == record.state,
    "the relaunch's state differs from the first launch's"
  doAssert snapRel.ring == record.ring,
    "the relaunch's ring differs from the first launch's"
  doAssert snapRel.qkv == record.qkv and snapRel.z == record.z and
    snapRel.a == record.a and snapRel.b == record.b and
    snapRel.conv == record.conv and snapRel.qn == record.qn and
    snapRel.kn == record.kn and snapRel.beta == record.beta and
    snapRel.y == record.y and snapRel.normed == record.normed and
    snapRel.blockOut == record.blockOut and snapRel.g == record.g,
    "the relaunch's judged sections differ from the first launch's"
  echo &"[mixer case {caseId}] relaunch bit-identical"

proc runMixer(engine: HwEngine) =
  ## 2 seeded cases, one mega launch each, the ring-roll, poison and counters
  ## checks per launch, plus a fresh relaunch, bit-identical.
  for caseId in 0 ..< NumCases:
    var rng = initPropRng(Seed + uint64(caseId) * CaseSeedStep)
    let w = buildWeights(rng)
    let carry0 = Carry(
      state: (proc(): seq[float32] =
        result = newSeq[float32](NumVHeads * HeadVDim * HeadKDim)
        for i in 0 ..< result.len:
          result[i] = rng.nextF32(-0.5'f32, 0.5'f32))(),
      ring: randBits(rng, ConvDim * RingWidth, -1.0'f32, 1.0'f32))
    # the norm1 row is a seeded fixture, preloaded as the projection
    # stages' shared operand (the mega entry's stage 1 compiled out)
    let norm1 = randBits(rng, Hidden, -1.0'f32, 1.0'f32)
    runMixerWalk(engine, w, carry0, norm1, caseId)
  echo &"CERAMIC MEGA GDN MIXER VERDICT: cases={NumCases} launches={NumCases * 2} " &
    &"relaunch bit-identical, poison and counters clean"

proc main =
  var engine = mixerInit()
  let t0 = epochTime()
  runMixer(engine)
  echo &"[mixer] wall clock {epochTime() - t0:.2f} s"

main()
