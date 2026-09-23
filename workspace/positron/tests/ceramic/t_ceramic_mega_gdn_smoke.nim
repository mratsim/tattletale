# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_smoke.nim
##
## One-launch fused GDN decoder layer smoke, one seeded launch
## of the `qwen35_moe` mega kernel at grid (950, 1, 1). Checks:
## - launch success, non-degenerate outputs, sync counters
## - untouched-memory sentinels, fresh-relaunch bit-identity
## - an informational naive-vs-mega diff
##
## The bands live in the comparison tier, this smoke owns none.
##
## No-copy binding:
## - page-aligned pointers with page-multiple byte lengths
## - anything else copy-ins, the kernel's in-place writes are lost
## - every buffer's byte length is asserted a `HostPageSize` multiple
##   before the launch
##
## Sequence: seeded inputs and weights → page buffers → one launch →
## sync counters, sentinels → snapshot → restore and relaunch →
## bit compare → informational naive diff.
##
## | check       | contract                                                                    |
## | ----------- | --------------------------------------------------------------------------- |
## | page fit    | every buffer's byte length is a `HostPageSize` multiple before the launch   |
## | launch      | one seeded launch completes, the outputs written over non-degenerate ranges |
## | wave sync   | the 13 stage counters land exactly on the per-stage threadgroup totals      |
## | sentinels   | kernel-written buffers stay in extent, kernel-read buffers bit-identical    |
## | determinism | the relaunch over restored state and counters is bit-identical              |
## | comparison  | informational max abs difference over the mega-vs-naive shared outputs      |

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/gdn_moe_decode_megakernel
import ../naive/naive_rng
import ../naive/naive_tensors
from ../naive/naive_qwen35_layer import naiveQwen35GdnLayer, LayerOut
import ceramic_pagebuf
import mega_bounded_wait
import ceramic_fam

# ─── Device entry, one launch of the mega's 13-stage dispatcher ───────

const MegaGdnMsl = metal:
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

# ─── Host, the seeded layer pass ─────────────────────────────────────

const Seed = 0xC04D0601'u64
const Eps = 1.0e-6'f32
const NumCounters = 13

type BigHost = object
  ## Seeded layer pass inputs and weights, bf16 bit patterns shared
  ## by the mega and the naive sides through their exact fp32 widenings.
  x, r: seq[uint16]
  norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W: seq[uint16]
  routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW: seq[uint16]
  aLog: seq[float32]
  dtBias: seq[uint16]
  state: seq[float32]
  ring: seq[uint16]

proc randBits(rng: var NaiveRng; n: int; lo, hi: float32): seq[uint16] =
  ## `n` bf16 bit patterns of uniform samples in [lo, hi].
  result = newSeq[uint16](n)
  for i in 0 ..< n:
    result[i] = f32ToBf16(rng.nextF32(lo, hi))

proc buildBigHost(seed: uint64): BigHost =
  ## Seeded inputs and weights at the Qwen bf16 class geometry, the same
  ## generation recipe as the naive composition suite so both sides see
  ## identical bits, modest magnitudes so no stage saturates.
  var rng = initNaiveRng(seed)
  result.x = randBits(rng, Hidden, -1.0'f32, 1.0'f32)
  result.r = randBits(rng, Hidden, -1.0'f32, 1.0'f32)
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
  result.state = newSeq[float32](NumVHeads * HeadVDim * HeadKDim)
  for i in 0 ..< NumVHeads * HeadVDim * HeadKDim:
    result.state[i] = rng.nextF32(-0.5'f32, 0.5'f32)
  result.ring = randBits(rng, ConvDim * RingWidth, -1.0'f32, 1.0'f32)

proc fillBf(buf: var PageBuf[uint16], src: seq[uint16]) =
  ## Copies bf16 bit patterns into a page buffer, the extent then the tail.
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

proc bfRangeMax(buf: PageBuf[uint16]; off, count: int): float32 =
  ## Widened magnitude maximum over one bf16 arena section.
  for i in off ..< off + count:
    result = max(result, abs(bf16ToF32(buf.hostPtr[i])))

proc maxDiff(megaBf: PageBuf[uint16]; megaOff: int;
    naive: seq[uint16]): float32 =
  ## Elementwise absolute difference maximum between a mega arena section
  ## and the naive side's row, both widened from bf16.
  doAssert megaBf.elems >= megaOff + naive.len
  for i in 0 ..< naive.len:
    let a = bf16ToF32(megaBf.hostPtr[megaOff + i])
    let b = bf16ToF32(naive[i])
    result = max(result, abs(a - b))

proc smokeChecks(engine: HwEngine, big: BigHost) =
  ## One seeded launch with the sync, sentinel, determinism
  ## and informational comparison checks around it.
  var
    counters = allocPageBuf[uint32](NumCounters)
    bfA = allocPageBuf[uint16](BfArenaLen)
    f32A = allocPageBuf[float32](F32ArenaLen)
    xPrev = allocPageBuf[uint16](Hidden)
    rPrev = allocPageBuf[uint16](Hidden)
    state = allocPageBuf[float32](NumVHeads * HeadVDim * HeadKDim)
    ring = allocPageBuf[uint16](ConvDim * RingWidth)
    norm1W = allocPageBuf[uint16](Hidden)
    qkvW = allocPageBuf[uint16](ConvDim * Hidden)
    zW = allocPageBuf[uint16](NumVHeads * HeadVDim * Hidden)
    aW = allocPageBuf[uint16](NumVHeads * Hidden)
    bW = allocPageBuf[uint16](NumVHeads * Hidden)
    convW = allocPageBuf[uint16](ConvDim * ConvKernel)
    onormW = allocPageBuf[uint16](NumVHeads * HeadVDim)
    outprojW = allocPageBuf[uint16](Hidden * NumVHeads * HeadVDim)
    norm2W = allocPageBuf[uint16](Hidden)
    routerW = allocPageBuf[uint16](NumExperts * Hidden)
    gateUpW = allocPageBuf[uint16](NumExperts * 2 * Inter * Hidden)
    downW = allocPageBuf[uint16](NumExperts * Hidden * Inter)
    sharedGW = allocPageBuf[uint16](Inter * Hidden)
    sharedUW = allocPageBuf[uint16](Inter * Hidden)
    sharedDW = allocPageBuf[uint16](Hidden * Inter)
    sharedGVW = allocPageBuf[uint16](Hidden)
    aLog = allocPageBuf[float32](NumVHeads)
    dtBias = allocPageBuf[uint16](NumVHeads)
  defer:
    freePageBuf(counters); freePageBuf(bfA); freePageBuf(f32A)
    freePageBuf(xPrev); freePageBuf(rPrev); freePageBuf(state); freePageBuf(ring)
    freePageBuf(norm1W); freePageBuf(qkvW); freePageBuf(zW); freePageBuf(aW)
    freePageBuf(bW); freePageBuf(convW); freePageBuf(onormW); freePageBuf(outprojW)
    freePageBuf(norm2W); freePageBuf(routerW); freePageBuf(gateUpW); freePageBuf(downW)
    freePageBuf(sharedGW); freePageBuf(sharedUW); freePageBuf(sharedDW)
    freePageBuf(sharedGVW); freePageBuf(aLog); freePageBuf(dtBias)

  fillBf(xPrev, big.x); fillBf(rPrev, big.r)
  fillBf(norm1W, big.norm1W); fillBf(qkvW, big.qkvW); fillBf(zW, big.zW)
  fillBf(aW, big.aW); fillBf(bW, big.bW); fillBf(convW, big.convW)
  fillBf(onormW, big.onormW); fillBf(outprojW, big.outprojW)
  fillBf(norm2W, big.norm2W); fillBf(routerW, big.routerW)
  fillBf(gateUpW, big.gateUpW); fillBf(downW, big.downW)
  fillBf(sharedGW, big.sharedGW); fillBf(sharedUW, big.sharedUW)
  fillBf(sharedDW, big.sharedDW); fillBf(sharedGVW, big.sharedGVW)
  fillBf(dtBias, big.dtBias)
  fillF32(aLog, big.aLog); fillF32(state, big.state)
  fillBf(ring, big.ring)

  # the no-copy contract, every byte length a page multiple (asserted in the fills)
  doAssert counters.elems * sizeof(uint32) mod HostPageSize == 0
  doAssert bfA.elems * sizeof(uint16) mod HostPageSize == 0
  doAssert f32A.elems * sizeof(float32) mod HostPageSize == 0

  var
    countersPA = counters.pa()
    bfAPA = bfA.pa()
    f32APA = f32A.pa()
    xPrevPA = xPrev.pa()
    rPrevPA = rPrev.pa()
    statePA = state.pa()
    ringPA = ring.pa()
    norm1WPA = norm1W.pa()
    qkvWPA = qkvW.pa()
    zWPA = zW.pa()
    aWPA = aW.pa()
    bWPA = bW.pa()
    convWPA = convW.pa()
    onormWPA = onormW.pa()
    outprojWPA = outprojW.pa()
    norm2WPA = norm2W.pa()
    routerWPA = routerW.pa()
    gateUpWPA = gateUpW.pa()
    downWPA = downW.pa()
    sharedGWPA = sharedGW.pa()
    sharedUWPA = sharedUW.pa()
    sharedDWPA = sharedDW.pa()
    sharedGVWPA = sharedGVW.pa()
    aLogPA = aLog.pa()
    dtBiasPA = dtBias.pa()

  proc launch(): bool {.gcsafe.} =
    # No host-side counter zeroing, the kernel re-zeroes all counters
    # at the launch's end and the page allocator's zero fill covers the first launch.
    # This launch pair is the relaunch-determinism proof without host zeroing.
    engine.run << (grid: (950, 1, 1), blk: (32, 1, 1)) >>
      ("qwen35_gdn_layer_bf16", countersPA,
        (bfAPA, f32APA, xPrevPA, rPrevPA, statePA, ringPA,
         norm1WPA, qkvWPA, zWPA, aWPA, bWPA, convWPA,
         onormWPA, outprojWPA, norm2WPA, routerWPA, gateUpWPA,
         downWPA, sharedGWPA, sharedUWPA, sharedDWPA,
         sharedGVWPA, aLogPA, dtBiasPA, Eps))
    result = true
  # The bounded wait on every launch, a wedged waveWait spin reports
  # the stuck stage's counters and exits, never an unbounded host spin.
  runMegaBounded(launch, counters.hostPtr, StageNames)

  # state and ring snapshots are the launch's pre-image,
  # the relaunch restores them
  # the arena snapshots are taken after the launch, they hold
  # the outputs the relaunch must reproduce bit for bit
  let stateSnap = readInto(state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let ringSnap = readInto(ring.hostPtr, ConvDim * RingWidth)

  discard launch()

  proc waveSyncCheck() =
    ## Post-launch, the kernel's launch-end reset has re-zeroed the counters.
    for i in 0 ..< NumCounters:
      doAssert counters.hostPtr[i] == 0'u32,
        &"stage counter {i} {counters.hostPtr[i]} want 0 (the launch-end reset)"

  proc outputRanges() =
    let moeMax = bfRangeMax(bfA, sMoeOut, Hidden)
    let h1Max = bfRangeMax(bfA, sH1, Hidden)
    let blockMax = bfRangeMax(bfA, sBlockOut, Hidden)
    let yMax = bfRangeMax(bfA, sY, NumVHeads * HeadVDim)
    echo &"[mega smoke] output maxima moeOut {moeMax:.4f} h1 {h1Max:.4f} " &
      &"blockOut {blockMax:.4f} y {yMax:.4f}"
    doAssert moeMax > 0.0'f32, "moeOut degenerate"
    doAssert h1Max > 0.0'f32, "h1 degenerate"
    doAssert blockMax > 0.0'f32, "blockOut degenerate"
    doAssert yMax > 0.0'f32, "y degenerate"
    doAssert moeMax < 1000.0'f32, "moeOut runaway"
    doAssert h1Max < 1000.0'f32, "h1 runaway"

  proc sentinels() =
    assertTailZero(bfA, BfArenaLen)
    assertTailZero(f32A, F32ArenaLen)
    for i in NumCounters ..< counters.elems:
      doAssert counters.hostPtr[i] == 0'u32, "counters tail written"
    assertReadUnchanged(xPrev, big.x)
    assertReadUnchanged(rPrev, big.r)
    assertReadUnchanged(norm1W, big.norm1W)
    assertReadUnchanged(convW, big.convW)
    assertReadUnchanged(onormW, big.onormW)
    assertReadUnchanged(routerW, big.routerW)
    assertReadUnchanged(gateUpW, big.gateUpW)
    assertReadUnchanged(downW, big.downW)
    assertReadUnchanged(qkvW, big.qkvW)
    assertReadUnchanged(zW, big.zW)
    assertReadUnchanged(aW, big.aW)
    assertReadUnchanged(bW, big.bW)
    assertReadUnchanged(outprojW, big.outprojW)
    assertReadUnchanged(norm2W, big.norm2W)
    assertReadUnchanged(sharedGW, big.sharedGW)
    assertReadUnchanged(sharedUW, big.sharedUW)
    assertReadUnchanged(sharedDW, big.sharedDW)
    assertReadUnchanged(sharedGVW, big.sharedGVW)
    for h in 0 ..< NumVHeads:
      doAssert aLog.hostPtr[h] == big.aLog[h], "kernel-read buffer modified"
      doAssert dtBias.hostPtr[h] == big.dtBias[h], "kernel-read buffer modified"

  waveSyncCheck()
  outputRanges()
  sentinels()
  let bfSnap = readInto(bfA.hostPtr, BfArenaLen)
  let f32Snap = readInto(f32A.hostPtr, F32ArenaLen)
  let statePost = readInto(state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let ringPost = readInto(ring.hostPtr, ConvDim * RingWidth)

  # the informational comparison, no band, the comparison tier owns it
  block comparison:
    var stateN = NaiveCube[float32](planes: NumVHeads, rows: HeadVDim,
      cols: HeadKDim)
    stateN.data = big.state
    var ringN = big.ring
    let t0 = epochTime()
    let naiveOut = naiveQwen35GdnLayer(stateN, ringN, big.x, big.r,
      big.norm1W, big.qkvW, big.zW, big.aW, big.bW, big.convW, big.onormW,
      big.outprojW, big.norm2W, big.routerW, big.gateUpW, big.downW,
      big.sharedGW, big.sharedUW, big.sharedDW, big.sharedGVW, big.aLog,
      big.dtBias, Eps)
    echo &"[mega smoke] naive walk {epochTime() - t0:.2f} s"
    let dMoe = maxDiff(bfA, sMoeOut, naiveOut.moeOut)
    let dH1 = maxDiff(bfA, sH1, naiveOut.h1)
    let dBlock = maxDiff(bfA, sBlockOut, naiveOut.blockOut)
    let dY = maxDiff(bfA, sY, naiveOut.y)
    echo &"[mega smoke] informational max abs diff vs naive " &
      &"moeOut {dMoe:.4f} h1 {dH1:.4f} blockOut {dBlock:.4f} y {dY:.4f}"

  # the relaunch, restored arenas, state and ring, zeroed counters
  for i in 0 ..< BfArenaLen: bfA.hostPtr[i] = bfSnap[i]
  for i in 0 ..< F32ArenaLen: f32A.hostPtr[i] = f32Snap[i]
  for i in 0 ..< NumVHeads * HeadVDim * HeadKDim: state.hostPtr[i] = stateSnap[i]
  for i in 0 ..< ConvDim * RingWidth: ring.hostPtr[i] = ringSnap[i]
  discard launch()
  waveSyncCheck()
  for i in 0 ..< BfArenaLen:
    doAssert bfA.hostPtr[i] == bfSnap[i], &"bf arena differs at {i}"
  for i in 0 ..< F32ArenaLen:
    doAssert f32A.hostPtr[i] == f32Snap[i], &"f32 arena differs at {i}"
  for i in 0 ..< NumVHeads * HeadVDim * HeadKDim:
    doAssert state.hostPtr[i] == statePost[i], &"state differs at {i}"
  for i in 0 ..< ConvDim * RingWidth:
    doAssert ring.hostPtr[i] == ringPost[i], &"ring differs at {i}"
  echo "[mega smoke] relaunch bit-identical, wave sync exact"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(MegaGdnMsl)
  let t0 = epochTime()
  smokeChecks(engine, buildBigHost(Seed))
  echo &"[mega smoke] wall clock {epochTime() - t0:.2f} s"
  echo "CERAMIC MEGA GDN SMOKE VERDICT: one launch, wave sync, sentinels, determinism"

main()
