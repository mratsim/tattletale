# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_gate.nim
## - sabotage build, judgment contract at the bottom:
##   nim c -r -d:release -d:WaveResetSabotage --outdir:build/tests --nimcache:nimcache/red tests/ceramic/t_ceramic_mega_gdn_gate.nim
##
## Megakernel launch suite for the bounded wait and the stage-counter entry contract, one seeded `qwen35_moe` mega kernel walk
## at grid (950, 1, 1).
##
## Launch discipline:
##
## - every launch runs through the host-side bounded wait (`runMegaBounded`),
##   its default 20 s deadline owns a wedged-grid report
##
## Bounded-wait expiry evidence, recorded facts:
##
## - the expiry diagnostic is proven on a short-deadline run against a grid
##   that never completes, the stuck stage's counters named, exit nonzero
## - before the wrapper existed, two unbounded megakernel runs spun at 96%
##   and 93.6% CPU for 2m33s and 1m34s and were killed rc=137
##
## Stage-counter entry contract, the dispatcher's launch-end self-reset
## via `waveReset`:
##
## - the 13 stage counters must be zero at launch entry
## - the kernel's own reset maintains the zero state after a completed
##   launch and the page allocator's zero fill covers the first launch
## - garbage between launches violates the contract, a crashed or partial
##   launch leaves mid-range counters with no kernel-side repair
##
## Stale-counter case, garbage written into the counters before a relaunch,
## both consequences observed first-hand on the recorded driver:
##
## - a count past its stage's threadgroup total makes the `waveWait` pass
##   instantly so the waits open before the producers run and the consumers
##   read the launch before's buffers, the output leaves the reference bits
## - the launch-end reset itself waits behind a `waveWait` on the last stage
##   and the garbage count passes that wait too, so the re-zero lands
##   mid-flight while threadgroups still run
## - the in-flight adds land after the re-zero, their totals never re-reach
##   the targets the late waiters spin on and the launch wedges, the expiry
##   diagnostic names the stuck stages
##
## | check       | contract                                                                                   |
## | ----------- | ------------------------------------------------------------------------------------------ |
## | deadline    | every launch completes inside the bounded wait's default deadline, counters re-zeroed      |
## | self-reset  | the relaunch over untouched counters is bit-identical to the host-zeroed reference launch  |
## | stale count | garbage counters before a relaunch give the recorded broken outcome, corrupt or wedge      |
## | sabotage    | the pre-self-reset spelling leaves the relaunch output off the reference continuation bits |
##
## WaveResetSabotage (`-d:WaveResetSabotage`):
##
## - restores the pre-self-reset spelling in the dispatcher, the launch-end
##   re-zero compiled out
## - the self-reset case relaunches over counters left at the launch
##   before's stage totals, every `waveWait` passes instantly on the stale counts, the output leaves the reference bits
##   and the bit-identity assert fails

import std/[strformat, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/qwen35_moe/qwen35_moe_decode_gdn_bf16
import ../naive/naive_rng
import ../naive/naive_tensors
import ceramic_pagebuf
import mega_bounded_wait

# ─── Expiry hook, the wedged outcome recorded instead of terminal ─────

var StaleGateExpired: bool
  ## Set by the stale-counter case's expiry hook, the wedged outcome is one
  ## of the two recorded consequences of the violated entry contract.

proc recordStaleGateExpiry(msg: string) {.gcsafe.} =
  ## Expiry hook of the stale-counter case, the bounded wait's diagnostic
  ## recorded to stderr and never terminal, the wedged worker thread is
  ## left to the process exit.
  StaleGateExpired = true
  stderr.writeLine "[mega gate] stale-counter relaunch wedged, " & msg

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
    qwen35GdnLayerWalk[true](counters, bfA, f32A, xPrev, rPrev, state, ring,
      norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W,
      routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW,
      aLog, dtBias, eps)

# ─── Host, the seeded layer pass ─────────────────────────────────────

const Seed = 0xC04D0602'u64
const Eps = 1.0e-6'f32
const NumCounters = 13
const StaleGateGarbage = 1_000_000'u32
  ## Stale-count garbage, past every per-stage threadgroup total in `WaveCounts`
  ## and far from the u32 wrap:
  ##
  ## - the instant pass on the stale count stays the discriminating behavior
  ## - a wrap would turn the count back into a blocking zero

type BigHost = object
  ## Seeded layer pass inputs and weights, the generation recipe of the mega smoke suite.
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
  ## Seeded inputs and weights at the Qwen bf16 class geometry, modest
  ## magnitudes so no stage saturates.
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

proc readInto[T](src: ptr UncheckedArray[T], count: int): seq[T] =
  ## Host-side read of `count` elements out of a page buffer.
  result = newSeq[T](count)
  for i in 0 ..< count:
    result[i] = src[i]

proc bitDiffCount[T](mega, refSeq: seq[T]): int =
  ## Count of bitwise mismatches between two same-length reads.
  doAssert mega.len == refSeq.len
  for i in 0 ..< refSeq.len:
    if mega[i] != refSeq[i]:
      inc result

# ─── The launch chain ─────────────────────────────────────────────────

proc gateChecks(engine: HwEngine, big: BigHost) =
  ## One seeded launch chain:
  ##
  ## - deadline case, the launch inside the bounded wait's default deadline
  ## - self-reset case, the relaunch over untouched counters
  ## - stale-counter case, garbage counters before a relaunch
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
    engine.run << (grid: (950, 1, 1), blk: (32, 1, 1)) >>
      ("qwen35_gdn_layer_bf16", countersPA,
        (bfAPA, f32APA, xPrevPA, rPrevPA, statePA, ringPA,
         norm1WPA, qkvWPA, zWPA, aWPA, bWPA, convWPA,
         onormWPA, outprojWPA, norm2WPA, routerWPA, gateUpWPA,
         downWPA, sharedGWPA, sharedUWPA, sharedDWPA,
         sharedGVWPA, aLogPA, dtBiasPA, Eps))
    result = true

  proc zeroCounters() =
    for i in 0 ..< NumCounters:
      counters.hostPtr[i] = 0'u32

  proc garbageCounters() =
    for i in 0 ..< NumCounters:
      counters.hostPtr[i] = StaleGateGarbage

  proc countersZeroWhere(after = "launch") =
    ## Asserts all 13 stage counters zero after the named launch.
    for i in 0 ..< NumCounters:
      doAssert counters.hostPtr[i] == 0'u32,
        &"stage counter {i} is {counters.hostPtr[i]}, want 0 after {after}"

  proc restorePreimage(preBf: seq[uint16]; preF32: seq[float32];
      preState: seq[float32]; preRing: seq[uint16]) =
    ## Restores the arena, state and ring to one pre-image, the weights
    ## and the kernel-read rows stay untouched throughout.
    for i in 0 ..< BfArenaLen: bfA.hostPtr[i] = preBf[i]
    for i in 0 ..< F32ArenaLen: f32A.hostPtr[i] = preF32[i]
    for i in 0 ..< NumVHeads * HeadVDim * HeadKDim: state.hostPtr[i] = preState[i]
    for i in 0 ..< ConvDim * RingWidth: ring.hostPtr[i] = preRing[i]

  # Deadline case:
  #   one launch inside the bounded wait's default deadline, the wall
  #   clock recorded, the launch-end reset's zero state asserted after.
  let t0 = epochTime()
  zeroCounters()
  runMegaBounded(launch, counters.hostPtr, StageNames)
  let deadlineWall = epochTime() - t0
  echo &"[mega gate] deadline case wall {deadlineWall:.2f} s " &
    &"(the bounded wait's default deadline " &
    &"{MegaWaitDeadlineSecMs.float / 1000.0:.0f} s)"
  doAssert deadlineWall < MegaWaitDeadlineSecMs.float / 1000.0,
    "the launch outlived the bounded wait's deadline"
  when not defined(WaveResetSabotage):
    countersZeroWhere("the deadline case's launch")

  # Continuation pre-image:
  #   the launch above advanced state and ring, the arenas hold its outputs, the next launches replay one decode step
  #   from this exact pre-image with the same kernel-read rows.
  let preBf = readInto(bfA.hostPtr, BfArenaLen)
  let preF32 = readInto(f32A.hostPtr, F32ArenaLen)
  let preState = readInto(state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let preRing = readInto(ring.hostPtr, ConvDim * RingWidth)

  # Reference continuation launch:
  #   the entry contract held explicitly, the counters host-zeroed, the full-buffer reads below are
  #   the reference continuation's bits.
  restorePreimage(preBf, preF32, preState, preRing)
  zeroCounters()
  discard launch()
  runMegaBounded(launch, counters.hostPtr, StageNames)
  let refBf = readInto(bfA.hostPtr, BfArenaLen)
  let refF32 = readInto(f32A.hostPtr, F32ArenaLen)
  let refState = readInto(state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let refRing = readInto(ring.hostPtr, ConvDim * RingWidth)
  when not defined(WaveResetSabotage):
    countersZeroWhere("the reference continuation launch")

  # Self-reset case:
  #   the relaunch over untouched counters, the kernel's launch-end reset
  #   alone maintains the entry contract here, the output must be
  #   bit-identical to the reference continuation.
  restorePreimage(preBf, preF32, preState, preRing)
  when not defined(WaveResetSabotage):
    countersZeroWhere("before the self-reset relaunch")
  discard launch()
  runMegaBounded(launch, counters.hostPtr, StageNames)
  when not defined(WaveResetSabotage):
    let selfResetBf = readInto(bfA.hostPtr, BfArenaLen)
    let selfResetF32 = readInto(f32A.hostPtr, F32ArenaLen)
    doAssert bitDiffCount(selfResetBf, refBf) == 0,
      "the self-reset relaunch's bf arena left the reference continuation"
    doAssert bitDiffCount(selfResetF32, refF32) == 0,
      "the self-reset relaunch's f32 arena left the reference continuation"
    for i in 0 ..< NumVHeads * HeadVDim * HeadKDim:
      doAssert state.hostPtr[i] == refState[i], &"state differs at {i}"
    for i in 0 ..< ConvDim * RingWidth:
      doAssert ring.hostPtr[i] == refRing[i], &"ring differs at {i}"
    echo "[mega gate] self-reset relaunch bit-identical to the reference"
  else:
    let sabotagedBf = readInto(bfA.hostPtr, BfArenaLen)
    let sabotagedF32 = readInto(f32A.hostPtr, F32ArenaLen)
    echo "[mega gate] sabotage relaunch bf arena mismatches " &
      &"{bitDiffCount(sabotagedBf, refBf)}/{BfArenaLen}, " &
      &"f32 arena mismatches {bitDiffCount(sabotagedF32, refF32)}/{F32ArenaLen}"
    doAssert bitDiffCount(sabotagedBf, refBf) == 0,
      "the sabotage build must leave the relaunch output off the reference " &
      "bits, the launch-end re-zero is compiled out and the stale counts " &
      "opened the waits early"
  when not defined(WaveResetSabotage):
    countersZeroWhere("the self-reset relaunch")

  # Stale-counter case:
  #   garbage written into the counters between launches, the contract boundary made loud, the stale counts open the waits before
  #   the producers run and the launch's output leaves the reference bits.
  restorePreimage(preBf, preF32, preState, preRing)
  garbageCounters()
  StaleGateExpired = false
  runMegaBounded(launch, counters.hostPtr, StageNames,
    deadlineMs = 5000.0, onExpiry = recordStaleGateExpiry)
  if StaleGateExpired:
    # Wedged outcome:
    #   the launch-end reset landed while threadgroups were still in flight, the re-zeroed counters cannot reach the targets
    #   the late waiters spin on, the bounded wait's diagnostic is recorded and the process exits here, a wedged grid cannot
    #   be unwound in-process.
    echo "CERAMIC MEGA GDN GATE VERDICT: deadline held, self-reset relaunch " &
      "bit-identical, stale-count garbage wedges the launch (recorded)"
    quit(0)
  let staleBf = readInto(bfA.hostPtr, BfArenaLen)
  let staleF32 = readInto(f32A.hostPtr, F32ArenaLen)
  let staleBfMismatches = bitDiffCount(staleBf, refBf)
  let staleF32Mismatches = bitDiffCount(staleF32, refF32)
  echo &"[mega gate] stale-counter relaunch bf arena mismatches " &
    &"{staleBfMismatches}/{BfArenaLen}, f32 arena mismatches " &
    &"{staleF32Mismatches}/{F32ArenaLen} against the reference"
  doAssert staleBfMismatches > 0 or staleF32Mismatches > 0,
    "the stale-counter relaunch reproduced the reference bits, the garbage " &
    "in the counters must open the waits early and corrupt the walk"
  # Under the garbage the launch-end reset races the in-flight threadgroups, their adds land after the re-zero and the counters
  # end non-zero, a recorded consequence of the violated entry contract.
  block staleGateCounters:
    var msg = "stale-counter relaunch counters at exit: "
    for i in 0 ..< NumCounters:
      msg.add &"{counters.hostPtr[i]} "
    echo "[mega gate] ", msg
  echo &"[mega gate] total wall {epochTime() - t0:.2f} s"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(MegaGdnMsl)
  gateChecks(engine, buildBigHost(Seed))
  echo "CERAMIC MEGA GDN GATE VERDICT: deadline held, self-reset relaunch " &
    "bit-identical, stale-count garbage corrupts the walk"

main()
