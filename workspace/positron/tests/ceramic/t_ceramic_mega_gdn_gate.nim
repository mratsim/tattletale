# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
##   nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_gate.nim
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
## | check       | contract                                                                                  |
## | ----------- | ----------------------------------------------------------------------------------------- |
## | deadline    | every launch completes inside the bounded wait's default deadline, counters re-zeroed     |
## | self-reset  | the relaunch over untouched counters is bit-identical to the host-zeroed reference launch |
## | stale count | garbage counters before a relaunch give the recorded broken outcome, corrupt or wedge     |
##
## Self-reset regression guard:
##
## - the pre-self-reset spelling (the launch-end re-zero compiled out)
##   relaunched over counters left at the launch before's stage totals
## - every `waveWait` passed instantly on the stale counts, the output
##   left the reference bits, the bit-identity assert fired
## - the launch-end self-reset spelling stands, this case its regression guard,
##   the defect-run proof living in git history

import std/[strformat, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/gdn_moe_decode_megakernel
import ceramic_pagebuf
import mega_bounded_wait
import ceramic_dtype
import mega_gdn_harness

# ─── Expiry callback, the wedge is a recorded contract outcome ────────

proc recordStaleGateExpiry(msg: string) {.gcsafe.} =
  ## Expiry callback of the stale-counter case, the bounded wait's
  ## diagnostic recorded and the wedge accepted as the outcome it is.
  ##
  ## Contract:
  ## - under the stale counts, the race between the launch-end reset
  ##   and the in-flight threadgroups ends either corrupt or wedged,
  ##   both recorded in the case's header, neither is a kernel-behavior failure
  ## - a wedged grid cannot be unwound in-process, so the callback
  ##   quits the process after one stuck-stage diagnostic, no case code
  ##   runs behind the wedged worker (its engine dispatch owns the pages)
  stderr.writeLine "[mega gate] stale-counter relaunch wedged, " & msg
  quit(0)

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

const Seed = 0xC04D0602'u64
const Eps = 1.0e-6'f32
const NumCounters = 13
const StaleGateGarbage = 1_000_000'u32
  ## Stale-count garbage, past every per-stage threadgroup total in `WaveCounts`
  ## and far from the u32 wrap:
  ##
  ## - the instant pass on the stale count stays the discriminating behavior
  ## - a wrap would turn the count back into a blocking zero

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
  var m = allocMegaGdn()
  defer: freeMegaGdn(m)
  fillGdnInputs(m, big)

  proc launch() {.gcsafe.} =
    runMegaGdn(engine, m, "qwen35_gdn_layer_bf16", 950, Eps)

  proc zeroCounters() =
    for i in 0 ..< NumCounters:
      m.counters.hostPtr[i] = 0'u32

  proc garbageCounters() =
    for i in 0 ..< NumCounters:
      m.counters.hostPtr[i] = StaleGateGarbage

  proc countersZeroWhere(after = "launch") =
    ## Asserts all 13 stage counters zero after the named launch.
    for i in 0 ..< NumCounters:
      doAssert m.counters.hostPtr[i] == 0'u32,
        &"stage counter {i} is {m.counters.hostPtr[i]}, want 0 after {after}"

  proc bfRangeMax(buf: PageBuf[uint16]; off, count: int): float32 =
    ## Widened magnitude maximum over one bf16 arena section.
    for i in off ..< off + count:
      result = max(result, abs(bf16ToF32(buf.hostPtr[i])))

  proc outputRanges() =
    ## Four shared outputs, each written over a non-degenerate range.
    let moeMax = bfRangeMax(m.bfA, sMoeOut, Hidden)
    let h1Max = bfRangeMax(m.bfA, sH1, Hidden)
    let blockMax = bfRangeMax(m.bfA, sBlockOut, Hidden)
    let yMax = bfRangeMax(m.bfA, sY, NumVHeads * HeadVDim)
    doAssert moeMax > 0.0'f32, "moeOut degenerate"
    doAssert h1Max > 0.0'f32, "h1 degenerate"
    doAssert blockMax > 0.0'f32, "blockOut degenerate"
    doAssert yMax > 0.0'f32, "y degenerate"

  proc sentinels() =
    ## Kernel-written buffers stay in extent, the page tails keep
    ## the zero fill, the kernel-read buffers stay bit-identical
    assertTailZero(m.bfA, BfArenaLen)
    assertTailZero(m.f32A, F32ArenaLen)
    for i in NumCounters ..< m.counters.elems:
      doAssert m.counters.hostPtr[i] == 0'u32, "counters tail written"
    assertReadUnchanged(m.xPrev, big.x)
    assertReadUnchanged(m.rPrev, big.r)
    assertReadUnchanged(m.norm1W, big.norm1W)
    assertReadUnchanged(m.convW, big.convW)
    assertReadUnchanged(m.onormW, big.onormW)
    assertReadUnchanged(m.routerW, big.routerW)
    assertReadUnchanged(m.gateUpW, big.gateUpW)
    assertReadUnchanged(m.downW, big.downW)
    assertReadUnchanged(m.qkvW, big.qkvW)
    assertReadUnchanged(m.zW, big.zW)
    assertReadUnchanged(m.aW, big.aW)
    assertReadUnchanged(m.bW, big.bW)
    assertReadUnchanged(m.outprojW, big.outprojW)
    assertReadUnchanged(m.norm2W, big.norm2W)
    assertReadUnchanged(m.sharedGW, big.sharedGW)
    assertReadUnchanged(m.sharedUW, big.sharedUW)
    assertReadUnchanged(m.sharedDW, big.sharedDW)
    assertReadUnchanged(m.sharedGVW, big.sharedGVW)
    for h in 0 ..< NumVHeads:
      doAssert m.aLog.hostPtr[h] == big.aLog[h], "kernel-read buffer modified"
      doAssert m.dtBias.hostPtr[h] == big.dtBias[h], "kernel-read buffer modified"

  proc restorePreimage(preBf: seq[uint16]; preF32: seq[float32];
      preState: seq[float32]; preRing: seq[uint16]) =
    ## Restores the arena, state and ring to one pre-image, the weights
    ## and the kernel-read rows stay untouched throughout.
    for i in 0 ..< BfArenaLen: m.bfA.hostPtr[i] = preBf[i]
    for i in 0 ..< F32ArenaLen: m.f32A.hostPtr[i] = preF32[i]
    for i in 0 ..< NumVHeads * HeadVDim * HeadKDim: m.state.hostPtr[i] = preState[i]
    for i in 0 ..< ConvDim * RingWidth: m.ring.hostPtr[i] = preRing[i]

  # Deadline case:
  #   one launch inside the bounded wait's default deadline, the wall
  #   clock recorded, the launch-end reset's zero state asserted after.
  let t0 = epochTime()
  zeroCounters()
  runMegaBounded(launch, m.counters.hostPtr, stageNames)
  let deadlineWall = epochTime() - t0
  echo &"[mega gate] deadline case wall {deadlineWall:.2f} s " &
    &"(the bounded wait's default deadline " &
    &"{TTT_MegaWaitDeadlineSecMs.float / 1000.0:.0f} s)"
  countersZeroWhere("the deadline case's launch")
  outputRanges()
  sentinels()

  # Continuation pre-image:
  #   the launch above advanced state and ring, the arenas hold its outputs, the next launches replay one decode step
  #   from this exact pre-image with the same kernel-read rows.
  let preBf = readRecord(m.bfA.hostPtr, BfArenaLen)
  let preF32 = readRecord(m.f32A.hostPtr, F32ArenaLen)
  let preState = readRecord(m.state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let preRing = readRecord(m.ring.hostPtr, ConvDim * RingWidth)

  # Reference continuation launch:
  #   one bounded launch over the restored pre-image, the m.counters host-zeroed,
  #   the full-buffer reads below are the reference continuation's bits.
  restorePreimage(preBf, preF32, preState, preRing)
  zeroCounters()
  runMegaBounded(launch, m.counters.hostPtr, stageNames)
  let refBf = readRecord(m.bfA.hostPtr, BfArenaLen)
  let refF32 = readRecord(m.f32A.hostPtr, F32ArenaLen)
  let refState = readRecord(m.state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let refRing = readRecord(m.ring.hostPtr, ConvDim * RingWidth)
  countersZeroWhere("the reference continuation launch")

  # Self-reset case:
  #   one bounded relaunch over the restored pre-image, the kernel's
  #   launch-end reset alone maintains the entry contract here, the output
  #   must be bit-identical to the reference continuation.
  restorePreimage(preBf, preF32, preState, preRing)
  countersZeroWhere("before the self-reset relaunch")
  runMegaBounded(launch, m.counters.hostPtr, stageNames)
  let selfResetBf = readRecord(m.bfA.hostPtr, BfArenaLen)
  let selfResetF32 = readRecord(m.f32A.hostPtr, F32ArenaLen)
  doAssert bitDiffCount(selfResetBf, refBf) == 0,
    "the self-reset relaunch's bf arena left the reference continuation"
  doAssert bitDiffCount(selfResetF32, refF32) == 0,
    "the self-reset relaunch's f32 arena left the reference continuation"
  for i in 0 ..< NumVHeads * HeadVDim * HeadKDim:
    doAssert m.state.hostPtr[i] == refState[i], &"state differs at {i}"
  for i in 0 ..< ConvDim * RingWidth:
    doAssert m.ring.hostPtr[i] == refRing[i], &"ring differs at {i}"
  echo "[mega gate] self-reset relaunch bit-identical to the reference"
  countersZeroWhere("the self-reset relaunch")

  # Stale-counter case:
  #   garbage written into the m.counters between launches, the contract boundary made loud, the stale counts open the waits before
  #   the producers run and the launch's output leaves the reference bits.
  restorePreimage(preBf, preF32, preState, preRing)
  garbageCounters()
  runMegaBounded(launch, m.counters.hostPtr, stageNames,
    deadlineMs = 5000.0, onExpiry = recordStaleGateExpiry)
  let staleBf = readRecord(m.bfA.hostPtr, BfArenaLen)
  let staleF32 = readRecord(m.f32A.hostPtr, F32ArenaLen)
  let staleBfMismatches = bitDiffCount(staleBf, refBf)
  let staleF32Mismatches = bitDiffCount(staleF32, refF32)
  echo &"[mega gate] stale-counter relaunch bf arena mismatches " &
    &"{staleBfMismatches}/{BfArenaLen}, f32 arena mismatches " &
    &"{staleF32Mismatches}/{F32ArenaLen} against the reference"
  # A wedged run exits the process from the expiry callback, the code
  # behind this point only ever sees the corrupt outcome.
  doAssert staleBfMismatches > 0 or staleF32Mismatches > 0,
    "the stale-counter relaunch reproduced the reference bits, the garbage " &
    "in the counters must open the waits early and corrupt the walk or wedge"
  # Under the garbage the launch-end reset races the in-flight threadgroups, their adds land after the re-zero and the m.counters
  # end non-zero, a recorded consequence of the violated entry contract.
  block staleGateCounters:
    var msg = "stale-counter relaunch counters at exit: "
    for i in 0 ..< NumCounters:
      msg.add &"{m.counters.hostPtr[i]} "
    echo "[mega gate] ", msg
  echo &"[mega gate] total wall {epochTime() - t0:.2f} s"
  echo &"[mega gate] VERDICT: deadline wall {deadlineWall:.2f} s, " &
    &"self-reset bit-exact {BfArenaLen + F32ArenaLen}/" &
    &"{BfArenaLen + F32ArenaLen}, stale outcome corrupt " &
    "(the wedge branch exits the process at the expiry callback), " &
    "stale bf " & &"{staleBfMismatches}/{BfArenaLen} f32 {staleF32Mismatches}/{F32ArenaLen}"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(MegaGdnMsl)
  gateChecks(engine, buildBigHost(Seed, f32ToBf16))

main()
