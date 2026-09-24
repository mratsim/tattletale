# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_fp16.nim
##
## One-launch fused GDN decoder layer check, the fp16 row.
## The same seeded generation recipe stored as fp16 bit patterns.
##
## The mega kernel's fp16 instantiation over the full `HaveNorm = true` walk:
##
## - launch success under the bounded wait, non-degenerate outputs, sync counters
## - read-unchanged sentinels, fresh-relaunch bit-identity

import std/[strformat, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/gdn_moe_decode_megakernel
import ceramic_pagebuf
import ceramic_dtype
import mega_bounded_wait
import mega_gdn_harness

const GridThreads: int = block:
  ## One launch's threadgroup count, the 13 stage blocks summed.
  var threadCount: int = 0
  for c in WaveCounts:
    threadCount += c.int
  threadCount

# ─── Device entry, one launch of the mega's fp16 instantiation ────────

const MegaGdnFp16Msl = metal:
  proc gdn_moe_layer_fp16(
      counters: ptr UncheckedArray[uint32],
      fpA: ptr UncheckedArray[float16],
      f32A: ptr UncheckedArray[float32],
      xPrev, rPrev: ptr UncheckedArray[float16],
      state: ptr UncheckedArray[float32],
      ring: ptr UncheckedArray[float16],
      norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W,
      routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW:
        ptr UncheckedArray[float16],
      aLog: ptr UncheckedArray[float32],
      dtBias: ptr UncheckedArray[float16],
      eps: float32) {.global.} =
    gdnMoeLayerWalk[float16, true](counters, fpA, f32A, xPrev, rPrev, state,
      ring, norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W,
      routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW,
      aLog, dtBias, eps)

# ─── Host, the seeded layer pass in fp16 bits ────────────────────────

const Seed = 0xC04D0602'u64
const Eps = 1.0e-6'f32
const NumCounters = 13

proc fp16Bits(x: float32): uint16 {.nimcall.} =
  ## One fp32 sample's fp16 bit pattern, the shared recipe's rounding.
  fp32ToFp16(x)

proc fp16RangeMax(buf: PageBuf[uint16]; off, count: int): float32 =
  ## Widened magnitude maximum over one fp16 arena section.
  for i in off ..< off + count:
    result = max(result, abs(fp16ToFp32(buf.hostPtr[i])))

proc fp16Checks(engine: HwEngine, big: BigHost) =
  ## One seeded fp16 launch, the sync, sentinel and determinism checks
  ## around it, plus the relaunch over restored arenas, state and ring.
  var m = allocMegaGdn()
  defer: freeMegaGdn(m)
  fillGdnInputs(m, big)

  proc launch() {.gcsafe.} =
    runMegaGdn(engine, m, "gdn_moe_layer_fp16", GridThreads, Eps)
  runMegaBounded(launch, m.counters.hostPtr, stageNames)

  let stateSnap = readRecord(m.state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let ringSnap = readRecord(m.ring.hostPtr, ConvDim * RingWidth)
  launch()

  proc waveSyncCheck() =
    ## Post-launch, the kernel's launch-end reset has re-zeroed the counters.
    for i in 0 ..< NumCounters:
      doAssert m.counters.hostPtr[i] == 0'u32,
        &"stage counter {i} {m.counters.hostPtr[i]} want 0 (the launch-end reset)"

  proc outputRanges() =
    let moeMax = fp16RangeMax(m.bfA, sMoeOut, Hidden)
    let h1Max = fp16RangeMax(m.bfA, sH1, Hidden)
    let blockMax = fp16RangeMax(m.bfA, sBlockOut, Hidden)
    let yMax = fp16RangeMax(m.bfA, sY, NumVHeads * HeadVDim)
    echo &"[mega fp16] output maxima moeOut {moeMax:.4f} h1 {h1Max:.4f} " &
      &"blockOut {blockMax:.4f} y {yMax:.4f}"
    doAssert moeMax > 0.0'f32, "moeOut degenerate"
    doAssert h1Max > 0.0'f32, "h1 degenerate"
    doAssert blockMax > 0.0'f32, "blockOut degenerate"
    doAssert yMax > 0.0'f32, "y degenerate"

  proc sentinels() =
    assertTailZero(m.bfA, BfArenaLen)
    assertTailZero(m.f32A, F32ArenaLen)
    assertMegaInputsUnchanged(m, big)
    for h in 0 ..< NumVHeads:
      doAssert m.aLog.hostPtr[h] == big.aLog[h], "kernel-read buffer modified"
      doAssert m.dtBias.hostPtr[h] == big.dtBias[h], "kernel-read buffer modified"

  waveSyncCheck()
  outputRanges()
  sentinels()
  let fpSnap = readRecord(m.bfA.hostPtr, BfArenaLen)
  let f32Snap = readRecord(m.f32A.hostPtr, F32ArenaLen)
  let statePost = readRecord(m.state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let ringPost = readRecord(m.ring.hostPtr, ConvDim * RingWidth)

  # the relaunch, restored arenas, state and ring, zeroed m.counters
  for i in 0 ..< BfArenaLen: m.bfA.hostPtr[i] = fpSnap[i]
  for i in 0 ..< F32ArenaLen: m.f32A.hostPtr[i] = f32Snap[i]
  for i in 0 ..< NumVHeads * HeadVDim * HeadKDim: m.state.hostPtr[i] = stateSnap[i]
  for i in 0 ..< ConvDim * RingWidth: m.ring.hostPtr[i] = ringSnap[i]
  launch()
  waveSyncCheck()
  for i in 0 ..< BfArenaLen:
    doAssert m.bfA.hostPtr[i] == fpSnap[i], &"fp16 arena differs at {i}"
  for i in 0 ..< F32ArenaLen:
    doAssert m.f32A.hostPtr[i] == f32Snap[i], &"f32 arena differs at {i}"
  for i in 0 ..< NumVHeads * HeadVDim * HeadKDim:
    doAssert m.state.hostPtr[i] == statePost[i], &"state differs at {i}"
  for i in 0 ..< ConvDim * RingWidth:
    doAssert m.ring.hostPtr[i] == ringPost[i], &"ring differs at {i}"
  echo "[mega fp16] relaunch bit-identical, wave sync exact"
  echo &"CERAMIC MEGA GDN FP16 VERDICT: launches=2 " &
    &"bit-exact relaunch " &
    &"{fpSnap.len + f32Snap.len + statePost.len + ringPost.len}/" &
    &"{fpSnap.len + f32Snap.len + statePost.len + ringPost.len}"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(MegaGdnFp16Msl)
  let t0 = epochTime()
  fp16Checks(engine, buildBigHost(Seed, fp16Bits))
  echo &"[mega fp16] wall clock {epochTime() - t0:.2f} s"

main()
