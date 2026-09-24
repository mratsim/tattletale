# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_fp16.nim
##
## One-launch fused GDN decoder layer check, the fp16 row.
## The same seeded generation recipe stored as fp16 bit patterns.
##
## The mega kernel's fp16 instantiation against the fp16 naive chain:
##
## - launch success under the bounded wait, non-degenerate outputs, sync counters
## - read-unchanged sentinels, fresh-relaunch bit-identity
## - an informational fp16 naive-vs-mega diff over the shared outputs
##
## This suite asserts generous sanity bounds only, the bf16 row's class.
##
## The fp16 chain's per-op round keeps its reassociation
## and transcendental differences within the bf16 row's class.

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/gdn_moe_decode_megakernel
import ../naive/naive_rng
import ../naive/naive_tensors
from ../naive/naive_qwen35_layer import naiveQwen35GdnLayer, LayerOut
from ../naive/naive_grouped_mm import GmmFamily, gmmF16
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

proc yWorstDiff(megaBf: PageBuf[uint16]; megaOff: int; naive: seq[uint16]): string =
  ## Failure diagnostic of the y sanity bound, computed only on failure:
  ## the y row's worst element against the naive row, widened from fp16.
  var yArg = 0
  for i in 0 ..< naive.len:
    if abs(fp16ToFp32(megaBf.hostPtr[megaOff + i]) - fp16ToFp32(naive[i])) >
        abs(fp16ToFp32(megaBf.hostPtr[megaOff + yArg]) - fp16ToFp32(naive[yArg])):
      yArg = i
  &"element {yArg}, mega {fp16ToFp32(megaBf.hostPtr[megaOff + yArg]):.6f}, " &
    &"naive {fp16ToFp32(naive[yArg]):.6f}, worst " &
    &"{abs(fp16ToFp32(megaBf.hostPtr[megaOff + yArg]) - fp16ToFp32(naive[yArg])):.5f}"

proc maxDiffF16(megaBf: PageBuf[uint16]; megaOff: int; naive: seq[uint16]): float32 =
  ## Elementwise absolute difference maximum between a mega arena section
  ## and the naive side's row, both widened from fp16.
  doAssert megaBf.elems >= megaOff + naive.len
  for i in 0 ..< naive.len:
    let a = fp16ToFp32(megaBf.hostPtr[megaOff + i])
    let b = fp16ToFp32(naive[i])
    result = max(result, abs(a - b))

proc fp16Checks(engine: HwEngine, big: BigHost) =
  ## One seeded fp16 launch, the sync, sentinel, determinism
  ## and naive-comparison checks around it.
  var m = allocMegaGdn()
  defer: freeMegaGdn(m)
  fillGdnInputs(m, big)

  proc launch() {.gcsafe.} =
    runMegaGdn(engine, m, "gdn_moe_layer_fp16", GridThreads, Eps)
  runMegaBounded(launch, m.counters.hostPtr, StageNames)

  let stateSnap = readRecord(m.state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let ringSnap = readRecord(m.ring.hostPtr, ConvDim * RingWidth)
  launch()

  var worstSanityUse = 0.0'f64

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

  block comparison:
    var stateN = NaiveCube[float32](planes: NumVHeads, rows: HeadVDim, cols: HeadKDim)
    stateN.data = big.state
    var ringN = big.ring
    let t0 = epochTime()
    let naiveOut = naiveQwen35GdnLayer(stateN, ringN, big.x, big.r,
      big.norm1W, big.qkvW, big.zW, big.aW, big.bW, big.convW, big.onormW,
      big.outprojW, big.norm2W, big.routerW, big.gateUpW, big.downW,
      big.sharedGW, big.sharedUW, big.sharedDW, big.sharedGVW, big.aLog,
      big.dtBias, Eps, gmmF16)
    echo &"[mega fp16] naive walk {epochTime() - t0:.2f} s"
    let dMoe = maxDiffF16(m.bfA, sMoeOut, naiveOut.moeOut)
    let dH1 = maxDiffF16(m.bfA, sH1, naiveOut.h1)
    let dBlock = maxDiffF16(m.bfA, sBlockOut, naiveOut.blockOut)
    let dY = maxDiffF16(m.bfA, sY, naiveOut.y)
    worstSanityUse = max(max(max(dMoe, dH1), dBlock), dY) / 0.1
    echo &"[mega fp16] informational max abs diff vs naive " &
      &"moeOut {dMoe:.5f} h1 {dH1:.5f} blockOut {dBlock:.5f} y {dY:.5f}"
    doAssert dMoe < 0.1'f32,
      &"moeOut outside the sanity bound, worst diff {dMoe:.5f}"
    doAssert dH1 < 0.1'f32,
      &"h1 outside the sanity bound, worst diff {dH1:.5f}"
    doAssert dBlock < 0.1'f32,
      &"blockOut outside the sanity bound, worst diff {dBlock:.5f}"
    doAssert dY < 0.1'f32,
      &"y outside the sanity bound: the worst element {yWorstDiff(m.bfA, sY, naiveOut.y)}"

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
  echo &"CERAMIC MEGA GDN FP16 VERDICT: launches=2 worst sanity usage " &
    &"{worstSanityUse:.3f}, bit-exact relaunch " &
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
