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
## One-launch fused GDN decoder layer smoke, the fp16 family row.
## Same launch shape as the bf16 row, `t_ceramic_mega_gdn_smoke.nim`.
## The same seeded generation recipe stored as fp16 bit patterns.
##
## The mega kernel's fp16 instantiation against the fp16 naive chain:
##
## - launch success under the bounded wait, non-degenerate outputs, sync counters
## - read-unchanged sentinels, fresh-relaunch bit-identity
## - an informational fp16 naive-vs-mega diff over the shared outputs
##
## The band model lives in the comparison tier, `ceramic_mega_gdn_composition.nim`,
## the bf16 rows. This suite asserts a generous sanity bound only.
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
import ceramic_fam
import mega_bounded_wait

const GridThreads: int = block:
  ## One launch's threadgroup count, the 13 stage blocks summed.
  var t: int = 0
  for c in WaveCounts:
    t += c.int
  t

# ─── Device entry, one launch of the mega's fp16 instantiation ────────

const MegaGdnFp16Msl = metal:
  proc gdn_moe_layer_fp16(
      counters: ptr UncheckedArray[uint32],
      bfA: ptr UncheckedArray[float16],
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
    gdnMoeLayerWalk[float16, true](counters, bfA, f32A, xPrev, rPrev, state,
      ring, norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W,
      routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW,
      aLog, dtBias, eps)

# ─── Host, the seeded layer pass in fp16 bits ────────────────────────

const Seed = 0xC04D0602'u64
const Eps = 1.0e-6'f32
const NumCounters = 13

type BigHost = object
  ## Seeded layer pass inputs and weights, fp16 bit patterns shared
  ## with the naive side through the exact fp32 widenings.
  x, r: seq[uint16]
  norm1W, qkvW, zW, aW, bW, convW, onormW, outprojW, norm2W: seq[uint16]
  routerW, gateUpW, downW, sharedGW, sharedUW, sharedDW, sharedGVW: seq[uint16]
  aLog: seq[float32]
  dtBias: seq[uint16]
  state: seq[float32]
  ring: seq[uint16]

proc randBits(rng: var NaiveRng; n: int; lo, hi: float32): seq[uint16] =
  ## `n` fp16 bit patterns of uniform samples in [lo, hi].
  result = newSeq[uint16](n)
  for i in 0 ..< n:
    result[i] = fp32ToFp16(rng.nextF32(lo, hi))

proc buildBigHost(seed: uint64): BigHost =
  ## Seeded inputs and weights at the same generation recipe as the bf16
  ## smoke (modest magnitudes so no stage saturates), stored as fp16 bits.
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
  ## Copies family-dtype bit patterns into a page buffer.
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

proc famRangeMax(buf: PageBuf[uint16]; off, count: int): float32 =
  ## Widened magnitude maximum over one fp16 arena section.
  for i in off ..< off + count:
    result = max(result, abs(fp16ToFp32(buf.hostPtr[i])))

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
    engine.run << (grid: (GridThreads, 1, 1), blk: (32, 1, 1)) >>
      ("gdn_moe_layer_fp16", countersPA,
        (bfAPA, f32APA, xPrevPA, rPrevPA, statePA, ringPA,
         norm1WPA, qkvWPA, zWPA, aWPA, bWPA, convWPA,
         onormWPA, outprojWPA, norm2WPA, routerWPA, gateUpWPA,
         downWPA, sharedGWPA, sharedUWPA, sharedDWPA,
         sharedGVWPA, aLogPA, dtBiasPA, Eps))
    result = true
  runMegaBounded(launch, counters.hostPtr, StageNames)

  let stateSnap = readInto(state.hostPtr, NumVHeads * HeadVDim * HeadKDim)
  let ringSnap = readInto(ring.hostPtr, ConvDim * RingWidth)
  discard launch()

  proc waveSyncCheck() =
    ## Post-launch, the kernel's launch-end reset has re-zeroed the counters.
    for i in 0 ..< NumCounters:
      doAssert counters.hostPtr[i] == 0'u32,
        &"stage counter {i} {counters.hostPtr[i]} want 0 (the launch-end reset)"

  proc outputRanges() =
    let moeMax = famRangeMax(bfA, sMoeOut, Hidden)
    let h1Max = famRangeMax(bfA, sH1, Hidden)
    let blockMax = famRangeMax(bfA, sBlockOut, Hidden)
    let yMax = famRangeMax(bfA, sY, NumVHeads * HeadVDim)
    echo &"[mega fp16] output maxima moeOut {moeMax:.4f} h1 {h1Max:.4f} " &
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
    let dMoe = maxDiffF16(bfA, sMoeOut, naiveOut.moeOut)
    let dH1 = maxDiffF16(bfA, sH1, naiveOut.h1)
    let dBlock = maxDiffF16(bfA, sBlockOut, naiveOut.blockOut)
    let dY = maxDiffF16(bfA, sY, naiveOut.y)
    var yArg = 0
    for i in 0 ..< naiveOut.y.len:
      if abs(fp16ToFp32(bfA.hostPtr[sY + i]) - fp16ToFp32(naiveOut.y[i])) >
          abs(fp16ToFp32(bfA.hostPtr[sY + yArg]) - fp16ToFp32(naiveOut.y[yArg])):
        yArg = i
    echo &"[mega fp16] y argmax {yArg} mega {fp16ToFp32(bfA.hostPtr[sY + yArg]):.6f} " &
      &"naive {fp16ToFp32(naiveOut.y[yArg]):.6f}"
    echo &"[mega fp16] informational max abs diff vs naive " &
      &"moeOut {dMoe:.5f} h1 {dH1:.5f} blockOut {dBlock:.5f} y {dY:.5f}"
    doAssert dMoe < 0.1'f32, "moeOut outside the sanity bound"
    doAssert dH1 < 0.1'f32, "h1 outside the sanity bound"
    doAssert dBlock < 0.1'f32, "blockOut outside the sanity bound"
    doAssert dY < 0.1'f32, "y outside the sanity bound"

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
  echo "[mega fp16] relaunch bit-identical, wave sync exact"

proc main =
  echo "device: ", bkMetal.init().deviceName()
  var engine = bkMetal.init()
  engine.ingest(MegaGdnFp16Msl)
  let t0 = epochTime()
  fp16Checks(engine, buildBigHost(Seed))
  echo &"[mega fp16] wall clock {epochTime() - t0:.2f} s"
  echo "CERAMIC MEGA GDN FP16 VERDICT: one fp16 launch, wave sync, sentinels, determinism, naive-chain row"

main()
