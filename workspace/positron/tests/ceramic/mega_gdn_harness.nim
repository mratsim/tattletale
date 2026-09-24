# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared host harness for the mega-GDN suites.
##
## One kernel binding, three suite members:
##
## - fp16, the full walk stored in fp16 bits
## - launch contract, the bounded-wait discipline
## - mixer, the stage-internal composition
##
## All three share this module's 25-buffer page set, the fill
## chorus and the launch-call pointer chorus.
##
## - the suites keep their scenario bodies (checks, records, verdicts)
## - this module holds the shared buffer set and launch plumbing
## - printed output stays the suites' own, nothing here echoes

import std/[strformat]
import workspace/crucible
import ../../src/mega_kernels/decode_layers/gdn_moe_decode_megakernel
import ../naive/naive_rng
import ceramic_pagebuf

const NumCounters = 13

# ─── Seeded host inputs, the generation recipe ────────────────────────

type BigHost* = object
  ## Seeded layer pass inputs and weights, bit patterns both the mega
  ## and the naive side widen exactly to fp32.
  x*, r*: seq[uint16]
  norm1W*, qkvW*, zW*, aW*, bW*, convW*, onormW*, outprojW*, norm2W*: seq[uint16]
  routerW*, gateUpW*, downW*, sharedGW*, sharedUW*, sharedDW*, sharedGVW*: seq[uint16]
  aLog*: seq[float32]
  dtBias*: seq[uint16]
  state*: seq[float32]
  ring*: seq[uint16]

proc randBits*(rng: var NaiveRng; n: int; lo, hi: float32;
    toBits: proc(x: float32): uint16 {.nimcall.}): seq[uint16] =
  ## `n` bit patterns of uniform samples in [lo, hi], the dtype's rounding.
  result = newSeq[uint16](n)
  for i in 0 ..< n:
    result[i] = toBits(rng.nextF32(lo, hi))

proc buildBigHost*(seed: uint64;
    toBits: proc(x: float32): uint16 {.nimcall.}): BigHost =
  ## Seeded inputs and weights at the Qwen decode class geometry, modest
  ## magnitudes so no stage saturates, the dtype's rounding applied.
  var rng = initNaiveRng(seed)
  result.x = randBits(rng, Hidden, -1.0'f32, 1.0'f32, toBits)
  result.r = randBits(rng, Hidden, -1.0'f32, 1.0'f32, toBits)
  result.norm1W = randBits(rng, Hidden, -0.05'f32, 0.05'f32, toBits)
  result.qkvW = randBits(rng, ConvDim * Hidden, -0.02'f32, 0.02'f32, toBits)
  result.zW = randBits(rng, NumVHeads * HeadVDim * Hidden, -0.02'f32, 0.02'f32, toBits)
  result.aW = randBits(rng, NumVHeads * Hidden, -0.02'f32, 0.02'f32, toBits)
  result.bW = randBits(rng, NumVHeads * Hidden, -0.02'f32, 0.02'f32, toBits)
  result.convW = randBits(rng, ConvDim * ConvKernel, -0.1'f32, 0.1'f32, toBits)
  result.onormW = randBits(rng, NumVHeads * HeadVDim, -0.05'f32, 0.05'f32, toBits)
  result.outprojW = randBits(rng, Hidden * NumVHeads * HeadVDim, -0.02'f32, 0.02'f32, toBits)
  result.norm2W = randBits(rng, Hidden, -0.05'f32, 0.05'f32, toBits)
  result.routerW = randBits(rng, NumExperts * Hidden, -0.02'f32, 0.02'f32, toBits)
  result.gateUpW = randBits(rng, NumExperts * 2 * Inter * Hidden, -0.02'f32, 0.02'f32, toBits)
  result.downW = randBits(rng, NumExperts * Hidden * Inter, -0.02'f32, 0.02'f32, toBits)
  result.sharedGW = randBits(rng, Inter * Hidden, -0.02'f32, 0.02'f32, toBits)
  result.sharedUW = randBits(rng, Inter * Hidden, -0.02'f32, 0.02'f32, toBits)
  result.sharedDW = randBits(rng, Hidden * Inter, -0.02'f32, 0.02'f32, toBits)
  result.sharedGVW = randBits(rng, Hidden, -0.02'f32, 0.02'f32, toBits)
  result.aLog = newSeq[float32](NumVHeads)
  for h in 0 ..< NumVHeads:
    result.aLog[h] = rng.nextF32(-2.0'f32, -0.1'f32)
  result.dtBias = randBits(rng, NumVHeads, -0.1'f32, 0.1'f32, toBits)
  result.state = newSeq[float32](NumVHeads * HeadVDim * HeadKDim)
  for i in 0 ..< NumVHeads * HeadVDim * HeadKDim:
    result.state[i] = rng.nextF32(-0.5'f32, 0.5'f32)
  result.ring = randBits(rng, ConvDim * RingWidth, -1.0'f32, 1.0'f32, toBits)

# ─── Page-buffer fills, the no-copy extents ───────────────────────────

proc fillBf*(buf: var PageBuf[uint16], src: seq[uint16]) =
  ## Copies bit patterns into a page buffer.
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

# ─── The buffer set, one launch's page allocations ────────────────────

type MegaGdnBufs* = object
  ## One launch's page buffers and launch pointers, allocated once.
  counters*: PageBuf[uint32]
  bfA*: PageBuf[uint16]
  f32A*: PageBuf[float32]
  xPrev*, rPrev*: PageBuf[uint16]
  state*: PageBuf[float32]
  ring*: PageBuf[uint16]
  norm1W*, qkvW*, zW*, aW*, bW*, convW*, onormW*, outprojW*, norm2W*: PageBuf[uint16]
  routerW*, gateUpW*, downW*, sharedGW*, sharedUW*, sharedDW*, sharedGVW*:
    PageBuf[uint16]
  aLog*: PageBuf[float32]
  dtBias*: PageBuf[uint16]

proc allocMegaGdn*(): MegaGdnBufs =
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

proc freeMegaGdn*(m: var MegaGdnBufs) =
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

proc fillGdnInputs*(m: var MegaGdnBufs; big: BigHost) =
  ## One seeded layer pass's inputs and weights, written once.
  fillBf(m.xPrev, big.x)
  fillBf(m.rPrev, big.r)
  fillBf(m.norm1W, big.norm1W)
  fillBf(m.qkvW, big.qkvW)
  fillBf(m.zW, big.zW)
  fillBf(m.aW, big.aW)
  fillBf(m.bW, big.bW)
  fillBf(m.convW, big.convW)
  fillBf(m.onormW, big.onormW)
  fillBf(m.outprojW, big.outprojW)
  fillBf(m.norm2W, big.norm2W)
  fillBf(m.routerW, big.routerW)
  fillBf(m.gateUpW, big.gateUpW)
  fillBf(m.downW, big.downW)
  fillBf(m.sharedGW, big.sharedGW)
  fillBf(m.sharedUW, big.sharedUW)
  fillBf(m.sharedDW, big.sharedDW)
  fillBf(m.sharedGVW, big.sharedGVW)
  fillBf(m.dtBias, big.dtBias)
  fillF32(m.aLog, big.aLog)
  fillF32(m.state, big.state)
  fillBf(m.ring, big.ring)

# ─── The launch call, the shared binding's pointer chorus ─────────────

proc runMegaGdn*(engine: HwEngine; m: var MegaGdnBufs;
    kernelName: string; gridThreads: int; eps: float32) =
  ## One direct launch of the shared GDN binding:
  ##
  ## - the 25 page pointers and the epsilon go in the kernel's argument order
  ## - the bounded wait stays the caller's, the suites' launch scenarios
  ##   need their own deadlines and expiry callbacks
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
  proc launch() {.gcsafe.} =
    engine.run << (grid: (gridThreads, 1, 1), blk: (32, 1, 1)) >>
      (kernelName, cPA,
        (bfAPA, f32APA, xPA, rPA, stPA, rgPA, n1PA, qkvPA, zPA, aPA,
         bPA, cvPA, onPA, opPA, n2PA, rtPA, guPA, dnPA,
         sgPA, suPA, sdPA, gvPA, alPA, dbPA, eps))
  launch()
