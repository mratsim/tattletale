# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Jagged attention bf16 replay machinery shared by the drivers, per backend.
##
## Host-generated inputs, f64 reference over the same bf16-rounded values. The recorded fixtures stay at D = 64, page_size = 32, fp16.
## The Qwen3-0.6B axes (D = 128, page_size = 256, bfloat16) are new here.
##
## - runtime parts: ordinary procs, one copy total
## - the runner monomorphizes per scenario, all in one compile

import std/[os, strformat, random]
from std/math import sqrt, exp
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_tile_sdpa
import workspace/positron/src/kernels/k_jagged_attention
import ../positron_testutils

export tile_algebra, layout_algebra, tensors, int_tuples

# ─── Pass band, stated before measurement ─────────────────────────────
#
# Element band, per output element: |got - want| <= Rtol*|want| + Abstol
#
# bf16 band (rtol 1e-2, abstol 1e-2):
#
# Online softmax rounds the QK dot in f32 accumulation over 128 bf16 products, then the exp2 rescale chain and the PV reduction
# run in f32, with P downcast to bfloat16 before the PV mma. The dominant term is bf16 quantization of P (2^-9 relative),
# amplified by the softmax-weighted PV sum, plus one bf16 ulp (2^-9) on the output store.
#
# - the fp16 fixture band (rtol 1e-3, abstol 1e-2) covers the same chain
#   at fp16's 2^-11 quantization, bf16 is 8x coarser
# - rtol 1e-2 absorbs the accumulated f32 chain at bf16 coarseness
# - abstol 1e-2 covers outputs near zero
const
  Bf16Rtol* = 1e-2'f64
  Bf16Abstol* = 1e-2'f64

const
  JaggedAttnTile = 32
  ## 32x32x32 mma schedule tile, page_size 256 is a multiple of the 32-token
  ## KV tile so each KV tile lies inside one page
  PadTokens = 2
  ## zero sentinel tokens after the logical output, written entries there fail the replay

proc bf16AttnCombo*(o, q, kPool, vPool: distinct TensorView,
                    qOffsets: distinct TensorView,
                    kv: JaggedKv,
                    scale: float32, layer, H, Nkv: int,
                    PageSize, D: static int,
                    windowed: static bool = false,
                    windowSize: int = -2147483648) =
  ## Passes the QK mma schedule as the kernel's trailing static.
  ##
  ## Contract:
  ##
  ## - `windowed`/`windowSize` are passed explicitly: the kernel's `mmaQK`
  ##   has no default, an argument list relying on the two defaults above
  ##   it leaves the schedule unbound
  ## - one call-site line per binding so the engine's monomorphization key
  ##   separates the element dtypes
  jagged_attn_fwd(o, q, kPool, vPool, qOffsets, kv,
                  scale, layer, H, Nkv, PageSize, D,
                  windowed, windowSize,
                  tiledMma(float32, q.T, kPool.T, JaggedAttnTile, JaggedAttnTile, D))

proc rank1Host[T](t: typedesc[T], vals: openArray[int]): auto =
  ## Host staging of a rank-1 int32 metadata array.
  ##
  ## `makeHostTensor` rejects rank-1 shapes, its `originOffset` dies on the `(n,)` layout. The layout is built here instead.
  let layout = make_layout((vals.len,))
  result = HostTensor[T, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[T]](allocShared(vals.len * sizeof(T))),
    offset: 0, layout: layout)
  for i in 0 ..< vals.len:
    result.data[i] = T(vals[i])

type Bf16JaggedCase* = object
  ## One replay scenario's geometry and page table, host-authored.
  ##
  ## - qLens/kvLens: per segment, kvLen counts the segment's new tokens too
  ##   (prefill: kvLen == qLen at sequence start, decode: qLen == 1)
  ## - pageIds: flat page table, segment s's pages are
  ##   pageIds[kvPageBounds[s] ..< kvPageBounds[s + 1]]
  ## - numPages: pool depth, pages without an owner hold garbage
  qLens*: seq[int]
  kvLens*: seq[int]
  pageIds*: seq[int]
  kvPageBounds*: seq[int]
  numPages*: int
  numLayers*: int
  layer*: int

proc bf16JaggedReplayCase*(engine: MetalEngine, name: string, scenario: Bf16JaggedCase, gridX = 0, windowSize = 0) =
  ## Replays one host-authored scenario against an f64 host reference.
  ##
  ## - one launch over the production grid, per-element band judgment
  ## - a relaunch judged bit-identical on the raw output bits
  ## - gridX > 0 overrides the production x dim: an undersized grid drives
  ##   the kernel's grid-stride query-tile loop and its per-tile resets
  ## - windowSize > 0 launches the windowed variant
  ##
  ## Preconditions:
  ## - the output staging tensor includes `PadTokens` zero sentinel tokens
  ##   after num_qo_tokens, checked after the launch
  ## - covered pool tokens are random bf16, everything beyond a segment's
  ##   kvLen is NaN garbage: the kernel must bound at the covered tokens,
  ##   a 0·NaN in the P·V mma would poison the output
  const
    H = 16
    Nkv = 8
    D = 128
    PageSize = 256
  doAssert scenario.qLens.len == scenario.kvLens.len
  doAssert scenario.kvPageBounds.len == scenario.qLens.len + 1
  doAssert scenario.kvPageBounds[0] == 0
  doAssert scenario.numLayers >= scenario.layer + 1

  let numSeqs = scenario.qLens.len
  var numQoTokens = 0
  var qOffsets: seq[int] = @[0]
  for s in 0 ..< numSeqs:
    numQoTokens += scenario.qLens[s]
    qOffsets.add numQoTokens
  var numIndices = 0
  for s in 0 ..< numSeqs:
    numIndices += scenario.kvPageBounds[s + 1] - scenario.kvPageBounds[s]

  # capacity bounds the metadata arrays, one slot per page-table entry
  # (the fixture machinery's new(JaggedKvHost, capacity = numIndices)),
  # not the pool depth
  var kvHost = new(JaggedKvHost, capacity = scenario.pageIds.len)
  var pageIds32 = newSeq[int32](scenario.pageIds.len)
  for i in 0 ..< scenario.pageIds.len:
    pageIds32[i] = int32(scenario.pageIds[i])
  for s in 0 ..< numSeqs:
    kvHost.addSequence(tokens = scenario.kvLens[s],
      pageIds = pageIds32.toOpenArray(scenario.kvPageBounds[s],
                                      scenario.kvPageBounds[s + 1] - 1))
  doAssert kvHost.batchSize == numSeqs, name & ": kv batch size"

  # seeded inputs: q and the covered pool tokens random in [-1, 1],
  # every non-covered pool slot NaN garbage
  var rng = initRand(20261109)
  template randBf16(): bfloat16 =
    rng.rand(-1.0 .. 1.0).to(bfloat16)
  template nanBf16(): bfloat16 =
    cast[bfloat16](uint16(0x7FC0))

  var
    qStage = makeHostTensor(bfloat16, (numQoTokens, H, D), LayoutRight)
    kStage = makeHostTensor(bfloat16, (scenario.numPages, scenario.numLayers, PageSize, Nkv, D), LayoutRight)
    vStage = makeHostTensor(bfloat16, (scenario.numPages, scenario.numLayers, PageSize, Nkv, D), LayoutRight)
    oStage = makeHostTensor(bfloat16, (numQoTokens + PadTokens, H, D), LayoutRight)
  zeroMem(cast[pointer](oStage.data), oStage.len() * sizeof(bfloat16))
  for i in 0 ..< qStage.len():
    qStage.data[i] = randBf16()
  for i in 0 ..< kStage.len():
    kStage.data[i] = nanBf16()
    vStage.data[i] = nanBf16()
  # covered tokens: segment s's token t lives at pool position
  # pageIds[bound + t div PageSize] * PageSize + t mod PageSize, one layer
  for s in 0 ..< numSeqs:
    for t in 0 ..< scenario.kvLens[s]:
      let
        pageIdx = scenario.kvPageBounds[s] + t div PageSize
        poolPos = (scenario.pageIds[pageIdx] * scenario.numLayers + scenario.layer) *
                  PageSize + t mod PageSize
        covered = poolPos * Nkv * D
      for i in 0 ..< Nkv * D:
        kStage.data[covered + i] = randBf16()
        vStage.data[covered + i] = randBf16()

  # f64 host reference over the same bf16-rounded inputs:
  # - prefill (qLen >= 2): query token j attends keys [0, kvLen - qLen + j],
  #   flash-attn varlen semantics
  # - decode (qLen == 1): attends keys [0, kvLen)
  # - GQA: kv_head = q_head div (H div Nkv)
  # - the softmax runs in the exp form, one log2(e) factor from the kernel's
  #   exp2 shape, identical in exact arithmetic
  var refO = newSeq[float64](numQoTokens * H * D)
  let scale = 1.0 / sqrt(float64(D))
  for s in 0 ..< numSeqs:
    let
      q0 = qOffsets[s]
      qLen = scenario.qLens[s]
      kvLen = scenario.kvLens[s]
    for h in 0 ..< H:
      let kvHead = h div (H div Nkv)
      for j in 0 ..< qLen:
        let lastKey = (if qLen == 1: kvLen - 1 else: kvLen - qLen + j)
        var scores = newSeq[float64](lastKey + 1)
        var m = -1.0e300
        for k in 0 .. lastKey:
          var acc = 0.0'f64
          for d in 0 ..< D:
            let pageIdx = scenario.kvPageBounds[s] + k div PageSize
            let poolPos = ((scenario.pageIds[pageIdx] * scenario.numLayers + scenario.layer) *
                           PageSize + k mod PageSize) * Nkv + kvHead
            acc += float64(qStage.data[((q0 + j) * H + h) * D + d].to(float32)) *
                   float64(kStage.data[poolPos * D + d].to(float32))
          scores[k] = acc * scale
          m = max(m, scores[k])
        var l = 0.0'f64
        var ps = newSeq[float64](lastKey + 1)
        for k in 0 .. lastKey:
          ps[k] = exp(scores[k] - m)
          l += ps[k]
        for d in 0 ..< D:
          var acc = 0.0'f64
          for k in 0 .. lastKey:
            let pageIdx = scenario.kvPageBounds[s] + k div PageSize
            let poolPos = ((scenario.pageIds[pageIdx] * scenario.numLayers + scenario.layer) *
                           PageSize + k mod PageSize) * Nkv + kvHead
            acc += ps[k] * float64(vStage.data[poolPos * D + d].to(float32))
          refO[((q0 + j) * H + h) * D + d] = acc / l

  proc launchOnce(): seq[uint32] =
    ## One launch over fresh bridges, returns the raw output bits.
    let
      qDev = engine.bridge(qStage)
      kDev = engine.bridge(kStage)
      vDev = engine.bridge(vStage)
      oDev = engine.bridge(oStage)
      qOffB = engine.bridge(rank1Host(int32, qOffsets))
      kvB = engine.bridge(kvHost)
    let prodGridX = (max(scenario.qLens) + JaggedAttnTile - 1) div JaggedAttnTile
    let gx = (if gridX > 0: gridX else: prodGridX)
    # windowed launches select their branch at compile time, the smoke scenarios run non-windowed (Qwen3-0.6B has no sliding window)
    if windowSize > 0:
      engine.launchDeviceFn bf16AttnCombo(
          oDev, qDev, kDev, vDev, qOffB, kvB,
          float32(scale), scenario.layer, H, Nkv,
          PageSize, D, true, windowSize) do:
        grid = (gx, H, numSeqs)
        blk = (32, 1)
    else:
      engine.launchDeviceFn bf16AttnCombo(
          oDev, qDev, kDev, vDev, qOffB, kvB,
          float32(scale), scenario.layer, H, Nkv,
          PageSize, D, false, -2147483648) do:
        grid = (gx, H, numSeqs)
        blk = (32, 1)
    engine.copyToHostFromDevice(oStage.data.toOpenArray(0, oStage.len() - 1), oDev.data)
    result = newSeq[uint32](oStage.len())
    for i in 0 ..< oStage.len():
      result[i] = uint32(cast[uint16](oStage.data[i]))

  var worst = 0.0'f64
  var oBad, oTotal = 0
  proc judge() =
    ## Per-element band judgment plus the sentinel-token check.
    for i in 0 ..< numQoTokens * H * D:
      let got = float64(oStage.data[i].to(float32))
      let want = refO[i]
      let bar = Bf16Rtol * abs(want) + Bf16Abstol
      let diff = abs(got - want)
      # NaN counts as out-of-band only under the not-(<=) form, a `>` comparison would let a NaN output pass judgment silently
      if not (diff <= bar):
        inc oBad
      elif diff / bar > worst:
        worst = diff / bar
      inc oTotal
    for i in numQoTokens * H * D ..< oStage.len():
      doAssert cast[uint16](oStage.data[i]) == 0'u16,
        name & ": o written into its sentinel tokens at element " & $i

  let firstBits = launchOnce()
  judge()
  let secondBits = launchOnce()
  for i in 0 ..< oStage.len():
    doAssert secondBits[i] == firstBits[i],
      name & ": o differs run to run at element " & $i

  echo &"{name}: seqs={numSeqs} qo={numQoTokens} pages={scenario.numPages} " &
    &"worstBar={worst:.2f} mismatches={oBad}/{oTotal} relaunch bit-identical"
  doAssert oBad == 0,
    name & ": " & $oBad & " of " & $oTotal & " output elements outside the band"
