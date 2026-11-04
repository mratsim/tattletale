# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Jagged attention replay machinery shared by the drivers, per backend,
## plus the dense cross-check against the dense reference path.
## The runtime parts are ordinary procs, one copy total.
## Only the runner monomorphizes per element dtype, all in one compile.

import std/[os, strformat]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_sdpa
import workspace/positron/src/kernels/k_jagged_attention
import ../positron_testutils
import jagged_attention_fixtures

export tile_algebra, layout_algebra, tensors, int_tuples
export jagged_attention_fixtures

# ─── Pass bands, stated before measurement ────────────────────────────
#
# Element band, per output element: |got - want| <= Rtol*|want| + Abstol
#
# f16 band (rtol 1e-3, abstol 1e-2):
# - the online-softmax attention chain rounds the QK dot in f32 accumulation
#   over 64 f16 products, then the exp2 rescale chain and the PV reduction run in f32, with P downcast to fp16 before the PV mma
# - the dominant term is the fp16 quantization of the inputs and of P, one
#   fp16 ulp (2^-11) of relative error per rounded element
#   amplified by the softmax denominator division
# - rtol 1e-3 absorbs the accumulated f32 chain, abstol 1e-2 covers outputs
#   near zero where the input quantization dominates
#
# f32 band (rtol 1e-5, abstol 1e-4):
# - every accumulation stays f32 with no P downcast
# - the QK dot over 64 f32 products and the PV reduction round at 2^-24 each,
#   and the exp2 rescale chain adds a handful more
# - the accumulated relative error sits around 1e-7 at unit-scale inputs
# - rtol 1e-5 leaves two orders of headroom over that error, abstol 1e-4
#   covers outputs near zero

const
  Fp16Rtol* = 1e-3'f64
  Fp16Abstol* = 1e-2'f64
  Fp32Rtol* = 1e-5'f64
  Fp32Abstol* = 1e-4'f64

const
  JaggedAttnTile = 32
  ## 32x32x32 mma schedule tile, one KV tile per page at the fixture page size
  PadTokens = 2
  ## zero sentinel tokens after the logical output, written entries there fail the replay

proc jaggedAttnCombo*(o, q, kPool, vPool: distinct TensorView,
                      qOffsets: distinct TensorView,
                      kv: JaggedKv,
                      scale: float32, layer, H, Nkv: int,
                      PageSize, D: static int) =
  ## Zips the two mma schedule tiles into the kernel's trailing statics.
  ##
  ## Contract:
  ##
  ## - one call-site line per binding so the engine's monomorphization key
  ##   separates the element dtypes
  jagged_attn_fwd(o, q, kPool, vPool, qOffsets, kv,
                  scale, layer, H, Nkv, PageSize, D,
                  tiledMma(float32, q.T, kPool.T, JaggedAttnTile, JaggedAttnTile, D),
                  tiledMma(float32, o.T, vPool.T, JaggedAttnTile, JaggedAttnTile, D))

proc jaggedAttnReplayCase*(engine: MetalEngine, Elem: typedesc, name: string): seq[uint32] =
  ## Replays one fixture, with relaunch:
  ##
  ## - one launch over the production grid, per-element band judgment
  ##   against the recorded f64 reference
  ## - a relaunch judged bit-identical on the raw output bits
  ##
  ## Returns the first launch's raw output bits over the staging tensor
  ## (logical tokens plus the sentinel tokens) as the jagged side of the dense cross-check comparison.
  ##
  ## Preconditions:
  ## - the fixtures use D = 64, page_size = JaggedAttnTile, num_layers = 2, checked below
  ## - the output staging tensor includes `PadTokens` zero sentinel tokens after num_qo_tokens, checked after the launch
  ## - the attention body reads every argument, so the inputs bridge fresh per launch
  ## - the write coverage comes from the band judgment plus the sentinel tokens and the relaunch bit-identity
  const
    wantDtype = when Elem is float16: "float16" else: "float32"
    Rtol = when Elem is float16: Fp16Rtol else: Fp32Rtol
    Abstol = when Elem is float16: Fp16Abstol else: Fp32Abstol
  let dir = JaggedAttnFixturesDir / name
  let meta = parseJaggedAttnMetadata(dir)
  doAssert meta.dtype == wantDtype, meta.name & ": dtype " & meta.dtype
  doAssert meta.D == 64, meta.name & ": head dim " & $meta.D & " != 64"
  doAssert meta.pageSize == JaggedAttnTile,
    meta.name & ": page size " & $meta.pageSize & " != the mma tile " & $JaggedAttnTile
  doAssert meta.numLayers == 2, meta.name & ": num_layers " & $meta.numLayers
  doAssert meta.H mod meta.Nkv == 0,
    "fixture " & meta.name & ": H " & $meta.H & " is not a multiple of Nkv " & $meta.Nkv
  doAssert meta.qOffsets.len == meta.numSeqs + 1 and meta.qOffsets[0] == 0 and
    meta.qOffsets[^1] == meta.numQoTokens, meta.name & ": q_offsets inconsistent"
  doAssert meta.kvOffsets.len == meta.numSeqs + 1 and meta.kvOffsets[0] == 0 and
    meta.kvOffsets[^1] == meta.numIndices, meta.name & ": kv_offsets inconsistent"
  doAssert meta.kvSeqlens.len == meta.numSeqs, meta.name & ": kv_seqlens length"

  let
    qBits = loadFixtureTensor(dir, "q.bin", (meta.numQoTokens, meta.H, meta.D),
      when Elem is float16: uint16 else: uint32)
    kBits = loadFixtureTensor(dir, "k.bin",
      (meta.numPages, meta.numLayers, meta.pageSize, meta.Nkv, meta.D),
      when Elem is float16: uint16 else: uint32)
    vBits = loadFixtureTensor(dir, "v.bin",
      (meta.numPages, meta.numLayers, meta.pageSize, meta.Nkv, meta.D),
      when Elem is float16: uint16 else: uint32)
    expected = loadFixtureTensor(dir, "o.bin", (meta.numQoTokens, meta.H, meta.D), float32)
    qOffHost = loadRank1FixtureTensor(dir, "q_offsets.bin", meta.numSeqs + 1)
    kvOffHost = loadRank1FixtureTensor(dir, "kv_offsets.bin", meta.numSeqs + 1)
    kvIdxHost = loadRank1FixtureTensor(dir, "kv_indices.bin", meta.numIndices)
    kvSeqHost = loadRank1FixtureTensor(dir, "kv_seqlens.bin", meta.numSeqs)
  doAssert xorHash(qBits) == meta.xorhashChecksum("q.bin"), meta.name & ": q.bin checksum"
  doAssert xorHash(kBits) == meta.xorhashChecksum("k.bin"), meta.name & ": k.bin checksum"
  doAssert xorHash(vBits) == meta.xorhashChecksum("v.bin"), meta.name & ": v.bin checksum"
  doAssert xorHash(expected) == meta.xorhashChecksum("o.bin"), meta.name & ": o.bin checksum"
  doAssert xorHash(qOffHost) == meta.xorhashChecksum("q_offsets.bin"), meta.name & ": q_offsets.bin checksum"
  doAssert xorHash(kvOffHost) == meta.xorhashChecksum("kv_offsets.bin"), meta.name & ": kv_offsets.bin checksum"
  doAssert xorHash(kvIdxHost) == meta.xorhashChecksum("kv_indices.bin"), meta.name & ": kv_indices.bin checksum"
  doAssert xorHash(kvSeqHost) == meta.xorhashChecksum("kv_seqlens.bin"), meta.name & ": kv_seqlens.bin checksum"

  # addSequence registers the fixture metadata, engine.bridge projects it to the kernel descriptor
  var kvHost = new(JaggedKvHost, capacity = meta.numIndices)
  for s in 0 ..< meta.numSeqs:
    kvHost.addSequence(tokens = int(kvSeqHost.data[s]),
      pageIds = kvIdxHost.data.toOpenArray(int(kvOffHost.data[s]), int(kvOffHost.data[s + 1]) - 1))
  doAssert kvHost.batchSize == meta.numSeqs, meta.name & ": kv batch size"

  # staging: the inputs copy from the recorded bits and the output staging is
  # zero-filled with the sentinel tokens after the logical output starting at zero
  var
    qStage = makeHostTensor(Elem, (meta.numQoTokens, meta.H, meta.D), LayoutRight)
    kStage = makeHostTensor(Elem, (meta.numPages, meta.numLayers, meta.pageSize, meta.Nkv, meta.D), LayoutRight)
    vStage = makeHostTensor(Elem, (meta.numPages, meta.numLayers, meta.pageSize, meta.Nkv, meta.D), LayoutRight)
    oStage = makeHostTensor(Elem, (meta.numQoTokens + PadTokens, meta.H, meta.D), LayoutRight)
  zeroMem(cast[pointer](oStage.data), oStage.len() * sizeof(Elem))
  for i in 0 ..< qStage.len():
    qStage.data[i] = cast[Elem](qBits.data[i])
  for i in 0 ..< kStage.len():
    kStage.data[i] = cast[Elem](kBits.data[i])
  for i in 0 ..< vStage.len():
    vStage.data[i] = cast[Elem](vBits.data[i])

  let logicalElems = meta.numQoTokens * meta.H * meta.D
  var maxQLen = 0
  for s in 0 ..< meta.numSeqs:
    maxQLen = max(maxQLen, meta.qOffsets[s + 1] - meta.qOffsets[s])
  let qBlocks = (maxQLen + JaggedAttnTile - 1) div JaggedAttnTile

  proc launchOnce(): seq[uint32] =
    ## One launch over fresh bridges, returns the raw output bits.
    let
      qDev = engine.bridge(qStage)
      kDev = engine.bridge(kStage)
      vDev = engine.bridge(vStage)
      oDev = engine.bridge(oStage)
      qOffDev = engine.bridge(qOffHost)
      kvB = engine.bridge(kvHost)
    engine.launchDeviceFn jaggedAttnCombo(
        oDev, qDev, kDev, vDev, qOffDev, kvB,
        meta.scale, meta.layer, meta.H, meta.Nkv,
        JaggedAttnTile, 64) do:
      grid = (qBlocks, meta.H, meta.numSeqs)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr oStage.data[0], oDev.data, oStage.len())
    result = newSeq[uint32](oStage.len())
    for i in 0 ..< oStage.len():
      when Elem is float16:
        result[i] = uint32(cast[uint16](oStage.data[i]))
      else:
        result[i] = cast[uint32](oStage.data[i])

  var worst = 0.0'f64
  var oBad, oTotal = 0
  proc judge() =
    ## Per-element band judgment plus the sentinel-token check.
    for i in 0 ..< logicalElems:
      when Elem is float16:
        let got = float16(cast[uint16](oStage.data[i])).to(float32).float64
      else:
        let got = oStage.data[i].float64
      let want = expected.data[expected.offset + i].float64
      let bar = Rtol * abs(want) + Abstol
      let diff = abs(got - want)
      # NaN counts as out-of-band only under the not-(<=) form, a `>` comparison
      # would let a NaN output pass the judgment silently
      if not (diff <= bar):
        inc oBad
      elif diff / bar > worst:
        worst = diff / bar
      inc oTotal
    for i in logicalElems ..< oStage.len():
      when Elem is float16:
        doAssert cast[uint16](oStage.data[i]) == 0'u16,
          name & ": o written into its sentinel tokens at element " & $i
      else:
        doAssert oStage.data[i] == 0.0'f32,
          name & ": o written into its sentinel tokens at element " & $i

  let firstBits = launchOnce()
  judge()
  let secondBits = launchOnce()
  for i in 0 ..< oStage.len():
    doAssert secondBits[i] == firstBits[i],
      name & ": o differs run to run at element " & $i
  result = firstBits

  echo &"{name}: seqs={meta.numSeqs} qo={meta.numQoTokens} pages={meta.numPages} " &
    &"worstBar={worst:.2f} mismatches={oBad}/{oTotal} relaunch bit-identical"
  doAssert oBad == 0,
    name & ": " & $oBad & " of " & $oTotal & " output elements outside the band"


# ─── Dense cross-check: the dense reference path on identical inputs ──

proc denseAttnAllPairs*(mmaQK, mmaPV: static TiledMma,
                        oRef, qPacked, kDense, vDense: distinct TensorView,
                        qOffsets, kvSeqlens: distinct TensorView,
                        scale: float32, numSeqs, H, Nkv, maxKv: int,
                        Dk, Dv: static int) =
  ## Dense reference path over every (sequence, head) pair of one launch:
  ## per pair, `scaled_dot_product_attention` over the segment's KV tokens, causal.
  ##
  ## - bottom-right alignment gives query token j the keys [0, kvLen - qLen + j],
  ##   the same band the jagged kernel masks
  ## - a decode pair (qLen = 1) attends [0, kvLen)
  ## - the GQA mapping degenerates per pair: one query head reads one kv head
  ## - the pair loop runs inside one threadgroup, `scaled_dot_product_attention`
  ##   owns the grid's x dimension for its own q tiles
  for pair in 0 ..< numSeqs * H:
    let
      seqId = pair div H
      head = pair mod H
      q0 = int(qOffsets[seqId])
      qLen = int(qOffsets[seqId + 1]) - q0
      kvLen = int(kvSeqlens[seqId])
      one = 1
      hdRow = H * Dk
      kvHead = head div (H div Nkv)
      kvRowStride = Nkv * Dk
      dCols = Dk
      dvCols = Dv
      qView = make_view(qPacked.data +% (q0 * H + head) * dCols,
                        make_layout((one, one, qLen, dCols), (0, 0, hdRow, 1)))
      oView = make_view(oRef.data +% (q0 * H + head) * dCols,
                        make_layout((one, one, qLen, dvCols), (0, 0, hdRow, 1)))
      kView = make_view(kDense.data +% (seqId * maxKv * Nkv + kvHead) * dCols,
                        make_layout((one, one, kvLen, dCols), (0, 0, kvRowStride, 1)))
      vView = make_view(vDense.data +% (seqId * maxKv * Nkv + kvHead) * dCols,
                        make_layout((one, one, kvLen, dvCols), (0, 0, kvRowStride, 1)))
    scaled_dot_product_attention(mmaQK, mmaPV, oView, qView, kView, vView,
                                 scale, Dk, Dv, isCausal = true)

proc jaggedAttnDenseCrossCheck*(engine: MetalEngine, Elem: typedesc, name: string, jaggedBits: seq[uint32]) =
  ## Dense cross-check of the jagged kernel on the dense-degenerate fixture.
  ##
  ## - the fixture's KV bits repack host-side through the recorded page table
  ##   into the (num_seqs, max_kv, Nkv, D) dense layout
  ## - the dense production path (`scaled_dot_product_attention`) runs per
  ##   (sequence, head) pair over the same q tokens
  ## - the judgment compares dense-vs-jagged and dense-vs-f64-reference within
  ##   the pre-stated band, reporting the dense-vs-jagged exact-element count
  ##
  ## `jaggedBits` is the jagged replay's first-launch output over the same
  ## inputs (logical tokens first, so element i of both sides corresponds).
  const
    Rtol = when Elem is float16: Fp16Rtol else: Fp32Rtol
    Abstol = when Elem is float16: Fp16Abstol else: Fp32Abstol
  let dir = JaggedAttnFixturesDir / name
  let meta = parseJaggedAttnMetadata(dir)
  doAssert meta.dtype == (when Elem is float16: "float16" else: "float32"),
    name & ": dtype " & meta.dtype
  doAssert meta.D == 64, name & ": head dim " & $meta.D & " != 64"

  let
    qBits = loadFixtureTensor(dir, "q.bin", (meta.numQoTokens, meta.H, meta.D),
      when Elem is float16: uint16 else: uint32)
    kBits = loadFixtureTensor(dir, "k.bin",
      (meta.numPages, meta.numLayers, meta.pageSize, meta.Nkv, meta.D),
      when Elem is float16: uint16 else: uint32)
    vBits = loadFixtureTensor(dir, "v.bin",
      (meta.numPages, meta.numLayers, meta.pageSize, meta.Nkv, meta.D),
      when Elem is float16: uint16 else: uint32)
    expected = loadFixtureTensor(dir, "o.bin", (meta.numQoTokens, meta.H, meta.D), float32)
    kvOffHost = loadRank1FixtureTensor(dir, "kv_offsets.bin", meta.numSeqs + 1)
    kvIdxHost = loadRank1FixtureTensor(dir, "kv_indices.bin", meta.numIndices)
    kvSeqHost = loadRank1FixtureTensor(dir, "kv_seqlens.bin", meta.numSeqs)
    qOffHost = loadRank1FixtureTensor(dir, "q_offsets.bin", meta.numSeqs + 1)

  var
    qStage = makeHostTensor(Elem, (meta.numQoTokens, meta.H, meta.D), LayoutRight)
    kStage = makeHostTensor(Elem, (meta.numPages, meta.numLayers, meta.pageSize, meta.Nkv, meta.D), LayoutRight)
    vStage = makeHostTensor(Elem, (meta.numPages, meta.numLayers, meta.pageSize, meta.Nkv, meta.D), LayoutRight)
    oDense = makeHostTensor(Elem, (meta.numQoTokens, meta.H, meta.D), LayoutRight)
  zeroMem(cast[pointer](oDense.data), oDense.len() * sizeof(Elem))
  for i in 0 ..< qStage.len():
    qStage.data[i] = cast[Elem](qBits.data[i])
  for i in 0 ..< kStage.len():
    kStage.data[i] = cast[Elem](kBits.data[i])
  for i in 0 ..< vStage.len():
    vStage.data[i] = cast[Elem](vBits.data[i])

  # dense repack: the recorded page walk places segment s's token t at (s, t, kv_head, d)
  var maxKv = 0
  for s in 0 ..< meta.numSeqs:
    maxKv = max(maxKv, meta.kvSeqlens[s])
  var
    kDense = makeHostTensor(Elem, (meta.numSeqs, maxKv, meta.Nkv, meta.D), LayoutRight)
    vDense = makeHostTensor(Elem, (meta.numSeqs, maxKv, meta.Nkv, meta.D), LayoutRight)
  for s in 0 ..< meta.numSeqs:
    for t in 0 ..< meta.kvSeqlens[s]:
      let page = kvIdxHost.data[kvOffHost.data[s] + t div meta.pageSize]
      let pos = t mod meta.pageSize
      for kvh in 0 ..< meta.Nkv:
        for d in 0 ..< meta.D:
          let poolIdx = (((page * meta.numLayers + meta.layer) * meta.pageSize + pos) *
                         meta.Nkv + kvh) * meta.D + d
          let denseIdx = ((s * maxKv + t) * meta.Nkv + kvh) * meta.D + d
          kDense.data[denseIdx] = kStage.data[poolIdx]
          vDense.data[denseIdx] = vStage.data[poolIdx]

  let
    qDev = engine.bridge(qStage)
    kDenseDev = engine.bridge(kDense)
    vDenseDev = engine.bridge(vDense)
    oDenseDev = engine.bridge(oDense)
    qOffDev = engine.bridge(qOffHost)
    kvSeqDev = engine.bridge(kvSeqHost)
  engine.launchDeviceFn denseAttnAllPairs(
      tiledMma(float32, Elem, Elem, JaggedAttnTile, JaggedAttnTile, 64),
      tiledMma(float32, Elem, Elem, JaggedAttnTile, JaggedAttnTile, 64),
      oDenseDev, qDev, kDenseDev, vDenseDev, qOffDev, kvSeqDev,
      meta.scale, meta.numSeqs, meta.H, meta.Nkv, maxKv, 64, 64) do:
    grid = (1, 1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr oDense.data[0], oDenseDev.data, oDense.len())

  var
    exact, djBad, doBad = 0
  for i in 0 ..< oDense.len():
    when Elem is float16:
      let got = float16(cast[uint16](oDense.data[i])).to(float32).float64
      let jag = float16(uint16(jaggedBits[i])).to(float32).float64
    else:
      let got = oDense.data[i].float64
      let jag = cast[float32](jaggedBits[i]).float64
    let want = expected.data[expected.offset + i].float64
    let bar = Rtol * abs(want) + Abstol
    if got == jag:
      inc exact
    # NaN counts as out-of-band only under the not-(<=) form, a `>` comparison
    # would let a NaN output pass the judgment silently
    if not (abs(got - jag) <= bar):
      inc djBad
    if not (abs(got - want) <= bar):
      inc doBad
  echo &"{name}: dense cross-check exact={exact}/{oDense.len()} " &
    &"dense-vs-jagged band mismatches={djBad} dense-vs-oracle band mismatches={doBad}"
  doAssert djBad == 0 and doBad == 0,
    name & ": the dense reference path disagrees with the jagged kernel or the oracle"
