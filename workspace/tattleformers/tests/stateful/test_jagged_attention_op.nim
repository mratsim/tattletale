# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Jagged attention op suite (Metal): the bf16 smoke scenarios through the op.
##
## Test that tn_jagged_attention reproduces the kernel smoke's results
## through the erased face over decode, prefill and the mixed
## shared-prefix batch, judged per element against an f64 host reference
## computed over the same bf16-rounded inputs (rtol 1e-2, abstol 1e-2).
##
## Staging: covered pool tokens random in [-1, 1] and every other slot
## NaN garbage, so a 0*NaN in the P*V mma would poison the output.
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/wipM07attnop workspace/tattleformers/tests/stateful/test_jagged_attention_op.nim

from std/math import sqrt, exp
import std/random
import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/tensorcery/src/device_view
import workspace/tensorcery/src/tensor_datatypes
from workspace/tattleformers/src/stateful/kvcache import TokensPerPage
import workspace/tattleformers/src/stateful/page_pool
import workspace/tattleformers/src/ops_Ni1o/attention

{.experimental: "callOperator".}

proc k_primer*(z, x: distinct DeviceView, shift: static int) =
  ## z = x + shift, one element.
  ##
  ## Why this kernel exists:
  ## - the compilation must hold one enqueue before the first erased-Tensor call
  ## - Tensor views dispatch through the erased interface, whose vtable
  ##   instantiation compiles eval and its batch submit
  ## - submit's finalize macro is a compile-time check erroring on an empty
  ##   codegen frontend
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let s = xv[0].to(float32) + float32(shift)
  zv[0] = s.to(bfloat16)

const
  H = 16
  Nkv = 8
  D = 128
  PageSize = TokensPerPage # 256
  NumLayers = 2
  NumPages = 4
  Bf16Rtol = 1e-2'f64
  Bf16Abstol = 1e-2'f64

type Scenario = object
  ## One replay scenario's geometry and page table, host-authored,
  ## the bf16 smoke's Bf16JaggedCase shape.
  qLens*: seq[int]
  kvLens*: seq[int]
  pageIds*: seq[int]
  kvPageBounds*: seq[int] # per segment, [start, end) into pageIds
  layer*: int

proc poolPos(pageId, layer, pos, kvHead: int): int =
  (((pageId * NumLayers + layer) * PageSize + pos) * Nkv + kvHead) * D

proc stageSlabs(engine: AnyEngine, pool: PagePool, sc: Scenario):
    tuple[k: seq[bfloat16], v: seq[bfloat16]] =
  ## Fills both slabs for one scenario: the scenario's covered tokens
  ## (its segments, its layer) random in [-1, 1] and every other slot
  ## NaN garbage. The kernel must bound at the covered tokens.
  let slabElems = NumPages * NumLayers * PageSize * Nkv * D
  result.k = newSeq[bfloat16](slabElems)
  result.v = newSeq[bfloat16](slabElems)
  var rng = initRand(20261109)
  template randBf16(): bfloat16 =
    rng.rand(-1.0 .. 1.0).to(bfloat16)
  template nanBf16(): bfloat16 =
    cast[bfloat16](uint16(0x7FC0))
  for i in 0 ..< slabElems:
    result.k[i] = nanBf16()
    result.v[i] = nanBf16()
  for s in 0 ..< sc.qLens.len:
    for t in 0 ..< sc.kvLens[s]:
      let
        pageIdx = sc.kvPageBounds[s] + t div PageSize
        base = poolPos(sc.pageIds[pageIdx], sc.layer, t mod PageSize, 0)
      for h in 0 ..< Nkv:
        for d in 0 ..< D:
          let idx = base + h * D + d
          result.k[idx] = randBf16()
          result.v[idx] = randBf16()
  pool.kBuffer().copyToDeviceFromHost(result.k)
  pool.vBuffer().copyToDeviceFromHost(result.v)

proc runCase[E: HwEngine](engine: E, pool: PagePool, sc: Scenario, name: string) =
  ## One scenario through the erased face, judged per element against
  ## an f64 host reference computed over the same bf16-rounded inputs.
  doAssert sc.kvPageBounds.len == sc.qLens.len + 1
  doAssert sc.kvPageBounds[0] == 0
  let numSeqs = sc.qLens.len
  var qTotal = 0
  var qOffsets = @[0]
  for s in 0 ..< numSeqs:
    qTotal += sc.qLens[s]
    qOffsets.add qTotal
  var kvMeta = JaggedKvMeta(
    offsets: @[0], indices: @[], seqlens: @[], batchSize: numSeqs)
  for s in 0 ..< numSeqs:
    # offsets are the cumulative page counts, the page ids of segment s are
    # kvMeta.indices[offsets[s] ..< offsets[s + 1]]
    kvMeta.offsets.add kvMeta.offsets[^1] + sc.kvPageBounds[s + 1] - sc.kvPageBounds[s]
    for b in sc.kvPageBounds[s] ..< sc.kvPageBounds[s + 1]:
      kvMeta.indices.add sc.pageIds[b]
    kvMeta.seqlens.add sc.kvLens[s]

  # seeded inputs: q random in [-1, 1], the pool's covered tokens random,
  # everything else NaN garbage (stageSlabs)
  var rng = initRand(20261110)
  var qHost = newSeq[bfloat16](qTotal * H * D)
  for i in 0 ..< qHost.len:
    qHost[i] = rng.rand(-1.0 .. 1.0).to(bfloat16)
  let (kHost, vHost) = stageSlabs(engine, pool, sc)

  # f64 host reference over the same bf16-rounded inputs:
  # - prefill (qLen >= 2): query token j attends keys [0, kvLen - qLen + j],
  #   flash-attn varlen semantics
  # - decode (qLen == 1): attends keys [0, kvLen)
  # - GQA: kv_head = q_head div (H div Nkv)
  # - the softmax runs in the exp form, one log2(e) factor from the kernel's
  #   exp2 shape, identical in exact arithmetic
  var refO = newSeq[float64](qTotal * H * D)
  let scale = 1.0 / sqrt(float64(D))
  for s in 0 ..< numSeqs:
    let
      q0 = qOffsets[s]
      qLen = sc.qLens[s]
      kvLen = sc.kvLens[s]
    for h in 0 ..< H:
      let kvHead = h div (H div Nkv)
      for j in 0 ..< qLen:
        let lastKey = (if qLen == 1: kvLen - 1 else: kvLen - qLen + j)
        var scores = newSeq[float64](lastKey + 1)
        var m = -1.0e300
        for k in 0 .. lastKey:
          var acc = 0.0'f64
          for d in 0 ..< D:
            let pageIdx = sc.kvPageBounds[s] + k div PageSize
            let poolPos = poolPos(sc.pageIds[pageIdx], sc.layer,
                                  k mod PageSize, kvHead)
            acc += float64(qHost[((q0 + j) * H + h) * D + d].to(float32)) *
                   float64(kHost[poolPos + d].to(float32))
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
            let pageIdx = sc.kvPageBounds[s] + k div PageSize
            let poolPos = poolPos(sc.pageIds[pageIdx], sc.layer,
                                  k mod PageSize, kvHead)
            acc += ps[k] * float64(vHost[poolPos + d].to(float32))
          refO[((q0 + j) * H + h) * D + d] = acc / l

  # op call: one erased-face eval over the staged tensors
  let
    qDev = newTensor(engine, @[qTotal, H, D], kBf16)
    oNode = tn_jagged_attention(engine, qDev.node, pool.kBuffer().node,
      pool.vBuffer().node, qOffsets, kvMeta,
      float32(scale), sc.layer, H, Nkv)
  qDev.copyToDeviceFromHost(qHost)
  var roots = @[oNode]
  engine.eval(roots.toSpan())

  var got = newSeq[bfloat16](qTotal * H * D)
  copyToHostFromDevice(got, Tensor(engine: engine, node: oNode))

  var worst = 0.0'f64
  var oBad, oTotal = 0
  for i in 0 ..< qTotal * H * D:
    let g = float64(got[i].to(float32))
    let want = refO[i]
    let bar = Bf16Rtol * abs(want) + Bf16Abstol
    let diff = abs(g - want)
    # NaN counts as out-of-band only under the not-(<=) form, a `>` comparison
    # would let a NaN output pass judgment silently
    if not (diff <= bar):
      inc oBad
    elif diff / bar > worst:
      worst = diff / bar
    inc oTotal
  echo "  " & name & ": worstBar=" & $worst & " mismatches=" & $oBad & "/" & $oTotal
  doAssert oBad == 0,
    name & ": " & $oBad & " of " & $oTotal & " output elements outside the band"

proc runMain[E: HwEngine](engine: E) =
  # primer enqueue: textually precede the first erased-Tensor call
  block:
    let
      zDev = engine.allocDevice(bfloat16, 1)
      xDev = engine.allocDevice(bfloat16, 1)
      zb = TensorBridge[typeof(zDev), (int,), (int,)](
        data: zDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
      xb = TensorBridge[typeof(xDev), (int,), (int,)](
        data: xDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
    var batch = engine.recordBatch()
    batch.enqueue<<(1, 1)>> k_primer(zb, xb, 3)
    batch.submit()

  let pool = PagePool.init(engine, NumPages, NumLayers, Nkv, D)

  # decode: one segment, q_len 1, kv_len 300 over 2 pages, partial last page
  runCase(engine, pool, Scenario(
    qLens: @[1], kvLens: @[300], pageIds: @[0, 1, 2], kvPageBounds: @[0, 3],
    layer: 0), "op_decode_partial_last_page")

  # prefill: one segment, q_len 8 == kv_len 8
  runCase(engine, pool, Scenario(
    qLens: @[8], kvLens: @[8], pageIds: @[0], kvPageBounds: @[0, 1],
    layer: 1), "op_prefill_layer1")

  # mixed shared-prefix batch: prefill + prefill crossing pages + decode,
  # all three segments share page 0's prefix
  # page table: 1 + 2 + 3 = 6 page-table entries over the 4-page pool.
  # Segments 2 and 3 reuse pages 0 and 2. Each segment covers the token
  # ranges of its own entries, so sharing a page is legal.
  runCase(engine, pool, Scenario(
    qLens: @[8, 8, 1], kvLens: @[8, 264, 520],
    pageIds: @[0, 2, 1, 0, 3, 2], kvPageBounds: @[0, 1, 3, 6],
    layer: 0), "op_mixed_shared_prefix")

  echo "jagged attention op (Metal): all checks passed"

proc main =
  let engine = gMetalEngine()
  runMain(engine)

when isMainModule:
  main()
