# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## Page pool write op suite (Metal): chunk decomposition and coordinates.
##
## Test that tn_pool_write places post-rope K/V entries at the right
## (page, layer, in-page) coordinates through one eval, single chunk,
## multi-page and cached-prefix skip.
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/wipM07pw workspace/tattleformers/tests/stateful/test_pool_write.nim

import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/tensorcery/src/device_view
import workspace/tensorcery/src/tensor_datatypes
from workspace/tattleformers/src/stateful/kvcache import TokensPerPage
import workspace/tattleformers/src/stateful/page_pool
import workspace/tattleformers/src/ops_Ni1o/pool_write

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
  NumPages = 4
  NumLayers = 2
  KvHeads = 2
  HeadDim = 4
  PageSize = TokensPerPage # 256
  SlotElems = PageSize * KvHeads * HeadDim
  SlabElems = NumPages * NumLayers * SlotElems

proc uploadSlab(engine: AnyEngine, slab: Tensor, seed: int): seq[bfloat16] =
  ## Fills the whole slab with known sentinel values, returns the host copy.
  result = newSeq[bfloat16](SlabElems)
  for i in 0 ..< SlabElems:
    result[i] = (float32((i * 31 + seed) mod 251) * 0.125'f32 - 8.0'f32).to(bfloat16)
  slab.copyToDeviceFromHost(result)

proc downloadSlab(slab: Tensor): seq[bfloat16] =
  result = newSeq[bfloat16](SlabElems)
  copyToHostFromDevice(result, slab)

proc srcRows(engine: AnyEngine, seqLen: int, seed: int): tuple[k, v: Tensor, kh, vh: seq[bfloat16]] =
  ## Builds (1, seq_len, kv_heads, head_dim) src entries, device and host copies.
  let shape = @[1, seqLen, KvHeads, HeadDim]
  result.kh = newSeq[bfloat16](seqLen * KvHeads * HeadDim)
  result.vh = newSeq[bfloat16](seqLen * KvHeads * HeadDim)
  for i in 0 ..< result.kh.len:
    result.kh[i] = (float32((i * 7 + seed) mod 97) * 0.25'f32 - 4.0'f32).to(bfloat16)
    result.vh[i] = (float32((i * 11 + seed) mod 89) * 0.25'f32 - 4.0'f32).to(bfloat16)
  result.k = newTensor(engine, shape, kBf16)
  result.v = newTensor(engine, shape, kBf16)
  result.k.copyToDeviceFromHost(result.kh)
  result.v.copyToDeviceFromHost(result.vh)

proc slabIdx(page, layer, pos, h, d: int): int =
  (((page * NumLayers + layer) * PageSize + pos) * KvHeads + h) * HeadDim + d

proc srcIdx(t, h, d: int): int =
  (t * KvHeads + h) * HeadDim + d

proc runWrite(pool: PagePool, layer: int, offset, seqLen, cachedTokens: int, kSrc, vSrc: Tensor) =
  ## One eval of the pool write op over the layer's views.
  let (lk, lv) = pool.layerView(layer)
  let node = tn_pool_write(pool.kBuffer().engine, lk.node, lv.node,
    kSrc.node, vSrc.node, offset, seqLen, cachedTokens)
  var roots = @[node]
  pool.kBuffer().engine.eval(roots.toSpan())

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

  let pool = PagePool.init(engine, NumPages, NumLayers, KvHeads, HeadDim)

  # single chunk: offset 0, 10 entries, one page, both layers write independently
  block:
    let kSlab = uploadSlab(engine, pool.kBuffer(), 1)
    let vSlab = uploadSlab(engine, pool.vBuffer(), 2)
    let (kSrc, vSrc, kh, vh) = srcRows(engine, 10, 3)
    runWrite(pool, 0, 0, 10, 0, kSrc, vSrc)
    # Layer 0 stores the entries, layer 1 keeps its sentinels
    let gotK = downloadSlab(pool.kBuffer())
    let gotV = downloadSlab(pool.vBuffer())
    for t in 0 ..< 10:
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(0, 0, t, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
            "K layer0 (" & $t & "," & $h & "," & $d & ")"
          doAssert gotV[slabIdx(0, 0, t, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
            "V layer0 (" & $t & "," & $h & "," & $d & ")"
          doAssert gotK[slabIdx(0, 1, t, h, d)].to(float32) == kSlab[slabIdx(0, 1, t, h, d)].to(float32),
            "K layer1 sentinel (" & $t & "," & $h & "," & $d & ")"
          doAssert gotV[slabIdx(0, 1, t, h, d)].to(float32) == vSlab[slabIdx(0, 1, t, h, d)].to(float32),
            "V layer1 sentinel (" & $t & "," & $h & "," & $d & ")"
    # Positions outside the write range keep their sentinels
    for pos in 10 ..< 16:
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(0, 0, pos, h, d)].to(float32) == kSlab[slabIdx(0, 0, pos, h, d)].to(float32),
            "K sentinel past range (" & $pos & "," & $h & "," & $d & ")"

  # multi-page: offset 200, 600 entries crossing pages 0..2, page-boundary chunks
  block:
    let kSlab = uploadSlab(engine, pool.kBuffer(), 4)
    let vSlab = uploadSlab(engine, pool.vBuffer(), 5)
    let (kSrc, vSrc, kh, vh) = srcRows(engine, 600, 6)
    runWrite(pool, 1, 200, 600, 0, kSrc, vSrc)
    let gotK = downloadSlab(pool.kBuffer())
    let gotV = downloadSlab(pool.vBuffer())
    for t in 0 ..< 600:
      let globalPos = 200 + t
      let page = globalPos div PageSize
      let pos = globalPos mod PageSize
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          doAssert gotK[slabIdx(page, 1, pos, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
            "K multi (" & $t & " @" & $globalPos & "," & $h & "," & $d & "): got " &
            $gotK[slabIdx(page, 1, pos, h, d)].to(float32) & " want " & $kh[srcIdx(t, h, d)].to(float32)
          doAssert gotV[slabIdx(page, 1, pos, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
            "V multi (" & $t & " @" & $globalPos & "," & $h & "," & $d & ")"
    # layer 0 keeps its sentinels across all touched pages
    for page in 0 .. 2:
      for pos in 0 ..< PageSize:
        for h in 0 ..< KvHeads:
          for d in 0 ..< HeadDim:
            doAssert gotK[slabIdx(page, 0, pos, h, d)].to(float32) == kSlab[slabIdx(page, 0, pos, h, d)].to(float32),
              "K multi layer0 sentinel (" & $page & "," & $pos & ")"

  # cached-prefix skip: offset 100, 10 entries, cachedTokens 105 -> entries 0..5 skipped
  block:
    let kSlab = uploadSlab(engine, pool.kBuffer(), 7)
    let (kSrc, vSrc, kh, vh) = srcRows(engine, 10, 8)
    runWrite(pool, 0, 100, 10, 105, kSrc, vSrc)
    let gotK = downloadSlab(pool.kBuffer())
    let gotV = downloadSlab(pool.vBuffer())
    # writeStart = max(0, 105 - 100) = 5: global positions 105..109 written,
    # 100..104 keep their sentinels
    for t in 0 ..< 10:
      let globalPos = 100 + t
      let pos = globalPos mod PageSize
      for h in 0 ..< KvHeads:
        for d in 0 ..< HeadDim:
          if t < 5:
            doAssert gotK[slabIdx(0, 0, pos, h, d)].to(float32) == kSlab[slabIdx(0, 0, pos, h, d)].to(float32),
              "K cached sentinel (" & $t & "," & $h & "," & $d & ")"
          else:
            doAssert gotK[slabIdx(0, 0, pos, h, d)].to(float32) == kh[srcIdx(t, h, d)].to(float32),
              "K cached (" & $t & "," & $h & "," & $d & ")"
            doAssert gotV[slabIdx(0, 0, pos, h, d)].to(float32) == vh[srcIdx(t, h, d)].to(float32),
              "V cached (" & $t & "," & $h & "," & $d & ")"

  echo "pool write op (Metal): all checks passed"

proc main =
  let engine = gMetalEngine()
  runMain(engine)

when isMainModule:
  main()
