# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## PagePool suite (Metal): borrow/return lifecycle, slab aliasing and layer views.
##
## Test that the pool borrows and recycles page slots through the Page destructor,
## that page views alias their slab window and that layerView places the layer
## base at the right element offset.
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/ttf --outdir:build/ttf --path:"$PWD" workspace/tattleformers/tests/stateful/test_page_pool.nim
##

import workspace/tensorcery/src/tensorcery
import workspace/tattleformers/src/stateful/kvcache
import workspace/tattleformers/src/stateful/page_pool

proc uploadSlab(slab: Tensor, modulus: int): seq[bfloat16] =
  ## Fills the whole slab with known sentinel values, returns the host copy.
  result = newSeq[bfloat16](getNumElems(slab.shape()))
  for i in 0 ..< result.len:
    result[i] = (float32(i mod modulus) * 0.125'f32).to(bfloat16)
  slab.upload(result)

proc runMain() =
  # borrow/return lifecycle: dropping a Page ref recycles its slot, LIFO
  block:
    let pool = PagePool.init(num_pages = 4, num_layers = 2, kv_heads = 8, head_dim = 128)
    doAssert pool.pagesAvailable() == 4
    var p0 = pool.borrow()
    var p1 = pool.borrow()
    doAssert pool.pagesAvailable() == 2
    doAssert p0.pageIndex() != p1.pageIndex()
    doAssert p0.pageIndex() >= 0 and p1.pageIndex() >= 0
    let idx0 = p0.pageIndex()
    let idx1 = p1.pageIndex()
    # dropping recycles both slots, the free stack pops LIFO
    p0 = nil
    doAssert pool.pagesAvailable() == 3
    var p2 = pool.borrow()
    doAssert p2.pageIndex() == idx0, "the freed slot must recycle first"
    p1 = nil
    var p3 = pool.borrow()
    doAssert p3.pageIndex() == idx1
    doAssert pool.pagesAvailable() == 2

    # exhaustion raises once every slot is out
    var pa = pool.borrow()
    doAssert pool.pagesAvailable() == 1
    var pb = pool.borrow()
    doAssert pool.pagesAvailable() == 0
    doAssertRaises ResourceExhaustedError:
      discard pool.borrow()
    pb = nil
    doAssert pool.pagesAvailable() == 1
    pa = nil
    doAssert pool.pagesAvailable() == 2

  # page views alias their slab window: the page view of page idx covers
  # elements [idx * (num_layers*PAGE_SIZE*kv_heads*head_dim), ... + slotSize)
  block:
    const
      NumPages = 3
      NumLayers = 2
      KvHeads = 2
      HeadDim = 4
      PageSize = TokensPerPage
      SlotElems = NumLayers * PageSize * KvHeads * HeadDim
    let pool = PagePool.init(NumPages, NumLayers, KvHeads, HeadDim)
    let page = pool.borrow()
    doAssert page.pageIndex() == 0, "the free stack pops the lowest index first (seeded descending)"
    doAssert page.k_view.shape().len == 4
    doAssert page.k_view.shape()[0] == NumLayers
    doAssert page.k_view.shape()[1] == PageSize
    doAssert page.k_view.shape()[2] == KvHeads
    doAssert page.k_view.shape()[3] == HeadDim
    let pageStrides = @[PageSize * KvHeads * HeadDim, KvHeads * HeadDim, HeadDim, 1]
    doAssert page.k_view.strides() == pageStrides.toSpan()

    # readback through the strided gather: the page view covers
    # elements [idx*SlotElems, (idx+1)*SlotElems) of the densely ordered slab
    let slabHost = uploadSlab(pool.kBuffer(), 97)
    let gotPage = page.k_view.toSeq(bfloat16)
    let base = page.pageIndex().int * SlotElems
    for i in 0 ..< SlotElems:
      doAssert gotPage[i].to(float32) == slabHost[base + i].to(float32),
        "page view element " & $i & " must match slab element " & $(base + i)

  # layerView: the (num_pages, PAGE_SIZE, kv_heads, head_dim) window,
  # layer base offset, same strides, page stride num_layers*PAGE_SIZE*kv_heads*head_dim
  block:
    const
      NumPages = 3
      NumLayers = 2
      KvHeads = 2
      HeadDim = 4
      PageSize = TokensPerPage
      LayerElems = NumPages * PageSize * KvHeads * HeadDim
    let pool = PagePool.init(NumPages, NumLayers, KvHeads, HeadDim)
    let (kView, vView) = pool.layerView(1)
    doAssert kView.shape().len == 4
    doAssert kView.shape()[0] == NumPages
    doAssert kView.shape()[1] == PageSize
    let layerStrides = @[NumLayers * PageSize * KvHeads * HeadDim, KvHeads * HeadDim, HeadDim, 1]
    doAssert kView.strides() == layerStrides.toSpan()

    # readback through the strided gather: the layer view element
    # (page, pos, head, dim) must match the slab at (page, layer=1, pos, ...)
    let slabHost = uploadSlab(pool.kBuffer(), 53)
    let gotLayer = kView.toSeq(bfloat16)
    for p in 0 ..< NumPages:
      for pos in 0 ..< PageSize:
        for h in 0 ..< KvHeads:
          for d in 0 ..< HeadDim:
            let viewIdx = ((p * PageSize + pos) * KvHeads + h) * HeadDim + d
            let slabIdx = (((p * NumLayers + 1) * PageSize + pos) * KvHeads + h) * HeadDim + d
            doAssert gotLayer[viewIdx].to(float32) == slabHost[slabIdx].to(float32),
              "layerView (" & $p & "," & $pos & "," & $h & "," & $d & ")"

  echo "page pool (Metal): all checks passed"

runMain()
