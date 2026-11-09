# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## Page pool write kernel.
##
## - `k_pool_write`: device-to-device copy of post-rope K/V entries into pool pages,
##   both buffers in one launch
## - one thread per (token, head, dim) element, grid-stride over the entries,
##   the page decomposition runs in-kernel: one launch writes the whole range
## - pure elementwise copy, no arithmetic: values are stored bit-exact

import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/device_view

proc k_pool_write*(dstK, srcK, dstV, srcV: distinct DeviceView, meta: ptr UncheckedArray[int32]) =
  ## Copies post-rope K/V entries into their pool slots, both pool windows
  ## fully overwritten over the covered positions.
  ##
  ## Expected input:
  ## - srcK, srcV: (seq_len, kv_heads, head_dim) entries of the post-rope K and V
  ## - dstK, dstV: (num_pages, page_size, kv_heads, head_dim) pool layer views
  ## - meta: the pointer-formal metadata buffer pattern of k_embed_gather.
  ##   Entry layout: offset, seqLen, cachedTokens, pageSize, then one int32
  ##   pool slot per sequence page
  ## - pageSlots[i] is the pool slot of sequence page i, a cached page
  ##   occupies any pool slot
  ##
  ## Output:
  ## - dstK, dstV fully written over positions [offset, offset + seq_len):
  ##   source element (t, h, d) writes the pool entry
  ##   (page, withinPage, h, d) at slot pageSlots[globalPos div page_size]
  ##
  ## Positions before max(0, cachedTokens - offset) keep their previous content.
  ## Sequence page = globalPos div page_size, withinPage = globalPos mod
  ## page_size, globalPos = offset + t.
  let
    dvK = make_view(dstK.data +% dstK.offset, dstK.layout)
    svK = make_view(srcK.data +% srcK.offset, srcK.layout)
    dvV = make_view(dstV.data +% dstV.offset, dstV.layout)
    svV = make_view(srcV.data +% srcV.offset, srcV.layout)
  let
    offset = meta[0]
    seqLen = meta[1]
    cachedTokens = meta[2]
    pageSize = meta[3]
    kvHeadDim = int(svK.shape[1]) * int(svK.shape[2])
    numel = int(svK.shape[0]) * kvHeadDim
    tid = thread_position_in_grid.x
    stride = getGridSize()
  # ── Write new KV into page slots ──
  # Each page covers TokensPerPage token positions.
  # page.k_view[layer_idx] is (PAGE_SIZE, kv_heads, head_dim)
  #
  # offset = ctx.kv_position avoids a GPU->CPU synchronous read every forward pass.
  # position_ids.min().item(int) would force that sync every pass.
  #
  # Lifecycle overview:
  #   1. Prefill:  startSequence sets kv_position=0
  #                forward writes at offset=0
  #                generate() calls setKvPosition(ids.len)
  #   2. Decode:   appendToken sets position_ids WITHOUT incrementing.
  #                forward writes at offset=kv_position, pos_ids.min() matches.
  #                generate() increments kv_position after forward.
  #   => Invariant: kv_position == position_ids.min() during forward.
  # Skip writing cached prefix positions (already in trie from COW)
  # TODO: how to test usage of cache?
  let writeStart = max(0'i32, cachedTokens - offset)

  for i in countup(tid, int32(numel - 1), int32(stride)):
    let
      t = i div int32(kvHeadDim)
      rem = i mod int32(kvHeadDim)
      h = rem div int32(svK.shape[2])
      d = rem mod int32(svK.shape[2])
    if t >= writeStart:
      let globalPos = offset + t
      let page = meta[4 + globalPos div pageSize]
      let withinPage = globalPos mod pageSize
      dvK[int(page), int(withinPage), int(h), int(d)] = svK[int(t), int(h), int(d)]
      dvV[int(page), int(withinPage), int(h), int(d)] = svV[int(t), int(h), int(d)]
