# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Embedding gather kernel.
##
## - `k_embed_gather`: o[t, :] = w[ids[t], :]. This is the Embedding forward, also `index_select(0, ids)`
## - one threadgroup per token, grid-stride over tokens, lanes stride the hidden columns
## - a pure copy without arithmetic: the element dtype passes through bit-identically (bfloat16 included)
## - token ids are metadata: a typed int32 device buffer binds to the pointer formal, outside the erased tensor face

import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/tensorcery/src/device_view

proc k_embed_gather*(o, w: distinct DeviceView, ids: ptr UncheckedArray[int32]) =
  ## Gathers each token's weight vector: o[t, :] = w[ids[t], :].
  ##
  ## Expected input:
  ## - o: (numTokens, hidden), a contiguous output view
  ## - w: (vocab, hidden) weight view, any strides
  ## - ids: numTokens token ids in [0, vocab)
  ##
  ## Output:
  ## - o fully written, token t's vector copied from weight entry ids[t]
  let
    ov = make_view(o.data +% o.offset, o.layout)
    wv = make_view(w.data +% w.offset, w.layout)
  let
    numTokens = int(ov.shape[0])
    hidden = int(wv.shape[1])
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for token in countup(group0, numTokens - 1, groupStride):
    let entry = int(ids[token])
    for col in countup(lane, hidden - 1, lanes):
      ov[token, col] = wv[entry, col]
