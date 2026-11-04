# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## applyAttentionMask: the attention band as one fused element loop,
## the mask shape the sdpa and jagged attention kernels share.
##
##   qk (mt, nt) f32    masked(qPos, kPos)    qk (mt, nt) f32
##        │                  │                        │
##        └─► [ band: tile(bool, L), c = masked ] ─► [ qk.where(band, -inf, qk) ]
##
## - Call sites pass the predicate as an untyped body over qPos and kPos,
##   anchored at baseQ and baseK
## - the body splices into the element loop and inlines at the call site,
##   FlexAttention's mask_mod without the block sparsity pass
## - a lambda would not lower: the body has `when` branches over kernel
##   statics and closes over runtime locals
## - the tile-range trim skipping fully masked blocks stays in each kernel body

import ./tiles
import ./tile_foreach
import ./tile_ops_ternary
import ./tile_matmul
import ./tile_tvm_reductions
import workspace/ceramic/src/layout_algebra

template applyAttentionMask*(qk: var Tile, bandLayout: static auto,
    qkMap: static auto, thrID: int, baseQ, baseK: int, masked: untyped) {.dirty.} =
  ## Scores where `masked` is true are set to -inf.
  ## Every attention backend applies this additive mask before the softmax.
  ##
  ## - bandLayout is the score tile's layout, the kernel's qkCLayout
  ## - baseQ and baseK anchor the tile: sdpa passes qBase and kv * nt,
  ##   the jagged kernel passes q0Local and kvPos

  let
    thrCrd = idx2crd(qkMap.getThreadPartitioning(), thrID)
    regCrd = qkMap.layout.slice(((thrCrd, _), _))
    crdBase = qkMap.layout.crd2idx(((thrCrd, _), _))
  var band: tile(bool, bandLayout)
  forEach idx, var c in band:
    let (qRow, kCol) = crdBase + regCrd.crd2idx(idx)
    let qPos = baseQ + qRow
    let kPos = baseK + kCol
    c = masked
  qk.where(band, negInf(float32), qk)
