# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/ceramic
import workspace/crucible
import ./tensors_internal_types

export tensors_internal_types

# ############################################################
#
#      TensorBasicView - device view over a LayoutBasic
#
# ############################################################
#
# Device-side view of a LayoutBasic tensor: the kernel argument of the memory-bound ops.
# Field structure matches TensorView. The frontend derives the kernel's device struct
# from the bridge argument and the element type stays reachable as `.T` via its generic parameter.

type TensorBasicView*[T] = object
  data*: ptr UncheckedArray[T]
  offset*: int
  layout*: LayoutBasic

func size*(v: TensorBasicView): int {.inline.} =
  result = 1
  for i in 0 ..< v.layout.rank:
    result *= v.layout.shape[i]

func isContiguous*(v: TensorBasicView): bool =
  ## - the device frontend lowers countup loops only
  var acc = 1
  for i in 0 ..< v.layout.rank:
    let d = v.layout.rank - 1 - i
    if v.layout.strides[d] != acc:
      return false
    acc *= v.layout.shape[d]
  result = true

func isNil*[T](v: TensorBasicView[T]): bool {.inline.} =
  v.data.isNil()

func flatIndex(shape, strides: array[MaxRank, int], rank: int, pos: int): int {.inline.} =
  ## Precondition: pos < product(shape[0 ..< rank]) so every dim divides.
  result = 0
  var p = pos
  for i in 0 ..< rank:
    let d = rank - 1 - i
    result += (p mod shape[d]) * strides[d]
    p = p div shape[d]

func alongAxisAt*[T](v: TensorBasicView[T], axis: int, idx: int): TensorBasicView[T] {.inline.} =
  ## Precondition: axis in 0 ..< v.layout.rank, idx < product of the other dims.
  result.data = v.data
  result.offset = v.offset
  result.layout.rank = 1
  result.layout.shape[0] = v.layout.shape[axis]
  result.layout.strides[0] = v.layout.strides[axis]
  var p = idx
  for d in 0 ..< v.layout.rank:
    if d != axis:
      result.offset += (p mod v.layout.shape[d]) * v.layout.strides[d]
      p = p div v.layout.shape[d]

func `[]`*[T](v: TensorBasicView[T], pos: int): T {.inline.} =
  v.data[v.offset + flatIndex(v.layout.shape, v.layout.strides, v.layout.rank, pos)]

func `[]=`*[T](v: var TensorBasicView[T], pos: int, val: T) {.inline.} =
  v.data[v.offset + flatIndex(v.layout.shape, v.layout.strides, v.layout.rank, pos)] = val

# ═══════════════════════════════════════════════════════════════════════
#  Tile load / store over a TensorBasicView
# ═══════════════════════════════════════════════════════════════════════
#
# Element-position tile load/store equivalent to the tile algebra over a TensorView.

proc load*[T, L](dst: var Tile[T, L], src: TensorBasicView, threadIdx: int, tileID: int, padWith: T = zero(T)) {.inline.} =
  let
    n = src.size()
    tiled = make_layout(n).logical_divide(dst.getLayout())
    sel = ((threadIdx, _), tileID)
    box = tiled.slice(sel)
    pos0 = tiled.crd2idx(sel)
  let dense = src.isContiguous()
  if pos0 + box.size() <= n:
    for e in 0 ..< box.size():
      let pos = pos0 + box.crd2idx(e)
      if dense:
        dst.data[e] = src.data[src.offset + pos]
      else:
        dst.data[e] = src.data[src.offset + flatIndex(src.layout.shape, src.layout.strides, src.layout.rank, pos)]
  else:
    for e in 0 ..< box.size():
      let pos = pos0 + box.crd2idx(e)
      if pos < n:
        if dense:
          dst.data[e] = src.data[src.offset + pos]
        else:
          dst.data[e] = src.data[src.offset + flatIndex(src.layout.shape, src.layout.strides, src.layout.rank, pos)]
      else:
        dst.data[e] = padWith

proc load*[T, L](_: typedesc[Tile[T, L]], src: TensorBasicView, threadIdx: int, tileID: int, padWith: T = zero(T)): Tile[T, L] {.inline.} =
  load(result, src, threadIdx, tileID, padWith)

proc store*[T, L](dst: TensorBasicView, src: Tile[T, L], threadIdx: int, tileID: int) {.inline.} =
  let
    n = dst.size()
    tiled = make_layout(n).logical_divide(src.getLayout())
    sel = ((threadIdx, _), tileID)
    box = tiled.slice(sel)
    pos0 = tiled.crd2idx(sel)
  let dense = dst.isContiguous()
  if pos0 + box.size() <= n:
    for e in 0 ..< box.size():
      let pos = pos0 + box.crd2idx(e)
      if dense:
        dst.data[dst.offset + pos] = src.data[e]
      else:
        dst.data[dst.offset + flatIndex(dst.layout.shape, dst.layout.strides, dst.layout.rank, pos)] = src.data[e]
  else:
    for e in 0 ..< box.size():
      let pos = pos0 + box.crd2idx(e)
      if pos < n:
        if dense:
          dst.data[dst.offset + pos] = src.data[e]
        else:
          dst.data[dst.offset + flatIndex(dst.layout.shape, dst.layout.strides, dst.layout.rank, pos)] = src.data[e]
