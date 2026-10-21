# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/ceramic/src/int_tuples/int_tuples_unsanctioned_helpers
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/tensors
import ./tile_matmul
import ./tiles

# ############################################################
#
#                    Data movement
#
# ############################################################

# ═══════════════════════════════════════════════════════════════
#  Guarded access
# ═══════════════════════════════════════════════════════════════

template tileInBounds*(tiled, tileCrd, sel, domain): bool =
  ## Whether the tile selected by `sel` **fully** fits inside `domain`.
  let farCorner = tiled.crd2idx(sel) + tileCrd.crd2idx(tileCrd.size() - 1)
  farCorner.inBounds(domain.shape)

func paddedLoad*[T, Sh, St](src: TensorView[T, Sh, St], crd: int or tuple, padWith: T): T {.inline.} =
  ## Guarded read of `src` at `crd`: in bounds reads the element,
  ## out of bounds reads `padWith`.
  if crd.inBounds(src.shape):
    src[crd]
  else:
    padWith

func guardedStore[T, Sh, St](dst: TensorView[T, Sh, St], crd: int or tuple, val: T) {.inline.} =
  ## Guarded store into `dst` at `crd`: in bounds stores `val`,
  ## out of bounds drops the store.
  if crd.inBounds(dst.shape):
    dst[crd] = val

# ═══════════════════════════════════════════════════════════════
#  Tile load / store for tensor cores
# ═══════════════════════════════════════════════════════════════

proc load*[T, ShD, StD, ShS, StS, Sh, St, TSh](
    dst: var Tile[T, ShD, StD], src: TensorView[T, ShS, StS],
    tvm: static ThreadValueMap[Sh, St, TSh], threadIdx: int,
    tileID: (int, int), padWith: T = zero(T)) =
  ## Tile load, global memory -> register tile, through the operand's
  ## thread->value map `tvm`.
  ##
  ## Say `src` is a tensor and `tvm` maps (thread, value) slots of one
  ## atom to coordinates of an atom-shaped subtile of `src`, repeated
  ## over the tile `tileID` addresses, the calling thread's slots
  ## selected by `threadIdx`.
  ##
  ## The tile's element (m, k) has coordinate tileID*tileShape + (m, k).
  ##
  ## Args:
  ##   - dst: the register tile, sized by `tvm`'s register layout
  ##   - src: the global tensor
  ##   - padWith: the read past bounds
  let tiled = src.logical_divide(tvm.layout)
  let thrCrd = idx2crd(tvm.getThreadPartitioning(), threadIdx)
  let thrSel = (((thrCrd, _), _), tileID)
  let srcReg = tiled.slice(thrSel)
  let tiledCrd = src.getIdentityLayout().logical_divide(tvm.layout)
  let regCrd = tiledCrd.slice(thrSel)

  let tileSel = (((_, _), _), tileID)
  let tileCrd = tiledCrd.slice(tileSel)
  if tileInBounds(tiledCrd, tileCrd, tileSel, src):
    for i in 0 ..< srcReg.size():
      dst[i] = srcReg[i]
  else:
    let crdBase = tiledCrd.crd2idx(thrSel)
    for i in 0 ..< srcReg.size():
      dst[i] = src.paddedLoad(crdBase + regCrd.crd2idx(i), padWith)

proc store*[T, ShD, StD, ShS, StS, Sh, St, TSh](
    dst: TensorView[T, ShD, StD], src: Tile[T, ShS, StS],
    tvm: static ThreadValueMap[Sh, St, TSh], threadIdx: int,
    tileID: (int, int)) =
  ## Tile store, register tile -> global memory, through the operand's
  ## thread->value map `tvm`.
  ##
  ## Say `dst` is a tensor and `tvm` maps (thread, value) slots of one
  ## atom to coordinates of an atom-shaped subtile of `dst`, repeated
  ## over the tile `tileID` addresses, the calling thread's slots
  ## selected by `threadIdx`.
  ##
  ## The tile's element (m, n) has coordinate tileID*tileShape + (m, n).
  ##
  ## Args:
  ##   - dst: the global tensor stored into
  ##   - src: the register tile, sized by `tvm`'s register layout
  let tiled = dst.logical_divide(tvm.layout)
  let thrCrd = idx2crd(tvm.getThreadPartitioning(), threadIdx)
  let thrSel = (((thrCrd, _), _), tileID)
  let dstReg = tiled.slice(thrSel)
  let tiledCrd = dst.getIdentityLayout().logical_divide(tvm.layout)
  let regCrd = tiledCrd.slice(thrSel)

  let tileSel = (((_, _), _), tileID)
  let tileCrd = tiledCrd.slice(tileSel)
  if tileInBounds(tiledCrd, tileCrd, tileSel, dst):
    for i in 0 ..< dstReg.size():
      dstReg[i] = src[i]
  else:
    let crdBase = tiledCrd.crd2idx(thrSel)
    for i in 0 ..< dstReg.size():
      dst.guardedStore(crdBase + regCrd.crd2idx(i), src[i])

# ═══════════════════════════════════════════════════════════
#  Tile load / store for general case
# ═══════════════════════════════════════════════════════════

proc load*[T, Sh, St, ShS, StS](
    dst: var Tile[T, Sh, St], src: TensorView[T, ShS, StS],
    threadIdx: int, tileID: int, padWith: T = zero(T)) {.inline.} =
  ## Tile load over the element positions: the register tile's map
  ## divides the position list, each position reads the element through
  ## `src`'s own layout. Out-of-bounds positions read `padWith`.
  ##
  ## Args:
  ## - dst: the register tile, typed with the thread->element map
  ## - src: the global tensor
  ## - threadIdx: the thread index inside the threadgroup
  ## - tileID: the flat tile number over the tile grid
  let
    tiled = make_layout(src.size()).logical_divide(dst.getLayout())
    sel = ((threadIdx, _), tileID)
    box = tiled.slice(sel)
    offset = tiled.crd2idx(sel)
  if tileInBounds(tiled, box, sel, src):
    for e in 0 ..< box.size():
      dst.data[e] = src[offset + box.crd2idx(e)]
  else:
    for e in 0 ..< box.size():
      dst.data[e] = src.paddedLoad(offset + box.crd2idx(e), padWith)

proc store*[T, Sh, St, ShD, StD](
    dst: TensorView[T, ShD, StD], src: Tile[T, Sh, St],
    threadIdx: int, tileID: int) {.inline.} =
  ## Tile store over the element positions: out-of-bounds positions drop.
  ##
  ## Args:
  ## - dst: the global tensor stored into
  ## - src: the register tile, typed with the thread->element map
  ## - threadIdx: the thread index inside the threadgroup
  ## - tileID: the flat tile number over the tile grid
  let
    tiled = make_layout(dst.size()).logical_divide(src.getLayout())
    sel = ((threadIdx, _), tileID)
    box = tiled.slice(sel)
    offset = tiled.crd2idx(sel)
  if tileInBounds(tiled, box, sel, dst):
    for e in 0 ..< box.size():
      dst[offset + box.crd2idx(e)] = src.data[e]
  else:
    for e in 0 ..< box.size():
      dst.guardedStore(offset + box.crd2idx(e), src.data[e])

proc load*[T, ShD, StD, ShS, StS, Sh, St, TSh](
    _: typedesc[Tile[T, ShD, StD]], src: TensorView[T, ShS, StS],
    tvm: static ThreadValueMap[Sh, St, TSh], threadIdx: int,
    tileID: (int, int), padWith: T = zero(T)): Tile[T, ShD, StD] {.inline.} =
  load(result, src, tvm, threadIdx, tileID, padWith)

proc load*[T, Sh, St, ShS, StS](
    _: typedesc[Tile[T, Sh, St]], src: TensorView[T, ShS, StS],
    threadIdx: int, tileID: int, padWith: T = zero(T)): Tile[T, Sh, St] {.inline.} =
  load(result, src, threadIdx, tileID, padWith)

# ═══════════════════════════════════════════════════════════════
#  Vectorization width
# ═══════════════════════════════════════════════════════════════

func collectLeaves[Sh, St, N](shapes, strides: var array[N, int], idx: var int, s: Sh, d: St) =
  ## Appends the flat leaf modes as (shape, stride) int pairs, size-1
  ## modes dropped: their strides are junk and would break the sequence.
  when Sh is tuple:
    staticFor i, 0, Sh.rank():
      collectLeaves(shapes, strides, idx, s[i], d[i])
  else:
    if s.toInt() > 1:
      shapes[idx] = s.toInt()
      strides[idx] = d.toInt()
      idx += 1

proc maxCommonVector*[Sh, St](layout: Layout[Sh, St]): int =
  ## Vectorization width: the longest sequence of consecutive addresses among the layout's modes.
  const nLeaves: int = default(Sh).flatten().rank().toInt()  # toint-allow
  var shapes: array[nLeaves, int]
  var strides: array[nLeaves, int]
  var idx = 0
  collectLeaves(shapes, strides, idx, layout.shape, layout.stride)
  var used: array[nLeaves, bool]
  var seqLen = 1
  var expect = 1
  var alive = true
  for round in 0 ..< idx:
    var best = -1
    for i in 0 ..< idx:
      if (not used[i]) and (best == -1 or strides[i] < strides[best]):
        best = i
    used[best] = true
    let step = alive and (strides[best] == expect)
    seqLen = if step: seqLen * shapes[best] else: seqLen
    expect = if step: seqLen else: expect
    alive = step
  return seqLen
