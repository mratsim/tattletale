# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/tensors
import ./tile_matmul
import ./tiles

# ═══════════════════════════════════════════════════════════════
#  Guarded access
# ═══════════════════════════════════════════════════════════════

func guardedLoad*[T, Sh, St](src: TensorView[T, Sh, St], crd: tuple, padWith: T): T {.inline.} =
  ## Guarded read of `src` at `crd`: in bounds reads the element,
  ## out of bounds reads `padWith`.
  if crd.inBounds(src.getLayout().shape):
    src.data[src.getLayout().crd2idx(crd)]
  else:
    padWith

func guardedStore*[T, Sh, St](dst: TensorView[T, Sh, St], crd: tuple, val: T) {.inline.} =
  ## Guarded store into `dst` at `crd`: in bounds stores `val`,
  ## out of bounds drops the store.
  if crd.inBounds(dst.getLayout().shape):
    dst.data[dst.getLayout().crd2idx(crd)] = val

# ═══════════════════════════════════════════════════════════════
#  Tile load / store
# ═══════════════════════════════════════════════════════════════

proc loadTile*[T, ShD, StD, ShS, StS, Sh, St, TSh](
    dst: var TensorOwned[T, ShD, StD], src: TensorView[T, ShS, StS],
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
  ## The tile's element (m, k) sits at coordinate tileID*tileShape + (m, k).
  ##
  ## Args:
  ##   - dst: the register tile, sized by `tvm`'s register layout
  ##   - src: the global tensor
  ##   - padWith: the read past bounds
  let srcLayout = src.getLayout()
  let tiled = srcLayout.zipped_divide(tvm.layout)
  let thrBits = idx2crd(tvm.getThreadPartitioning(), threadIdx)
  let thrSel = (((thrBits, _), _), tileID)
  let srcReg = src.slice(tiled, thrSel)
  let tiledCrd = make_identity_layout(srcLayout.shape).zipped_divide(tvm.layout)
  let regCrd = tiledCrd.slice(thrSel)

  let tileSel = (((_, _), _), tileID)
  let tileCrd = tiledCrd.slice(tileSel)
  let farCorner = tiledCrd.crd2idx(tileSel) + tileCrd.crd2idx(tileCrd.size() - 1)
  if farCorner.inBounds(srcLayout.shape):
    for i in 0 ..< srcReg.size():
      dst[i] = srcReg[i]
  else:
    let crdBase = tiledCrd.crd2idx(thrSel)
    for i in 0 ..< srcReg.size():
      dst[i] = src.guardedLoad(crdBase + regCrd.crd2idx(i), padWith)

proc storeTile*[T, ShD, StD, ShS, StS, Sh, St, TSh](
    dst: TensorView[T, ShD, StD], src: TensorOwned[T, ShS, StS],
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
  ## The tile's element (m, n) sits at coordinate tileID*tileShape + (m, n).
  ##
  ## Args:
  ##   - dst: the global tensor stored into
  ##   - src: the register tile, sized by `tvm`'s register layout
  let dstLayout = dst.getLayout()
  let tiled = dstLayout.zipped_divide(tvm.layout)
  let thrBits = idx2crd(tvm.getThreadPartitioning(), threadIdx)
  let thrSel = (((thrBits, _), _), tileID)
  let dstReg = dst.slice(tiled, thrSel)
  let tiledCrd = make_identity_layout(dstLayout.shape).zipped_divide(tvm.layout)
  let regCrd = tiledCrd.slice(thrSel)

  let tileSel = (((_, _), _), tileID)
  let tileCrd = tiledCrd.slice(tileSel)
  let farCorner = tiledCrd.crd2idx(tileSel) + tileCrd.crd2idx(tileCrd.size() - 1)
  if farCorner.inBounds(dstLayout.shape):
    for i in 0 ..< dstReg.size():
      dstReg[i] = src[i]
  else:
    let crdBase = tiledCrd.crd2idx(thrSel)
    for i in 0 ..< dstReg.size():
      dst.guardedStore(crdBase + regCrd.crd2idx(i), src[i])
