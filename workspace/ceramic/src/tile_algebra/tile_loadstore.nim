# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
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

func guardedLoad*[T, Sh, St](src: TensorView[T, Sh, St], crd: int or tuple, padWith: T): T {.inline.} =
  ## Guarded read of `src` at `crd`: in bounds reads the element,
  ## out of bounds reads `padWith`.
  if crd.inBounds(src.getLayout().shape):
    src.data[src.getLayout().crd2idx(crd)]
  else:
    padWith

func guardedStore*[T, Sh, St](dst: TensorView[T, Sh, St], crd: int or tuple, val: T) {.inline.} =
  ## Guarded store into `dst` at `crd`: in bounds stores `val`,
  ## out of bounds drops the store.
  if crd.inBounds(dst.getLayout().shape):
    dst.data[dst.getLayout().crd2idx(crd)] = val

# ═══════════════════════════════════════════════════════════════
#  Tile load / store through the operand's thread->value map (tensor cores)
# ═══════════════════════════════════════════════════════════════

proc loadTile*[T, ShD, StD, ShS, StS, Sh, St, TSh](
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

# ═══════════════════════════════════════════════════════════════
#  Tile load / store, blocked convention
# ═══════════════════════════════════════════════════════════════
#
#  The tile is the lane's register block, its layout the only schedule.
#
#  Blocked convention:
#  - the block's last axis spreads across the simdgroup's 32 lanes
#  - lane t owns that axis's chunk [t*S, (t+1)*S)
#  - the other axes live in-thread
#  - one threadgroup covers one covered tile
#
#  The blocked thread->value map below plays the operands' `tvm` role:
#  the same zipped_divide / slice / coordinate machinery, no tensor cores.

func staticShape*(n: NimNode): seq[int] {.compileTime.} =
  ## Static ints of a shape node: a tuple literal, a tuple type, or a typedesc wrapping one
  var nn = n
  if nn.kind == nnkStmtListExpr:
    nn = nn[^1]
  if nn.kind in {nnkTupleConstr, nnkPar}:
    for e in nn:
      result.add e.getStaticInt()
    return
  var ty = nn.getTypeInst()
  if ty.kind == nnkBracketExpr and ty[0].kind == nnkSym and $ty[0] == "typeDesc":
    ty = ty[1]
  if ty.isTupleTy():
    for i in 0 ..< ty.len:
      result.add ty[i].getStaticInt()
  else:
    result.add ty.getStaticInt()

proc crdLeaf(n: NimNode, d: int): NimNode {.compileTime.} =
  ## d-th component of a coordinate expression, the expression itself
  ## when it is a scalar
  if n.getTypeInst().isTupleTy():
    nnkBracketExpr.newTree(n, newLit(d))
  else:
    n

macro blockMap*(blk: typed): untyped =
  ## Blocked thread->value map of `blk`: coordinate strides, the spread
  ## axis's basis carrying the chunk step.
  ##
  ## - mode 0: the 32-lane chunk index
  ## - mode 1: the lane's block
  ##
  ## Size-1-free by construction.
  let sh = staticShape(blk)
  let last = sh.len - 1
  let sLast = sh[last]
  # mode 0: the chunk index, stride E(S, last), scalar mode and stride
  let thrSh = newLit(32)
  let thrSt = bindSym"E".newCall(newLit(sLast), newLit(last))
  # mode 1: the block's modes with stride E(1, d). Size-1 modes require
  # plain-0 strides, a coordinate stride on a size-1 mode breaks compose
  var blkSt: NimNode
  var blkShLit: NimNode
  proc blockMode(d: int): NimNode {.compileTime.} =
    if sh[d] == 1: newLit(0)
    else: bindSym"E".newCall(newLit(1), newLit(d))
  if sh.len == 1:
    blkSt = blockMode(0)
    blkShLit = bindSym"makeIntTuple".newCall(nnkPar.newTree(newLit(sh[0])))
  else:
    var par = nnkPar.newTree()
    var shPar = nnkPar.newTree()
    for d in 0 ..< sh.len:
      par.add blockMode(d)
      shPar.add newLit(sh[d])
    blkSt = par
    blkShLit = bindSym"makeIntTuple".newCall(shPar)
  result = quote do:
    make_layout((`thrSh`, `blkShLit`), (`thrSt`, `blkSt`))

template blockMapFor(blk: typed): untyped =
  ## blockMap expanded per instantiation inside the generic procs
  blockMap(blk)

proc loadTile*[T, ShD, StD, ShS, StS](
    dst: var Tile[T, ShD, StD], src: TensorView[T, ShS, StS],
    threadIdx: int, tileID: tuple, padWith: T = zero(T)) {.inline.} =
  ## Tile load, global memory into the register tile, blocked convention
  ## with the block's last axis spread across the simdgroup's 32 lanes.
  ##
  ## - interior tiles (whole covered tile inside `src`) flat-copy
  ## - ragged tiles go through guardedLoad on the coordinate map
  ##
  ## Args:
  ## - dst: the register tile, the lane's block
  ## - src: the global tensor
  ## - threadIdx: the lane index inside the threadgroup
  ## - tileID: the covered tile's coordinates over `src`'s tile grid
  ##
  ## padWith is the value read past bounds.
  const map = blockMapFor(ShD)
  let srcLayout = src.getLayout()
  let tiled = srcLayout.zipped_divide(map)
  let sel = ((threadIdx, _), tileID)
  let regView = src.slice(tiled, sel)

  let tiledCrd = make_identity_layout(srcLayout.shape).zipped_divide(map)
  let boxOrigin = tiledCrd.crd2idx((_, tileID))
  let boxCrd = tiledCrd.slice((_, tileID))
  let farCorner = boxOrigin + boxCrd.crd2idx(boxCrd.size() - 1)
  if farCorner.inBounds(srcLayout.shape):
    for i in 0 ..< regView.size():
      dst[i] = regView[i]
  else:
    let crdBase = tiledCrd.crd2idx(sel)
    let laneCrd = tiledCrd.slice(sel)
    for i in 0 ..< regView.size():
      dst[i] = src.guardedLoad(crdBase + laneCrd.crd2idx(i), padWith)

proc storeTile*[T, ShD, StD, ShS, StS](
    dst: TensorView[T, ShD, StD], src: Tile[T, ShS, StS],
    threadIdx: int, tileID: tuple) {.inline.} =
  ## Tile store, register tile into global memory under the blocked convention
  ##
  ## Args:
  ## - dst: the global tensor
  ## - src: the register tile, the lane's block
  ## - threadIdx: the lane index inside the threadgroup
  ## - tileID: the covered tile's coordinates over `dst`'s tile grid
  const map = blockMapFor(ShS)
  let dstLayout = dst.getLayout()
  let tiled = dstLayout.zipped_divide(map)
  let sel = ((threadIdx, _), tileID)
  let regView = dst.slice(tiled, sel)

  let tiledCrd = make_identity_layout(dstLayout.shape).zipped_divide(map)
  let boxOrigin = tiledCrd.crd2idx((_, tileID))
  let boxCrd = tiledCrd.slice((_, tileID))
  let farCorner = boxOrigin + boxCrd.crd2idx(boxCrd.size() - 1)
  if farCorner.inBounds(dstLayout.shape):
    for i in 0 ..< regView.size():
      regView[i] = src[i]
  else:
    let crdBase = tiledCrd.crd2idx(sel)
    let laneCrd = tiledCrd.slice(sel)
    for i in 0 ..< regView.size():
      dst.guardedStore(crdBase + laneCrd.crd2idx(i), src[i])

macro reducedStoreCrd*(blk: typed, tileID: tuple, c: typed): untyped =
  ## Destination coordinate of reduced-block coordinate `c` for covered
  ## tile `tileID`, the spread axis collapsed to the tile grid position.
  ##
  ## Contract:
  ## - `c` is a tuple for rank >= 2 reduced blocks, a scalar for rank 1
  let sh = staticShape(blk)
  var fields: seq[NimNode] = @[]
  for d in 0 ..< sh.len - 1:
    fields.add nnkInfix.newTree(bindSym"+",
      nnkInfix.newTree(bindSym"*", crdLeaf(tileID, d),
        newLit(sh[d])),
      crdLeaf(c, d))
  fields.add crdLeaf(tileID, sh.len - 1)
  result = nnkPar.newTree(fields)

template reducedStoreCrdFor*(blk: typed, tileID: tuple, c: typed): untyped =
  ## Per-instantiation expansion of reducedStoreCrd inside the generic procs
  reducedStoreCrd(blk, tileID, c)
