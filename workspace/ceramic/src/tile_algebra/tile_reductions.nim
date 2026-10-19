# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile-level reductions.
##
## Sums run the fixed 32-lane tree:
## - in-thread partials per output element
## - simdShuffleDown 16/8/4/2/1, then a lane-0 broadcast
##
## No lane id and no ownership derivation: the tree always sums dst slot d
## across all 32 lanes, each lane contributing the slot's own partial
## or the zero it filled with.
##
## Scope:
## - the spread axis (the block's last axis, distributed across the lanes)
## - the full tile
##
## A fully in-thread axis needs no tree and is out of scope.

import std/macros
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import ./tiles

func unwrapTypeDesc(t: NimNode): NimNode {.compileTime.} =
  ## Type behind a typeDesc-wrapped node
  if t.kind == nnkBracketExpr and t[0].kind == nnkSym and $t[0] == "typeDesc":
    t[1]
  else:
    t

macro sumConformance(dstSh, srcSh: typed, axis: static int) =
  ## Checks dst rank = src rank - 1, dst modes = src modes minus the axis mode
  let d = toSeqStaticInts(unwrapTypeDesc(dstSh.getTypeInst))
  let r = toSeqStaticInts(unwrapTypeDesc(srcSh.getTypeInst))
  doAssert r.len == d.len + 1, "sum: dst rank must be src rank - 1"
  doAssert axis >= 0 and axis < r.len, "sum: axis out of range"
  doAssert axis == r.len - 1,
    "sum: the reduced axis must be the spread axis (the block's last)"
  for i in 0 ..< d.len:
    let srcMode = (if i < axis: r[i] else: r[i + 1])
    doAssert d[i] == srcMode,
      "sum: dst mode " & $i & " must be src mode with axis removed (" & $d[i] &
      " vs " & $srcMode & ")"

macro dropMode(crd: typed, axis: static int): untyped =
  ## `crd` without its axis-th component: one bracket read per kept position,
  ## a scalar when one component remains
  var elems: seq[NimNode]
  let n = crd.getTypeInst().len
  doAssert axis >= 0 and axis < n, "dropMode: axis out of range"
  for i in 0 ..< n:
    if i != axis:
      elems.add nnkBracketExpr.newTree(crd, newLit(i))
  if elems.len == 1:
    result = elems[0]
  else:
    result = nnkTupleConstr.newTree(elems)

macro reducedShapeOf*(blk: typed): untyped =
  ## Reduced tile shape of `blk`: the block with the spread axis removed,
  ## as an inline Int-tuple literal
  let sh = staticShape(blk)
  var par = nnkPar.newTree()
  for i in 0 ..< sh.len - 1:
    par.add newLit(sh[i])
  result = bindSym"makeIntTuple".newCall(par)

macro reducedAxisOf*(blk: typed): untyped =
  ## Spread axis of a block: its last mode
  result = newLit(staticShape(blk).len - 1)

proc sum*[T, ShD, StD, ShS, StS](
    dst: var Tile[T, ShD, StD],
    src: Tile[T, ShS, StS], axis: static int) {.inline.} =
  ## Per-row sums over the spread axis: in-thread partials, then a fixed
  ## 32-lane tree and lane-0 broadcast, every lane holding the totals.
  ##
  ## Args:
  ## - dst: rank(src) - 1, src's modes with the axis removed
  ## - src: the tile to reduce
  ## - axis: the spread axis, the block's last
  static: sumConformance(ShD, ShS, axis)
  dst.fillWith(zero(T))
  for i in 0 ..< src.size():
    let crd = idx2crd(src.getLayout(), i)
    let dFlat = crd2idx(dst.getLayout(), dropMode(crd, axis)).toInt
    dst.data[dFlat] = dst.data[dFlat] + src.data[i]
  for d in 0 ..< dst.size():
    var acc = dst.data[d]
    acc = acc + simdShuffleDown(acc, 16)
    acc = acc + simdShuffleDown(acc, 8)
    acc = acc + simdShuffleDown(acc, 4)
    acc = acc + simdShuffleDown(acc, 2)
    acc = acc + simdShuffleDown(acc, 1)
    acc = simdShuffle(acc, 0)
    dst.data[d] = acc

proc sum*[T, Sh, St](src: Tile[T, Sh, St]): T {.inline.} =
  ## Full-tile sum: the lane sums its block, then the fixed 32-lane tree
  ## and lane-0 broadcast. Every lane returns the tile total.
  var acc = zero(T)
  for i in 0 ..< src.size():
    acc = acc + src.data[i]
  acc = acc + simdShuffleDown(acc, 16)
  acc = acc + simdShuffleDown(acc, 8)
  acc = acc + simdShuffleDown(acc, 4)
  acc = acc + simdShuffleDown(acc, 2)
  acc = acc + simdShuffleDown(acc, 1)
  simdShuffle(acc, 0)
