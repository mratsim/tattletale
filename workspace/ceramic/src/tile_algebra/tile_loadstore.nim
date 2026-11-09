# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
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

# ═══════════════════════════════════════════════════════════════════════
#  Thread->value map offset unfolding
# ═══════════════════════════════════════════════════════════════════════

type
  TvmCoeff = tuple[axis, coeff: int]
    ## One stride-leaf contribution: `coeff` per coordinate unit, added to coordinate axis `axis`.
  TvmLeaf = object
    ## One leaf of a thread->value map layout: the dimension and the coordinate contributions of one unit.
    dim: int
    coeffs: seq[TvmCoeff]

proc flattenTvmCoeffs(n: NimNode, axis: var int, coeffs: var seq[TvmCoeff]) {.compileTime.} =
  ## Flattens a coordinate stride's coefficient tree into (axis, coeff) pairs, axes numbered in tree order.
  if n.kind in {nnkTupleConstr, nnkPar}:
    for child in n:
      flattenTvmCoeffs(child, axis, coeffs)
  else:
    coeffs.add (axis, n.intVal.int)
    axis.inc

proc collectTvmLeaves(sh, st: NimNode, leaves: var seq[TvmLeaf]) {.compileTime.} =
  ## Flattens a (shape, stride) type tree into its leaves, tree order.
  ## Stride leaves:
  ## - a coordinate-stride leaf contributes one pair per flat coefficient
  ## - a scalar static stride leaf contributes its value on axis 0
  if sh.kind in {nnkTupleConstr, nnkPar, nnkTupleTy}:
    for i in 0 ..< sh.len:
      collectTvmLeaves(sh[i], st[i], leaves)
  else:
    var leaf: TvmLeaf
    leaf.dim = sh.getStaticInt()
    if st.isCoordStride():
      var axis = 0
      flattenTvmCoeffs(st.getCoordStrideDescriptor().coeffs, axis, leaf.coeffs)
    else:
      leaf.coeffs = @[(0, st.getStaticInt())]
    leaves.add leaf

proc tvmAxisCount(leaves: seq[TvmLeaf]): int {.compileTime.} =
  ## Number of coordinate axes the leaves spread over.
  result = 1
  for leaf in leaves:
    for (axis, _) in leaf.coeffs:
      result = max(result, axis + 1)

proc tvmLayoutAxes(layoutTy: NimNode): int {.compileTime.} =
  ## Coordinate-axis count of a thread->value map layout.
  ## The count is the widest flat coefficient vector over all stride leaves,
  ## one shared rank for the thread part and the register part.
  var leaves: seq[TvmLeaf] = @[]
  collectTvmLeaves(layoutTy[1], layoutTy[2], leaves)
  tvmAxisCount(leaves)

proc emitTvmUnfold(stmts: NimNode, leaves: seq[TvmLeaf], nAxes: int, idx: NimNode) {.compileTime.} =
  ## Appends the offset computation of one flat index over `leaves`:
  ## - the index decomposes layout-left over the leaf dimensions
  ##   (`idx2crd` order), every quotient carried to the next dimension.
  ##   The last dimension keeps its quotient unmodded
  ## - each coordinate scales its leaf's nonzero coefficients
  ## - one accumulator per coordinate axis over `nAxes` with the offsets
  ##   returned as a tuple
  let rem = genSym(nskVar, "tvmRem")
  stmts.add quote do:
    var `rem` = `idx`
  var accs: seq[NimNode] = @[]
  for axis in 0 ..< nAxes:
    let acc = genSym(nskVar, "tvmAcc" & $axis)
    stmts.add quote do:
      var `acc` = 0
    accs.add acc
  for k in 0 ..< leaves.len:
    let leaf = leaves[k]
    let crd = genSym(nskLet, "tvmCrd" & $k)
    if k < leaves.len - 1:
      let dim = newLit(leaf.dim)
      stmts.add quote do:
        let `crd` = `rem` mod `dim`
        `rem` = `rem` div `dim`
    else:
      stmts.add quote do:
        let `crd` = `rem`
    for (axis, coeff) in leaf.coeffs:
      if coeff != 0:
        let co = newLit(coeff)
        let target = accs[axis]
        stmts.add quote do:
          `target` = `target` + `crd` * `co`
  var parts = nnkTupleConstr.newTree()
  for acc in accs:
    parts.add acc
  stmts.add parts

proc tvmRegisterTrees(layoutTy: NimNode): tuple[sh, st: NimNode] {.compileTime.} =
  ## (value dims, cell-repetition dims) of a thread->value map layout type:
  ## the register part that the map loads into.
  result.sh = nnkTupleConstr.newTree(layoutTy[1][0][1], layoutTy[1][1])
  result.st = nnkTupleConstr.newTree(layoutTy[2][0][1], layoutTy[2][1])

macro tvmThreadOffset(layout: Layout, threadIdx: int): untyped =
  ## Coordinate offset of a thread's slots in a thread->value map,
  ## unfolded from the map layout's static type tree at compile time.
  ##
  ## Structure of a thread->value map layout:
  ## `((thread dims, value dims), cell-repetition dims)` with the thread dims at `layout.shape[0][0]`.
  ##
  ## The flat thread index decomposes over the thread dims layout-left,
  ## each dimension's coordinate scales its stride leaf's coefficients
  ## (leaf contributions: `collectTvmLeaves`, offset accumulation: `emitTvmUnfold`).
  ##
  ## Returns a tuple of runtime ints with one entry per coordinate axis.
  ## `crd2idx` over runtime coordinates cannot evaluate these layouts.
  ## Composition leaves a zero stride as an `Int[0]` beside `CoordStride` leaves.
  ## The runtime sum has no `+` across the two.
  let ty = layout.getTypeInst()
  var leaves: seq[TvmLeaf] = @[]
  collectTvmLeaves(ty[1][0][0], ty[2][0][0], leaves)
  var stmts = nnkStmtList.newTree()
  emitTvmUnfold(stmts, leaves, tvmLayoutAxes(ty), threadIdx)
  result = nnkBlockExpr.newTree(newEmptyNode(), stmts)

macro tvmRegisterSize(layout: Layout): int =
  ## Register slot count of a thread->value map. The count is the product
  ## of the value and cell-repetition dimensions.
  let ty = layout.getTypeInst()
  var leaves: seq[TvmLeaf] = @[]
  let trees = tvmRegisterTrees(ty)
  collectTvmLeaves(trees.sh, trees.st, leaves)
  var product = 1
  for leaf in leaves:
    product *= leaf.dim
  result = newLit(product)

macro tvmRegisterOffset(layout: Layout, regIdx: int): untyped =
  ## Coordinate offset of one register slot in a thread->value map.
  ## The flat register index decomposes over the non-thread dimensions.
  ## The leaf contributions are the same as for `tvmThreadOffset`.
  ## Returns the offset as a tuple of runtime ints with one entry per coordinate axis.
  let ty = layout.getTypeInst()
  var leaves: seq[TvmLeaf] = @[]
  let trees = tvmRegisterTrees(ty)
  collectTvmLeaves(trees.sh, trees.st, leaves)
  var stmts = nnkStmtList.newTree()
  emitTvmUnfold(stmts, leaves, tvmLayoutAxes(ty), regIdx)
  result = nnkBlockExpr.newTree(newEmptyNode(), stmts)

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
  let thrBase = tvmThreadOffset(tvm.layout, threadIdx)
  let regSize = tvmRegisterSize(tvm.layout)
  let tileBase = (tileID[0] * tvm.tileShape[0], tileID[1] * tvm.tileShape[1])
  let farCorner = tileBase + tvm.layout.crd2idx(tvm.layout.size() - 1)
  if farCorner.inBounds(src.shape):
    for i in 0 ..< regSize:
      dst[i] = src[tileBase + thrBase + tvmRegisterOffset(tvm.layout, i)]
  else:
    for i in 0 ..< regSize:
      dst[i] = src.paddedLoad(tileBase + thrBase + tvmRegisterOffset(tvm.layout, i), padWith)

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
  let thrBase = tvmThreadOffset(tvm.layout, threadIdx)
  let regSize = tvmRegisterSize(tvm.layout)
  let tileBase = (tileID[0] * tvm.tileShape[0], tileID[1] * tvm.tileShape[1])
  let farCorner = tileBase + tvm.layout.crd2idx(tvm.layout.size() - 1)
  if farCorner.inBounds(dst.shape):
    for i in 0 ..< regSize:
      dst[tileBase + thrBase + tvmRegisterOffset(tvm.layout, i)] = src[i]
  else:
    for i in 0 ..< regSize:
      dst.guardedStore(tileBase + thrBase + tvmRegisterOffset(tvm.layout, i), src[i])

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
