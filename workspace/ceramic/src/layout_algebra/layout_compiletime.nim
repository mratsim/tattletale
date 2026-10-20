# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/ceramic/src/int_tuples
import std/importutils
import ./layouts_unsanctioned_helpers
import ./layouts_datatypes

export layouts_unsanctioned_helpers
export layouts_datatypes
import ./ism_coord_strides

# ═══════════════════════════════════════════════════════════════
#  rank
# ═══════════════════════════════════════════════════════════════

func rank*(n: NimNode): int =
  ## Number of top-level modes of a type or value node
  ## - a bare Int[N] or int is rank 1
  ## - a tuple has one mode per member
  ## - a Layout[Sh, St] type ranks its shape
  ## - an expression ranks its type
  case n.kind
  of nnkBracketExpr:
    if n[0].kind == nnkSym and n[0].strVal == "Layout":
      n[1].rank()
    else:
      1
  of nnkTupleTy, nnkTupleConstr, nnkPar:
    if n.len == 1 and n[0].kind in {nnkPar, nnkTupleTy, nnkTupleConstr}:
      n[0].rank()
    else:
      n.len
  else:
    n.getTypeInst().rank()

# ═══════════════════════════════════════════════════════════════
#  hier_unzip, split a layout dimension by dimension, gather tiles and rest
# ═══════════════════════════════════════════════════════════════

macro hier_unzip(splitter: untyped, layout: typed, tiler: typed): untyped =
  ## Split `layout` by `tiler` through `splitter` and gather the parts into one rank-2 Layout:
  ## - dimension 0 carries the tile parts of every tiler element
  ## - dimension 1 carries the rest parts plus the leftover dimensions, PyCute hier_unzip chain semantics
  ## - a scalar (int, Int) or Layout tiler becomes `splitter(layout, tiler)` verbatim, a sub-tuple tiler element recurses
  ## Usage:
  ##   let r = hier_unzip(logical_divide, make_layout((4, 8), (1, 4)), (2, 4))
  ##   doAssert r === (((2, 4), (2, 2)), ((1, 4), (2, 16)))
  let tilerInner = if tiler.kind == nnkStmtListExpr: tiler[^1]
                   else: tiler
  let tlrTy = tilerInner.getTypeInst()
  if tlrTy.kind notin {nnkTupleTy, nnkTupleConstr}:
    return splitter.newCall(layout, tiler)
  var stmts = newStmtList()
  let (sh, st) = stmts.destructureLayout(layout)
  let (shapeTy, strideTy) = layoutTypeArgs(layout)

  proc huzTyAt(ty: NimNode, path: seq[int]): NimNode {.compileTime.} =
    result = ty
    for i in path:
      if result.isTupleTy():
        result = result[i]
      else:
        break

  proc huzAt(e, ty: NimNode, path: seq[int]): NimNode {.compileTime.} =
    result = e
    var t = ty
    for i in path:
      if not t.isTupleTy():
        break
      t = t[i]
      result = getTupleIndex(result, i)

  template huzArity(e, ty: NimNode, path: seq[int]): int =
    let n = huzAt(e, ty, path)
    if n.kind in {nnkTupleConstr, nnkPar}:
      n.len
    else:
      let t = huzTyAt(ty, path)
      if t.isTupleTy():
        t.len
      else: 1

  template huzProj(fld: string, dsh, dst, dtl: NimNode): NimNode =
    splitter.newCall(ident"make_layout".newCall(dsh, dst), dtl).newDotExpr(ident(fld))

  let R = huzArity(sh, shapeTy, @[])
  doAssert tlrTy.len <= R,
    "hier_unzip: tiler has more dimensions (" & $tlrTy.len & ") than the layout (" & $R & ")"

  var tileParts = TupleBuilderNested.new(2)
  var restParts = TupleBuilderNested.new(2)
  var levelState: seq[tuple[tilerLen, arity: int]]
  for ev in tiler.tupleStream():
    case ev.kind
    of kOpen:
      let tilerLen = huzArity(tilerInner, tlrTy, ev.path)
      let arity = huzArity(sh, shapeTy, ev.path)
      doAssert tilerLen <= arity,
        "hier_unzip: tiler has more dimensions (" & $tilerLen &
        ") than the layout dimension (" & $arity & ")"
      levelState.add (tilerLen, arity)
      tileParts.append(ev, ev)
      restParts.append(ev, ev)
    of kLeaf:
      let dsh = huzAt(sh, shapeTy, ev.path)
      let dst = huzAt(st, strideTy, ev.path)
      # spliced pure, once per projection, both copies stay pure expressions
      # the Nim and C compiler constnat fold and common sub-expression elimination
      # while when composed, the whole AST can be discarded if unused.
      let lsh = huzProj("shape", dsh, dst, ev.leaf)
      let lst = huzProj("stride", dsh, dst, ev.leaf)
      template splitTy(fld: string): NimNode =
        quote do: typeof(`lsh`.newDotExpr(ident(fld)))
      tileParts.append((getTupleIndex(lsh, 0), splitTy("shape")),
                       (getTupleIndex(lst, 0), splitTy("stride")))
      restParts.append((getTupleIndex(lsh, 1), splitTy("shape")),
                       (getTupleIndex(lst, 1), splitTy("stride")))
    of kClose:
      # dimensions past the tiler at this level pass through whole
      let lvl = levelState.pop()
      for j in lvl.tilerLen ..< lvl.arity:
        restParts.append((huzAt(sh, shapeTy, ev.path & j), huzTyAt(shapeTy, ev.path & j)),
                         (huzAt(st, strideTy, ev.path & j), huzTyAt(strideTy, ev.path & j)))
      tileParts.append(ev, ev)
      restParts.append(ev, ev)

  let tileShape = tileParts.emit(0, emitScalarForSize1 = true)
  let tileStride = tileParts.emit(1, emitScalarForSize1 = true)
  let restShape = restParts.emit(0, emitScalarForSize1 = true)
  let restStride = restParts.emit(1, emitScalarForSize1 = true)
  stmts.add ident"make_layout".newCall(
    nnkTupleConstr.newTree(tileShape, restShape),
    nnkTupleConstr.newTree(tileStride, restStride))
  result = nnkBlockExpr.newTree(newEmptyNode(), stmts)

# ═══════════════════════════════════════════════════════════════
#  emitLayout, builder-to-layout constructor
# ═══════════════════════════════════════════════════════════════

proc emitLayoutLit*(shapeVal, shapeTy, strideVal, strideTy: NimNode): NimNode {.compileTime.} =
  ## Layout[shapeTy, strideTy](shape: shapeVal, stride: strideVal),
  ## all parts already literal nodes.
  let LayoutSym = bindSym"Layout"
  result = quote do:
    `LayoutSym`[`shapeTy`, `strideTy`](shape: `shapeVal`, stride: `strideVal`)

func emitLayout*(tb: TupleBuilderFlat or TupleBuilderNested, emitScalarForSize1 = true): NimNode {.compileTime.} =
  let sh = tb.emit(0, emitScalarForSize1 = emitScalarForSize1)
  let st = tb.emit(1, emitScalarForSize1 = emitScalarForSize1)
  if sh.kind in {nnkPar, nnkTupleConstr} and sh.len == 0:
    result = ident"make_layout".newCall(IntCT(1), newLit(0))
  else:
    result = ident"make_layout".newCall(sh, st)


# ═══════════════════════════════════════════════════════════════


macro zippedToTiledPartsImpl*(tileShape, restShape, tileStride, restStride: typed): untyped =
  ## Regroup the zipped (tile, rest) parts into the tiled form:
  ## - the tile part kept whole
  ## - the rest part unpacked one level, a scalar kept whole
  result = newStmtList()
  var builder = TupleBuilderFlat.new(2)
  builder.append((tileShape, tileShape.getTypeInst()),
                 (tileStride, tileStride.getTypeInst()))
  let restShapeType = restShape.getTypeInst()
  let restStrideType = restStride.getTypeInst()
  if restShapeType.kind in {nnkTupleTy, nnkTupleConstr} and restShapeType.len > 0:
    for i in 0 ..< restShapeType.len:
      # the rest's static fields carry literal types, lifted to Int[v]
      # so the ty tuple stays uniform with the tile's Int brackets
      let shTyI = getTupleIndex(restShapeType, i)
      let stTyI = getTupleIndex(restStrideType, i)
      builder.append((getTupleIndex(restShape, i),
                      if shTyI.kind in {nnkIntLit .. nnkUInt64Lit}: intTy(shTyI.intVal.int)
                      else: shTyI),
                     (getTupleIndex(restStride, i),
                      if stTyI.kind in {nnkIntLit .. nnkUInt64Lit}: intTy(stTyI.intVal.int)
                      else: stTyI))
  else:
    builder.append((restShape, restShapeType), (restStride, restStrideType))
  result.add builder.emitLayout()

macro zippedToFlatPartsImpl*(tileShape, restShape, tileStride, restStride: typed): untyped =
  ## Flatten the zipped (tile, rest) parts, every leaf at one level
  result = newStmtList()
  var builder = TupleBuilderFlat.new(2)
  for (shapeEvent, strideEvent) in tileShape.tupleDimsStream().zip(tileStride.tupleDimsStream()):
    builder.onLeaves(shapeEvent):
      builder.append((shapeEvent.leaf, shapeEvent.leafTy),
                     (strideEvent.leaf, strideEvent.leafTy))
  for (shapeEvent, strideEvent) in restShape.tupleDimsStream().zip(restStride.tupleDimsStream()):
    builder.onLeaves(shapeEvent):
      builder.append((shapeEvent.leaf, shapeEvent.leafTy),
                     (strideEvent.leaf, strideEvent.leafTy))
  result.add builder.emitLayout()

macro zippedToTiledImpl*(zipped: typed): untyped =
  ## Reassemble the zipped (tile, rest) layout into the tiled form,
  ## the pair impl over the destructured dimensions
  result = newStmtList()
  let (zippedShape, zippedStride) = result.destructureLayout(zipped)
  result.add getAst(zippedToTiledPartsImpl(
    getTupleIndex(zippedShape, 0), getTupleIndex(zippedShape, 1),
    getTupleIndex(zippedStride, 0), getTupleIndex(zippedStride, 1)))

macro zippedToFlatImpl*(zipped: typed): untyped =
  ## Flatten the zipped (tile, rest) layout:
  ## - destructure, the pair impl over the dimensions
  result = newStmtList()
  let (zippedShape, zippedStride) = result.destructureLayout(zipped)
  result.add getAst(zippedToFlatPartsImpl(
    getTupleIndex(zippedShape, 0), getTupleIndex(zippedShape, 1),
    getTupleIndex(zippedStride, 0), getTupleIndex(zippedStride, 1)))

# ═══════════════════════════════════════════════════════════════
#  streamDims, takeDimensionsImpl, outer-dimension streams and slices
# ═══════════════════════════════════════════════════════════════

proc streamDims*(sh, st: NimNode): tuple[
    shapes, shapeTys, strides, strideTys: seq[NimNode], count: int] {.compileTime.} =
  var shapeStream = sh.tupleDimsStream()
  var strideStream = st.tupleDimsStream()
  while not shapeStream.done():
    let shapeEvent = shapeStream.next()
    let strideEvent = strideStream.next()
    result.shapes.add shapeEvent.leaf
    result.shapeTys.add shapeEvent.leafTy
    result.strides.add strideEvent.leaf
    result.strideTys.add strideEvent.leafTy
  result.count = result.shapes.len

macro takeDimensionsImpl*(originalLayout, sh, st: typed, B, E: static int): untyped =
  let (shapeLeaves, shapeTys, strideLeaves, strideTys, dimCount) = streamDims(sh, st)
  doAssert B >= 0, "takeDimensions: B must be a valid dimension index"
  if B <= 0 and E >= dimCount:
    return originalLayout
  var builder = TupleBuilderFlat.new(2)
  for i in B ..< min(E, dimCount):
    builder.append((shapeLeaves[i], shapeTys[i]), (strideLeaves[i], strideTys[i]))
  result = builder.emitLayout()

# ═══════════════════════════════════════════════════════════════
#  basisReprStream
# ═══════════════════════════════════════════════════════════════

func basisReprStream*(stride: NimNode): TupleStream =
  ## Unified stream for strides, regular scalars or coordinate strides
  privateAccess(TupleStream)
  let ty = stride.getTypeOf()
  let csDesc = ty.getCoordStrideDescriptor()
  if csDesc.kind == caNone:
    result.pending = TupleStreamEvent(path: @[], kind: kLeaf, leaf: stride, leafTy: ty)
    result.hasPending = true
  else:
    return csDesc.coeffs.tupleStream()