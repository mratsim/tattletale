# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/ceramic/src/int_tuples
import ./layouts_unsanctioned_helpers

# ═══════════════════════════════════════════════════════════════
#  emitLayout, builder-to-layout constructor
# ═══════════════════════════════════════════════════════════════

func emitLayout*(tb: TupleBuilderFlat or TupleBuilderNested, ctor: NimNode = nil, emitScalarForSize1 = true): NimNode {.compileTime.} =
  ## Emit a flat layout from an arity-2 tuple builder.
  ## If no `ctor` is passed, "make_layout(accumulated_shape, accumulated_stride)" will be emitted
  ##
  ## An empty builder emits make_layout(1, 0).
  let sh = tb.emit(0, emitScalarForSize1 = emitScalarForSize1)
  let st = tb.emit(1, emitScalarForSize1 = emitScalarForSize1)
  if sh.kind in {nnkPar, nnkTupleConstr} and sh.len == 0:
    ident"make_layout".newCall(IntCT(1), newLit(0))
  elif ctor.isNil():
    ident"make_layout".newCall(sh, st)
  else:
    ctor.newCall(sh, st)

proc appendDimension*(builder: var TupleBuilderNested, pairs: seq[tuple[shape, stride: NimNode]]) {.compileTime.} =
  ## Append a fold's pair set as one dimension slot.
  if pairs.len == 1:
    builder.append(pairs[0].shape, pairs[0].stride)
    return
  builder.append(TupleStreamEvent(path: @[], kind: kOpen),
                 TupleStreamEvent(path: @[], kind: kOpen))
  for p in pairs:
    builder.append(p.shape, p.stride)
  builder.append(TupleStreamEvent(path: @[], kind: kClose),
                 TupleStreamEvent(path: @[], kind: kClose))

# ═══════════════════════════════════════════════════════════════
#  destructureLayout, layout AST -> (shape, stride) expressions
# ═══════════════════════════════════════════════════════════════

func destructureLayout*(resultStmt: var NimNode, layoutAst: NimNode): tuple[shape, strides: NimNode] =
  ## Destructure a typed Layout into (shape, stride) tuple expressions,
  ## without forcing a Layout materialization when the AST already
  ## carries the base tuples:
  ## - lvalue emits `layout.shape` / `layout.stride` field reads
  ## - Layout object constructor, possibly wrapped in a StmtListExpr:
  ##   strips it, the base tuples pass through as-is
  ## - layout-valued call or a `typeof(make_layout(...))` alias-typed
  ##   value materializes once through evalOnceAs, the binding is
  ##   appended to `resultStmt`
  ##
  ## Args:
  ##   - resultStmt
  ##     statement list extended with materialization bindings
  ##   - layoutAst
  ##     a typed Layout expression
  ## Returns:
  ##   - shape, carries the shape tuple expression of the destructured layout
  ##   - strides, carries the stride tuple expression of the destructured layout
  ## Pattern:
  ##   - a one-line template and getAst delegate the (shape, stride) pair to a typed macro, right after the call
  ##   - the Impl walks the pair's leaves with tupleStream, `leafTy` carries each leaf's type
  ##   - an unexpanded make_layout(shape, stride) call, say a getAst expansion, strips to its base tuple arguments, no type needed
  ## Precondition:
  ##   - layoutAst semantically type-checks as a Layout, unless it is an unexpanded make_layout(shape, stride) call
  var inner = layoutAst
  while inner.kind in {nnkStmtList, nnkStmtListExpr, nnkBlockExpr}:
    inner = inner[^1]
  if inner.kind == nnkCall and inner[0].eqIdent("make_layout") and inner.len == 3:
    result.shape = inner[1]
    result.strides = inner[2]
    return
  if layoutAst.kind notin {nnkSym, nnkIdent}:
    let typ = layoutAst.getTypeInst()
    let layoutTy = if typ.kind == nnkVarTy: typ[0] else: typ
    doAssert (layoutTy.kind == nnkBracketExpr and layoutTy[0].eqIdent("Layout")) or layoutTy.kind == nnkSym,
      "destructureLayout: expected a Layout, got " & typ.repr
    if inner.kind == nnkObjConstr:
      for field in inner:
        if field.kind == nnkExprColonExpr:
          if field[0].eqIdent("shape"):
            result.shape = field[1]
          elif field[0].eqIdent("stride"):
            result.strides = field[1]
      # the semchecked constructor fields wrap the base tuples
      # in a hidden conversion, unwrap it.
      while result.shape.kind == nnkHiddenSubConv:
        result.shape = result.shape[^1]
      while result.strides.kind == nnkHiddenSubConv:
        result.strides = result.strides[^1]
      doAssert result.shape != nil and result.strides != nil,
        "destructureLayout: Layout constructor without shape/stride fields"
      return
    if layoutTy.kind == nnkSym or layoutAst.kind in {nnkCall, nnkCommand} or
        inner.kind in {nnkCall, nnkCommand}:
      # A layout-valued call, or a value typed through
      # a `typeof(make_layout(...))` alias symbol.
      let alias = ident("destructuredLayout")
      resultStmt.add bindSym"evalOnceAs".newCall(alias, layoutAst)
      result.shape = alias.newDotExpr(ident"shape")
      result.strides = alias.newDotExpr(ident"stride")
      return

  result.shape = layoutAst.newDotExpr(ident"shape")
  result.strides = layoutAst.newDotExpr(ident"stride")

# ═══════════════════════════════════════════════════════════════
#  hier_unzip, split a layout dimension by dimension, gather tiles and rest
# ═══════════════════════════════════════════════════════════════

macro hier_unzip*(splitter: untyped, layout: typed, tiler: typed): untyped =
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
      tileParts.append(getTupleIndex(lsh, 0), getTupleIndex(lst, 0))
      restParts.append(getTupleIndex(lsh, 1), getTupleIndex(lst, 1))
    of kClose:
      # dimensions past the tiler at this level pass through whole
      let lvl = levelState.pop()
      for j in lvl.tilerLen ..< lvl.arity:
        restParts.append(huzAt(sh, shapeTy, ev.path & j), huzAt(st, strideTy, ev.path & j))
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


macro zippedToTiledPairImpl*(tileShape, restShape, tileStride, restStride: typed): untyped =
  ## Regroup the zipped (tile, rest) parts into the tiled form:
  ## - the tile part kept whole
  ## - the rest part unpacked one level, a scalar kept whole
  result = newStmtList()
  var builder = TupleBuilderFlat.new(2)
  builder.append(tileShape, tileStride)
  let restShapeType = restShape.getTypeInst()
  if restShapeType.kind in {nnkTupleTy, nnkTupleConstr} and restShapeType.len > 0:
    for i in 0 ..< restShapeType.len:
      builder.append(getTupleIndex(restShape, i),
                     getTupleIndex(restStride, i))
  else:
    builder.append(restShape, restStride)
  result.add builder.emitLayout()

macro zippedToFlatPairImpl*(tileShape, restShape, tileStride, restStride: typed): untyped =
  ## Flatten the zipped (tile, rest) parts, every leaf at one level
  result = newStmtList()
  var builder = TupleBuilderFlat.new(2)
  for (shapeEvent, strideEvent) in tileShape.tupleDimsStream().zip(tileStride.tupleDimsStream()):
    builder.onLeaves(shapeEvent):
      builder.append(shapeEvent.leaf, strideEvent.leaf)
  for (shapeEvent, strideEvent) in restShape.tupleDimsStream().zip(restStride.tupleDimsStream()):
    builder.onLeaves(shapeEvent):
      builder.append(shapeEvent.leaf, strideEvent.leaf)
  result.add builder.emitLayout()

macro zippedToTiledImpl*(zipped: typed): untyped =
  ## Reassemble the zipped (tile, rest) layout into the tiled form,
  ## the pair impl over the destructured dimensions
  result = newStmtList()
  let (zippedShape, zippedStride) = result.destructureLayout(zipped)
  result.add getAst(zippedToTiledPairImpl(
    getTupleIndex(zippedShape, 0), getTupleIndex(zippedShape, 1),
    getTupleIndex(zippedStride, 0), getTupleIndex(zippedStride, 1)))

macro zippedToFlatImpl*(zipped: typed): untyped =
  ## Flatten the zipped (tile, rest) layout:
  ## - destructure, the pair impl over the dimensions
  result = newStmtList()
  let (zippedShape, zippedStride) = result.destructureLayout(zipped)
  result.add getAst(zippedToFlatPairImpl(
    getTupleIndex(zippedShape, 0), getTupleIndex(zippedShape, 1),
    getTupleIndex(zippedStride, 0), getTupleIndex(zippedStride, 1)))

# ═══════════════════════════════════════════════════════════════
#  streamDims, takeDimensionsImpl, outer-dimension streams and slices
# ═══════════════════════════════════════════════════════════════

proc streamDims*(sh, st: NimNode): tuple[shapes, strides: seq[NimNode], count: int] {.compileTime.} =
  var shapeStream = sh.tupleDimsStream()
  var strideStream = st.tupleDimsStream()
  while not shapeStream.done():
    let shapeEvent = shapeStream.next()
    let strideEvent = strideStream.next()
    result.shapes.add shapeEvent.leaf
    result.strides.add strideEvent.leaf
  result.count = result.shapes.len

macro takeDimensionsImpl*(originalLayout, sh, st: typed, B, E: static int): untyped =
  let (shapeLeaves, strideLeaves, dimCount) = streamDims(sh, st)
  doAssert B >= 0, "takeDimensions: B must be a valid dimension index"
  if B <= 0 and E >= dimCount:
    result = originalLayout
    return
  var builder = TupleBuilderFlat.new(2)
  for i in B ..< min(E, dimCount):
    builder.append(shapeLeaves[i], strideLeaves[i])
  result = builder.emitLayout()

