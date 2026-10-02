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
#  LayoutCT, compile-time Layout accumulator for macros
# ═══════════════════════════════════════════════════════════════

type LayoutCT* = object
  shape*, stride*: seq[NimNode]

proc append*(ct: var LayoutCT, sh, st: NimNode) {.compileTime.} =
  ct.shape.add sh
  ct.stride.add st

func emit*(ct: LayoutCT): NimNode {.compileTime.} =
  ## Build make_layout from accumulated dimensions (no coalesce).
  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  var outSh = newNimNode(nnkPar)
  var outSt = newNimNode(nnkPar)
  for i in 0 ..< ct.shape.len:
    outSh.add ct.shape[i]
    outSt.add ct.stride[i]
  if ct.shape.len == 0:
    result = ident"make_layout".newCall(newLit(1), newLit(0))
  else:
    result = ident"make_layout".newCall(outSh, outSt)

# ═══════════════════════════════════════════════════════════════
#  emitLayout, builder-to-layout constructor
# ═══════════════════════════════════════════════════════════════

func emitLayout*(tb: TupleBuilderFlat or TupleBuilderNested, ctor: NimNode = nil):
    tuple[resultLayout: NimNode, verbatim: bool] {.compileTime.} =
  ## Emit a flat layout from an arity-2 tuple builder.
  ## If no `ctor` is passed, "make_layout(accumulated_shape, accumulated_stride)" will be emitted
  ##
  ## A verbatim flag is returned so the caller can use the original symbol
  ## if no transformation was applied to the stream.
  ## Otherwise the layout is reconstructed from elements.
  ##
  ## An empty builder emits make_layout(1, 0).
  let (sh, shV) = tb.emit(0, emitScalarForSize1 = true)
  let (st, stV) = tb.emit(1, emitScalarForSize1 = true)
  if sh.kind in {nnkPar, nnkTupleConstr} and sh.len == 0:
    (ident"make_layout".newCall(IntCT(1), newLit(0)), shV and stV)
  elif ctor.isNil():
    (ident"make_layout".newCall(sh, st), shV and stV)
  else:
    (ctor.newCall(sh, st), shV and stV)

proc appendDimension*(builder: var TupleBuilderNested, pairs: seq[tuple[shape, stride: NimNode]]) {.compileTime.} =
  ## Append a fold's pair set as one dimension slot.
  if pairs.len == 1:
    builder.append(pairs[0].shape, pairs[0].stride)
    return
  builder.append(TupleStreamEvent(path: @[], kind: kOpen, verbatim: true),
                 TupleStreamEvent(path: @[], kind: kOpen, verbatim: true))
  for p in pairs:
    builder.append(p.shape, p.stride)
  builder.append(TupleStreamEvent(path: @[], kind: kClose, verbatim: true),
                 TupleStreamEvent(path: @[], kind: kClose, verbatim: true))

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
  ## Precondition:
  ##   - layoutAst semantically type-checks as a Layout
  let typ = layoutAst.getTypeInst()
  let layoutTy = if typ.kind == nnkVarTy: typ[0] else: typ
  doAssert (layoutTy.kind == nnkBracketExpr and layoutTy[0].eqIdent("Layout")) or layoutTy.kind == nnkSym,
    "destructureLayout: expected a Layout, got " & typ.repr
  let inner = if layoutAst.kind == nnkStmtListExpr: layoutAst[^1] else: layoutAst
  if inner.kind == nnkObjConstr:
    for field in inner:
      if field.kind == nnkExprColonExpr:
        if field[0].eqIdent("shape"):
          result.shape = field[1]
        elif field[0].eqIdent("stride"):
          result.strides = field[1]
    # the semchecked constructor fields wrap the base tuples
    # in a hidden conversion, unwrap it.
    if result.shape.kind == nnkHiddenSubConv: result.shape = result.shape[^1]
    if result.strides.kind == nnkHiddenSubConv:
      result.strides = result.strides[^1]
    doAssert result.shape != nil and result.strides != nil,
      "destructureLayout: Layout constructor without shape/stride fields"
  elif layoutTy.kind == nnkSym or layoutAst.kind in {nnkCall, nnkCommand} or
      inner.kind in {nnkCall, nnkCommand}:
    # A layout-valued call, or a value typed through
    # a `typeof(make_layout(...))` alias symbol.
    let alias = ident("destructuredLayout")
    resultStmt.add bindSym"evalOnceAs".newCall(alias, layoutAst)
    result.shape = alias.newDotExpr(ident"shape")
    result.strides = alias.newDotExpr(ident"stride")
  else:
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

  let (tileShape, _) = tileParts.emit(0, emitScalarForSize1 = true)
  let (tileStride, _) = tileParts.emit(1, emitScalarForSize1 = true)
  let (restShape, _) = restParts.emit(0, emitScalarForSize1 = true)
  let (restStride, _) = restParts.emit(1, emitScalarForSize1 = true)
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
  builder.append(tileShape, tileStride, verbatim = false)
  let restShapeType = restShape.getTypeInst()
  if restShapeType.kind in {nnkTupleTy, nnkTupleConstr} and restShapeType.len > 0:
    for i in 0 ..< restShapeType.len:
      builder.append(getTupleIndex(restShape, i),
                     getTupleIndex(restStride, i), verbatim = false)
  else:
    builder.append(restShape, restStride, verbatim = false)
  result.add builder.emitLayout().resultLayout

macro zippedToFlatPairImpl*(tileShape, restShape, tileStride, restStride: typed): untyped =
  ## Flatten the zipped (tile, rest) parts, every leaf at one level
  result = newStmtList()
  var builder = TupleBuilderFlat.new(2)
  for (shapeEvent, strideEvent) in tileShape.tupleDimsStream().zip(tileStride.tupleDimsStream()):
    shapeEvent.onLeaves():
      builder.append(shapeEvent.leaf, strideEvent.leaf)
  for (shapeEvent, strideEvent) in restShape.tupleDimsStream().zip(restStride.tupleDimsStream()):
    shapeEvent.onLeaves():
      builder.append(shapeEvent.leaf, strideEvent.leaf)
  result.add builder.emitLayout().resultLayout

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
