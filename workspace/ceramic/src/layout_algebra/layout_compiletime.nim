# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/ceramic/src/int_tuples

# ═══════════════════════════════════════════════════════════════
#  layoutTypeArgs, layout dimensions+types extraction
# ═══════════════════════════════════════════════════════════════

func layoutTypeArgs*(layout: NimNode): tuple[shapeTy, strideTy: NimNode] {.compileTime.} =
  ## Extract the Layout type's shape and stride type nodes from a typed expression, resolving type aliases.
  ##
  ## Removability: a macro can often avoid this helper by destructuring
  ## the layout first and passing
  ## the destructured shape and stride to a typed macro:
  ##
  ## - the destructured nodes arrive semchecked
  ##   (the nested call's argument semcheck)
  ## - the inner fold works on nodes and per-leaf values directly
  ##   (coalesce took this route, complement's transplant candidate)
  ##
  ## Prefer the destructure route when the fold does not need
  ## the whole-profile type peek in one shot.
  let typ = layout.getTypeInst()
  if typ.kind == nnkBracketExpr and typ[0].eqIdent("Layout"):
    return (typ[1], typ[2])
  if typ.kind == nnkSym:
    let objTy = typ.getTypeImpl()
    if objTy.kind == nnkObjectTy:
      for field in objTy[2]:
        if field.kind == nnkIdentDefs and field[0].eqIdent("shape"):
          result.shapeTy = field[1]
        elif field.kind == nnkIdentDefs and field[0].eqIdent("stride"):
          result.strideTy = field[1]
      if result.shapeTy != nil and result.strideTy != nil:
        return
  error("layoutTypeArgs: cannot recover Layout type args from " & typ.repr)

# ═══════════════════════════════════════════════════════════════
#  LayoutCT, compile-time Layout accumulator for macros
# ═══════════════════════════════════════════════════════════════

type LayoutCT* = object
  shape*, stride*: seq[NimNode]

proc append*(ct: var LayoutCT; sh, st: NimNode) {.compileTime.} =
  ct.shape.add sh
  ct.stride.add st

func emit*(ct: LayoutCT): NimNode {.compileTime.} =
  ## Build make_layout from accumulated dimensions (no coalesce).
  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  var outSh = newNimNode(nnkPar)
  var outSt = newNimNode(nnkPar)
  for i in 0 ..< ct.shape.len:
    outSh.add ct.shape[i]; outSt.add ct.stride[i]
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

proc appendDimension*(builder: var TupleBuilderNested;
                      pairs: seq[tuple[shape, stride: NimNode]]) {.compileTime.} =
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

func destructureLayout*(resultStmt: var NimNode; layoutAst: NimNode): tuple[shape, strides: NimNode] =
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
  let splitterNode = splitter
  proc dimensionCall(e: NimNode, idx: int): NimNode =
    ## `e.dimension(idx)` as a method-call node.
    let dim = ident"dimension"
    result = newCall(nnkDotExpr.newTree(e, dim), newLit(idx))
  proc fieldElem(e: NimNode, f: string, idx: int): NimNode =
    ## Element `idx` of field `f` on `e`.
    let fld = nnkDotExpr.newTree(e, ident(f))
    result = nnkBracketExpr.newTree(fld, newLit(idx))
  proc unwrap(e0: NimNode): NimNode =
    ## Typed macro parameters of macro call arguments arrive wrapped in a statement list. The value is the last child.
    result = e0
    if e0.kind == nnkStmtListExpr:
      result = e0[^1]

  let layoutShapeTy = layoutTypeArgs(layout).shapeTy
  let R = dimCount(layoutShapeTy)
  let tlrTy = unwrap(tiler).getTypeInst()
  if tlrTy.kind notin {nnkTupleTy, nnkTupleConstr}:
    return splitterNode.newCall(layout, tiler)
  let tilerRank = tlrTy.len
  doAssert tilerRank <= R,
    "hier_unzip: tiler has more dimensions (" & $tilerRank & ") than the layout (" & $R & ")"

  var stmts = newStmtList()
  var bindingCount = 0
  proc freshAlias(): NimNode {.compileTime.} =
    inc bindingCount
    ident("huzSplit" & $bindingCount)

  type Parts = tuple[fsh, fst, ssh, sst: seq[NimNode]]

  proc walk(e, eShapeTy, tval, ty: NimNode, needBinding: static bool): Parts {.compileTime.} =
    ## Split the layout dimension `e` by the tiler element of type `ty`.
    ## `eShapeTy` carries the element type of `e`, `tval` carries the tiler value expression.
    ## - a leaf, a scalar or Layout sub-tiler, emits one `splitter` call
    ## - a sub-tuple tiler emits one `splitter` call per sub-element and binds the gathered rank-2 result once
    ## Returns the shape and stride of the first and second gathered dimensions,
    ## one element per sub-dimension plus one per leftover layout dimension.
    ## `needBinding` false at the top level, there the gathered parts form the final Layout directly.
    if ty.kind in {nnkTupleTy, nnkTupleConstr}:
      doAssert ty.len <= dimCount(eShapeTy),
        "hier_unzip: tiler has more dimensions (" & $ty.len &
        ") than the layout dimension (" & $dimCount(eShapeTy) & ")"
      for j in 0 ..< ty.len:
        let child = walk(dimensionCall(e, j), eShapeTy[j],
                         nnkBracketExpr.newTree(tval, newLit(j)), ty[j], true)
        result.fsh.add child.fsh
        result.fst.add child.fst
        result.ssh.add child.ssh
        result.sst.add child.sst
      for j in ty.len ..< dimCount(eShapeTy):
        result.ssh.add fieldElem(e, "shape", j)
        result.sst.add fieldElem(e, "stride", j)
      when needBinding:
        let nodeR = freshAlias()
        stmts.add newCall(bindSym"evalOnceAs", nodeR,
          bindSym"make_layout".newCall(
            nnkTupleConstr.newTree(nnkTupleConstr.newTree(result.fsh),
                                   nnkTupleConstr.newTree(result.ssh)),
            nnkTupleConstr.newTree(nnkTupleConstr.newTree(result.fst),
                                   nnkTupleConstr.newTree(result.sst))))
        result.fsh = @[fieldElem(nodeR, "shape", 0)]
        result.fst = @[fieldElem(nodeR, "stride", 0)]
        result.ssh = @[fieldElem(nodeR, "shape", 1)]
        result.sst = @[fieldElem(nodeR, "stride", 1)]
    else:
      let leafR = freshAlias()
      stmts.add bindSym"evalOnceAs".newCall(leafR,
        splitterNode.newCall(e, tval))
      result.fsh = @[fieldElem(leafR, "shape", 0)]
      result.fst = @[fieldElem(leafR, "stride", 0)]
      result.ssh = @[fieldElem(leafR, "shape", 1)]
      result.sst = @[fieldElem(leafR, "stride", 1)]

  let top = walk(ident"huzLyt", layoutShapeTy, ident"huzTlr", tlrTy, false)
  stmts.insert(0, newCall(bindSym"evalOnceAs", ident"huzLyt", layout))
  stmts.insert(1, nnkLetSection.newTree(
    nnkIdentDefs.newTree(ident"huzTlr", newEmptyNode(), tiler)))
  stmts.add bindSym"make_layout".newCall(
    nnkTupleConstr.newTree(nnkTupleConstr.newTree(top.fsh), nnkTupleConstr.newTree(top.ssh)),
    nnkTupleConstr.newTree(nnkTupleConstr.newTree(top.fst), nnkTupleConstr.newTree(top.sst)))
  result = nnkBlockExpr.newTree(newEmptyNode(), stmts)
