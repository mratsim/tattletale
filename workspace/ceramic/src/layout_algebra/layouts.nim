# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout transforms and selectors: dimension, filter_zeros, padRight/Left,
## mapLeavesWith, zipDimensions, groupDimensions, upcast/downcast, etc.
##
## Re-exports `layouts_datatypes` (Layout type, predicates) and
## `layout_constructors` (make_layout, col_major_strides, LayoutCT).

import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/macros/static_for
import ./layouts_datatypes
import ./layout_constructors

export layouts_datatypes
export layout_constructors

# ═══════════════════════════════════════════════════════════════
#  dimension, extract dimension as rank-1 Layout
# ═══════════════════════════════════════════════════════════════

template dimension*(layout: Layout; idx: static int): auto =
  ## Extract dimension `idx` as a standalone rank-1 Layout.
  ## For scalar layouts (rank-1), only idx=0 is valid.
  when layout.shape is tuple:
    make_layout(layout.shape[idx], layout.stride[idx])
  else:
    static: doAssert idx == 0
    layout

# ═══════════════════════════════════════════════════════════════
#  isCompact, check if strides match canonical col-major ordering
# ═══════════════════════════════════════════════════════════════

func isCompact*(layout: Layout): bool =
  ## True when strides match canonical column-major ordering.
  ## Does not coalesce first, size-1 dimensions may cause false negatives.
  layout === (layout.shape, col_major_strides(layout.shape))

func isCompact*(layout: static Layout): static bool =
  ## True when strides match canonical column-major ordering.
  ## Does not coalesce first, size-1 dimensions may cause false negatives.
  layout === (layout.shape, col_major_strides(layout.shape))

# ═══════════════════════════════════════════════════════════════
#  filter_zeros, replace stride-0 shapes with Int[1]
# ═══════════════════════════════════════════════════════════════

macro filterZerosFlat(sh, st: typed): untyped =
  ## Stride-0 dimension shapes become Int[1], everything else stays as-is.
  let stT = st.getTypeInst()
  let shT = sh.getTypeInst()
  # scalar path, single dimension
  if shT.kind != nnkTupleConstr:
    if stT.kind == nnkBracketExpr and $stT[0] == "Int" and stT[1].intVal == 0:
      result = IntCT(1)
    else:
      result = sh
    return
  # tuple path, one shape element per stride type
  result = newNimNode(nnkTupleConstr)
  for i in 0 ..< shT.len:
    let stN = stT[i]
    if stN.kind == nnkBracketExpr and $stN[0] == "Int" and stN[1].intVal == 0:
      result.add IntCT(1)
    else:
      result.add newTree(nnkBracketExpr, sh, newLit(i))

template filter_zeros*(layout: Layout): auto =
  ## Replace stride-0 shapes with 1; returns flat (both shape and stride flattened).
  let st = flatten(layout.stride)
  let sh = filterZerosFlat(flatten(layout.shape), st)
  make_layout(sh, st)

# ═══════════════════════════════════════════════════════════════
#  Padding
# ═══════════════════════════════════════════════════════════════


macro padRight*(layout: Layout; rank: static int): untyped =
  ## Extend layout to target rank by padding with identity dimensions (1, 0).
  let shTyp = layoutTypeArgs(layout).shapeTy
  let curRank = if shTyp.kind == nnkTupleConstr: shTyp.len else: 1

  if curRank >= rank:
    result = layout
    return

  var ct = LayoutCT()
  if shTyp.kind == nnkTupleConstr:
    for i in 0 ..< shTyp.len:
      ct.shape.add newTree(nnkBracketExpr, newTree(nnkDotExpr, layout, ident"shape"), newLit i)
      ct.stride.add newTree(nnkBracketExpr, newTree(nnkDotExpr, layout, ident"stride"), newLit i)
  else:
    ct.shape.add newTree(nnkDotExpr, layout, ident"shape")
    ct.stride.add newTree(nnkDotExpr, layout, ident"stride")
  for i in curRank ..< rank:
    ct.shape.add IntCT(1)
    ct.stride.add IntCT(0)
  result = ct.emit()



macro padLeft*(layout: Layout; rank: static int): untyped =
  ## Extend layout to target rank by prepending identity dimensions (1, 0).
  let shTyp = layoutTypeArgs(layout).shapeTy
  let curRank = if shTyp.kind == nnkTupleConstr: shTyp.len else: 1

  if curRank >= rank:
    result = layout
    return

  var ct = LayoutCT()
  for i in 0 ..< (rank - curRank):
    ct.shape.add IntCT(1)
    ct.stride.add IntCT(0)
  if shTyp.kind == nnkTupleConstr:
    for i in 0 ..< shTyp.len:
      ct.shape.add nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit i)
      ct.stride.add nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit i)
  else:
    ct.shape.add nnkDotExpr.newTree(layout, ident"shape")
    ct.stride.add nnkDotExpr.newTree(layout, ident"stride")
  result = ct.emit()

# ═══════════════════════════════════════════════════════════════
#  mapLeavesWith, apply body to each leaf (shape, stride) pair
# ═══════════════════════════════════════════════════════════════

proc mapLeavesRec(
      stmts: var NimNode;
      shExpr, shTyp,
      stExpr, stTyp,
      body: NimNode): tuple[shape, stride: NimNode] {.compileTime.} =
  ## Recursively walks shape and stride in parallel.
  ## At each leaf pair, substitutes `it_sh` (leaf shape) and `it_st` (leaf stride)
  ## in `body`, evaluates body via evalOnceAs, and combines results
  ## into a Layout with the same nesting structure.
  ##
  ## Body must return (new_shape, new_stride).
  ##
  ## Examples:
  ##   mapLeavesWith(make_layout((2, 3))): (it_sh * 2, it_st)
  ##   # → (4, 6):(1, 2)
  ##
  ##   mapLeavesWith(make_layout((2, 3))):
  ##     let x = it_sh * 10
  ##     let y = it_st + 7
  ##     (x * y, x div y)
  ##   # → (160, 270):(2, 3)
  if shTyp.kind == nnkTupleConstr:
    var outSh = nnkPar.newNimNode()
    var outSt = nnkPar.newNimNode()
    for i in 0 ..< shTyp.len:
      let subSh = nnkBracketExpr.newTree(shExpr, newLit(i))
      let subSt = nnkBracketExpr.newTree(stExpr, newLit(i))
      let (childSh, childSt) =
        mapLeavesRec(
          stmts,
          subSh, shTyp[i],
          subSt, stTyp[i],
          body)
      outSh.add childSh
      outSt.add childSt
    return (shape: outSh, stride: outSt)
  else:
    proc subst(n: NimNode): NimNode =
      if n.kind in {nnkIdent, nnkSym} and n.eqIdent("it_sh"):
        result = shExpr
      elif n.kind in {nnkIdent, nnkSym} and n.eqIdent("it_st"):
        result = stExpr
      else:
        result = n.copyNimTree()
        for j in 0 ..< n.len:
          result[j] = subst(n[j])
    let blockExpr = nnkBlockExpr.newTree(newEmptyNode(), subst(body))
    let tmp = ident("pairLeaves_" & $(stmts.len+1))
    stmts.add quote do:
      evalOnceAs(`tmp`, `blockExpr`)
    return (shape: nnkBracketExpr.newTree(tmp, newLit(0)),
            stride: nnkBracketExpr.newTree(tmp, newLit(1)))

macro mapLeavesWith*(layout: Layout; body: untyped): untyped =
  ## Apply `body` to each leaf (shape, stride) pair.
  ## Body receives `it_sh` and `it_st`, must return (new_shape, new_stride).
  let bodyExpr = if body.kind == nnkStmtList and body.len == 1: body[0] else: body
  let (shTyp, stTyp) = layoutTypeArgs(layout)
  let shExpr = newTree(nnkDotExpr, layout, ident"shape")
  let stExpr = newTree(nnkDotExpr, layout, ident"stride")
  var stmts = newStmtList()
  let (outSh, outSt) = mapLeavesRec(stmts, shExpr, shTyp, stExpr, stTyp, bodyExpr)
  stmts.add nnkCall.newTree(bindSym"make_layout", outSh, outSt)
  result = nnkBlockExpr.newTree(newEmptyNode(), stmts)


# ═══════════════════════════════════════════════════════════════
#  upcast / downcast, reinterpret layout at coarser/finer granularity
# ═══════════════════════════════════════════════════════════════

template upcast*(layout: Layout; N: static int): auto =
  ## Reinterpret layout from finer to coarser granularity.
  ##
  ## Every N consecutive elements become one coarse element.
  ## shape shrinks, stride adjusts (not simply shape/N, stride*N).
  ##
  ## Examples:
  ##   upcast<4>(make_layout(32, 1))  # → (8, 1)  32 int8 → 8 int32
  ##   upcast<4>(make_layout(8, 2))   # → (4, 1)  strided int8 → int32

  mapLeavesWith(layout):
    when it_st is Int:
      when it_st.V == 0:
        (it_sh, it_st)
      else:
        # The stride is a multiple of N or N is a multiple of the stride
        static:
          doAssert abs(it_st.V) mod N == 0 or N mod abs(it_st.V) == 0,
            "upcast: stride " & $it_st.V & " and granularity " & $N & " are not divisible"
        (
          ceil_div(
            it_sh,
            ceil_div(N, abs(it_st))
          ),
          sign(it_st) * ceil_div(abs(it_st), N)
        )
    else:
      (it_sh, ceil_div(it_st, N))

template downcast*(layout: Layout; N: static int): auto =
  ## Reinterpret layout from coarser to finer granularity.
  ##
  ## Each coarse element splits into N finer elements.
  ## shape grows, stride adjusts (not simply shape*N, stride/N).
  ##
  ## Examples:
  ##   downcast<4>(make_layout(8, 1))  # → (32, 1)  8 int32 → 32 int8
  ##   downcast<4>(make_layout(8, 2))  # → (8, 8)   strided int32 → int8

  mapLeavesWith(layout):
    when it_st is Int:
      when abs(it_st.V) == 1:
        (it_sh * N, it_st)
      else:
        (it_sh, it_st * N)
    else:
      block:
        let new_sh = (if abs(it_st) == 1: it_sh * N else: it_sh)
        let new_st = (if abs(it_st) == 1: it_st else: it_st * N)
        (new_sh, new_st)

# ═══════════════════════════════════════════════════════════════
#  zipDimensions, interleave corresponding dimensions of two layouts
# ═══════════════════════════════════════════════════════════════

macro zipDimensions*[A, B: Layout](a: A, b: B): untyped =
  ## Zip dimensions of two layouts: interleave corresponding dimensions pairwise.
  ##
  ##   Given layouts A with dimensions (a0, a1, ..., aN) and
  ##   B with dimensions (b0, b1, ..., bN), zipDimensions produces a layout
  ##   with dimensions ((a0,b0), (a1,b1), ..., (aN,bN)).
  ##
  ##   For rank-1 inputs: (a:b, x:y) → ((a,x):(b,y))

  let (aShT, aStT) = layoutTypeArgs(a)
  let (bShT, bStT) = layoutTypeArgs(b)
  let aShape = newTree(nnkDotExpr, a, ident"shape")
  let bShape = newTree(nnkDotExpr, b, ident"shape")
  let aStride = newTree(nnkDotExpr, a, ident"stride")
  let bStride = newTree(nnkDotExpr, b, ident"stride")

  proc zipElems(valA, valB, typA, typB: NimNode): NimNode =
    let aIsTuple = typA.kind == nnkTupleConstr
    let bIsTuple = typB.kind == nnkTupleConstr
    if not aIsTuple and not bIsTuple:
      result = newTree(nnkTupleConstr, valA, valB)
    elif aIsTuple and bIsTuple:
      result = newNimNode(nnkTupleConstr)
      for i in 0 ..< typA.len:
        let ai = newTree(nnkBracketExpr, valA, newLit i)
        let bi = newTree(nnkBracketExpr, valB, newLit i)
        let subA = typA[i].getTypeInst()
        let subB = typB[i].getTypeInst()
        result.add zipElems(ai, bi, subA, subB)
    else:
      error "zipDimensions: mismatched rank"

  let zShape = zipElems(aShape, bShape, aShT, bShT)
  let zStride = zipElems(aStride, bStride, aStT, bStT)
  result = newCall(bindSym"make_layout", zShape, zStride)

# ═══════════════════════════════════════════════════════════════
#  groupDimensions, wrap dimensions [B, E) into a nested sub-Layout
# ═══════════════════════════════════════════════════════════════

macro groupDimensions*(layout: Layout; B, E: static int): untyped =
  ## Wraps dimensions at indices `[B, E)` into a nested sub-tuple in both
  ## shape and stride, producing a higher-rank Layout.
  ##
  ## Examples:
  ##   groupDimensions(make_layout((2, 3, 5, 7)), 0, 2)
  ##   # → ((2, 3), 5, 7):((1, 2), 6, 30)
  var ct = LayoutCT()
  let shTyp = layoutTypeArgs(layout).shapeTy
  let R =
    if shTyp.kind == nnkTupleConstr:
      shTyp.len
    else: 1
  for i in 0 ..< B:
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit i),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit i))
  var gSh = nnkPar.newNimNode()
  var gSt = nnkPar.newNimNode()
  for i in B ..< E:
    gSh.add nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit i)
    gSt.add nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit i)
  ct.append(gSh, gSt)
  for i in E ..< R:
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit i),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit i))
  result = ct.emit()


# ═══════════════════════════════════════════════════════════════
#  takeDimensions, extract dimensions [B, E) into a new Layout
# ═══════════════════════════════════════════════════════════════

macro takeDimensions*(layout: Layout; B, E: static int): untyped =
  ## Extract dimensions in range `[B, E)` into a new Layout.
  ## Returns a scalar Layout if only one dimension is extracted.
  ##
  ## Examples:
  ##   takeDimensions(make_layout((2, 3, 5, 7)), 1, 3)
  ##   # → (3, 5):(2, 6)
  var ct = LayoutCT()
  let shTyp = layoutTypeArgs(layout).shapeTy
  let R = if shTyp.kind == nnkTupleConstr: shTyp.len else: 1
  for i in B ..< min(E, R):
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit(i)),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit(i)))
  result = ct.emit()

# ═══════════════════════════════════════════════════════════════
#  selectDimensions, extract specific dimension indices into a new Layout
# ═══════════════════════════════════════════════════════════════

macro selectDimensions*(layout: Layout, Is: varargs[int]{lit|`const`}): untyped =
  ## Extract specific dimension indices into a new Layout.
  var ct = LayoutCT()
  for i in 0 ..< Is.len:
    let idx = Is[i].intVal
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit(idx)),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit(idx)))
  result = ct.emit()

# ═══════════════════════════════════════════════════════════════
#  replaceDimension, replace a dimension with a sub-Layout
# ═══════════════════════════════════════════════════════════════

macro replaceDimension*(layout: Layout; x: typed; N: static int): untyped =
  ## Replace dimension N of layout with Layout x.
  let shTyp = layoutTypeArgs(layout).shapeTy
  let R = if shTyp.kind == nnkTupleConstr: shTyp.len else: 1
  var ct = LayoutCT()
  for i in 0 ..< R:
    if i == N:
      ct.append(newTree(nnkDotExpr, x, ident"shape"),
                 newTree(nnkDotExpr, x, ident"stride"))
    else:
      ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit(i)),
                 nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit(i)))
  result = ct.emit()

# ═══════════════════════════════════════════════════════════════
#  transform_layout, map a layout's modes, one or two at a time
# ═══════════════════════════════════════════════════════════════

proc dimensionExpr(l: NimNode; idx: int): NimNode {.compileTime.} =
  ## `l.dimension(idx)` as a method-call node.
  let dim = ident"dimension"
  result = newCall(nnkDotExpr.newTree(l, dim), newLit(idx))

proc substDims(n: NimNode; itL, itT: NimNode): NimNode {.compileTime.} =
  ## Replace `it_l` with `itL` and `it_t` with `itT`, a nil binding stays as-is.
  if n.kind in {nnkIdent, nnkSym}:
    if itL != nil and n.eqIdent("it_l"):
      return itL
    if itT != nil and n.eqIdent("it_t"):
      return itT
    return n
  result = n.copyNimTree()
  for j in 0 ..< n.len:
    result[j] = substDims(n[j], itL, itT)

proc dimCount(ty: NimNode): int {.compileTime.} =
  ## Top-level dimension count of a shape or layout type node.
  if ty.kind in {nnkTupleConstr, nnkTupleTy}:
    ty.len
  else:
    1

proc emitDimLet(stmts, accSh, accSt: var seq[NimNode]; name: NimNode;
                  dimExpr: NimNode) {.compileTime.} =
  ## Bind `dimExpr` to `name`, append its shape and stride to the accumulators.
  stmts.add nnkLetSection.newTree(
    nnkIdentDefs.newTree(name, newEmptyNode(), dimExpr))
  accSh.add name.newDotExpr(ident"shape")
  accSt.add name.newDotExpr(ident"stride")

proc emitMappedDims(bindings, stmts, accSh, accSt: seq[NimNode];
    flatTop: bool): NimNode {.compileTime.} =
  ## Rebuild the layout over the accumulated shape and stride.
  ## flatTop unwraps a single dimension to scalar shape and stride,
  ## false keeps a 1-tuple.
  result = nnkStmtListExpr.newNimNode()
  for b in bindings:
    result.add b
  for st in stmts:
    result.add st
  result.add bindSym"make_layout".newCall(
    newTree(if flatTop: nnkPar else: nnkTupleConstr, accSh),
    newTree(if flatTop: nnkPar else: nnkTupleConstr, accSt))

macro transform_layout*(layout: typed; tiler: typed; body: untyped): untyped =
  ## Map the dimensions of `layout` against the dimensions of `tiler` through `body`.
  ##
  ## - `it_l` binds to layout dimension i, a rank-1 Layout
  ## - `it_t` binds to tiler dimension i, a rank-1 Layout for a Layout tiler
  ##   and the raw element for a tuple tiler
  ## - dimensions past the shorter side pass through unchanged, a tuple tiler
  ##   longer than the layout is a compile-time error
  ##
  ## Returns the layout rebuilt dimension by dimension.
  ##
  ## CuTe: transform_layout(l, t, f)
  let R = dimCount(layout.getTypeInst()[1])
  let tilerInst = tiler.getTypeInst()
  let tilerIsLayout = tilerInst.kind == nnkBracketExpr and tilerInst[0].eqIdent("Layout")
  let tilerRank = if tilerIsLayout:
                    dimCount(layoutTypeArgs(tiler).shapeTy)
                  else:
                    tilerInst.len
  if not tilerIsLayout:
    doAssert tilerRank <= R,
      "transform_layout: tiler has more dimensions (" & $tilerRank &
      ") than layout (" & $R & ")"

  var accSh, accSt: seq[NimNode]
  var stmts: seq[NimNode]
  let tlrNode = genSym(nskLet, "tlr")
  for idx in 0 ..< max(R, tilerRank):
    if idx < min(R, tilerRank):
      let itT = if tilerIsLayout:
                  dimensionExpr(ident("tlr"), idx)
                else:
                  nnkBracketExpr.newTree(tlrNode, newLit(idx))
      stmts.emitDimLet(accSh, accSt, genSym(nskLet, "d"),
        substDims(body, dimensionExpr(ident("lyt"), idx), itT))
    elif idx < R:
      stmts.emitDimLet(accSh, accSt, genSym(nskLet, "m"), dimensionExpr(ident("lyt"), idx))
    else:
      stmts.emitDimLet(accSh, accSt, genSym(nskLet, "m"), dimensionExpr(ident("tlr"), idx))
  let bindings = if tilerIsLayout:
                   @[newCall(bindSym"evalOnceAs", ident"lyt", layout),
                     newCall(bindSym"evalOnceAs", ident"tlr", tiler)]
                 else:
                   @[newCall(bindSym"evalOnceAs", ident"lyt", layout)]
  if not tilerIsLayout:
    stmts.insert(nnkLetSection.newTree(
      nnkIdentDefs.newTree(tlrNode, newEmptyNode(), tiler)), 0)
  result = emitMappedDims(bindings, stmts, accSh, accSt, false)

macro mapDimensionsWith*[L: Layout](arg: L; body: untyped): untyped =
  ## Map each dimension of Layout `arg` through `body`.
  ##
  ## - `it_l` binds to the current dimension, a rank-1 Layout
  ## - `body` evaluates to the replacement per dimension
  ##
  ## Returns the layout rebuilt dimension by dimension.
  ##
  ## CuTe: transform_layout(l, f)
  let R = dimCount(layoutTypeArgs(arg).shapeTy)
  var accSh, accSt: seq[NimNode]
  var stmts: seq[NimNode]
  for idx in 0 ..< R:
    stmts.emitDimLet(accSh, accSt, ident("r" & $idx),
      substDims(body, dimensionExpr(ident("lyt"), idx), nil))
  result = emitMappedDims(
    @[newCall(bindSym"evalOnceAs", ident"lyt", arg)],
    stmts, accSh, accSt, true)
