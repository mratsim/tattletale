# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout transforms and selectors: mode, filter_zeros, padRight/Left,
## mapLeavesWith, zipModes, groupModes, upcast/downcast, etc.
##
## Re-exports `layouts_datatypes` (Layout type, predicates) and
## `layout_constructors` (make_layout, col_major_strides, LayoutCT).
##
## Reference:
##   - CuTe C++: layout.hpp

import std/macros
import ./int_tuples
import ./macros/static_for
import ./layouts_datatypes
import ./layout_constructors

export layouts_datatypes
export layout_constructors

# ═══════════════════════════════════════════════════════════════
#  mode — extract mode as rank-1 Layout
# ═══════════════════════════════════════════════════════════════

template mode*(layout: Layout; idx: static int): auto =
  ## Extract mode `idx` as a standalone rank-1 Layout.
  ## For scalar layouts (rank-1), only idx=0 is valid.
  when layout.shape is tuple:
    make_layout(layout.shape[idx], layout.stride[idx])
  else:
    static: doAssert idx == 0
    layout

# ═══════════════════════════════════════════════════════════════
#  isCompact — check if strides match canonical col-major ordering
# ═══════════════════════════════════════════════════════════════

func isCompact*(layout: Layout): bool =
  ## True when strides match canonical column-major ordering.
  ## Note: does NOT coalesce first — size-1 modes may cause false negatives.
  layout === (layout.shape, col_major_strides(layout.shape))

func isCompact*(layout: static Layout): static bool =
  ## True when strides match canonical column-major ordering.
  ## Note: does NOT coalesce first — size-1 modes may cause false negatives.
  layout === (layout.shape, col_major_strides(layout.shape))

# ═══════════════════════════════════════════════════════════════
#  filter_zeros — replace stride-0 shapes with Int[1]
# ═══════════════════════════════════════════════════════════════

macro filterZerosFlat(sh, st: typed): untyped =
  ## Stride-0 mode shapes → Int[1](), everything else as-is.
  let stT = st.getTypeInst()
  let shT = sh.getTypeInst()
  # ── scalar path: single mode ──
  if shT.kind != nnkTupleConstr:
    if stT.kind == nnkBracketExpr and $stT[0] == "Int" and stT[1].intVal == 0:
      result = IntCT(1)
    else:
      result = sh
    return
  # ── tuple path: iterate over modes ──
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
#  layoutTypeArgs — shape/stride TYPE extraction, nnkSym-safe
# ═══════════════════════════════════════════════════════════════

func layoutTypeArgs*(layout: NimNode): tuple[shapeTy, strideTy: NimNode] {.compileTime.} =
  ## Shape/stride TYPE nodes of a Layout-typed expression, nnkSym-safe.
  ## A module-scope `type A = typeof(make_layout(...))` alias (atoms_nvidia
  ## declares SM80_* exactly this way) makes getTypeInst return the alias
  ## symbol (nnkSym — no children), so `typ[1]` crashes. Recover the args
  ## from the alias's definition instead; kind-structure equivalent
  ## (tuple/scalar, nesting, .len) to the non-aliased case. Design:
  ## layoutTypeArgs recovers the Layout type arguments from an alias's
  ## typedef (nnkSym-safe), so compose sees the actual shape/stride types.
  let typ = layout.getTypeInst()
  if typ.kind == nnkBracketExpr and typ[0].eqIdent("Layout"):
    return (typ[1], typ[2])
  if typ.kind == nnkSym:
    let rhs = typ.getImpl()[2]          # typedef RHS: A = <type expr>
    let inner =                         # unwrap typeof(...)
      if rhs.kind in {nnkCall, nnkCommand} and rhs[0].eqIdent("typeof"): rhs[1]
      else: rhs
    if inner.kind in {nnkCall, nnkCommand} and inner[0].eqIdent("make_layout"):
      return (inner[1].getTypeInst(), inner[2].getTypeInst())
    if inner.kind == nnkBracketExpr and inner[0].eqIdent("Layout"):
      return (inner[1], inner[2])
  error("layoutTypeArgs: cannot recover Layout type args from " & typ.repr)

# ═══════════════════════════════════════════════════════════════
#  padRight — extend layout rank by padding with identity modes
# ═══════════════════════════════════════════════════════════════

#  CuTe: append<R>(layout) pads to rank R with (1, 0) modes.
#  Used by blocked_product / raked_product to equalize ranks.
#  Pads on the RIGHT (appends identity modes at the end).

macro padRight*(layout: Layout; rank: static int): untyped =
  ## Extend layout to target rank by padding with identity modes (1, 0).
  ## Identity modes are appended on the right.
  ## Zero-cost if layout is at least the target rank. The input AST is passed as-is
  ## No intermediate value is materialized.
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

#  padLeft — extend layout rank by prepending identity modes

#  CuTe: prepend<R>(layout) pads to rank R with (1, 0) modes.
#  Used by gemm.hpp to lift 2D operands to 3D.
#  Pads on the LEFT (prepends identity modes at the front).

macro padLeft*(layout: Layout; rank: static int): untyped =
  ## Extend layout to target rank by prepending identity modes (1, 0).
  ## Identity modes are prepended on the left.
  ## Zero-cost if layout is at least the target rank.
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
#  mapLeavesWith — apply body to each leaf (shape, stride) pair
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
#  upcast / downcast — reinterpret layout at coarser/finer granularity
# ═══════════════════════════════════════════════════════════════
#
#  CuTe: upcast<N>(layout), downcast<N>(layout)
#
#  Building block of recast_layout<OldType, NewType>.  upcast by N when
#  sizeof ratio = N (e.g. int8→int32 is upcast<4>); downcast when ratio < 1.

template upcast*(layout: Layout; N: static int): auto =
  ## Reinterpret layout from finer to coarser granularity.
  ##
  ## Every N consecutive elements become one coarse element.
  ## shape shrinks, stride adjusts (not simply shape/N, stride*N).
  ##
  ## Examples:
  ##   upcast<4>(make_layout(32, 1))  # → (8, 1)  32 int8 → 8 int32
  ##   upcast<4>(make_layout(8, 2))   # → (4, 1)  strided int8 → int32

  # ── Why not just shape/N and stride*N? ──
  # N consecutive elements at stride |d| span N·|d| memory units.
  # ceil_div(N, |d|) counts how many fit in one coarse slot:
  #   new_shape = ceil_div(sh, ceil_div(N, |d|))
  #   new_stride = ceil_div(|d|, N)
  # Broadcast (stride 0) is unchanged.  Dynamic strides keep shape
  # unchanged (no compile-time info), stride = ceil_div(st, N).
  mapLeavesWith(layout):
    when it_st is Int:
      when it_st.V == 0:
        (it_sh, it_st)
      else:
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

  # ── Why not just shape*N and stride/N? ──
  # If |stride| == 1 (contiguous): each coarse slot splits into N,
  #   shape*N, stride unchanged.
  # If |stride| > 1: stride was in coarse-element units; after splitting
  #   each coarse stride d becomes d·N in fine units, stride*N, shape unchanged.
  # Dynamic strides use a runtime check: if |st|==1 → shape*N else stride*N.
  # Broadcast (stride 0) is unchanged.
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
#  zipModes — interleave corresponding modes of two layouts
# ═══════════════════════════════════════════════════════════════

macro zipModes*[A, B: Layout](a: A, b: B): untyped =
  ## Zip modes of two layouts: interleave corresponding modes pairwise.
  ##
  ##   Given layouts A with modes (a0, a1, ..., aN) and
  ##   B with modes (b0, b1, ..., bN), zipModes produces a layout
  ##   with modes ((a0,b0), (a1,b1), ..., (aN,bN)).
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
      error "zipModes: mismatched rank"

  let zShape = zipElems(aShape, bShape, aShT, bShT)
  let zStride = zipElems(aStride, bStride, aStT, bStT)
  result = newCall(bindSym"make_layout", zShape, zStride)

# ═══════════════════════════════════════════════════════════════
#  groupModes — wrap modes [B, E) into a nested sub-Layout
# ═══════════════════════════════════════════════════════════════

macro groupModes*(layout: Layout; B, E: static int): untyped =
  ## Wraps modes at indices `[B, E)` into a nested sub-tuple in both
  ## shape and stride, producing a higher-rank Layout.
  ##
  ## CuTe: group<B,E>(layout) — layout.hpp:1011
  ## Python: group(layout, B, E) — algebra.py:319
  ##
  ## Examples:
  ##   groupModes(make_layout((2, 3, 5, 7)), 0, 2)
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

#  takeModes — extract modes [B, E) into a new Layout
# ═══════════════════════════════════════════════════════════════

macro takeModes*(layout: Layout; B, E: static int): untyped =
  ## Extract modes in range `[B, E)` into a new Layout.
  ## Returns a scalar Layout if only one mode is extracted.
  ##
  ## Examples:
  ##   takeModes(make_layout((2, 3, 5, 7)), 1, 3)
  ##   # → (3, 5):(2, 6)
  var ct = LayoutCT()
  let shTyp = layoutTypeArgs(layout).shapeTy
  let R = if shTyp.kind == nnkTupleConstr: shTyp.len else: 1
  for i in B ..< min(E, R):
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit(i)),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit(i)))
  result = ct.emit()

# ═══════════════════════════════════════════════════════════════
#  selectModes — extract specific mode indices into a new Layout
# ═══════════════════════════════════════════════════════════════

macro selectModes*(layout: Layout, Is: varargs[int]{lit|`const`}): untyped =
  ## Extract specific mode indices into a new Layout.
  var ct = LayoutCT()
  for i in 0 ..< Is.len:
    let idx = Is[i].intVal
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit(idx)),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit(idx)))
  result = ct.emit()

# ═══════════════════════════════════════════════════════════════
#  replaceMode — replace a mode with a sub-Layout
# ═══════════════════════════════════════════════════════════════

macro replaceMode*(layout: Layout; x: typed; N: static int): untyped =
  ## Replace mode N of layout with Layout x.
  ## CuTe: replace<N>(layout, x) — layout.hpp:1001
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
#  map — apply fn to each mode independently
#  zipWith — pairwise fn over modes of two Layouts
# ═══════════════════════════════════════════════════════════════

macro mapModesWith*[L: Layout](arg: L; body: untyped): untyped =
  ## Apply `body` to each mode of Layout `arg`. Within body, `it` is the current mode.
  ## `body` must evaluate to a Layout.
  ##
  ## Example:
  ##   mapModesWith(make_layout((2, 4), (1, 2))):
  ##     make_layout(it.shape, it.stride * 2)
  ##   # → (2, 4):(2, 4)
  let shTy = layoutTypeArgs(arg).shapeTy
  let R = if shTy.kind == nnkTupleConstr: shTy.len else: 1

  result = newStmtList()
  proc subst(n: NimNode; i: int; la: NimNode): NimNode =
    if n.kind in {nnkIdent, nnkSym} and n.eqIdent("it"):
      result = newCall(bindSym"mode", la, newLit(i))
    else:
      result = n.copyNimTree()
      for j in 0 ..< n.len:
        result[j] = subst(n[j], i, la)

  var ct = LayoutCT()
  for i in 0 ..< R:
    let bodyExpr = subst(body, i, arg)
    let resName = ident("r" & $i)
    result.add newLetStmt(resName, bodyExpr)
    ct.append(newTree(nnkDotExpr, resName, ident"shape"),
               newTree(nnkDotExpr, resName, ident"stride"))
  result.add ct.emit()

macro zipModesWith*[A, B: Layout](a: A; b: B; body: untyped): untyped =
  ## Zip modes of two layouts pairwise via body, appending leftovers from the longer one.
  ##
  ## Within body, `it_a` is the current mode of `a` and `it_b` the current mode of `b`.
  ## Body must return a Layout.
  ##
  ## For the first `min(rank(a), rank(b))` modes, both `it_a` and `it_b` are
  ## available — the body combines them. Any remaining modes from the longer
  ## layout are appended unchanged.
  ##
  ## Example (a shorter, b longer):
  ##   let a = make_layout((2,), (1,))           # rank-1
  ##   let b = make_layout(((2, 2), (2, 8)), ((1, 4), (2, 8)))  # rank-2
  ##   let r = zipModesWith(a, b):
  ##     make_layout(it_a.shape, it_b.stride)   # shape from a, stride from b's 1st mode
  ##   # mode 0 = zip result:  (2):(1, 4)       — shape from a (2), stride from b's 1st (1, 4)
  ##   # mode 1 = b's 2nd mode leftover:  (2, 8):(2, 8)
  ##
  ## Example (same rank):
  ##   let a = make_layout((2, 4), (1, 2))
  ##   let b = make_layout((3, 5), (10, 20))
  ##   let r = zipModesWith(a, b):
  ##     make_layout(it_a.shape, it_b.stride)   # take shape from a, stride from b
  ##   # r == (2, 4):(10, 20)
  let (shA, _) = layoutTypeArgs(a)
  let (shB, _) = layoutTypeArgs(b)
  let RA = if shA.kind == nnkTupleConstr: shA.len else: 1
  let RB = if shB.kind == nnkTupleConstr: shB.len else: 1
  let rMin = min(RA, RB)
  let rMax = max(RA, RB)

  proc subst(n: NimNode; i: int; la, lb: NimNode): NimNode =
    if n.kind in {nnkIdent, nnkSym} and n.eqIdent("it_a"):
      result = newCall(bindSym"mode", la, newLit(i))
    elif n.kind in {nnkIdent, nnkSym} and n.eqIdent("it_b"):
      result = newCall(bindSym"mode", lb, newLit(i))
    else:
      result = n.copyNimTree()
      for j in 0 ..< n.len:
        result[j] = subst(n[j], i, la, lb)

  var ct = LayoutCT()
  result = newStmtList()
  for i in 0 ..< rMax:
    if i < rMin:
      let bodyExpr = subst(body, i, a, b)
      let resName = ident("r" & $i)
      result.add newLetStmt(resName, bodyExpr)
      ct.append(newTree(nnkDotExpr, resName, ident"shape"),
                 newTree(nnkDotExpr, resName, ident"stride"))
    elif i < RA:
      let mName = ident("m" & $i)
      result.add newLetStmt(mName, newCall(bindSym"mode", a, newLit(i)))
      ct.append(newTree(nnkDotExpr, mName, ident"shape"),
                 newTree(nnkDotExpr, mName, ident"stride"))
    else:
      let mName = ident("m" & $i)
      result.add newLetStmt(mName, newCall(bindSym"mode", b, newLit(i)))
      ct.append(newTree(nnkDotExpr, mName, ident"shape"),
                 newTree(nnkDotExpr, mName, ident"stride"))
  result.add ct.emit()
