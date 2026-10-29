# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout data types: Layout[Sh, St], basic accessors, and type-level predicates.

import std/macros
import std/typetraits
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import ./layouts_unsanctioned_helpers

# ═══════════════════════════════════════════════════════════════
#  Layout[Sh, St], typed shape + stride pair
# ═══════════════════════════════════════════════════════════════

type Layout*[Sh, St] = object
  ## A compile-time-typed layout: `Layout[Shape, Stride]`.
  ## Both Sh and St can be int, Int[N], or tuples thereof.
  shape*: Sh
  stride*: St

# ═══════════════════════════════════════════════════════════════
#  destructureLayout, layout AST -> (shape, stride) expressions
# ═══════════════════════════════════════════════════════════════

proc pickShapeStride*(fields: NimNode, shape, strides: var NimNode) {.compileTime.} =
  ## Collect the shape/stride members of a Layout field list, object
  ## constructor fields and object type fields share the layout
  for def in fields:
    if def.kind in {nnkExprColonExpr, nnkIdentDefs} and def.len >= 2:
      if def[0].eqIdent("shape"):
        shape = def[1]
      elif def[0].eqIdent("stride"):
        strides = def[1]

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
    # Save statements into the result as they may introduce intermediate values
    let body = if inner.kind == nnkBlockExpr: inner[1] else: inner
    for i in 0 ..< body.len - 1:
      resultStmt.add body[i]
    inner = body[^1]
  if inner.kind == nnkCall and inner[0].eqIdent("make_layout") and inner.len == 3:
    result.shape = inner[1]
    result.strides = inner[2]
    return
  if layoutAst.kind in {nnkCall, nnkCommand, nnkBracketExpr}:
    # Deferred call in case the call hasn't been semchecked yet,
    # e.g. a `t[0]` element read of a symbol-bound tiler
    let alias = genSym(nskTemplate, "destructuredLayout")
    resultStmt.add bindSym"evalOnceAs".newCall(alias, layoutAst)
    result.shape = alias.newDotExpr(ident"shape")
    result.strides = alias.newDotExpr(ident"stride")
    return
  if layoutAst.kind notin {nnkSym, nnkIdent}:
    let typ = layoutAst.getTypeInst()
    let layoutTy = if typ.kind == nnkVarTy: typ[0] else: typ
    doAssert (layoutTy.kind == nnkBracketExpr and layoutTy[0].sameType(bindSym"Layout")) or layoutTy.kind == nnkSym,
      "destructureLayout: expected a Layout, got " & typ.repr
    if inner.kind == nnkObjConstr:
      pickShapeStride(inner, result.shape, result.strides)
      # the semchecked constructor fields wrap the base tuples
      # in a hidden conversion, unwrap it.
      while result.shape != nil and result.shape.kind == nnkHiddenSubConv:
        result.shape = result.shape[^1]
      while result.strides != nil and result.strides.kind == nnkHiddenSubConv:
        result.strides = result.strides[^1]
      doAssert result.shape != nil and result.strides != nil,
        "destructureLayout: Layout constructor without shape/stride fields"
      return
    if layoutTy.kind == nnkSym or layoutAst.kind in {nnkCall, nnkCommand} or
        inner.kind in {nnkCall, nnkCommand}:
      # A layout-valued call, or a value typed through
      # a `typeof(make_layout(...))` alias symbol.
      let alias = genSym(nskTemplate, "destructuredLayout")
      resultStmt.add bindSym"evalOnceAs".newCall(alias, layoutAst)
      result.shape = alias.newDotExpr(ident"shape")
      result.strides = alias.newDotExpr(ident"stride")
      return

  result.shape = layoutAst.newDotExpr(ident"shape")
  result.strides = layoutAst.newDotExpr(ident"stride")


func `===`*(a: Layout, b: tuple): bool {.inline.} =
  ## Deep comparison against a (shape, stride) tuple.
  ## This handles static Int checks against int checks
  ## and also size-1 tuples against int/Int
  (a.shape === b[0]) and (a.stride === b[1])

func `===`*[A, B: Layout](a: A, b: B): bool {.inline.} =
  ## Deep comparison between two Layouts.
  a.shape === b.shape and a.stride === b.stride

func `$`*(layout: Layout): string =
  ## CuTe-style representation: "(shape):(stride)".
  ##   ($make_layout(4,1))  →  "(4):(1)"
  ##   ($make_layout((4,8),(1,4)))  →  "((4,8)):((1,4))"
  $layout.shape & ":" & $layout.stride

template rank*(layout: Layout): static int =
  ## Number of dimensions in layout (compile-time constant).
  layout.shape.rank()

template rank*[Sh, St](_: typedesc[Layout[Sh, St]]): static int =
  ## Number of dimensions in a layout type (compile-time constant).
  Sh.rank()

template size*(layout: Layout): auto =
  product(layout.shape)

template size*(t: IntOrIntTuple): auto =
  product(t)

func isLayoutTy*(n: NimNode): bool {.compileTime.} =
  ## True for Layout values, the generic head carries the match
  let ty = n.getTypeInst()
  ty.sameType(bindSym"Layout") or
    (ty.kind == nnkBracketExpr and ty[0].sameType(bindSym"Layout"))

# ═══════════════════════════════════════════════════════════════
#  cosize, max offset + 1 of a layout
# ═══════════════════════════════════════════════════════════════

macro cosizeImpl(sh, st: typed): untyped =
  ## Cosize emit, one term per shape leaf
  let one = IntCT(1)
  result = one
  for (shEv, stEv) in zip(sh.tupleStream(), st.tupleStream()):
    if shEv.kind == kLeaf:
      result = result + (shEv.leaf - one) * abs(stEv.leaf)

func cosize*(layout: Layout): auto =
  ## Compute cosize = sum_i ((sh_i - 1) * |st_i|) + 1.
  cosizeImpl(layout.shape, layout.stride)

func cosize*[A, B](_: typedesc[Layout[A, B]]): static int {.inline.} =
  ## Compile-time cosize from the Layout type alone.
  ## Precondition, the shape and stride are all-static Int[N] leaves.
  ## Dynamic layouts produce a compile error.
  var tmp {.noInit.}: Layout[A, B]
  cosize(tmp).toInt()

macro coprofileImpl*(bStrides: typed): untyped =
  result = IntCT(0)
  for stEv in bStrides.tupleStream():
    if stEv.kind == kLeaf:
      result = result + stEv.leaf

macro coprofile*(layout: Layout): untyped =
  ## Codomain profile of the layout:
  ## `profile = Σ leaves(layout.stride)`
  result = newStmtList()
  let (_, strides) = result.destructureLayout(layout)
  result.add quote do:
    coprofileImpl(`strides`)

# ═══════════════════════════════════════════════════════════════
#  StrideOrder
# ═══════════════════════════════════════════════════════════════

type StrideOrder* = enum
  LayoutLeft
    ## Leftmost dimension is contiguous (stride 1),
    ## equivalent to col-major / prefix_product.
    ##
    ## Example:
    ##   make_layout((M, N), LayoutLeft) -> (M, N) : (1, M)
    ##   make_layout((3, 4, 5), LayoutLeft) -> (3, 4, 5) : (1, 3, 12)

  LayoutRight
    ## Rightmost dimension is contiguous (stride 1),
    ## equivalent to row-major / suffix_product.
    ##
    ## Example:
    ##   make_layout((M, N), LayoutRight) -> (M, N) : (N, 1)
    ##   make_layout((3, 4, 5), LayoutRight) -> (3, 4, 5) : (20, 5, 1)

# ═══════════════════════════════════════════════════════════════
#  shape-structure predicates over IntOrIntTuple
# ═══════════════════════════════════════════════════════════════

template congruent*[A, B: IntOrIntTuple](a: A, b: B): bool =
  ## True if `a` and `b` have the same hierarchical rank structure.
  ## Returns a bool typedesc, usable in both `static` and runtime contexts.
  when a is (int or Int) or b is (int or Int):
    a is (int or Int) and b is (int or Int)
  elif a is tuple and b is tuple:
    when a.rank() != b.rank():
      false
    else:
      block:
        var ok = true
        staticFor i, 0, rank(a):
          if not congruent(a[i], b[i]):
            ok = false
        ok
  else:
    false

func weakly_congruent*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if A's nesting is contained in B's structure.
  ## Scalar matches anything; tuple must have at least as much structure.
  when a is (int or Int):
    true
  elif b is (int or Int):
    false
  elif a is tuple and b is tuple:
    when a.rank() != b.rank():
      false
    else:
      block:
        var ok = true
        staticFor i, 0, rank(a):
          if not weakly_congruent(a[i], b[i]):
            ok = false
        ok
  else:
    false

func can_group_a_into_b_impl[A, B](a: A; aStartIdx: int; b: B): int =
  ## Find consecutive dimensions in `a` from `aStartIdx` whose product equals `b`.
  static: doAssert a isnot int, "scalar a should be handled by caller"
  let bVal = product(b).toInt()
  var acc = 1
  var aIdx = aStartIdx
  block accLoop:
    staticFor i, 0, rank(a):
      if i >= aStartIdx:
        if acc < bVal:
          acc *= product(a[i]).toInt()
          aIdx = i + 1
        else:
          aIdx = i
          break accLoop
  if acc == bVal: aIdx else: -1

func can_group_a_into_b*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## Check if shape `a` (flat) can be grouped into shape `b` (nested).
  static: doAssert a isnot int, "scalar a should be handled by caller"
  when b is int or b is Int:
    can_group_a_into_b_impl(a, 0, b) != -1
  else:
    var aIdx = 0
    staticFor j, 0, rank(b):
      aIdx = can_group_a_into_b_impl(a, aIdx, b[j])
      if aIdx == -1:
        return false
    aIdx == rank(a)

func compatible*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if `a` is structurally compatible with `b`: same total size, and
  ## a's nesting can address into b's structure.
  ## Supports grouping: (2,2,3) is compatible with (4,3).
  let aSize = product(a).toInt()
  let bSize = product(b).toInt()
  if aSize != bSize:
    return false
  when a is int or a is Int:
    true
  elif b is int or b is Int:
    false
  elif rank(a) == rank(b):
    block:
      var ok = true
      staticFor i, 0, rank(a):
        if not compatible(a[i], b[i]):
          ok = false
      ok
  else:
    can_group_a_into_b(a, b)
