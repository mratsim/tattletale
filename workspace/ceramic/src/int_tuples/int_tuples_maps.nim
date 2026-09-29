# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples_datatypes

proc leafAccess(e, t: NimNode; idx: int): NimNode {.compileTime.}

# ═══════════════════════════════════════════════════════════════════════
#  mapLeavesWith, recursive leaf-wise tuple map
# ═══════════════════════════════════════════════════════════════════════

macro mapLeavesWith*(t: IntOrIntTuple, body: untyped): untyped =
  ## Recursively walk `t` (int | Int[N] | tuple) and apply `body` to
  ## every leaf.  Returns a value of the same shape with leaves transformed.
  ##
  ## Identity:
  ## - when `body` maps every leaf to itself, the result is `t` verbatim,
  ##   no tuple reconstruction
  ## - a leaf body `when cond: replacement else: leaf` keeps this identity
  ##   whenever every `cond` folds false, decided at compile time
  ##   over one combined `when` statement
  ##
  ## Example spellings:
  ##
  ##   mapLeavesWith((1, (2, 3)), it * 10)  →  (10, (20, 30))
  ##   mapLeavesWith(5, it)                 →  5
  ##   mapLeavesWith((2, 3), it * 10)       →  (20, 30)

  # An untyped macro parameter captured by a nested proc closure is lazily semchecked
  # against the caller scope, silently degrading to a nil node when the leaf
  # placeholder `it` is unbound there. The body is therefore materialized
  # into a local before any nested proc closes over it.
  let rawBody = body

  proc replaceNodes(ast, what, by: NimNode): NimNode =
    proc inspect(node: NimNode): NimNode =
      case node.kind
      of {nnkIdent, nnkSym}:
        if node.eqIdent(what): return by
        return node
      of nnkEmpty, nnkLiterals:
        return node
      else:
        result = node.kind.newTree()
        for child in node:
          result.add inspect(child)
    result = inspect(ast)

  proc unwrapValueExpr(n: NimNode): NimNode =
    ## Single-statement wrapper around an expression value. Statement lists wrap expressions one level deep.
    result = n
    while result.kind in {nnkStmtList, nnkStmtListExpr} and result.len == 1:
      result = result[0]

  proc sameExpr(a, b: NimNode): bool =
    ## Structural equality of two expression nodes. Each side first unwraps
    ## its statement wrappers, then the treeRepr strings are compared for equality.
    let reprA = treeRepr(unwrapValueExpr(a))
    let reprB = treeRepr(unwrapValueExpr(b))
    reprA == reprB

  proc inExprSlot(n: NimNode): NimNode =
    ## A `when` statement occupies an expression slot only in block-wrapped form.
    if unwrapValueExpr(n).kind == nnkWhenStmt:
      newTree(nnkBlockStmt, newEmptyNode(), n)
    else:
      n

  proc leafIdentity(built, expect: NimNode): tuple[identity: bool, conds: seq[NimNode]] =
    ## Reduction of the leaf body `built` to the leaf expression `expect`.
    ##
    ## Returns:
    ## - identity true with no conditions, `built` is `expect` verbatim
    ## - identity true with conditions, `built` is a `when` whose every
    ##   branch condition folds false selects `expect`, the conditions are
    ##   returned verbatim for a combined compile-time fold
    ## - identity false, the leaf must be reconstructed
    let core = unwrapValueExpr(built)
    if sameExpr(core, expect):
      return (true, @[])
    if core.kind != nnkWhenStmt:
      return (false, @[])
    let last = core[^1]
    if last.kind notin {nnkElse, nnkElseExpr} or not sameExpr(last[^1], expect):
      return (false, @[])
    var conds: seq[NimNode]
    for branch in core:
      if branch.kind in {nnkElifBranch, nnkElifExpr}:
        conds.add branch[0]
    (true, conds)

  var leaves: seq[tuple[built, expect: NimNode]]

  proc walk(e, ty: NimNode): NimNode =
    ## Bottom-up build of the shape-preserving result. Nested tuple types recurse
    ## one element type at a time, every leaf gets `rawBody` spliced with `it`
    ## bound to the leaf access and recorded with its expected value.
    if ty.kind in {nnkTupleTy, nnkTupleConstr}:
      var elems: seq[NimNode]
      for idx in 0 ..< ty.len:
        elems.add inExprSlot(walk(leafAccess(e, ty, idx), ty[idx]))
      result = nnkTupleConstr.newTree(elems)
    else:
      result = rawBody.replaceNodes(ident"it", e)
      leaves.add (result, e)

  let built = walk(t, t.getTypeInst())

  var conds: seq[NimNode]
  for lf in leaves:
    let (identity, branchConds) = leafIdentity(lf.built, lf.expect)
    if not identity:
      result = built
      return
    conds.add branchConds

  if conds.len == 0:
    result = t
    return

  var combined = conds[0]
  for i in 1 ..< conds.len:
    combined = nnkInfix.newTree(ident"or", combined, conds[i])
  result = quote do:
    when `combined`:
      `built`
    else:
      `t`

# ═══════════════════════════════════════════════════════════════════════
#  flatMapLeaves, flat leaf-wise tuple map of a single pack
# ═══════════════════════════════════════════════════════════════════════

proc flatMapLeavesImpl(tNode: NimNode; body: NimNode): NimNode {.compileTime.} =
  ## Build the flat pack for `tNode`, one node per leaf, `body` with `it`
  ## replaced by the leaf access. Returns the untyped tuple construction.

  proc replaceNodes(ast, what, by: NimNode): NimNode =
    proc inspect(node: NimNode): NimNode =
      case node.kind
      of {nnkIdent, nnkSym}:
        if node.eqIdent(what): return by
        return node
      of nnkEmpty, nnkLiterals:
        return node
      else:
        result = node.kind.newTree()
        for child in node:
          result.add inspect(child)
    result = inspect(ast)

  proc isLeaf(t: NimNode): bool =
    (t.kind == nnkSym and $t == "int") or
    (t.kind == nnkBracketExpr and $t[0] == "Int")

  proc collect(acc: var NimNode; e: NimNode; t: NimNode) =
    if t.kind == nnkTupleConstr:
      for idx in 0 ..< t.len:
        let fd = t[idx]
        let fa = leafAccess(e, t, idx)
        if isLeaf(fd):
          acc.add body.replaceNodes(ident"it", fa)
        else:
          collect(acc, fa, fd)
    else:
      acc.add body.replaceNodes(ident"it", e)

  let tType = tNode.getTypeImpl()
  # nnkPar keeps a single-item result scalar and packs multi-item results
  # like nnkTupleConstr, avoiding an explicit `if result.len == 1` branch.
  result = newNimNode(nnkPar)
  collect(result, tNode, tType)

macro flatMapLeaves*(t: IntOrIntTuple, body: untyped): untyped =
  ## Map every leaf of a (possibly nested) tuple to exactly one output
  ## element and emit the flat result as a single tuple construction.
  ##
  ## - `body` is evaluated once per leaf with `it` bound to the leaf
  ## - The output is flat, the input's nesting is collapsed
  ##
  ## Returns the flat tuple construction.
  ##
  ## CuTe analogue, `transform_apply` + `tuple_cat(a...)` (tuple_algorithms.hpp).
  ##
  ## Example spellings:
  ##
  ##   flatMapLeaves(5, it)                 → 5
  ##   flatMapLeaves((1, (2, 3)), it)       → (1, 2, 3)
  ##
  ##   flatMapLeaves((2, 3), it * 10)       → (20, 30)
  flatMapLeavesImpl(t, body)

# ═══════════════════════════════════════════════════════════════════════
#  flatLeaves and flatLeavesRev, compile-time leaf streams
# ═══════════════════════════════════════════════════════════════════════

proc leafAccess(e, t: NimNode; idx: int): NimNode {.compileTime.} =
  ## Field access of element `idx` of tuple expression `e`.
  ## Tuple constructions splice the element directly, other expressions
  ## get bracket access.
  if e.kind in {nnkTupleConstr, nnkPar} and e.len == t.len:
    e[idx]
  else:
    newTree(nnkBracketExpr, e, newLit(idx))

type
  FlatLeafPair* = tuple[leaf, leafTy: NimNode]
    ## A leaf of a (possibly nested) IntOrIntTuple expression:
    ## the leaf access expression and its leaf type node.

proc unwrapStmtListExpr(e0: NimNode): NimNode {.compileTime.} =
  ## Typed macro parameters of macro/template call arguments arrive wrapped in a statement list. The value is the last child.
  if e0.kind == nnkStmtListExpr: e0[^1] else: e0

proc flatLeavesImpl(e0, t0: NimNode): seq[FlatLeafPair] {.compileTime.} =
  ## Forward leaf stream over the type tree of expression `e`.
  let e = unwrapStmtListExpr(e0)
  let t = if t0.kind == nnkExprColonExpr: t0[1] else: t0
  if t.kind in {nnkTupleConstr, nnkTupleTy}:
    for idx in 0 ..< t.len:
      let sub = leafAccess(e, t, idx)
      result.add flatLeavesImpl(sub, t[idx])
  else:
    result.add (e, t)

proc flatLeavesRevImpl(e0, t0: NimNode): seq[FlatLeafPair] {.compileTime.} =
  ## Reverse leaf stream over the type tree of expression `e`,
  ## the exact mirror order of `flatLeavesImpl`.
  let e = unwrapStmtListExpr(e0)
  let t = if t0.kind == nnkExprColonExpr: t0[1] else: t0
  if t.kind in {nnkTupleConstr, nnkTupleTy}:
    for idx in countdown(t.len - 1, 0):
      let sub = leafAccess(e, t, idx)
      result.add flatLeavesRevImpl(sub, t[idx])
  else:
    result.add (e, t)

func flatLeaves*(e: NimNode): seq[FlatLeafPair] {.compileTime.} =
  ## Stream the leaves of the (possibly nested) IntOrIntTuple expression
  ## `e` in order, each with its leaf type node.
  ##
  ## Returns one FlatLeafPair per leaf in document order.
  ## - Static `Int[N]` leaves keep their `Int[V]` type
  ## - Runtime `int` leaves keep their `int` type
  ##
  ## Nim rejects `{.compileTime.}` on iterators.
  ## The stream is a compile-time function, consumed with the same for-loop syntax.
  flatLeavesImpl(e, e.getTypeInst())

func flatLeavesRev*(e: NimNode): seq[FlatLeafPair] {.compileTime.} =
  ## Stream the leaves of the (possibly nested) IntOrIntTuple expression
  ## `e` in reverse order, the mirror of `flatLeaves`.
  flatLeavesRevImpl(e, e.getTypeInst())

macro countLeaves*(t: typed): untyped =
  ## Number of leaves of the (possibly nested) IntOrIntTuple `t`.
  var n = 0
  for _ in flatLeaves(t):
    inc n
  result = newLit(n)

# ═══════════════════════════════════════════════════════════════════════
#  concatFlat, the leaves of two tuples as one flat tuple
# ═══════════════════════════════════════════════════════════════════════

macro concatFlat*(a, b: typed): untyped =
  ## The leaves of `a` then the leaves of `b` as one flat tuple,
  ## no intermediate flattened tuple materializes.
  result = nnkTupleConstr.newTree()
  for (leaf, _) in flatLeaves(a):
    result.add leaf
  for (leaf, _) in flatLeaves(b):
    result.add leaf

# ═══════════════════════════════════════════════════════════════════════
#  mapDimensionsWith, top-level-only tuple map
# ═══════════════════════════════════════════════════════════════════════

macro mapDimensionsWith*(t: tuple; body: untyped): untyped =
  ## Apply `body` to each top-level element of tuple `t` (does NOT recurse into nested tuples).
  ## `it` binds to the current element.
  ##
  ## Example:
  ##   mapDimensionsWith((2, 4, 6)): it * 2  →  (4, 8, 12)
  let tt = getTypeInst(t)
  let n = tt.len

  proc subst(x: NimNode; i: int; ttup: NimNode): NimNode =
    if x.kind in {nnkIdent, nnkSym} and x.eqIdent("it"):
      result = nnkBracketExpr.newTree(ttup, newLit(i))
    else:
      result = x.copyNimTree()
      for j in 0 ..< x.len: result[j] = subst(x[j], i, ttup)

  var items: seq[NimNode]
  for i in 0 ..< n:
    items.add subst(body, i, t)
  # nnkTupleConstr: always preserve tuple structure (even for single-element results)
  result = nnkTupleConstr.newTree(items)
