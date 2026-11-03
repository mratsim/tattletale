# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Hierarchical tuples and stride scalars: the value vocabulary of the layout algebra.
##
## An `IntTuple` is a leaf or a tuple of IntTuples. A leaf is a `StrideScalar`:
##
## - an integer, or an `ArithTuple`: a Z^S element with elementwise addition,
##   integer scaling and zero-extended trailing positions
##
## Shapes and coordinates hold integer leaves. Strides may hold ArithTuple
## leaves such as the by-mode tiler basis strides `1@0` and `2@1`.
##
## Semantics are ported from the frozen python reference implementation
## (vendored pycute), so Nim trees compare and evaluate identically to it.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b4 \
##     --outdir:build/b4 workspace/tattletile/tests/test_layout_algebra_conformance.nim

import std/macros
import std/strutils

type
  LayoutError* = object of ValueError
    ## Refusal raised when an algebra operation hits a refused precondition:
    ## a divisibility violation, a non-injective layout, a rank mismatch

proc layoutError*(msg: string): ref LayoutError =
  ## Builds a LayoutError refusal with message `msg`.
  ## Call sites raise it, one construction site per message.
  newException(LayoutError, msg)

type
  ScalarKind* = enum
    ## Leaf kind of a stride scalar: an integer or a coordinate-axis element
    skInt
    skAtup

  StrideScalar* = object
    ## One stride leaf: an integer, or an ArithTuple over the codomain axes.
    ##
    ## An ArithTuple is a hierarchical tuple of stride scalars under elementwise
    ## addition and integer scaling, with implicit zero extension.
    ##
    ## - `(1, 0)` and `(1,)` are the same element
    ## - `1@0` is the unit of axis 0, `2@1` is two steps along axis 1
    case kind*: ScalarKind
    of skInt:
      intVal*: int
    of skAtup:
      data*: seq[StrideScalar]

  TupKind* = enum
    ## Node kind of a hierarchical tuple: a leaf or a tuple of children
    tkLeaf
    tkNode

  IntTuple* = object
    ## Hierarchical tuple: a leaf with a StrideScalar, or a tuple of IntTuples.
    ## Shape, stride, coordinate and profile trees are all IntTuples.
    case kind*: TupKind
    of tkLeaf:
      leaf*: StrideScalar
    of tkNode:
      children*: seq[IntTuple]

# ═══════════════════════════════════════════════════════════════════════
# Constructors
# ═══════════════════════════════════════════════════════════════════════

proc ssInt*(v: int): StrideScalar {.inline.} =
  ## Integer stride scalar `v`
  StrideScalar(kind: skInt, intVal: v)

proc ssAtup*(xs: seq[StrideScalar]): StrideScalar {.inline.} =
  ## ArithTuple with children `xs` verbatim
  StrideScalar(kind: skAtup, data: xs)

proc scaledBasis*(value: int, path: seq[int]): StrideScalar =
  ## Scaled basis element `value` at coordinate-axis `path`:
  ## `scaledBasis(2, @[0])` is `2@0`. An empty path collapses to the integer.
  result = ssInt(value)
  for idx in countdown(path.len - 1, 0):
    let i = path[idx]
    var padded = newSeq[StrideScalar](i + 1)
    padded[i] = result
    result = ssAtup(padded)

proc eUnit*(path: seq[int]): StrideScalar {.inline.} =
  ## Unit basis element at `path`: `eUnit(@[])` is 1, `eUnit(@[0])` is `1@0`
  scaledBasis(1, path)

proc itLeaf*(ss: StrideScalar): IntTuple {.inline.} =
  ## A hierarchical tuple leaf holding the stride scalar `ss`
  IntTuple(kind: tkLeaf, leaf: ss)

proc itv*(v: int): IntTuple {.inline.} =
  ## A hierarchical tuple leaf holding the integer `v`
  itLeaf(ssInt(v))

proc itNode*(xs: seq[IntTuple]): IntTuple {.inline.} =
  ## A hierarchical tuple node with children `xs`, an empty tuple when `xs` is empty
  IntTuple(kind: tkNode, children: xs)

proc itEmpty*: IntTuple {.inline.} =
  ## Empty hierarchical tuple `()`
  itNode(newSeq[IntTuple]())

proc asChild*(x: IntTuple): IntTuple {.inline.} =
  ## Pass an IntTuple child through to `itNode`
  x

proc asChild*(x: StrideScalar): IntTuple {.inline.} =
  ## Lift a stride scalar to an IntTuple leaf child of `itNode`
  itLeaf(x)

proc asChild*(x: int): IntTuple {.inline.} =
  ## Lift a runtime integer to an IntTuple leaf child of `itNode`
  itv(x)

macro it*(xs: varargs[untyped]): IntTuple =
  ## Literal builder: `it(8)` is the integer leaf 8, `it(4, 2)` the tuple
  ## `(4, 2)`, and nested calls compose: `it(it(2, 2, 2), it(2, 2, 2))`.
  ##
  ## - children may be integer literals, IntTuple or StrideScalar expressions
  ## - a single integer literal builds the leaf
  if xs.len == 1 and xs[0].kind in {nnkIntLit .. nnkUInt64Lit}:
    result = quote do: itv(int(`xs[0]`))
  else:
    var children = nnkBracket.newTree()
    for x in xs:
      if x.kind in {nnkIntLit .. nnkUInt64Lit}:
        children.add quote do: asChild(itv(int(`x`)))
      else:
        children.add quote do: asChild(`x`)
    result = nnkCall.newTree(ident("itNode"),
      nnkPrefix.newTree(ident("@"), children))

# ═══════════════════════════════════════════════════════════════════════
# Stride-scalar algebra: +, integer *, equality, colex order
# ═══════════════════════════════════════════════════════════════════════

proc ssAdd*(a, b: StrideScalar): StrideScalar =
  ## Elementwise addition with implicit zero extension.
  ##
  ## Returns:
  ## - the integer sum when both operands are integers
  ## - `a` unchanged when one operand is the integer 0
  ## - the elementwise sum, zero-extended, when both are ArithTuples
  ##
  ## Raises:
  ## - LayoutError for a nonzero integer added to an ArithTuple, a rank
  ##   incompatibility
  if a.kind == skInt and b.kind == skInt:
    return ssInt(a.intVal + b.intVal)
  if a.kind == skInt:
    if a.intVal == 0:
      return b
    raise layoutError("ArithTuple incompatibility: " & $a.intVal & " + coordinate stride")
  if b.kind == skInt:
    if b.intVal == 0:
      return a
    raise layoutError("ArithTuple incompatibility: coordinate stride + " & $b.intVal)
  let n = max(a.data.len, b.data.len)
  var out_data = newSeq[StrideScalar](n)
  for i in 0 ..< n:
    let x = if i < a.data.len: a.data[i] else: ssInt(0)
    let y = if i < b.data.len: b.data[i] else: ssInt(0)
    out_data[i] = ssAdd(x, y)
  result = ssAtup(out_data)

proc ssSub*(a, b: StrideScalar): StrideScalar =
  ## Elementwise subtraction with implicit zero extension, the inverse of ssAdd
  ##
  ## Raises:
  ## - LayoutError for a nonzero integer subtracted from an ArithTuple
  if a.kind == skInt and b.kind == skInt:
    return ssInt(a.intVal - b.intVal)
  if b.kind == skInt:
    if b.intVal == 0:
      return a
    raise layoutError("ArithTuple incompatibility: coordinate stride - " & $b.intVal)
  if a.kind == skInt:
    if a.intVal == 0:
      var neg = newSeq[StrideScalar](b.data.len)
      for i in 0 ..< b.data.len:
        neg[i] = ssSub(ssInt(0), b.data[i])
      return ssAtup(neg)
    raise layoutError("ArithTuple incompatibility: " & $a.intVal & " - coordinate stride")
  let n = max(a.data.len, b.data.len)
  var out_data = newSeq[StrideScalar](n)
  for i in 0 ..< n:
    let x = if i < a.data.len: a.data[i] else: ssInt(0)
    let y = if i < b.data.len: b.data[i] else: ssInt(0)
    out_data[i] = ssSub(x, y)
  result = ssAtup(out_data)

proc ssMul*(d: StrideScalar, k: int): StrideScalar =
  ## Integer scaling of a stride scalar: `2 * (1@0)` is the stride `2@0`
  case d.kind
  of skInt:
    ssInt(d.intVal * k)
  of skAtup:
    var out_data = newSeq[StrideScalar](d.data.len)
    for i in 0 ..< d.data.len:
      out_data[i] = ssMul(d.data[i], k)
    ssAtup(out_data)

proc ssEq*(a, b: StrideScalar): bool =
  ## Equality under implicit zero extension: the integer 0 equals the all-zero
  ## ArithTuple of any rank, otherwise rank-incompatible operands differ
  if a.kind == skInt and b.kind == skInt:
    return a.intVal == b.intVal
  if a.kind == skInt and a.intVal != 0:
    return false
  if b.kind == skInt and b.intVal != 0:
    return false
  let da = if a.kind == skAtup: a.data else: newSeq[StrideScalar]()
  let db = if b.kind == skAtup: b.data else: newSeq[StrideScalar]()
  let n = max(da.len, db.len)
  for i in 0 ..< n:
    let x = if i < da.len: da[i] else: ssInt(0)
    let y = if i < db.len: db[i] else: ssInt(0)
    if not ssEq(x, y):
      return false
  true

proc ssZero*(d: StrideScalar): bool =
  ## True when the stride scalar is the additive identity:
  ## the integer 0, or an ArithTuple whose every leaf is 0.
  case d.kind
  of skInt:
    d.intVal == 0
  of skAtup:
    for c in d.data:
      if not ssZero(c):
        return false
    true

proc colexLt*(a, b: StrideScalar): bool =
  ## Strict colexicographic order on stride scalars: the highest position
  ## dominates (an axis-1 element outranks every axis-0 element) and equal
  ## positions compare by coefficient.
  ##
  ## An integer compares with an ArithTuple only when the integer is 0,
  ## the zero extension.
  ##
  ## Raises:
  ## - LayoutError for a rank-incompatible nonzero integer
  if a.kind == skInt and b.kind == skInt:
    return a.intVal < b.intVal
  if a.kind == skInt and a.intVal != 0:
    raise layoutError("colex order: rank-incompatible " & $a.intVal)
  if b.kind == skInt and b.intVal != 0:
    raise layoutError("colex order: rank-incompatible " & $b.intVal)
  let da = if a.kind == skAtup: a.data else: newSeq[StrideScalar]()
  let db = if b.kind == skAtup: b.data else: newSeq[StrideScalar]()
  for i in countdown(max(da.len, db.len) - 1, 0):
    let x = if i < da.len: da[i] else: ssInt(0)
    let y = if i < db.len: db[i] else: ssInt(0)
    if colexLt(x, y):
      return true
    if colexLt(y, x):
      return false
  false

proc basisRepr*(d: StrideScalar): seq[tuple[val: int, path: seq[int]]] =
  ## Decomposition of a stride scalar into scaled basis elements: one
  ## `(value, path)` per nonzero leaf, or the single zero term `[(0, [])]`.

  proc walk(y: StrideScalar, prefix: seq[int], acc: var seq[tuple[val: int, path: seq[int]]]) =
    if y.kind == skAtup:
      for i in 0 ..< y.data.len:
        walk(y.data[i], prefix & @[i], acc)
    elif y.intVal != 0:
      acc.add (y.intVal, prefix)
  walk(d, @[], result)
  if result.len == 0:
    result.add (0, newSeq[int]())

proc projS*(d: StrideScalar, path: seq[int]): StrideScalar =
  ## Part of the stride scalar `d` at position `path`, `projS(2@1, @[1])` is 2.
  ##
  ## Raises:
  ## - LayoutError when the path leaves the element's rank
  result = d
  for p in path:
    if result.kind != skAtup:
      raise layoutError("proj: position " & $p & " indexes a rank-0 stride scalar")
    if p < 0 or p >= result.data.len:
      raise layoutError("proj: position " & $p & " out of range")
    result = result.data[p]

proc ssStr*(d: StrideScalar): string =
  ## Compact stride-scalar form.
  ##
  ## - a single nonzero leaf prints in basis form `2@1@0`: value, path reversed
  ## - a multi-term sum prints as its children `(1,2)`, the zero prints `0`
  let rep = basisRepr(d)
  if rep.len == 1:
    var parts = @[$rep[0].val]
    for i in countdown(rep[0].path.len - 1, 0):
      parts.add $rep[0].path[i]
    return parts.join("@")
  var parts: seq[string]
  for c in d.data:
    parts.add ssStr(c)
  return "(" & parts.join(",") & ")"

proc ssUnit*(d: StrideScalar): StrideScalar =
  ## Multiplicative unit of `d`'s algebra at `d`'s axis:
  ## `ssUnit(2@1)` is `1@1` and `ssUnit(5)` is 1.
  ##
  ## Raises:
  ## - LayoutError for a multi-term sum, which has no unit
  let rep = basisRepr(d)
  if rep.len != 1:
    raise layoutError("unit: multi-term stride sums have no unit")
  return scaledBasis(1, rep[0].path)

# ═══════════════════════════════════════════════════════════════════════
# Tree walks
# ═══════════════════════════════════════════════════════════════════════

proc isNode*(x: IntTuple): bool {.inline.} =
  ## True when `x` is a tuple node, false for a leaf
  x.kind == tkNode

proc flatScalars*(x: IntTuple): seq[StrideScalar] =
  ## Leaves of the tree, left to right. An ArithTuple leaf stays whole.
  case x.kind
  of tkLeaf:
    result = @[x.leaf]
  of tkNode:
    for c in x.children:
      result.add flatScalars(c)

proc leafCount*(x: IntTuple): int {.inline.} =
  ## Number of leaves of the tree
  flatScalars(x).len

proc wrapT*(x: IntTuple): IntTuple {.inline.} =
  ## `x` wrapped in a 1-tuple unless it already is a tuple
  if x.kind == tkNode:
    x
  else:
    itNode(@[x])

proc unwrapT*(x: IntTuple): IntTuple =
  ## Every enclosing 1-tuple stripped: `unwrapT(((((42,)),)))` is 42
  result = x
  while result.kind == tkNode and result.children.len == 1:
    result = result.children[0]

proc getModeT*(x: IntTuple, path: seq[int]): IntTuple =
  ## Subtree of `x` at `path`, one index per step.
  ## Negative indices count from the end.
  ##
  ## Example: `getModeT(it(1, (2, 3)), @[1, 0])` is 2.
  ##
  ## Raises:
  ## - LayoutError when the path indexes a leaf
  result = x
  for p in path:
    if result.kind != tkNode:
      raise layoutError("get: indexing a leaf with " & $p)
    let i = (p mod result.children.len + result.children.len) mod result.children.len
    result = result.children[i]

proc zipLeavesT*(a, b: IntTuple): seq[tuple[x, y: IntTuple]] =
  ## Corresponding leaf pairs of two weakly congruent trees. `a` drives the walk:
  ## where `a` has a leaf the matching subtree of `b` is yielded whole.
  ##
  ## Precondition:
  ## - weaklyCongruentT(a, b)
  if a.kind == tkNode:
    for i in 0 ..< a.children.len:
      result.add zipLeavesT(a.children[i], b.children[i])
  else:
    result.add (a, b)

proc congruentT*(a, b: IntTuple): bool =
  ## Same hierarchical profile: matching tuple or leaf structure at every level,
  ## whatever the leaves are
  if a.kind == tkNode and b.kind == tkNode:
    if a.children.len != b.children.len:
      return false
    for i in 0 ..< a.children.len:
      if not congruentT(a.children[i], b.children[i]):
        return false
    return true
  not (a.kind == tkNode or b.kind == tkNode)

proc weaklyCongruentT*(a, b: IntTuple): bool =
  ## `a` coarsens the profile of `b`: a leaf coarsens anything, a tuple never
  ## coarsens a leaf, tuples recurse
  if a.kind == tkNode and b.kind == tkNode:
    if a.children.len != b.children.len:
      return false
    for i in 0 ..< a.children.len:
      if not weaklyCongruentT(a.children[i], b.children[i]):
        return false
    return true
  a.kind != tkNode

proc unflattenT*(vals: seq[StrideScalar], profile: IntTuple): IntTuple =
  ## Tree of `profile` rebuilt from the flat leaf values `vals`, in order,
  ## the inverse of flatScalars.
  ##
  ## Raises:
  ## - LayoutError when `vals` is too short for the profile
  var idx = 0
  proc build(profile: IntTuple): IntTuple =
    if profile.kind == tkNode:
      var cs = newSeq[IntTuple](profile.children.len)
      for i in 0 ..< profile.children.len:
        cs[i] = build(profile.children[i])
      return itNode(cs)
    if idx >= vals.len:
      raise layoutError("unflatten: not enough values for the profile")
    inc idx
    itLeaf(vals[idx - 1])
  result = build(profile)

proc repeatLikeT*(v: int, profile: IntTuple): IntTuple =
  ## `v` replicated at every leaf of `profile`'s tree
  var vals: seq[StrideScalar]
  for _ in 0 ..< leafCount(profile):
    vals.add ssInt(v)
  unflattenT(vals, profile)

proc getSizeT*(x: IntTuple): int =
  ## Product of the integer leaves: the number of integral coordinates.
  ##
  ## Raises:
  ## - LayoutError for a non-integer leaf
  result = 1
  for v in flatScalars(x):
    if v.kind != skInt:
      raise layoutError("size: non-integer leaf " & ssStr(v))
    result *= v.intVal

proc prefixProductT*(a: IntTuple, base: StrideScalar): IntTuple =
  ## Exclusive prefix product of the leaves of `a`, seeded by the stride scalar
  ## `base` and congruent with `a`: `prefixProductT(it(3, 2, 4), 1)` is `(1, 3, 6)`.
  ##
  ## Raises:
  ## - LayoutError for a non-integer leaf of `a`
  var running = base
  var vals: seq[StrideScalar] = @[running]
  for v in flatScalars(a):
    if v.kind != skInt:
      raise layoutError("prefix_product: non-integer leaf " & ssStr(v))
    running = ssMul(running, v.intVal)
    vals.add running
  vals.setLen(leafCount(a))
  unflattenT(vals, a)

proc prefixProductT*(a: IntTuple, base: IntTuple): IntTuple =
  ## Exclusive prefix product with a per-mode tuple base: each mode is prefix
  ## producted independently from its own base leaf.
  ##
  ## Raises:
  ## - LayoutError when shape and base ranks differ at some level
  if base.kind == tkLeaf:
    return prefixProductT(a, base.leaf)
  if a.kind != tkNode or a.children.len != base.children.len:
    raise layoutError("prefix_product: shape and base ranks differ")
  var cs = newSeq[IntTuple](a.children.len)
  for i in 0 ..< a.children.len:
    cs[i] = prefixProductT(a.children[i], base.children[i])
  itNode(cs)

proc idx2crdT*(idx, shape: IntTuple): IntTuple =
  ## Natural coordinate of `shape` for any input coordinate.
  ## Decomposition is colexicographical, leftmost mode varying fastest.
  ##
  ## - the last mode keeps the full quotient
  ## - an out-of-bounds index accumulates in the last mode, which never wraps
  ##
  ## Example: `idx2crdT(itv(7), it(3, 2, 4))` is `(1, 0, 1)`.
  ##
  ## Raises:
  ## - LayoutError for a tuple coordinate of a scalar shape or a non-integer leaf
  if shape.kind == tkLeaf:
    if idx.kind != tkLeaf:
      raise layoutError("idx2crd: tuple coordinate of a scalar shape")
    if idx.leaf.kind != skInt:
      raise layoutError("idx2crd: non-integer coordinate leaf")
    return idx
  if idx.kind == tkLeaf and idx.leaf.kind == skAtup:
    # a coordinate-axis coordinate evaluates against the shape's axes, zero-extended to the shape's rank
    if shape.children.len < idx.leaf.data.len:
      raise layoutError("idx2crd: coordinate rank exceeds shape rank")
    var cs = newSeq[IntTuple](shape.children.len)
    for i in 0 ..< shape.children.len:
      if i < idx.leaf.data.len:
        if idx.leaf.data[i].kind != skInt:
          raise layoutError("idx2crd: non-integer coordinate-axis leaf")
        cs[i] = itv(idx.leaf.data[i].intVal)
      else:
        cs[i] = itv(0)
    return idx2crdT(itNode(cs), shape)
  if idx.kind == tkNode:
    if idx.children.len != shape.children.len:
      raise layoutError("idx2crd: incompatible rank")
    var cs = newSeq[IntTuple](shape.children.len)
    for i in 0 ..< shape.children.len:
      cs[i] = idx2crdT(idx.children[i], shape.children[i])
    return itNode(cs)
  # integer index over a tuple shape
  if idx.leaf.kind != skInt:
    raise layoutError("idx2crd: non-integer coordinate leaf")
  var i = idx.leaf.intVal
  let flat = flatScalars(shape)
  if flat.len == 0:
    raise layoutError("idx2crd: integer coordinate over an empty shape")
  var rem = newSeq[StrideScalar](flat.len)
  for j in 0 ..< flat.len - 1:
    if flat[j].kind != skInt:
      raise layoutError("idx2crd: non-integer shape leaf")
    if flat[j].intVal <= 0:
      raise layoutError("idx2crd: non-positive shape leaf " & $flat[j].intVal)
    rem[j] = ssInt(i mod flat[j].intVal)
    i = i div flat[j].intVal
  rem[flat.len - 1] = ssInt(i)
  unflattenT(rem, shape)

proc innerProductT*(coord, stride: IntTuple): StrideScalar =
  ## Sum of the leaf-wise products of two weakly congruent trees:
  ## a coordinate tree against a stride tree.
  ##
  ## Raises:
  ## - LayoutError for a non-integer coordinate leaf
  result = ssInt(0)
  for (c, d) in zipLeavesT(coord, stride):
    if c.leaf.kind != skInt:
      raise layoutError("inner_product: non-integer coordinate leaf")
    result = ssAdd(result, ssMul(d.leaf, c.leaf.intVal))

proc makeBasisLikeT*(profile: IntTuple, prefix: seq[int] = @[]): IntTuple =
  ## A `profile`-shaped tree of unit basis elements, one per leaf position:
  ## `makeBasisLikeT(it(10, 20))` is `(1@0, 1@1)`
  if profile.kind == tkNode:
    var cs = newSeq[IntTuple](profile.children.len)
    for i in 0 ..< profile.children.len:
      cs[i] = makeBasisLikeT(profile.children[i], prefix & @[i])
    return itNode(cs)
  itLeaf(eUnit(prefix))

proc leafPaths*(profile: IntTuple, prefix: seq[int] = @[]): seq[seq[int]] =
  ## Path of every leaf of `profile`, in left-to-right order
  if profile.kind == tkNode:
    for i in 0 ..< profile.children.len:
      result.add leafPaths(profile.children[i], prefix & @[i])
  else:
    result = @[prefix]

proc asTupleT*(d: StrideScalar): IntTuple =
  ## ArithTuple materialized as a tree: integers are leaves, ArithTuples become nodes of their converted children.
  if d.kind == skInt:
    return itv(d.intVal)
  var cs = newSeq[IntTuple](d.data.len)
  for i in 0 ..< d.data.len:
    cs[i] = asTupleT(d.data[i])
  itNode(cs)

# ═══════════════════════════════════════════════════════════════════════
# Equality and printing
# ═══════════════════════════════════════════════════════════════════════

proc eqTupleT*(a, b: IntTuple): bool =
  ## Tuple equality with python semantics.
  ##
  ## - nodes compare elementwise with equal length
  ## - leaves compare as stride scalars: integer 0 equals the all-zero ArithTuple
  ## - a node never equals a leaf
  if a.kind == tkNode and b.kind == tkNode:
    if a.children.len != b.children.len:
      return false
    for i in 0 ..< a.children.len:
      if not eqTupleT(a.children[i], b.children[i]):
        return false
    return true
  if a.kind == tkLeaf and b.kind == tkLeaf:
    return ssEq(a.leaf, b.leaf)
  false


proc strT*(x: IntTuple): string =
  ## Compact tuple form matching the python reference implementation:
  ## `(4, 8)` with spaces, a 1-tuple `(8,)` with a trailing comma, `()` empty
  case x.kind
  of tkLeaf:
    ssStr(x.leaf)
  of tkNode:
    if x.children.len == 0:
      "()"
    else:
      var parts: seq[string]
      for c in x.children:
        parts.add strT(c)
      if x.children.len == 1:
        "(" & parts[0] & ",)"
      else:
        "(" & parts.join(", ") & ")"
