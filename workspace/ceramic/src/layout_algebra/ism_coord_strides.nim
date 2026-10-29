# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/strutils
import std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/macros/replace_nodes

# ═══════════════════════════════════════════════════════════════
#   Integer Semi-Module - Coordinate Strides
# ═══════════════════════════════════════════════════════════════

type
  CoordStride*[AT: static tuple] = object

func `===`*[A, B: CoordStride](a: A, b: B): bool {.inline.} =
  A is B

func `===`*[T: tuple](a: T, b: CoordStride): bool {.inline.} =
  when T.rank() == 1:
    a[0] === b
  else:
    false

func `===`*(a: CoordStride, b: tuple): bool {.inline.} =
  when b.rank() == 1:
    a === b[0]
  else:
    false

# ═══════════════════════════════════════════════════════════════
#   NimNode helpers and syntactic sugar
# ═══════════════════════════════════════════════════════════════

proc unwrapTypedesc(n: NimNode): NimNode

macro `+`*[A: CoordStride, B: CoordStride](a: A, b: B): untyped =
  let ta = a.getTypeInst[1]
  let tb = b.getTypeInst[1]
  if not ta.isTupleTy() and not tb.isTupleTy():
    return nnkCall.newTree(nnkBracketExpr.newTree(
      bindSym"CoordStride", newIntLitNode(ta.intVal.int + tb.intVal.int)))
  var builder = TupleBuilderNested.new(1)
  if not ta.isTupleTy() or not tb.isTupleTy():
    # a scalar coordinate stride broadcasts over the other's coefficients
    let scalar = if ta.isTupleTy(): tb else: ta
    let tupleSide = if ta.isTupleTy(): ta else: tb
    for ev in tupleSide.tupleStream():
      builder.onLeaves(ev):
        let c = ev.leaf.intVal.int + scalar.intVal.int
        builder.append((newIntLitNode(c), intTy(c)))
  else:
    var zip = zipLongest(ta.tupleStream(), tb.tupleStream())
    while not zip.done():
      let (ev, other) = zip.next()
      builder.onLeaves(ev):
        let v = if ev.leaf.isNil: 0 else: ev.leaf.intVal.int
        let w = if other.leaf.isNil: 0 else: other.leaf.intVal.int
        let c = v + w
        builder.append((newIntLitNode(c), intTy(c)))
  return nnkCall.newTree(nnkBracketExpr.newTree(
    bindSym"CoordStride", builder.emit(0)))

macro `*`*[A: CoordStride](a: A, k: static int): untyped =
  let ta = a.getTypeInst[1]
  if not ta.isTupleTy():
    return nnkCall.newTree(nnkBracketExpr.newTree(
      bindSym"CoordStride", newIntLitNode(k * ta.intVal.int)))
  var builder = TupleBuilderNested.new(1)
  for ev in ta.tupleStream():
    builder.onLeaves(ev):
      let c = k * ev.leaf.intVal.int
      builder.append((newIntLitNode(c), intTy(c)))
  return nnkCall.newTree(nnkBracketExpr.newTree(
    bindSym"CoordStride", builder.emit(0)))

macro `*`*[V: static int, A: CoordStride](a: A, k: Int[V]): untyped =
  return a * newLit(V)

macro `*`*[V: static int, A: CoordStride](k: Int[V], a: A): untyped =
  return a * newLit(V)

macro `*`*[A: CoordStride](k: static int, a: A): untyped =
  return a * newLit(k)

macro `+`*[A: CoordStride](a: A, k: static int): untyped =
  let ta = a.getTypeInst[1]
  if not ta.isTupleTy():
    return nnkCall.newTree(nnkBracketExpr.newTree(
      bindSym"CoordStride", newIntLitNode(ta.intVal.int + k)))
  var builder = TupleBuilderNested.new(1)
  var first = true
  for ev in ta.tupleStream():
    builder.onLeaves(ev):
      let c = ev.leaf.intVal.int + (if first: k else: 0)
      builder.append((newIntLitNode(c), intTy(c)))
      first = false
  return nnkCall.newTree(nnkBracketExpr.newTree(
    bindSym"CoordStride", builder.emit(0)))

macro `+`*[A: CoordStride](k: static int, a: A): untyped =
  return a + newLit(k)

macro `+`*[V: static int, A: CoordStride](a: A, k: Int[V]): untyped =
  return a + newLit(V)

macro `+`*[V: static int, A: CoordStride](k: Int[V], a: A): untyped =
  return a + newLit(V)

macro `+`*[A: CoordStride](a: A, k: tuple): untyped =
  let ta = a.getTypeInst[1]
  var builder = TupleBuilderNested.new(1)
  if not k.getTypeInst.isTupleTy():
    # a scalar k broadcasts over every coeff leaf
    for ev in ta.tupleStream():
      builder.onLeaves(ev):
        builder.append((k + ev.leaf, k.getTypeInst()))
  else:
    var zip = zipLongest(ta.tupleStream(), k.tupleStream())
    while not zip.done():
      let (ea, eb) = zip.next()
      builder.onLeaves(ea):
        let c = if ea.leaf.isNil: newIntLitNode(0) else: ea.leaf
        let kExpr = if eb.leaf.isNil: newIntLitNode(0) else: eb.leaf
        if kExpr.kind == nnkIntLit and c.kind == nnkIntLit:
          let sum = kExpr.intVal.int + c.intVal.int
          builder.append((newIntLitNode(sum), intTy(sum)))
        else:
          let sumExpr = kExpr + c
          builder.appendTyped(sumExpr)
  return builder.emit(0)

macro `+`*[A: CoordStride](k: tuple, a: A): untyped =
  return nnkCall.newTree(bindSym"+", a, k)

macro `*`*[A: CoordStride, B: CoordStride](a: A, b: B): untyped =
  error("CoordStride *: cannot scale by a coordinate stride", b)

macro `div`*[A: CoordStride](a: A, k: static int): untyped =
  ## Divides a coordinate stride's coeffs by a static scalar
  var builder = TupleBuilderNested.new(1)
  for ev in a.getTypeInst[1].tupleStream():
    builder.onLeaves(ev):
      let c = ev.leaf.intVal.int div k
      builder.append((newIntLitNode(c), intTy(c)))
  return nnkCall.newTree(nnkBracketExpr.newTree(
    bindSym"CoordStride", builder.emit(0)))

macro `*`*[A: CoordStride](k: typed, a: A): untyped =
  let ta = a.getTypeInst[1]
  if not ta.isTupleTy():
    return k * newIntLitNode(ta.intVal.int)
  var builder = TupleBuilderNested.new(1)
  for ev in ta.tupleStream():
    builder.onLeaves(ev):
      builder.append((k * ev.leaf, k.getTypeInst()))
  return builder.emit(0)

# ═══════════════════════════════════════════════════════════════
#   Evaluation: runtime coordinate leaf x coordinate stride
# ═══════════════════════════════════════════════════════════════

macro `+`*(a, b: tuple): untyped =
  ## Semimodule completion, evaluation side: leafwise
  ## coordinate-tuple addition, evaluated at runtime.
  ##
  ## - the stream walks each argument's type for structure and yields
  ##   the argument's element expressions as leaves
  var zip = zipLongest(a.tupleStream(), b.tupleStream())
  var builder = TupleBuilderNested.new(1)
  while not zip.done():
    let (ea, eb) = zip.next()
    builder.onLeaves(ea):
      let x = if ea.leaf.isNil: newIntLitNode(0) else: ea.leaf
      let y = if eb.leaf.isNil: newIntLitNode(0) else: eb.leaf
      if x.kind == nnkIntLit and y.kind == nnkIntLit:
        let sum = x.intVal.int + y.intVal.int
        builder.append((newIntLitNode(sum), intTy(sum)))
      else:
        let sumExpr = x + y
        builder.appendTyped(sumExpr)
  return builder.emit(0)

# ═══════════════════════════════════════════════════════════════
#   Constructors
# ═══════════════════════════════════════════════════════════════

macro E*(mode: static int): untyped =
  ## Unit basis atom at `mode`: E(0) is 1@0, E(1) is 1@1.
  var coeffs = nnkTupleConstr.newTree()
  for _ in 0 ..< mode:
    coeffs.add(newIntLitNode(0))
  coeffs.add(newIntLitNode(1))
  return nnkCall.newTree(
    nnkBracketExpr.newTree(bindSym"CoordStride", coeffs))

macro E*(scale, mode: static int): untyped =
  ## Scaled basis atom: E(2, 0) is 2@0, E(-2, 1) is -2@1.
  var coeffs = nnkTupleConstr.newTree()
  for _ in 0 ..< mode:
    coeffs.add(newIntLitNode(0))
  coeffs.add(newIntLitNode(scale))
  return nnkCall.newTree(nnkBracketExpr.newTree(bindSym"CoordStride", coeffs))

macro E*(coeffs: typed): untyped =
  ## Multi-term atom from a coefficient vector: E((1, 1)) is 1@0 + 1@1.
  return nnkCall.newTree(nnkBracketExpr.newTree(bindSym"CoordStride", coeffs))

proc getCoordinates(path: seq[int]): NimNode =
  var node = newIntLitNode(1)
  for i in countdown(path.len - 1, 0):
    var level = nnkTupleConstr.newTree()
    for j in 0 ..< path[i] + 1:
      if j == path[i]:
        level.add(node)
      else:
        level.add(newIntLitNode(0))
    node = level
  return node

macro make_basis_like*(profile: typed): untyped =
  ## Builds a (nested) tuple of coordinates describing the input `profile`.
  ##
  ## The coordinates are represented as:
  ##     value@path.
  ##
  ## Value is either 1 (a value exist at those coordinates)
  ## or 0 (out-of-bounds)
  ##
  ## Examples:
  ##
  ##   make_basis_like((10, 20))       ==  (1@0, 1@1)
  ##   make_basis_like((10, (20, 30))) ==  (1@0, (1@0@1, 1@1@1))
  ##   make_basis_like(10)             ==  1
  ##
  ## In particular, the following:
  ##
  ##   make_basis_like((10, (20, 30))) ==  (1@0, (1@0@1, 1@1@1))
  ##
  ## reads '20' exists at inner index 0 of outer tuple index 1
  ##
  ## Note:
  ##   For the life of me, I have no idea why coordinates/paths are printed
  ##   innermost index first in the pycute / CuTe reference.

  if not profile.getTypeInst().isTupleTy():
    return newIntLitNode(1)
  var builder = TupleBuilderNested.new(1)
  for ev in profile.tupleStream():
    builder.onLeaves(ev):
      let coords = ev.path.getCoordinates()
      builder.append((nnkCall.newTree(
        nnkBracketExpr.newTree(bindSym"CoordStride", coords)),
        nnkBracketExpr.newTree(bindSym"CoordStride", coords)))
  return builder.emit(0)

# ═══════════════════════════════════════════════════════════════
#   Pretty-printing
# ═══════════════════════════════════════════════════════════════

proc pretty(t: NimNode, path: seq[int]): string =
  # TODO:
  #   For the life of me, I have no idea why pycute / CuTe
  #   is printing `E` paths innermost first.

  # 1. classify - depth-first count of nonzero leaves
  var
    nzCount = 0
    nzValue: BiggestInt
    nzPath: seq[int]
  proc walk(n: NimNode, p: seq[int]) =
    if n.kind == nnkTupleConstr:
      for i in 0 ..< n.len:
        walk(n[i], p & i)
    else:
      if n.intVal != 0:
        nzCount += 1
        nzValue = n.intVal
        nzPath = p
  walk(t, path)

  # 2. zero element
  if nzCount == 0:
    return "0"

  # 3. single basis leaf - value@p_n@...@p_0, innermost first
  if nzCount == 1:
    var s = $nzValue
    for i in countdown(nzPath.len - 1, 0):
      s.add("@")
      s.add($nzPath[i])
    return s

  # 4. multi-term - nested tuple, recursive pretty printing
  var parts: seq[string]
  for i in 0 ..< t.len:
    if t[i].kind == nnkTupleConstr:
      parts.add(pretty(t[i], @[]))
    else:
      parts.add($t[i].intVal)
  return "(" & parts.join(",") & ")"

macro `$`*(T: typedesc[CoordStride]): string =
  return newStrLitNode(pretty(T.unwrapTypedesc(), @[]))

macro `$`*(a: CoordStride): string =
  return newStrLitNode(pretty(a.unwrapTypedesc(), @[]))

# ═══════════════════════════════════════════════════════════════
#   Compile-time helpers for layout algebra
# ═══════════════════════════════════════════════════════════════

type
  CoordStrideKind* = enum
    caNone    # not a CoordStride leaf: an int or dynamic stride
    caSingle  # one nonzero coefficient: scale@path
    caMulti   # a coefficient vector with several nonzero terms

  CoordStrideDescriptor* = object
    kind*: CoordStrideKind
    path*: seq[int]
    scale*: int
    coeffs*: NimNode # caMulti: the coefficient tuple, values as int literals

proc isCoordStride*(leafTy: NimNode): bool =
  ## True when the type is a coordinate stride instantiation, false for nil
  if leafTy == nil:
    return false
  var t = leafTy
  if t.kind == nnkSym:
    let impl = t.getImpl
    if impl.kind == nnkTypeDef and impl[2].kind == nnkBracketExpr and
        impl[2][0].kind == nnkSym and impl[2][0].strVal == "typeDesc":
      t = impl[2][1]
    else:
      return false
  return t.kind == nnkBracketExpr and t[0].kind == nnkSym and
    t[0].strVal == "CoordStride"

proc getCoordStrideDescriptor*(leafTy: NimNode): CoordStrideDescriptor =
  ## Extracts the compile-time descriptor of one stride leaf from its type.
  if not isCoordStride(leafTy):
    return CoordStrideDescriptor(kind: caNone)
  var t = leafTy
  if t.kind == nnkSym:
    t = t.getImpl[2][1]
  let coeffs = t[1]
  var
    nzCount = 0
    nzValue: int
    nzPath: seq[int]
  proc walk(n: NimNode, p: seq[int]) =
    if n.kind in {nnkTupleConstr, nnkPar}:
      for i in 0 ..< n.len:
        walk(n[i], p & i)
    else:
      if n.intVal != 0:
        nzCount += 1
        nzValue = int(n.intVal)
        nzPath = p
  walk(coeffs, @[])
  if nzCount == 0:
    CoordStrideDescriptor(kind: caMulti, coeffs: coeffs)
  elif nzCount == 1:
    CoordStrideDescriptor(kind: caSingle, path: nzPath, scale: nzValue, coeffs: coeffs)
  else:
    CoordStrideDescriptor(kind: caMulti, coeffs: coeffs)

proc proj*(slots: var seq[tuple[path: seq[int], cur: int, curNode: NimNode]], basis: seq[int]): int =
  ## Slot position: the record's position the basis path implies.
  ## Returns:
  ##     the record's index, the record created on first address
  ##     with its span seeded at 1
  for k in 0 ..< slots.len:
    if slots[k].path == basis:
      return k
  slots.add (basis, 1, IntCT(1))
  slots.len - 1

func proj*(leafTy: NimNode): tuple[path: seq[int], scale: int] {.compileTime.} =
  ## Extract from a stride leaf the part its own position implies.
  ## An atom yields its basis path and scale.
  ## Returns:
  ## - an integer leaf: the rank-zero identity, an empty path,
  ##   the scale is its static value
  ## - a single-term atom: its basis path and scale
  ## Precondition:
  ## - a multi-term atom raises
  if not isCoordStride(leafTy):
    return (@[], leafTy.getStaticInt())
  let d = getCoordStrideDescriptor(leafTy)
  if d.kind == caMulti:
    error "proj: multi-term coordinate strides are not supported"
  (d.path, d.scale)

macro `<`*[A: CoordStride, B: CoordStride](a: A, b: B): bool =
  ## Colex order over coordinate strides, ascending on (axis, scale),
  ## the key order of the complement's stride sort
  let da = a.getTypeInst().getCoordStrideDescriptor()
  let db = b.getTypeInst().getCoordStrideDescriptor()
  doAssert da.kind == caSingle and db.kind == caSingle,
    "CoordStride <: only single-term atoms order"
  return newLit((da.path[^1], da.scale) < (db.path[^1], db.scale))

proc csScaledEq(a, b: NimNode, s: int): bool =
  ## Elementwise b == s*a over (possibly nested) int-literal tuples,
  ## false on any structural mismatch.
  if a.kind in {nnkTupleConstr, nnkPar} and b.kind in {nnkTupleConstr, nnkPar}:
    if a.len != b.len:
      return false
    for i in 0 ..< a.len:
      if not csScaledEq(a[i], b[i], s):
        return false
    return true
  if a.kind in {nnkTupleConstr, nnkPar} or b.kind in {nnkTupleConstr, nnkPar}:
    return false
  b.intVal == s * a.intVal

func csCanMerge*(a, b: CoordStrideDescriptor, s: int): bool {.compileTime.} =
  ## Merge test for adjacent stride leaves, symbolic `s * d_a == d_b`:
  ##
  ##   single-term:  path_a == path_b  and  scale_b == s * scale_a
  ##   multi-term:   coeff_b[i] == s * coeff_a[i]  for every i
  ##
  ## Returns:
  ##   true when the leaves merge. The merged dimension's stride is
  ##   the first leaf's atom d_a, unchanged.
  case a.kind
  of caNone:
    false
  of caSingle:
    b.kind == caSingle and a.path == b.path and b.scale == s * a.scale
  of caMulti:
    b.kind == caMulti and csScaledEq(a.coeffs, b.coeffs, s)

proc unwrapTypedesc(n: NimNode): NimNode =
  var t = n.getTypeInst
  if t.kind == nnkBracketExpr and t[0].kind == nnkSym and t[0].strVal == "typeDesc":
    t = t[1]
  if t.kind == nnkBracketExpr and t[0].kind == nnkSym and t[0].strVal == "CoordStride":
    return t[1]
  if t.kind == nnkSym:
    return t.getImpl[2][1]
  error("CoordStride: not an CoordStride instantiation: " & t.repr, n)

