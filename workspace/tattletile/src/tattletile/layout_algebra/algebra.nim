# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout algebra: coalesce, complement, compose, divide, right inverse, product.
##
## The transformation vocabulary the tile passes use to resolve layouts.
##
## - every op is a value-level port of the frozen python reference implementation, vendored pycute algebra.py plus the Layout methods
## - Nim outputs match the reference byte for byte: same shapes, same strides,
##   same basis-stride terms
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b4 \
##     --outdir:build/b4 workspace/tattletile/tests/test_layout_algebra_conformance.nim

import std/algorithm
import std/sequtils
import std/math
import std/tables

import ./htuple
import ./layout

# ═══════════════════════════════════════════════════════════════════════
# make_layout over sublayouts, tiler promotion
# ═══════════════════════════════════════════════════════════════════════

proc concatLayout*(ls: seq[Layout]): Layout =
  ## Concatenate sublayouts: each input becomes one mode of the result.
  ## An empty list gives the empty layout `():()`.
  if ls.len == 0:
    return layoutSet(itEmpty(), itEmpty())
  var sh = newSeq[IntTuple](ls.len)
  var st = newSeq[IntTuple](ls.len)
  for i in 0 ..< ls.len:
    sh[i] = ls[i].shape
    st[i] = ls[i].stride
  layoutSet(itNode(sh), itNode(st))

proc tilerToLayout*(t: IntTuple, e: StrideScalar = ssInt(1)): Layout =
  ## A tiler (an IntTuple of integer sizes) transformed into the layout acting
  ## identically under composition: every leaf size becomes one mode whose
  ## stride is the unit basis element of the leaf position.
  ##
  ## - `tilerToLayout(it(8, 8))` is `(8, 8):(1@0, 1@1)`
  ## - a scalar leaf becomes the rank-1 layout `t:e`
  ##
  ## Raises:
  ## - LayoutError for a non-integer tiler leaf
  if t.kind == tkLeaf:
    if t.leaf.kind != skInt:
      raise layoutError("tiler_to_layout: non-integer tiler leaf")
    return layoutSet(t, itLeaf(e))
  var ls = newSeq[Layout](t.children.len)
  for i in 0 ..< t.children.len:
    ls[i] = tilerToLayout(t.children[i], eUnit(@[i]))
  concatLayout(ls)

# ═══════════════════════════════════════════════════════════════════════
# Profiles and the coalesce fold
# ═══════════════════════════════════════════════════════════════════════

proc coprofile*(L: Layout): IntTuple =
  ## Profile of the layout's codomain, read straight off the strides.
  ##
  ## - the result materializes the sum of the stride leaves as a tree
  ## - an integer codomain gives an integer leaf, a Z^S codomain one leaf per axis
  var acc = ssInt(0)
  for d in flatScalars(L.stride):
    acc = ssAdd(acc, d)
  asTupleT(acc)

proc coalesceZCore*(shape, stride: IntTuple): tuple[s, d: IntTuple] =
  ## Size-1-preserving coalesce fold over flat leaf pairs.
  ##
  ## Adjacent modes `(sa, sb):(da, db)` merge into `sa*sb:da` exactly when the map stays linear across the pair, checked at two points:
  ##
  ## - `sa*da == db` (linearity at `(0, 1)`)
  ## - `(sa-1)*da + db == (2*sa-1)*da` (linearity at `(sa-1, 1)`)
  ##
  ## Refusing to merge is always sound: it gives a longer but equivalent layout.
  var rs = newSeq[IntTuple]()
  var rd = newSeq[IntTuple]()
  for (sb, db) in zipLeavesT(shape, stride):
    while rs.len > 0 and rs[^1].kind == tkLeaf and rs[^1].leaf.kind == skInt and
        rs[^1].leaf.intVal == 1:
      rs.del(rs.len - 1)
      rd.del(rd.len - 1)
    if rs.len > 0:
      let sa = rs[^1].leaf.intVal
      let da = rd[^1].leaf
      if ssEq(ssMul(da, sa), db.leaf) and
          ssEq(ssAdd(ssMul(da, sa - 1), db.leaf),
               ssMul(da, 2 * sa - 1)):
        rs[^1] = itv(sa * sb.leaf.intVal)
        continue
    rs.add sb
    rd.add db
  result = (itNode(rs), itNode(rd))

proc coalesceZLayout(shape, stride: IntTuple): Layout =
  ## An empty shape gives `1:0`, enclosing 1-tuples strip off.
  let (ns, nd) = coalesceZCore(shape, stride)
  if ns.kind == tkNode and ns.children.len == 0:
    return layoutSet(itv(1), itv(0))
  layoutSet(unwrapT(ns), unwrapT(nd))

proc coalesceZ*(L: Layout, profile: IntTuple): Layout =
  ## Coalesce `L` per `profile`. A scalar profile coalesces the whole layout
  ## and a tuple profile coalesces each mode against its profile entry.
  ## Modes beyond the profile length stay unchanged.
  ##
  ## Postconditions:
  ## - `size(coalesceZ(L, profile)) == size(L)`
  ## - `coalesceZ(L, profile)(i) == L(i)` for every integer `i`
  if profile.kind == tkLeaf:
    return coalesceZLayout(L.shape, L.stride)
  if rank(L) < profile.children.len:
    raise layoutError("Rank mismatch: coalesce_z(" & $L & ", " & strT(profile) & ")")
  var ls = newSeq[Layout]()
  let ms = L.modes()
  for i in 0 ..< ms.len:
    if i < profile.children.len:
      ls.add coalesceZ(ms[i], profile.children[i])
    else:
      ls.add ms[i]
  concatLayout(ls)

proc coalesce*(L: Layout, profile: IntTuple): Layout =
  ## Coalesce `L` per `profile` and drop a trailing size-1 mode of the whole result, scalar profile only.
  ##
  ## Postcondition: `L(i)` is preserved for every in-bounds integer `i`.
  if profile.kind == tkNode:
    if rank(L) < profile.children.len:
      raise layoutError("Rank mismatch: coalesce(" & $L & ", " & strT(profile) & ")")
    var ls = newSeq[Layout]()
    let ms = L.modes()
    for i in 0 ..< ms.len:
      if i < profile.children.len:
        ls.add coalesce(ms[i], profile.children[i])
      else:
        ls.add ms[i]
    return concatLayout(ls)
  let (ns, nd) = coalesceZCore(L.shape, L.stride)
  if ns.kind == tkNode and ns.children.len == 0:
    return layoutSet(itv(1), itv(0))
  if ns.children.len > 1 and ns.children[^1].kind == tkLeaf and
      ns.children[^1].leaf.kind == skInt and ns.children[^1].leaf.intVal == 1:
    return layoutSet(unwrapT(itNode(ns.children[0 ..< ^1])),
                     unwrapT(itNode(nd.children[0 ..< ^1])))
  layoutSet(unwrapT(ns), unwrapT(nd))

proc coalesce*(L: Layout): Layout =
  ## Merge contiguous dimensions of `L` into the simpler equivalent layout.
  ##
  ## - size-1 modes drop
  ## - a mode merges into the open chain when the chain's span reaches its stride
  coalesce(L, itv(1))

# ═══════════════════════════════════════════════════════════════════════
# Complement
# ═══════════════════════════════════════════════════════════════════════

type
  AxisAcc = object
    ## Per-codomain-axis gap-fill accumulator of the complement walk.
    ## Fields: the gap shapes `s` and the running strides `d`, both integers
    s: seq[int]
    d: seq[int]

proc complementBuildAxis(coprof: IntTuple, path: seq[int], acc: Table[seq[int], AxisAcc]): Layout =
  if coprof.kind == tkNode:
    var ls = newSeq[Layout](coprof.children.len)
    for i in 0 ..< coprof.children.len:
      ls[i] = complementBuildAxis(coprof.children[i], path & @[i], acc)
    return concatLayout(ls)
  let a = acc[path]
  var sh = newSeq[IntTuple](a.s.len + 1)
  for i in 0 ..< a.s.len:
    sh[i] = itv(a.s[i])
  sh[^1] = itv(1)
  var st = newSeq[IntTuple](a.d.len)
  for i in 0 ..< a.d.len:
    st[i] = itv(a.d[i])
  coalesceZLayout(itNode(sh), itNode(st))

proc complementScaleAxis(L: Layout, coprof: IntTuple, path: seq[int]): Layout =
  ## A scalar codomain keeps its integer strides, its unit basis element being 1.
  if coprof.kind == tkNode:
    let ms = L.modes()
    var ls = newSeq[Layout](coprof.children.len)
    for i in 0 ..< coprof.children.len:
      ls[i] = complementScaleAxis(ms[i], coprof.children[i], path & @[i])
    return concatLayout(ls)
  proc scaleStride(x: IntTuple): IntTuple =
    if x.kind == tkLeaf:
      return itLeaf(ssMul(eUnit(path), x.leaf.intVal))
    var cs = newSeq[IntTuple](x.children.len)
    for i in 0 ..< x.children.len:
      cs[i] = scaleStride(x.children[i])
    itNode(cs)
  layoutSet(L.shape, scaleStride(L.stride))

proc complementExtendAxis(L: Layout, coprof, extend: IntTuple, path: seq[int]): Layout =
  ## Last mode of each axis grows to `ceil(extend / last stride)`, the extension bound.
  if coprof.kind == tkNode:
    let ms = L.modes()
    var ls = newSeq[Layout](coprof.children.len)
    for i in 0 ..< coprof.children.len:
      ls[i] = complementExtendAxis(ms[i], coprof.children[i], extend,
                                   path & @[i])
    return concatLayout(ls)
  let shapeA = getModeT(extend, path).leaf.intVal
  let lastStride = wrapT(L.stride).children[^1]
  var sizeC = basisRepr(lastStride.leaf)[0].val
  var shapeR = @[shapeA]
  for i in 0 ..< shapeR.len:
    let s = shapeR[i]
    shapeR[i] = (s + sizeC - 1) div sizeC
    sizeC = (s + sizeC - 1) div s
  let wrapped = wrapT(L.shape)
  var cs = wrapped.children[0 ..< ^1]
  cs.add itupleNode(shapeR)
  makeLayout(itNode(cs), L.stride).coalesce()

proc complement*(L: Layout, extend: IntTuple): Layout =
  ## Complement of `L`, extended to cover the shape `extend`: the layout filling
  ## the codomain holes of `L`.
  ##
  ## - weakly congruent to the codomain
  ## - strictly ordered, `result(i-1) < result(i)`
  ## - disjoint from `L`
  ##
  ## The extension grows the last mode of each axis to the target size.
  ##
  ## Raises:
  ## - LayoutError when the strides overlap (a non-injective layout)
  let coprof = coprofile(L)
  var acc = initTable[seq[int], AxisAcc]()
  for p in leafPaths(coprof):
    acc[p] = AxisAcc(s: @[], d: @[1])
  # stride leaves sort by colex order on the stride scalars,
  # ties broken by the original order (a stable sort)
  let shLeaves = flatScalars(L.shape)
  let stLeaves = flatScalars(L.stride)
  var pairs: seq[tuple[de: StrideScalar, s: int, idx: int]] = @[]
  for i in 0 ..< stLeaves.len:
    if shLeaves[i].kind != skInt:
      raise layoutError("complement: non-integer shape leaf")
    pairs.add (stLeaves[i], shLeaves[i].intVal, i)
  proc cmpPairs(a, b: tuple[de: StrideScalar, s: int, idx: int]): int =
    if colexLt(a.de, b.de):
      return -1
    if colexLt(b.de, a.de):
      return 1
    cmp(a.idx, b.idx)
  pairs.sort(cmpPairs)
  for (de, s, _) in pairs:
    if ssZero(de) or s == 1:
      continue
    let rep = basisRepr(de)
    let path = rep[0].path
    let d = rep[0].val
    var a = acc[path]
    if d < a.d[^1]:
      raise layoutError("complement(" & $L & "): Non-injective layout")
    a.s.add(d div a.d[^1])
    a.d.add(d * s)
    acc[path] = a
  var res = complementBuildAxis(coprof, @[], acc)
  res = complementScaleAxis(res, coprof, @[])
  result = complementExtendAxis(res, coprof, extend, @[])

proc complement*(L: Layout): Layout =
  ## Minimal complement of `L`, no extension bound
  let coprof = coprofile(L)
  var acc = initTable[seq[int], AxisAcc]()
  for p in leafPaths(coprof):
    acc[p] = AxisAcc(s: @[], d: @[1])
  let shLeaves = flatScalars(L.shape)
  let stLeaves = flatScalars(L.stride)
  var pairs: seq[tuple[de: StrideScalar, s: int, idx: int]] = @[]
  for i in 0 ..< stLeaves.len:
    if shLeaves[i].kind != skInt:
      raise layoutError("complement: non-integer shape leaf")
    pairs.add (stLeaves[i], shLeaves[i].intVal, i)
  proc cmpPairs(a, b: tuple[de: StrideScalar, s: int, idx: int]): int =
    if colexLt(a.de, b.de):
      return -1
    if colexLt(b.de, a.de):
      return 1
    cmp(a.idx, b.idx)
  pairs.sort(cmpPairs)
  for (de, s, _) in pairs:
    if ssZero(de) or s == 1:
      continue
    let rep = basisRepr(de)
    let path = rep[0].path
    let d = rep[0].val
    var a = acc[path]
    if d < a.d[^1]:
      raise layoutError("complement(" & $L & "): Non-injective layout")
    a.s.add(d div a.d[^1])
    a.d.add(d * s)
    acc[path] = a
  var res = complementBuildAxis(coprof, @[], acc)
  result = complementScaleAxis(res, coprof, @[])

# ═══════════════════════════════════════════════════════════════════════
# Right inverse
# ═══════════════════════════════════════════════════════════════════════

type
  ChainEntry = object
    ## One stride-chain entry of the right inverse.
    ##
    ## - `de`: the stride
    ## - `s`: the mode shape
    ## - `pps`: the mode's domain index in the original order
    de: StrideScalar
    s: int
    pps: int

proc rightInverseInvertAxis(e: StrideScalar, flatD: IntTuple, chain: seq[ChainEntry]): Layout =
  ## Inverts one codomain axis along its stride chain, the chain visited in
  ## increasing stride order.
  var currD = e
  for c in flatD.children:
    let de = c.leaf
    if not ssZero(de) and ssEq(ssUnit(de), e):
      currD = ssUnit(de)
      break
  let one = projS(currD, basisRepr(currD)[0].path)
  var rs = newSeq[IntTuple]()
  var rd = newSeq[StrideScalar]()
  for entry in chain:
    let de = entry.de
    let s = entry.s
    if ssZero(de) or s == 1:
      continue
    let residue = ssSub(currD, de)
    if not ssEq(ssAdd(residue, residue), ssInt(0)):
      continue
    let reach = if ssZero(residue): residue else: ssMul(residue, s - 1)
    if not colexLt(reach, currD):
      continue
    var stride = ssMul(one, entry.pps)
    if not ssZero(residue):
      let crd = idx2crdT(itv(residue.intVal), itNode(rs))
      stride = ssAdd(stride, innerProductT(crd, itNode(rd.mapIt(itLeaf(it)))))
    rs.add itv(s)
    rd.add stride
    currD = ssMul(currD, s)
  coalesce(layoutSet(itNode(rs), itNode(rd.mapIt(itLeaf(it)))))

proc rightInverse*(L: Layout): Layout =
  ## Largest right inverse of `L`: the largest injective layout `R` satisfying
  ## `L(R(k)) == k` over the contiguous portion of the image.
  ##
  ## Each codomain axis inverts independently: a mode continues its chain only
  ## when its stride is exactly the running size. A smaller stride overlaps
  ## covered ground and a larger one leaves holes.
  ##
  ## Raises:
  ## - LayoutError when the layout's shape has non-integer leaves
  let (flatS, flatD) = coalesceZCore(L.shape, L.stride)
  let pps = prefixProductT(flatS, ssInt(1))
  var chain: seq[ChainEntry] = @[]
  for i in 0 ..< flatD.children.len:
    chain.add ChainEntry(de: flatD.children[i].leaf,
                         s: flatS.children[i].leaf.intVal,
                         pps: pps.children[i].leaf.intVal)
  proc cmpChain(a, b: ChainEntry): int =
    if colexLt(a.de, b.de):
      return -1
    if colexLt(b.de, a.de):
      return 1
    cmp(a.s, b.s)
  chain.sort(cmpChain)
  rightInverseInvertAxis(ssInt(1), flatD, chain)

# ═══════════════════════════════════════════════════════════════════════
# Greatest common domain and layout addition
# ═══════════════════════════════════════════════════════════════════════

proc greatestCommonDomain*(A, B: Layout): Layout =
  ## Layout selecting the greatest common domain of two shapes.
  ## The result factors, in order, the divisor shared by `shape(A)` and `shape(B)`.
  ##
  ## - the stride records where each common factor appears
  ## - depends only on the shapes
  ## - coprime leaf walks give the trivial singleton `1:0`
  var a = flatScalars(A.shape).mapIt(it.intVal)
  var b = flatScalars(B.shape).mapIt(it.intVal)
  var result_s = newSeq[int]()
  var result_d = newSeq[int]()
  var pA = 1
  var pB = 1
  var i = 0
  var j = 0
  while i < a.len and j < b.len:
    if a[i] == 1:
      inc i
      continue
    if b[j] == 1:
      inc j
      continue
    let lcm_pab = lcm(pA, pB)
    let rA = lcm_pab div pA
    let rB = lcm_pab div pB
    if a[i] mod rA == 0 and b[j] mod rB == 0:
      let gcd_rab = gcd(a[i] div rA, b[j] div rB)
      if gcd_rab != 1:
        result_s.add(gcd_rab)
        result_d.add(lcm_pab)
        a[i] = a[i] div (rA * gcd_rab)
        b[j] = b[j] div (rB * gcd_rab)
        pA = lcm_pab * gcd_rab
        pB = pA
        continue
    let eA = pA * a[i]
    let eB = pB * b[j]
    let qA = eB mod eA == 0
    let qB = eA mod eB == 0
    if qA or not qB:
      pA = eA
      inc i
    if qB or not qA:
      pB = eB
      inc j
  if result_s.len == 0:
    return layoutSet(itNode(@[itv(1)]), itNode(@[itv(0)]))
  layoutSet(itupleNode(result_s), itupleNode(result_d))

proc layoutAdd*(A, B: Layout): Layout =
  ## Coordinate-wise addition of two layouts of equal size: the layout `R`
  ## with `R(i) == A(i) + B(i)` for every `i`. `A` and `B` need not be compatible.
  ##
  ## Raises:
  ## - LayoutError on a size mismatch or when no common refinement exists
  if size(A) != size(B):
    raise layoutError("layout_add: size mismatch size(A)=" & $size(A) &
      " vs size(B)=" & $size(B))
  let aCo = coalesce(A)
  let bCo = coalesce(B)
  let g = greatestCommonDomain(aCo, bCo)
  if size(g) != size(A):
    raise layoutError("layout_add: A and B have no common refinement of size " &
      $size(A))
  var vals: seq[StrideScalar]
  for d in flatScalars(g.stride):
    if d.kind != skInt:
      raise layoutError("layout_add: non-integer domain stride")
    vals.add ssAdd(aCo.call(d.intVal), bCo.call(d.intVal))
  coalesce(layoutSet(g.shape, unflattenT(vals, g.stride)))

# ═══════════════════════════════════════════════════════════════════════
# Composition
# ═══════════════════════════════════════════════════════════════════════

proc compositionImpl(A: Layout, B: Layout): Layout =
  ## Group composition `A o B` for a layout `B`: the layout whose domain is
  ## `B` and whose values are `A` evaluated at `B`'s values.
  ##
  ## Raises:
  ## - LayoutError on the shape or stride divisibility conditions
  let a2 = coalesceZ(A, coprofile(B))
  if B.shape.kind == tkNode:
    var ls = newSeq[Layout]()
    for b in B.modes():
      ls.add compositionImpl(a2, b)
    return concatLayout(ls)
  # B is a scalar-shape layout, N:M
  let bShape = B.shape.leaf.intVal
  let bStride = B.stride.leaf
  if ssZero(bStride):
    return layoutSet(B.shape, itv(0))
  if bShape == 1:
    # A evaluated at one point, the stride itself is the coordinate
    return layoutSet(B.shape, itLeaf(a2.call(itLeaf(bStride))))
  var haveResult = false
  var resultL: Layout
  for (strideB, basisB) in basisRepr(bStride):
    let ab = a2.getMode(basisB)
    var rs = wrapT(ab.shape).children.mapIt(it.leaf.intVal)
    var rd = wrapT(ab.stride).children.mapIt(it.leaf)
    var sb = strideB
    # divide out the first strideB elements of the A layout
    while rs.len > 1:
      let q = sb div rs[0]
      let r = sb mod rs[0]
      if r != 0:
        break
      sb = q
      rs.delete(0)
      rd.delete(0)
    rd[0] = ssMul(rd[0], sb)
    if rs.len > 1:
      let q = rs[0] div sb
      let r = rs[0] mod sb
      if r == 0 and q > 0:
        rs[0] = q
      elif not (q >= bShape - 1):
        raise layoutError("Stride divisibility condition violated: composition(" &
          $A & ", " & $B & ")")
    # keep the first B.shape elements of what remains
    rs[^1] = bShape
    for i in 0 ..< rs.len - 1:
      let q = rs[^1] div rs[i]
      let r = rs[^1] mod rs[i]
      rs[^1] = q
      if q == 0:
        rs[i] = r
        rs = rs[0 .. i]
        rd = rd[0 .. i]
        break
      if r != 0:
        raise layoutError("Shape divisibility condition violated: composition(" &
          $A & ", " & $B & ")")
    var shape = newSeq[IntTuple]()
    for s in rs:
      shape.add itv(s)
    var stride = newSeq[IntTuple]()
    for d in rd:
      stride.add itLeaf(d)
    let term = layoutSet(itNode(shape), itNode(stride))
    if haveResult:
      resultL = layoutAdd(resultL, term)
    else:
      resultL = term
      haveResult = true
  coalesce(resultL)

proc compose*(A: Layout, B: Layout): Layout =
  ## Group composition `A o B`: the layout `R` with `R(i) == A(B(i))` for every
  ## `i` in the domain of `B`. A tuple `B` maps mode by mode, one mode of `A`
  ## per mode of `B`.
  ##
  ## Examples:
  ##   compose(makeLayout(it(20), itv(2)), makeLayout(it(5, 4), it(4, 1)))
  ##   # -> (5, 4):(8, 2)
  ##
  ## Raises:
  ## - LayoutError on the composition divisibility conditions
  compositionImpl(A, B)

proc compose*(t: IntTuple, B: Layout): Layout =
  ## Composition with a tiler tuple, promoted to its layout first:
  ## `compose(it(8, 8), tv) == compose(tilerToLayout(it(8, 8)), tv)`.
  compositionImpl(tilerToLayout(t), B)

type
  PerMode* = object
    ## One right-hand mode of a by-mode composition: a layout, or the absent
    ## mode that leaves the corresponding mode of the left side unchanged.
    case present*: bool
    of true:
      layout*: Layout
    of false:
      discard

proc someMode*(l: Layout): PerMode {.inline.} =
  ## A by-mode composition right side holding the layout `l`
  PerMode(present: true, layout: l)

proc noneMode*: PerMode {.inline.} =
  ## Absent by-mode composition right side, a no-op on that mode.
  PerMode(present: false)

proc composeModes*(A: Layout, bs: openArray[PerMode]): Layout =
  ## By-mode composition of `A` with the right sides `bs`, one mode of `A` per entry.
  ## `(A0, A1, ...) o (X, None)` equals `(A0 o X, A1)`.
  ##
  ## An absent entry leaves its mode whole.
  ##
  ## Raises:
  ## - LayoutError when `A` has fewer modes than `bs` has entries
  if rank(A) < bs.len:
    raise layoutError("Rank mismatch: composition(" & $A & ", by-mode tiler)")
  let ms = A.modes()
  var ls = newSeq[Layout](bs.len)
  for i in 0 ..< bs.len:
    if bs[i].present:
      ls[i] = compositionImpl(ms[i], bs[i].layout)
    else:
      ls[i] = ms[i]
  concatLayout(ls)

# ═══════════════════════════════════════════════════════════════════════
# Divide and product
# ═══════════════════════════════════════════════════════════════════════

proc logicalDivide*(A: Layout, tiler: Layout): Layout =
  ## Split layout `A` by the tile `B`: mode 0 of the result is the tile
  ## (`compose(A, B)`) and mode 1 numbers the tiles. Every element of `A`
  ## appears exactly once.
  ##
  ## Raises:
  ## - LayoutError when the tiler does not divide `A`
  compose(A, concatLayout(@[tiler, complement(tiler, A.shape)]))

proc logicalDivide1(A: Layout, t: int): Layout =
  ## Rank-1 logical divide of a scalar-shape layout by an integer,
  ## the tile pair `(t, ceil(shape/t)):(stride, stride*t)`.
  if A.shape.kind == tkLeaf:
    if A.stride.kind != tkLeaf:
      raise layoutError("logical_divide: non-scalar stride of a scalar shape")
    let s = A.shape.leaf.intVal
    let d = A.stride.leaf
    let sh = itNode(@[itv(t), itv(ceilDiv(s, t))])
    let st = itNode(@[itLeaf(d), itLeaf(ssMul(d, t))])
    return makeLayout(sh, st)
  logicalDivide(A, makeLayout(itv(t)))

proc logicalDivide*(A: Layout, tiler: IntTuple): Layout =
  ## Logical divide by a tiler tuple: an integer divides the rank-1 layout
  ## directly and a tuple divides by mode, one mode of `A` per tiler entry.
  ##
  ## Raises:
  ## - LayoutError when `A` has fewer modes than the tiler or the tiler nests
  if tiler.kind == tkLeaf:
    if tiler.leaf.kind != skInt:
      raise layoutError("logical_divide: non-integer tiler leaf")
    return logicalDivide1(A, tiler.leaf.intVal)
  let ms = A.modes()
  if ms.len < tiler.children.len:
    raise layoutError("Rank mismatch: logical_divide(" & $A & ", " & strT(tiler) & ")")
  var ls = newSeq[Layout]()
  for i in 0 ..< tiler.children.len:
    let t = tiler.children[i]
    if t.kind != tkLeaf or t.leaf.kind != skInt:
      raise layoutError("logical_divide: nested tuple tiler entries are outside the supported subset")
    ls.add logicalDivide1(ms[i], t.leaf.intVal)
  concatLayout(ls)

proc zippedDivide*(A: Layout, tiler: Layout): Layout =
  ## Logical divide of `A` by the tiler layout `B`: a layout tiler is already
  ## zipped and the result is `((tile...), (rest...))`.
  logicalDivide(A, tiler)

proc zippedDivide*(A: Layout, tiler: IntTuple): Layout =
  ## Logical divide of `A` by an integer-size tiler tuple. The tiler is first
  ## promoted to a layout: `zippedDivide(A, it(8, 8))` partitions both modes at once
  logicalDivide(A, tilerToLayout(tiler))

proc logicalProduct*(A: Layout, tiler: Layout): Layout =
  ## Reproduce the block `A` over the grid the tiler describes:
  ## `A x B = (A, A* o B)`. The rank-2 result has `A` itself as mode 0 plus
  ## copy numbering `complement(A) o B` as mode 1.
  ##
  ## Raises:
  ## - LayoutError when the copy grid cannot tile without overlap
  concatLayout(@[A, compose(complement(A), tiler)])

proc logicalProduct1(A: Layout, t: int): Layout =
  logicalProduct(A, layoutSet(itv(t), itv(1)))

proc logicalProduct*(A: Layout, tiler: IntTuple): Layout =
  ## Reproduce the block `A` over an integer tiler tuple, one mode of `A` per
  ## tiler entry. An integer entry tiles that mode directly.
  ##
  ## Raises:
  ## - LayoutError when `A` has fewer modes than the tiler or the tiler nests
  if tiler.kind == tkLeaf:
    if tiler.leaf.kind != skInt:
      raise layoutError("logical_product: non-integer tiler leaf")
    return logicalProduct(A, layoutSet(tiler, itv(1)))
  let ms = A.modes()
  if ms.len < tiler.children.len:
    raise layoutError("Rank mismatch: logical_product(" & $A & ", " & strT(tiler) & ")")
  var ls = newSeq[Layout]()
  for i in 0 ..< tiler.children.len:
    let t = tiler.children[i]
    if t.kind == tkLeaf and t.leaf.kind == skInt:
      ls.add logicalProduct1(ms[i], t.leaf.intVal)
    else:
      raise layoutError("logical_product: nested tuple tiler entries are outside the supported subset")
  concatLayout(ls)
