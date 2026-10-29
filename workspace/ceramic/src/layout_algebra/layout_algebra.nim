# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout algebra: coalesce, complement, compose, divide, inverses, products.

import std/macros
import std/algorithm
import std/sequtils
import std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/math
import workspace/ceramic/src/int_tuples/int_tuples_unsanctioned_helpers
import ./layouts_unsanctioned_helpers
import ./layouts
import ./layout_compiletime {.all.}
import ./layouts_unsanctioned_helpers
import ./ism_coord_strides
import ./layout_indexing
import ./layout_indexing_gpu

# ═══════════════════════════════════════════════════════════════
#  getIndicesSortedByStride
# ═══════════════════════════════════════════════════════════════

proc getIndicesSortedByStride(strides: seq[int]): seq[int] {.compileTime.} =
  ## Return indices sorted by stride ascending.
  ## Data stays in original arrays, iterate this permutation.
  result = newSeq[int](strides.len)
  for i in 0 ..< result.len:
    result[i] = i
  for i in 0 ..< result.len:
    for j in i + 1 ..< result.len:
      if strides[result[i]] > strides[result[j]]:
        swap result[i], result[j]

# ═══════════════════════════════════════════════════════════════
#  coalesce
# ═══════════════════════════════════════════════════════════════
#
# Pseudo code
#
#   coalesce(A):   # simplify the layout without changing it
#                  # as a function from integers to integers
#
#     result = the first mode s₀:d₀ of A, flattened
#     for each next mode s₁:d₁:
#         result = result ++ s₁:d₁
#     return result
#
#     s₀:d₀ ++ s₁:d₁, four cases:
#     1. s₁ == 1:      s₀:d₀                    # a size-1 mode is ignored
#     2. s₀ == 1:      s₁:d₁                    # a size-1 mode is ignored
#     3. d₁ == s₀*d₀:  s₀*s₁:d₀                 # no gap between the modes
#     4. else:         (s₀,s₁):(d₀,d₁)
#
#     the size is unchanged and every input integer maps to the
#     same output integer

macro coalesceImpl(sh, st: typed, dropTrailing: static bool = false): untyped =
  ## Merges adjacent dimensions as long as there are no gaps between their elements.
  var builder = TupleBuilderFlat.new(2)
  var span = 0                 # the open chain's shape product, 0 = no open chain
  var headTy: NimNode          # the open chain's head stride type
  var pending: NimNode         # a trailing size-1 leaf's stride, nil when none
  var pendingTy: NimNode       # the pending stride's type
  var pendingVal = DynamicSentinel  # the pending stride's static value

  template appendChain =
    let headVal = headTy.getStaticInt()
    if headVal != DynamicSentinel:
      builder.append((IntCT(span), intTy(span)), (IntCT(headVal), intTy(headVal)))
    else:
      var t = headTy
      if t.kind == nnkSym:
        t = t.getImpl[2][1]
      builder.append((IntCT(span), intTy(span)),
        (bindSym"E".newCall(t[1]), nnkBracketExpr.newTree(bindSym"CoordStride", t[1])))

  var shS = sh.tupleStream()
  var stS = st.tupleStream()
  while not shS.done():
    let shapeEv = shS.next()
    let strideEv = stS.next()
    if shapeEv.kind != kLeaf:
      continue
    let shapeVal = shapeEv.leafTy.getStaticInt()
    let strideVal = strideEv.leafTy.getStaticInt()
    let headVal = if headTy.isNil: DynamicSentinel else: headTy.getStaticInt()

    let strideCrd = strideEv.leafTy.getCoordStrideDescriptor()
    let headCrd = if headTy.isNil: CoordStrideDescriptor(kind: caNone)
                  else: headTy.getCoordStrideDescriptor()
    let crdReachable =
      strideCrd.kind != caNone and headCrd.kind != caNone and
      shapeVal != DynamicSentinel and csCanMerge(headCrd, strideCrd, span)

    if not headTy.isNil and
        ((shapeVal != DynamicSentinel and strideVal != DynamicSentinel and
          headVal != DynamicSentinel and span * headVal == strideVal) or
         crdReachable):
      # the chain's span reaches this dimension's stride, merge frontward
      span *= shapeVal
      pending = nil
      pendingVal = DynamicSentinel
    elif shapeVal == 1:
      pending = strideEv.leaf
      pendingTy = strideEv.leafTy
      pendingVal = if strideCrd.kind == caNone: strideVal else: DynamicSentinel

    else: # Flush
      if not headTy.isNil:
        appendChain()
      if shapeVal != DynamicSentinel and
          (strideVal != DynamicSentinel or strideCrd.kind != caNone):
        span = shapeVal
        headTy = strideEv.leafTy
      else:
        builder.append((shapeEv.leaf, shapeEv.leafTy),
          (if strideVal != DynamicSentinel: (IntCT(strideVal), intTy(strideVal))
           else: (strideEv.leaf, strideEv.leafTy)))
        span = 0
        headTy = nil
      pending = newEmptyNode()
      pendingVal = DynamicSentinel

  if not headTy.isNil:
    appendChain()
  elif pending.isNil:
    builder.append((IntCT(1), intTy(1)), (IntCT(0), intTy(0)))
  if not pending.isNil and pending.kind != nnkEmpty:
    # the discard drops the trailing size-1 mode only when other
    # modes precede it, a lone size-1 result stays
    if not dropTrailing or builder.count == 0:
      builder.append((IntCT(1), intTy(1)),
        (if pendingVal != DynamicSentinel: (IntCT(pendingVal), intTy(pendingVal))
         else: (pending, pendingTy)))
  return builder.emitLayout()

macro coalesce*(layout: Layout): untyped =
  ## Merge contiguous dimensions.
  ##
  ## - size-1 dimensions are dropped
  ## - a dimension merges into the chain when the chain's span
  ##   (shape times stride) equals that dimension's stride.
  ##
  ##   (2, 4):(1, 2)  folds to (8):(1), the (2,1) chain's span 2
  ##                  reaches the second dimension's stride 2
  ##   (4, 1):(1, 0)  folds to (4):(1), the trailing broadcast drops,
  ##                  a chain reaching its stride 0 absorbs it
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  template coalesceDelegate(shape, stride: typed): untyped =
    coalesceImpl(shape, stride, dropTrailing = true)
  result.add getAst(coalesceDelegate(sh, st))

# ═══════════════════════════════════════════════════════════════
#  complement
# ═══════════════════════════════════════════════════════════════

proc slotAt*(slots: var seq[tuple[path: seq[int], cur: int, curNode, leaf: NimNode]], basis: seq[int], leaf: NimNode): int =
  ## Slot position: the record's position the basis path implies.
  ## Returns:
  ##     the record's index, the record created on first address
  ##     with its span seeded at 1 and its terminal leaf at `leaf`
  for k in 0 ..< slots.len:
    if slots[k].path == basis:
      return k
  slots.add (basis, 1, IntCT(1), leaf)
  slots.len - 1

proc complementGenericFold(sh, st: NimNode, bound: NimNode, dropTrailing: bool): NimNode {.compileTime.} =
  ## Fills the gaps a layout leaves uncovered in its address range.
  ##
  ## Args:
  ## - `sh`, `st`: the layout's shape and stride expressions
  ## - `bound`: the cotarget, one value per axis, `IntCT(1)`
  ##   for the minimal complement
  ## Returns:
  ##     a layout covering every address the input misses,
  ##     gaps append flat, coalesce decides per axis
  var pairs: seq[tuple[stride: NimNode, path: seq[int], scale,
                       shapeVal: int]] = @[]
  var lastStride: NimNode
  var lastStrideTy: NimNode
  var z = st.tupleStream().zipLongest(sh.tupleStream())
  while not z.done():
    let (stEv, shEv) = z.next()
    if shEv.kind != kLeaf:
      continue
    # a lone stride leaf repeats for every shape leaf,
    # a nil leaf on the padded side keeps the last one
    if stEv.kind == kLeaf and stEv.leaf != nil:
      lastStride = stEv.leaf
      lastStrideTy = stEv.leafTy
    # proj reads each leaf once, a multi-term atom raises in proj
    let (path, scale) = proj(lastStrideTy)
    if scale == DynamicSentinel:
      error "complement: dynamic strides are not supported"
    let shapeVal = shEv.leafTy.getStaticInt()
    if shapeVal == DynamicSentinel:
      error "complement: dynamic shapes are not supported"
    pairs.add (lastStride, path, scale, shapeVal)

  pairs.sort(proc(a, b: tuple[stride: NimNode, path: seq[int], scale,
                              shapeVal: int]): int =
    let c = cmp(a.path, b.path)
    if c != 0: c else: cmp(a.scale, b.scale))

  # Cotarget leaves: one per codomain axis, the extras multiply
  # into the last axis's terminal
  var boundLeafs: seq[NimNode] = @[]
  for bEv in bound.tupleStream():
    if bEv.kind == kLeaf:
      boundLeafs.add bEv.leaf

  var builder = TupleBuilderFlat.new(2)
  # Slot records: one per distinct basis path, slotAt routes by the atom's path
  var slots: seq[tuple[path: seq[int], cur: int, curNode, leaf: NimNode]]

  for p in pairs:
    # a fresh slot's index is the leaf it consumes
    let si = slots.slotAt(p.path, (
      if slots.len < boundLeafs.len:
        boundLeafs[slots.len]
      else:
        IntCT(1)))
    # a stride-0 leaf maps every coordinate to offset 0,
    # a size-1 dimension covers one offset, neither opens a gap
    if p.scale == 0 or p.shapeVal == 1:
      continue
    if p.scale < slots[si].cur:
      error "complement: non-injective layout, stride " & $p.scale &
        " overlaps the covered span " & $slots[si].cur & " on slot " & $p.path
    # every gap appends and size-1 gaps die in coalesce.
    # A gap shape is a number, a span stride is the leaf's node scaled
    builder.append((IntCT(p.scale div slots[si].cur), nil), (slots[si].curNode, nil))
    slots[si].cur = p.scale * p.shapeVal
    slots[si].curNode = p.stride * IntCT(p.shapeVal)

  # leftover leaves multiply into the last slot's terminal leaf
  var remaining = IntCT(1)
  for k in slots.len ..< boundLeafs.len:
    remaining = remaining * boundLeafs[k]
  slots[^1].leaf = slots[^1].leaf * remaining
  # Terminals: one per slot, each bound leaf ceils against the covered span
  for s in slots:
    builder.append((s.leaf.ceil_div(IntCT(s.cur)), nil), (s.curNode, nil))

  return bindSym"coalesceImpl".newCall(builder.emit(0), builder.emit(1), newLit(dropTrailing))

macro complement*(layout: Layout): untyped =
  ## Layout complement, the codomain gap filler.
  ##
  ## Say `layout` is the element order of a tensor covering
  ## part of a contiguous range, say the even offsets.
  ##
  ## The complement is the layout over the free offsets:
  ## not set subtraction, but the stride-gap fill that lets
  ## `make_layout(layout, result)` cover the range.
  ##
  ## Contract:
  ## - strictly ordered, `result(i-1) < result(i)`, the result is unique
  ## - when the stride chain divides, the pair
  ##   `make_layout(layout, complement)` is a bijection
  ##   onto a contiguous range, otherwise it under-fills:
  ##   the largest ordered, disjoint layout that fits
  ##
  ## Skip semantics:
  ## - a stride-0 dimension maps every coordinate to offset 0,
  ##   a size-1 dimension covers a single offset, neither opens a gap
## - the minimal complement's terminal is 1 per codomain axis.
## A cotarget ceil-chains against the axis's covered span
  ##
  ##    offsets  ════ layout ════▶  its offsets, every second slot
  ##    offsets  ──── complement ─▶  the free slots, gap-filled in order
  ##    offsets  ═══════ make_layout(layout, complement) ═══════▶  the range
  ##
  ## Examples:
  ##
  ##    4:2                     → (2, 1):(1, 8)
  ##    4:2 with 16             → (2, 2):(1, 8)
  ##    (2, 2):(1, 4) with 16   → (2, 2):(2, 8)
  let (sh, st) = result.destructureLayout(layout)
  # minimal complement: every terminal is 1
  result = complementGenericFold(sh, st, IntCT(1), false)

macro complement*(layout: Layout, cosizeBound: typed): untyped =
  ## Complement extended to a cotarget: an `Int`, a runtime `int`,
  ## a shape-tuple bound. The tuple binds one value per codomain axis.
  let bound =
    if cosizeBound.kind == nnkIntLit:
      # an int literal wraps to the static Int leaf
      nnkCall.newTree(nnkBracketExpr.newTree(bindSym"Int", cosizeBound))
    else:
      cosizeBound
  let (sh, st) = result.destructureLayout(layout)
  result = complementGenericFold(sh, st, bound, true)

# ═══════════════════════════════════════════════════════════════
#  compose
# ═══════════════════════════════════════════════════════════════
#
# Pseudo code
#
#   compose(A, B):   # A o B,  (A o B)(c) = A(B(c))
#
#     # B as a tuple of modes composes mode by mode
#     # (left-distributivity over concatenation)
#     if B is a tuple of modes <B0, B1, ...>:
#         return make_layout(compose(A[0], B0), compose(A[1], B1), ...)
#
#     A' = coalesce_z(A, coprofile(B))    # A flattened and coalesced
#
#     if B.shape is a tuple:
#         return make_layout(compose(A', b) for each mode b of B)
#
#     # from here B = N:M, one shape and one stride
#     if M == 0:  return N:0
#     if N == 1:  return 1:A'(M)          # A' evaluated at one point
#     if A' is a single mode a:b:  return N:(M*b)
#
#     # general case, per coefficient Mⱼ of M
#     # (one turn when M is one number):
#     Strided = A' / Mⱼ                   # every Mⱼ-th element of A'
#     Kept    = Strided % N               # its first N elements
#     result  = concat(result, Kept)
#     return coalesce(result)             # per-mode coalesce
#
#     A' / d is the layout of every d-th element of A':
#         q = d
#         for each mode sᵢ:dᵢ of A':
#             dᵢ = q * dᵢ
#             if q % sᵢ == 0:   sᵢ = 1, q = q / sᵢ
#             elif sᵢ % q == 0: sᵢ = sᵢ / q, stop
#             else: the stride divisibility condition fails
#
#     Strided % N is the first N elements of Strided:
#         q = N
#         for each mode sᵢ of Strided:
#             q, r = divmod(q, sᵢ)
#             if q == 0:   sᵢ = r, the modes after sᵢ become 1, stop
#             elif r != 0: the shape divisibility condition fails
#             else: keep sᵢ

# ═══════════════════════════════════════════════════════════════
#  strided divide/modulo mode arithmetic
# ═══════════════════════════════════════════════════════════════
#
#  qᵢ = M div pᵢ, 1 when the running prefix exhausts M

macro stridedFold(sh, st, n, m: typed): untyped =
  result = newStmtList()

  var builder = TupleBuilderFlat.new(2)
  var q = m
  var remaining = quote do: `n` div Int[1]()
  var pendingS, pendingD: NimNode
  for (shapeEv, strideEv) in sh.tupleDimsStream().zip(st.tupleDimsStream()):
    if not pendingS.isNil:
      let modeShape = quote do: min(`pendingS`.ceil_div(`q`), `remaining`)
      let modeStride = quote do: `q` * `pendingD`
      builder.append(modeShape, modeStride)
      q = quote do: `q`.ceil_div(`pendingS`)
      remaining = quote do: `remaining` div `modeShape`
    pendingS = shapeEv.leaf
    pendingD = strideEv.leaf
  let lastShape = quote do: `remaining`
  let lastStride = quote do: `q` * `pendingD`
  builder.append(lastShape, lastStride)
  result.add builder.emitLayout()


proc pathSelect(node: NimNode, path: seq[int]): NimNode {.compileTime.} =
  ## Index `node` at `path`, one bracket read per level. A scalar mode
  ## absorbs the rest of the path, it is its own single mode
  result = node
  for i in path:
    if result.getTypeOf().isTupleTy():
      result = nnkBracketExpr.newTree(result, newLit(i))
    else:
      break

template foldDelegate(aSh, aSt, n, m: typed): untyped =
  stridedFold(aSh, aSt, n, m)

macro composeImpl(aShape, aStride, bShape, bStride: typed): untyped =
  ## Composition with a scalar B = N:M.
  ##
  ## - N == 1: 1:A'(M), A' evaluated at one point
  ## - otherwise: the term loop, one walk per stride term scaled,
  ##   the strides sum, one coalesce
  ##
  ## Divisibility is the caller's responsibility, it is not checked.
  result = newStmtList()
  if bStride.getTypeOf().getStaticInt() == 0:
    # a zero stride maps every coordinate to offset 0
    result.add make_layout(bShape, IntCT(0))
    return
  var nVal = bShape.getTypeOf().getStaticInt()
  if nVal == 1:
    # A' evaluated at one point, evaluation is coalesce-invariant
    result.add make_layout(IntCT(1),
      bindSym"crd2idx".newCall(make_layout(aShape, aStride), bStride))
    return
  let stmts = newStmtList()
  var shapeAst, strideAst: NimNode = nil
  var termIndex = 0
  for termEv in bStride.basisReprStream():
    if termEv.kind != kLeaf:
      continue
    # a zero coefficient contributes nothing, the node read covers
    # int-literal CoordStride coeffs
    if termEv.leaf.getStaticInt() == 0:
      continue
    let pSh = pathSelect(aShape, termEv.path)
    let pSt = pathSelect(aStride, termEv.path)
    # projected parts coalesce once per term
    let ap = ident("aPrime" & $termIndex)
    inc termIndex
    stmts.add quote do:
      evalOnceAs(`ap`, coalesceImpl(`pSh`, `pSt`))
    let (aSh, aSt) = result.destructureLayout(ap)
    let termAst = getAst(foldDelegate(aSh, aSt, bShape, termEv.leaf))
    let alias = genSym(nskLet, "composeTerm")
    stmts.add quote do:
      let `alias` = `termAst`
    let (termShape, termStride) = result.destructureLayout(alias)
    if shapeAst.isNil:
      shapeAst = termShape
      strideAst = termStride
    else:
      strideAst = strideAst + termStride
  if strideAst.isNil:
    # every coefficient is zero, the LHS stays untouched
    stmts.add make_layout(bShape, IntCT(0))
  else:
    stmts.add quote do:
      coalesce(make_layout(`shapeAst`, `strideAst`))
  result.add quote do:
    block:
      `stmts`

macro composeImpl(aShape, aStride: typed, bShape, bStride: tuple): untyped =
  ## Composition on destructured parts, B a tuple of modes:
  ## left-distributivity, each mode composes separately.
  result = newStmtList()
  template composeModeDelegate(aSh, aSt, bSh, bSt: typed): untyped =
    composeImpl(aSh, aSt, bSh, bSt)
  var builder = TupleBuilderFlat.new(2)
  for (shEv, stEv) in bShape.tupleDimsStream().zip(bStride.tupleDimsStream()):
    let modeAst = getAst(composeModeDelegate(`aShape`, `aStride`, shEv.leaf, stEv.leaf))
    let (modeShape, modeStride) = result.destructureLayout(modeAst)
    builder.append(modeShape, modeStride)
  result.add builder.emitLayout()

macro composeImpl[A, B: Layout](a: A, tiler: B): untyped =
  ## Composition with a Layout tiler. Both sides destructure.
  ## The shape kind routes to the term loop or right-distributivity.
  result = newStmtList()
  let (aShape, aStride) = result.destructureLayout(a)
  let (bShape, bStride) = result.destructureLayout(tiler)
  template composeModeDelegate(aSh, aSt, bSh, bSt: typed): untyped =
    composeImpl(aSh, aSt, bSh, bSt)
  result.add getAst(composeModeDelegate(aShape, aStride, bShape, bStride))

macro composeImpl[A: Layout, B: tuple](a: A, tiler: B): untyped =
  ## Composition with a tuple tiler. Distribution is mode-wise:
  ## element i composes against layout mode i, the modes concatenate.
  ##
  ## - a keep mark passes the mode through
  ## - a scalar element becomes (B):(1) in the mode's domain
  ## - a Layout element composes as-is
  ##
  ## Each element dispatches to composeImpl, the modes concatenate.
  ##
  ## An un-inlined tuple producer is evaluated once, its value read
  ## through an alias. Per-mode structure comes from the layout type,
  ## destructured parts stay untyped until a nested typed call
  ## semchecks their members.
  result = newStmtList()
  var tilerEval = tiler
  let tilerTy = tiler.getTypeOf()
  if tiler.kind notin {nnkTupleConstr, nnkPar, nnkSym, nnkBracketExpr, nnkDotExpr}:
    result.add bindSym"evalOnceAs".newCall(ident"tilerEval", tiler)
    tilerEval = ident"tilerEval"
  let (aShape, aStride) = result.destructureLayout(a)
  let (aShapeTy, aStrideTy) = layoutTypeArgs(a)
  template composeModeDelegate(aSh, aSt, bSh, bSt: typed): untyped =
    composeImpl(aSh, aSt, bSh, bSt)
  var builder = TupleBuilderFlat.new(2)
  var ti = 0
  for tEv in tiler.tupleDimsStream(tilerTy):
    if ti >= (if aShapeTy.isTupleTy(): aShapeTy.len else: 1):
      error("tiler has more modes than the layout", tEv.leaf)
    let tLeaf = tEv.leaf
    let tLeafTy = tEv.leafTy
    let aSh = if aShapeTy.isTupleTy(): aShape.getTupleIndex(ti) else: aShape
    let aSt = if aStrideTy.isTupleTy(): aStride.getTupleIndex(ti) else: aStride
    if tLeaf.kind in {nnkSym, nnkIdent} and $tLeaf == "_":
      # pass-through: the element is the mode itself
      builder.append(aSh, aSt)
    else:
      # a scalar element is (B):(1) in the mode's domain
      # a Layout entry composes as-is, both dispatch to composeImpl
      let (bSh, bSt) = if tLeafTy.isLayoutTy():
        result.destructureLayout(tLeaf)
      else:
        if not isScalarTy(tLeafTy) and tLeaf.kind notin {nnkIntLit .. nnkUInt64Lit}:
          error("tiler leaf is neither an int, an Int nor a Layout", tLeaf)
        (tLeaf, IntCT(1))
      let modeAst = getAst(composeModeDelegate(aSh, aSt, bSh, bSt))
      let (modeShape, modeStride) = result.destructureLayout(modeAst)
      builder.append(modeShape, modeStride)
    inc ti
  result.add builder.emitLayout(emitScalarForSize1 = false)

macro compose*[A: Layout, B: Layout or tuple](a: A, tiler: B): untyped =
  ## Layout composition, `A o B`.
  ##
  ## Say `A` is a layout, the element order of a tensor,
  ## and `B` an access pattern, say take every second element.
  ##
  ## Composition applies the pattern on the layout.
  ##
  ## The result is itself a layout, so patterns defined once
  ## work on any tensor, without spelling out the resulting
  ## indexing by hand.
  ##
  ## Returns a layout `R` such that `R(i) = A(B(i))` for all
  ## `i` in `0 ..< cosize(B)`.
  ## The caller is responsible for ensuring divisibility.
  ##
  ##    domain  ──── B ────▶  A's domain  ──── A ────▶  values
  ##    domain  ════════════ R ══════════════▶  values
  ##
  ## With B = (5, 4):(4, 1) and A = 20:2, R = (5, 4):(8, 2):
  ##
  ##    B(i) = 4·(i mod 5) + i div 5
  ##    A(B(i)) = 2·B(i)
  ##    R(i) = 8·(i mod 5) + 2·(i div 5)
  ##
  ## Examples:
  ##
  ##    (20, 2)       ∘ (5, 4):(4, 1) → (5, 4):(8, 2)
  ##
  ##    (6, 2):(8, 2) ∘ (4, 3):(3, 1) → ((2, 2), 3):((24, 2), 8)

  result = newStmtList()
  template composeDelegate(al, tEv: typed): untyped =
    composeImpl(al, tEv)
  result.add getAst(composeDelegate(a, tiler))

# ═══════════════════════════════════════════════════════════════
#  logical_product
# ═══════════════════════════════════════════════════════════════

macro logicalProductFinish(a, rest: typed): untyped =
  result = newStmtList()
  var grid = rest
  # the composed layout arrives as `block: let m0 = ...; <layout>`,
  # keep the let statements, the layout expression reads those variables
  while grid.kind in {nnkStmtListExpr, nnkBlockExpr}:
    let body = if grid.kind == nnkBlockExpr: grid[1] else: grid
    for i in 0 ..< body.len - 1:
      result.add(body[i])
    grid = body[^1]
  let (aShape, aStrides) = result.destructureLayout(a)
  let (rShape, rStrides) = result.destructureLayout(grid)
  result.add make_layout(
    nnkTupleConstr.newTree(aShape, rShape),
    nnkTupleConstr.newTree(aStrides, rStrides)
  )

macro logical_product*[A, B: Layout](a: A, tiler: B): untyped =
  ## Logical product, `a x tiler = (a, a* ∘ tiler)`.
  ##
  ## Say you need a small tile written out at every grid position,
  ## the product is the layout of that write-out.
  ## Reproduce the block `a` over the grid the tiler describes:
  ## - a copy of `a` (dimension 0) lands at every position the tiler's
  ##   offset map selects
  ## - the copy positions are numbered by `a`'s complement composed
  ##   with the tiler (dimension 1)
  ##
  ## Returns a rank-2 layout `R` such that `R(t, i) = a(t) + a*(tiler(i))`:
  ## - dimension 0 is the block itself, `R[0] == a`
  ## - dimension 1 numbers the copies, `R[1] = complement(a, size(a) * cosize(tiler)) ∘ tiler`
  ## - `size(R) == size(a) * size(tiler)`
  ##
  ## Inverse of logical_divide: the divide factors `a` into tiles
  ## `a ∘ (tiler, complement(tiler, size(a)))`, the product reassembles
  ## from the block and the copy grid.
  ##
  ##    domain ──── a ════════▶  block positions    (dim 0)
  ##    domain ──── a* ∘ tiler ─▶  copy numbers     (dim 1)
  ##    domain ════════════ R ════════════▶  (block, copy)
  ##
  ## Examples:
  ##
  ##    logical_product(make_layout((2, 2), (4, 1)), make_layout(6, 1))
  ##    → ((2, 2), (2, 3)):((4, 1), (2, 8))
  ##
  ##    logical_product(make_layout((2, 2), (1, 2)), make_layout((3, 4), (4, 1)))
  ##    → ((2, 2), (3, 4)):((1, 2), (16, 4))
  ##
  ## Copy-grid semantics:
  ## - the complement extends to the bound `size(a) * cosize(tiler)`, the span one contiguous copy grid covers
  ## - a divisible block stride chain gives distinct, non-overlapping copy slots, one per tiler position
  ## - an under-filling block keeps the largest ordered, disjoint copy grid that fits

  template logicalProductDelegate(a_layout, tiler_layout) =
    # Please read "Implementation note for direct AST->AST transformation of constructors"
    # in layout_constructors on why no evalOnceAs
    logicalProductFinish(a_layout, compose(complement(a_layout, size(a_layout) * cosize(tiler_layout)), tiler_layout))
  result = getAst(logicalProductDelegate(a, tiler))

# ═══════════════════════════════════════════════════════════════
#  zipped_product, tiled_product, flat_product
# ═══════════════════════════════════════════════════════════════

template zipped_product*(blk: Layout, tiler: auto): auto =
  ## Reproduce the block `blk` over the grid the tiler describes:
  ## the block dimensions and the copy dimensions zipped into one rank-2
  ## layout with both sides gathered.
  ##
  ## Say you reproduce a per-thread tile over a thread grid and one
  ## index must select the thread's tile: dimension 0 reads inside
  ## the tile and dimension 1 selects the tile.
  ##
  ## `zipped_product` applies `logical_product` and gathers the split
  ## dimensions into two dimensions:
  ## - dimension 0 = the block dimensions, one leaf per tiler element
  ## - dimension 1 = the copy dimensions, each numbered by `blk* ∘ tiler`
  ##   over its tiler element, `blk* = complement(blk, size(blk) * cosize(tiler))`
  ##
  ## Returns a layout `R` = `((M, N, ...), (TileM, TileN, ...))`,
  ## `size(R) == size(blk) * size(tiler)`.
  ##
  ## `zipped_divide` runs the mirrored gather:
  ## - `zipped_divide(a, b)[0] = compose(a, b)`, the tile sides gathered
  ## - `zipped_product(a, b)[1] = compose(complement(a, size(a) * cosize(b)), b)`,
  ##   the copy sides gathered
  ##
  ## Example:
  ##   zipped_product(make_layout(4, 1), make_layout(3, 1))
  ##   # → (4, 3):(1, 4)
  ##
  ##   zipped_product(make_layout((2, 4), (1, 2)), make_layout(((3, 1), (1, 3)), ((1, 3), (3, 3))))
  ##   # → ((2, 4), ((3, 1), (1, 3))):((1, 2), ((8, 24), (24, 24)))
  hier_unzip(logical_product, blk, tiler)

macro tiled_product*(blk: Layout, tiler: typed): untyped =
  ## Reproduce a block layout across the positions a tiler describes,
  ## the block dimensions grouped as dimension 0, the reproduction
  ## dimensions flat after them.
  ##
  ## Say a block's threads fill a shared-memory buffer and each
  ## whole tile must sit flat in the layout, block first.
  ##
  ## Returns a layout `R` = `((BlkM, BlkN), RepM, RepN, ...)` where
  ## every tiler position carries one copy of `blk`:
  ## - dimension 0 = the block, the block dimensions grouped together
  ## - dimensions 1+ = the reproduction dimensions, flat
  ##
  ## Contrast `zipped_product`, which keeps the reproduction dimensions
  ## grouped as a second dimension.
  ##
  ## Example:
  ##   tiled_product(make_layout((2, 4), (1, 2)), make_layout(3, 1))
  ##   # → ((2, 4), 3):((1, 2), 8)
  template productSplitter(l, b: auto): auto =
    # getAst indirection to delegate overload resolution to the Nim compiler
    logical_product(l, b)
  let hierUnzipAst = getAst(hier_unzip(productSplitter, blk, tiler))
  if hierUnzipAst.kind == nnkBlockExpr:
    result = newStmtList()
    # the composed layout arrives as `block: let m0 = ...; <layout>`
    let huzBody = hierUnzipAst[1]
    result.add huzBody[0 ..< huzBody.len - 1]
    let makeLayoutCall = huzBody[^1]
    let (tileShape, restShape, tileStride, restStride) = (makeLayoutCall[1][0], makeLayoutCall[1][1], makeLayoutCall[2][0], makeLayoutCall[2][1])
    result.add bindSym"zippedToTiledPartsImpl".newCall(tileShape, restShape, tileStride, restStride)
  else:
    result = newStmtList()
    let (zippedShape, zippedStride) = result.destructureLayout(hierUnzipAst)
    result.add bindSym"zippedToTiledPartsImpl".newCall(
      getTupleIndex(zippedShape, 0), getTupleIndex(zippedShape, 1),
      getTupleIndex(zippedStride, 0), getTupleIndex(zippedStride, 1))

macro flat_product*(blk: Layout, tiler: typed): untyped =
  ## Reproduce a block layout across the positions a tiler describes,
  ## every dimension at one level, no grouping anywhere.
  ##
  ## Say the copies' positions must index flat with no grouping,
  ## one dimension per level.
  ##
  ## Returns a layout `R` = `(BlkM, BlkN, RepM, RepN, ...)` where
  ## every tiler position carries one copy of `blk`.
  ##
  ## Contrast `tiled_product`, which groups the block dimensions
  ## as dimension 0:
  ## - `zipped_product` groups the reproduction dimensions too
  ##
  ## Example:
  ##   flat_product(make_layout((2, 4), (1, 2)), make_layout(3, 1))
  ##   # → (2, 4, 3):(1, 2, 8)
  template productSplitter(l, b: auto): auto =
    # getAst indirection to delegate overload resolution to the Nim compiler
    logical_product(l, b)
  let hierUnzipAst = getAst(hier_unzip(productSplitter, blk, tiler))
  if hierUnzipAst.kind == nnkBlockExpr:
    result = newStmtList()
    # the composed layout arrives as `block: let m0 = ...; <layout>`
    let huzBody = hierUnzipAst[1]
    result.add huzBody[0 ..< huzBody.len - 1]
    let makeLayoutCall = huzBody[^1]
    let (tileShape, restShape, tileStride, restStride) = (makeLayoutCall[1][0], makeLayoutCall[1][1], makeLayoutCall[2][0], makeLayoutCall[2][1])
    result.add bindSym"zippedToFlatPartsImpl".newCall(tileShape, restShape, tileStride, restStride)
  else:
    result = newStmtList()
    let (zippedShape, zippedStride) = result.destructureLayout(hierUnzipAst)
    result.add bindSym"zippedToFlatPartsImpl".newCall(
      getTupleIndex(zippedShape, 0), getTupleIndex(zippedShape, 1),
      getTupleIndex(zippedStride, 0), getTupleIndex(zippedStride, 1))

# ═══════════════════════════════════════════════════════════════
#  blocked_product / raked_product
# ═══════════════════════════════════════════════════════════════

macro productZipImpl(prodCtor: typed, raked: static bool): untyped =
  result = newStmtList()
  # the composed layout arrives as `block: let m0 = ...; <layout>`
  var prodAst = prodCtor
  while prodAst.kind in {nnkStmtListExpr, nnkBlockExpr}:
    let body = if prodAst.kind == nnkBlockExpr: prodAst[1] else: prodAst
    for i in 0 ..< body.len - 1:
      result.add(body[i])
    prodAst = body[^1]
  let (prodShape, prodStride) = result.destructureLayout(prodAst)
  let firstShape = getTupleIndex(prodShape, if raked: 1 else: 0)
  let secondShape = getTupleIndex(prodShape, if raked: 0 else: 1)
  let firstStride = getTupleIndex(prodStride, if raked: 1 else: 0)
  let secondStride = getTupleIndex(prodStride, if raked: 0 else: 1)
  var firstShapeStream = firstShape.tupleDimsStream()
  var secondShapeStream = secondShape.tupleDimsStream()
  var firstStrideStream = firstStride.tupleDimsStream()
  var secondStrideStream = secondStride.tupleDimsStream()
  var builder = TupleBuilderNested.new(2)
  builder.append(TupleStreamEvent(path: @[], kind: kOpen),
                 TupleStreamEvent(path: @[], kind: kOpen))
  while not firstShapeStream.done():
    let firstShapeEvent = firstShapeStream.next()
    let secondShapeEvent = secondShapeStream.next()
    let firstStrideEvent = firstStrideStream.next()
    let secondStrideEvent = secondStrideStream.next()
    builder.append(
      (nnkPar.newTree(firstShapeEvent.leaf, secondShapeEvent.leaf),
       nnkPar.newTree(firstShapeEvent.leafTy, secondShapeEvent.leafTy)),
      (nnkPar.newTree(firstStrideEvent.leaf, secondStrideEvent.leaf),
       nnkPar.newTree(firstStrideEvent.leafTy, secondStrideEvent.leafTy)))
  builder.append(TupleStreamEvent(path: @[], kind: kClose),
                 TupleStreamEvent(path: @[], kind: kClose))
  result.add builder.emitLayout()

macro productZipDelegate[A, B: Layout](blk: A, tiler: B, raked: static bool): untyped =
  let rakedLit = newLit(raked)
  result = quote do:
    block:
      const rankMax = max(`blk`.rank(), `tiler`.rank())
      productZipImpl(
        logical_product(padRight(`blk`, rankMax), padRight(`tiler`, rankMax)),
        `rakedLit`)

macro blocked_product*[A, B: Layout](blk: A, tiler: B): untyped =
  ## Repeat block over tiler grid, each block contiguous.
  ## Returns:
  ## - ((BLK_A, TILER_A), (BLK_B, TILER_B), ...), each block contiguous
  ##
  ## Say each tile copy must stay contiguous, one whole tile before
  ## the grid steps to the next.
  quote do:
    productZipDelegate(`blk`, `tiler`, false)

macro raked_product*[A, B: Layout](blk: A, tiler: B): untyped =
  ## Repeat block over tiler grid, blocks interleaved.
  ## Returns:
  ## - ((TILER_A, BLK_A), (TILER_B, BLK_B), ...), blocks interleaved
  ##
  ## Say the tile's elements must spread across grid slots instead,
  ## a cyclic fill over the grid.
  quote do:
    productZipDelegate(`blk`, `tiler`, true)

# ═══════════════════════════════════════════════════════════════
#  tile_to_shape
# ═══════════════════════════════════════════════════════════════

template tile_to_shape*(blk: Layout, target_shape: typed, ord_shape: static StrideOrder = LayoutLeft): auto =
  ## Repeat a block layout to fill a target shape.
  ##
  ## Say you have an atom layout and must fill a whole buffer tile
  ## with repeats of it, repeat order per dimension from `ord_shape`.
  ##
  ## Returns:
  ##   the block tiled over the target shape, ceil_div repeats per
  ##   target dimension, repeat order per dimension from ord_shape
  ##
  ## Example:
  ##   let tile = tile_to_shape(make_layout((2,3), (1,2)), (6, 12))
  ##   # block (2,3) repeated to fill (6,12) in 3 columns:
  ##   # ((2,3),3):((1,2),6)
  const R = static(target_shape.rank())
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(ts, target_shape)
    let padded_blk = padRight(bk, R)
    let blk_shape = product_each(padded_blk.shape)
    let trg_flat = product_each(ts)
    let product_shape = zipDimensionsWith(trg_flat, blk_shape): ceil_div(it_a, it_b)
    let tiler = make_layout(product_shape, ord_shape)
    blocked_product(padded_blk, tiler)

# ═══════════════════════════════════════════════════════════════
#  logical_divide
# ═══════════════════════════════════════════════════════════════

template divideRank1(l, t) =
  when l.shape isnot tuple:
    make_layout((t, ceil_div(l.shape, t)), (l.stride, l.stride * t))
  else:
    logical_divide(l, make_layout(t))

macro logical_divide*[A, B: Layout](layout: A, tiler: B): untyped =
  ## Logical divide: split a layout into a (tile, rest) pair.
  ##
  ## Say you walk the elements of `layout` in chunks shaped like
  ## `tiler`, say 4 consecutive elements. The divide answers
  ## two questions:
  ## - the tile dimension steps through positions inside a chunk
  ## - the rest dimension steps through the chunks
  ##
  ## Reading element `(t, c)` of the result reads element
  ## `tiler(t) + tiles(c)` of `layout`: `tiles` numbers
  ## the chunks, the complement of the tiler.
  ##
  ## One-line definition:
  ##
  ##    logical_divide(A, B) = compose(A, make_layout(B, complement(B, size(A))))
  ##
  ## Contract:
  ## - a Layout tiler returns a 2-dimension layout, dimension 0
  ##   is `compose(layout, tiler)`, dimension 1 numbers the tiles
  ## - every element of `layout` appears exactly once, the divide reorders elements without dropping any
  ## - the tiler must divide the layout, a caller precondition,
  ##   checked only on compile-time values
  ##
  ##    elements ══ tiler ══▶  chunk positions       (tile)
  ##    elements ── complement ─▶  chunk numbers     (rest)
  ##    elements ════════════ the divide result ══▶  (position, chunk)
  ##
  ## Examples:
  ##
  ##    logical_divide(make_layout(16, 3), 4)
  ##    # → (4, 4):(3, 12)
  ##
  ##    logical_divide(make_layout((4, 2, 3), (2, 1, 8)), make_layout(4, 2))
  ##    # → ((2, 2), (2, 3)):((4, 1), (2, 8))
  result = newStmtList()
  let (shA, stA) = result.destructureLayout(layout)
  let (shB, stB) = result.destructureLayout(tiler)
  # one-line definition, complement expands once via destructureLayout
  let cmp = genSym(nskLet, "divCmp")
  result.add quote do:
    let `cmp` = complement(make_layout(`shB`, `stB`), `shA`)
  let (cmpSh, cmpSt) = result.destructureLayout(cmp)
  result.add quote do:
    compose(make_layout(`shA`, `stA`),
      make_layout((`shB`, `cmpSh`), (`stB`, `cmpSt`)))

macro logical_divide*[L: Layout](layout: L, tiler: int): untyped =
  ## Logical divide by an int tiler
  getAst(divideRank1(layout, tiler))

macro logical_divide*[L: Layout, V: static int](layout: L, tiler: Int[V]): untyped =
  ## Logical divide by a static int tiler, see the int overload
  getAst(divideRank1(layout, tiler))

macro divideTupleImpl(sh, st, tiler: typed): untyped =
  ## Per-dimension divide over the destructured layout.
  ##
  ## - one tiler element per layout dimension
  ## - dimensions beyond the tiler length pass through
  ## - a divided dimension carries the (tile, rest) pair.
  template divideTupleDimShape(dsh, dst, dtl) =
    logical_divide(make_layout(dsh, dst), dtl).shape
  template divideTupleDimStride(dsh, dst, dtl) =
    logical_divide(make_layout(dsh, dst), dtl).stride
  let shTy = sh.getTypeOf()
  let layoutRank = if shTy.isTupleTy(): shTy.len else: 1
  let tilerTy = tiler.getTypeOf()
  let tilerRank = if tilerTy.isTupleTy(): tilerTy.len else: 1
  if tilerRank > layoutRank:
    error "logical_divide: tiler has more dimensions (" & $tilerRank &
      ") than the layout (" & $layoutRank & ")"
  var builder = TupleBuilderFlat.new(2)
  var tilerDims = tiler.tupleDimsStream()
  for (shEv, stEv) in sh.tupleDimsStream().zip(st.tupleDimsStream()):
    builder.onLeaves(shEv):
      if not tilerDims.done():
        # the per-dim divide is emitted twice, once per projection.
        # Both copies stay pure expressions the C compiler folds and deduplicates,
        # a binding would instead materialize a wall of temporaries + temp types
        # that might be harder to optimize away on certain backends (Vulkan / WebGPU that don't use LLVM for example)
        let tilerEv = tilerDims.next()
        let divSh = getAst(divideTupleDimShape(shEv.leaf, stEv.leaf, tilerEv.leaf))
        let divSt = getAst(divideTupleDimStride(shEv.leaf, stEv.leaf, tilerEv.leaf))
        let divShTy = quote do: typeof(`divSh`)
        let divStTy = quote do: typeof(`divSt`)
        builder.append((divSh, divShTy), (divSt, divStTy))
      else:
        # a pass-through dimension, the dimension arrives whole
        builder.append((shEv.leaf, shEv.leafTy), (stEv.leaf, stEv.leafTy))
  return builder.emitLayout()

macro logical_divide*(layout: Layout, tiler: tuple): untyped =
  ## Logical divide by a tuple tiler, one tiler element per layout dimension:
  ## - tiler elements matched positionally to layout dimensions,
  ##   dimensions beyond the tiler length pass through undivided
  ## - a divided dimension becomes the (tile, rest) pair, each
  ##   pair carries the Layout-tiler contract
  ## - an empty tiler divides nothing, every dimension passes through

  proc unwrapSLE(e: NimNode): NimNode =
    result = e
    while result.kind == nnkStmtListExpr:
      result = result[^1]
  if unwrapSLE(tiler).getTypeOf().len == 0:
    # an empty tiler divides nothing, the layout passes through verbatim
    result = newStmtList()
    result.add layout
  else:
    result = newStmtList()
    let (sh, st) = result.destructureLayout(layout)
    template divideTupleDelegate(sh2, st2, tiler2) =
      divideTupleImpl(sh2, st2, tiler2)
    result.add getAst(divideTupleDelegate(sh, st, tiler))

template zipped_divide*(layout: Layout, tiler: auto): auto =
  ## Divides layout by tiler, zips tile/rest dimensions into rank-2 result.
  ##
  ## Say a kernel level needs the tile and the rest at once, one
  ## index into dimension 1 selects which tile, dimension 0 walks
  ## positions inside it.
  ##
  ## Returns:
  ## - dimension 0 carries the tile, dimension 1 the rest, zipped
  hier_unzip(logical_divide, layout, when tiler is Layout: tiler else: makeIntTuple(tiler))

# ═══════════════════════════════════════════════════════════════
#  tiled_divide / flat_divide
# ═══════════════════════════════════════════════════════════════

macro tiled_divide*(layout: Layout, tiler: typed): untyped =
  ## Split a layout into a tile and number the tiles.
  ##
  ## Say a block of a kernel owns one tile, dimension 0 reads
  ## positions inside it, dimensions 1+ count the block's tile.
  ##
  ## Returns a layout `R` such that reading element `(t, r)` of `R`
  ## reads element `tiler(t) + tiles(r)` of `layout`:
  ## - dimension 0 = the tile, the tile dimensions grouped together
  ## - dimensions 1+ = the rest dimensions and undivided dimensions, flat
  ##
  ## `R = ((TileM, TileN), RestM, RestN, ...)`.
  ##
  ## Contrast `zipped_divide`, which keeps the rest dimensions grouped
  ## as a second dimension, and `flat_divide`, which unpacks the tile too.
  ##
  ## Divisibility of the layout by the tiler is a caller precondition.
  ##
  ## Example:
  ##   tiled_divide(make_layout((4, 8), (1, 4)), (2, 4))
  ##   # → ((2, 4), 2, 2):((1, 4), 2, 16)
  ##
  ##    layout (4, 8):(1, 4)
  ##    tile (2, 4) ══▶ dimension 0, positions inside a chunk
  ##    rest 2, 2  ──▶ dimensions 1, 2, the chunk numbers
  template divideSplitter(l, t: auto): auto =
    # getAst indirection to delegate overload resolution to the Nim compiler
    logical_divide(l, t)
  let hierUnzipAst = getAst(hier_unzip(divideSplitter, layout, tiler))
  if hierUnzipAst.kind == nnkBlockExpr:
    result = newStmtList()
    # the composed layout arrives as `block: let m0 = ...; <layout>`
    let huzBody = hierUnzipAst[1]
    result.add huzBody[0 ..< huzBody.len - 1]
    let makeLayoutCall = huzBody[^1]
    let (tileShape, restShape, tileStride, restStride) = (makeLayoutCall[1][0], makeLayoutCall[1][1], makeLayoutCall[2][0], makeLayoutCall[2][1])
    result.add bindSym"zippedToTiledPartsImpl".newCall(tileShape, restShape, tileStride, restStride)
  else:
    result = bindSym"zippedToTiledImpl".newCall(hierUnzipAst)

macro flat_divide*(layout: Layout, tiler: typed): untyped =
  ## Split a layout into a tile and number the tiles, every dimension
  ## at one level, no grouping anywhere.
  ##
  ## Say a kernel indexes its tile flat, tile positions first,
  ## tile counts after, one dimension per level.
  ##
  ## Returns a layout `R` such that reading element `(t, r)` of `R`
  ## reads element `tiler(t) + tiles(r)` of `layout`, every dimension
  ## at one level, no grouping anywhere:
  ## - the tile dimensions first
  ## - then the rest dimensions and undivided dimensions
  ##
  ## `R = (TileM, TileN, RestM, RestN, ...)`.
  ##
  ## Contrast `tiled_divide`, which groups the tile dimensions
  ## as dimension 0:
  ## - `zipped_divide` groups the rest dimensions too
  ##
  ## Divisibility of the layout by the tiler is a caller precondition.
  ##
  ## Example:
  ##   flat_divide(make_layout((4, 8), (1, 4)), (2, 4))
  ##   # → (2, 4, 2, 2):(1, 4, 2, 16)
  template divideSplitter(l, t: auto): auto =
    # getAst indirection to delegate overload resolution to the Nim compiler
    logical_divide(l, t)
  let hierUnzipAst = getAst(hier_unzip(divideSplitter, layout, tiler))
  if hierUnzipAst.kind == nnkBlockExpr:
    result = newStmtList()
    # the composed layout arrives as `block: let m0 = ...; <layout>`
    let huzBody = hierUnzipAst[1]
    result.add huzBody[0 ..< huzBody.len - 1]
    let makeLayoutCall = huzBody[^1]
    let (tileShape, restShape, tileStride, restStride) = (makeLayoutCall[1][0], makeLayoutCall[1][1], makeLayoutCall[2][0], makeLayoutCall[2][1])
    result.add bindSym"zippedToFlatPartsImpl".newCall(tileShape, restShape, tileStride, restStride)
  else:
    result = bindSym"zippedToFlatImpl".newCall(hierUnzipAst)

# ═══════════════════════════════════════════════════════════════
#  right_inverse
# ═══════════════════════════════════════════════════════════════

macro rightInverseImpl(sh, st: typed): untyped =
  ## right_inverse core, emits coalesced contiguous chains in stride order:
  ## - a dimension joins when its stride equals the chain span
  ## - a dynamic shape ends the chain and keeps its own value node
  ## - the chain stride is the product of the shapes before the dimension,
  ##   a runtime prefix expression once a dynamic leaf precedes it
  ## - an empty chain collapses to the empty layout (1, 0)
  var dims: seq[tuple[stride, shape, prefix: int, prefixNode, leaf, leafTy: NimNode]]
  var prefix = 1
  # prefixNode carries the prefix as an expression once it turns runtime,
  # so a stride after a dynamic leaf can still be emitted
  var prefixNode: NimNode = nil
  for (shEv, stEv) in sh.tupleStream().zip(st.tupleStream()):
    if shEv.kind != kLeaf:
      continue
    let shape = shEv.leafTy.getStaticInt()
    dims.add (stEv.leafTy.getStaticInt(), shape, prefix, prefixNode, shEv.leaf, shEv.leafTy)
    let factor = if shape == DynamicSentinel: shEv.leaf
                 else: IntCT(shape)
    if shape == DynamicSentinel:
      # past a dynamic shape the prefix is unknown, no sentinel arithmetic
      prefix = DynamicSentinel
      prefixNode = if prefixNode.isNil: factor
                   else: prefixNode * factor
    else:
      if prefixNode != nil:
        # the prefix stays runtime after a dynamic leaf, fold the factor into the prefix
        prefixNode = prefixNode * factor
      if prefix != DynamicSentinel:
        prefix = prefix * shape

  var builder = TupleBuilderFlat.new(2)
  var curr = 1
  for idx in getIndicesSortedByStride(dims.mapIt(it.stride)):
    let dim = dims[idx]
    if dim.stride == curr:
      let shLeaf = if dim.shape == DynamicSentinel: dim.leaf
                   else: IntCT(dim.shape)
      let shLeafTy = if dim.shape == DynamicSentinel: dim.leafTy
                     else: intTy(dim.shape)
      let stLeaf = if dim.prefix == DynamicSentinel: dim.prefixNode
                   else: IntCT(dim.prefix)
      let stLeafTy = if dim.prefix == DynamicSentinel: nnkCall.newTree(ident"typeof", dim.prefixNode)
                     else: intTy(dim.prefix)
      builder.append((shLeaf, shLeafTy), (stLeaf, stLeafTy))
      if dim.shape == DynamicSentinel:
        break
      curr = dim.stride * dim.shape
  return bindSym"coalesce".newCall(builder.emitLayout())

macro right_inverse*(layout: typed): untyped =
  ## Quasi-inverse, the largest injective R with L(R(i)) == i.
  ##
  ## Say you know a target offset and need the coordinate reaching it,
  ## the inverse maps offsets back to coordinates.
  ## Returns:
  ## - a coalesced Layout, typically lower rank than L
  ## - (1, 0) when no chain exists
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  template rightInverseDelegate(sh2, st2) =
    ## Expands the inverse core on the destructured tuples at the use site.
    rightInverseImpl(sh2, st2)
  result.add getAst(rightInverseDelegate(sh, st))

# ═══════════════════════════════════════════════════════════════
#  left_inverse
# ═══════════════════════════════════════════════════════════════

macro leftInverseImpl(sh, st: typed): untyped =
  ## Left-inverse dimensions built from stride ratios:
  ##
  ##   result_shape[i]  = stride / size_so_far
  ##   result_stride[i] = shape prefix of the previous stride-sorted dimension
  ##
  ## All strides must be static (compile-time assert).
  var dims: seq[tuple[stride, shape, prefix: int, prefixNode, leaf, leafTy: NimNode]]
  var prefix = 1
  # prefixNode carries the prefix as an expression once it turns runtime,
  # so a stride after a dynamic leaf can still be emitted
  var prefixNode: NimNode = nil
  for (shEv, stEv) in sh.tupleStream().zip(st.tupleStream()):
    if shEv.kind != kLeaf:
      continue
    let shape = shEv.leafTy.getStaticInt()
    dims.add (stEv.leafTy.getStaticInt(), shape, prefix, prefixNode, shEv.leaf, shEv.leafTy)
    let factor = if shape == DynamicSentinel: shEv.leaf
                 else: IntCT(shape)
    if shape == DynamicSentinel:
      # past a dynamic shape the prefix is unknown, no sentinel arithmetic
      prefix = DynamicSentinel
      prefixNode = if prefixNode.isNil: factor
                   else: prefixNode * factor
    else:
      if prefixNode != nil:
        # the prefix stays runtime after a dynamic leaf, fold the factor into the prefix
        prefixNode = prefixNode * factor
      if prefix != DynamicSentinel:
        prefix = prefix * shape

  var builder = TupleBuilderFlat.new(2)
  var sizeSoFar = 1
  var prevIdx = -1
  var prevPrefix = 0
  var prevPrefixNode: NimNode = nil
  for idx in getIndicesSortedByStride(dims.mapIt(it.stride)):
    let dim = dims[idx]
    if dim.stride == 0:
      continue
    doAssert dim.stride != DynamicSentinel,
      "left_inverse: dynamic strides are not chainable"
    doAssert dim.stride mod sizeSoFar == 0,
      "left_inverse: stride " & $dim.stride & " not divisible by " & $sizeSoFar
    let gap = dim.stride div sizeSoFar
    # a unit gap marks no hole, the final coalesce would drop the shape-1 head
    if gap != 1:
      let prevStLeaf = if prevPrefix == DynamicSentinel: prevPrefixNode
                       else: IntCT(prevPrefix)
      let prevStLeafTy = if prevPrefix == DynamicSentinel: nnkCall.newTree(ident"typeof", prevPrefixNode)
                         else: intTy(prevPrefix)
      builder.append((IntCT(gap), intTy(gap)), (prevStLeaf, prevStLeafTy))
    sizeSoFar = dim.stride
    prevIdx = idx
    prevPrefix = dim.prefix
    prevPrefixNode = dim.prefixNode
  if prevIdx >= 0:
    # tail = the last stride-sorted dimension's own shape
    let dim = dims[prevIdx]
    let shLeaf = if dim.shape == DynamicSentinel: dim.leaf
                 else: IntCT(dim.shape)
    let shLeafTy = if dim.shape == DynamicSentinel: dim.leafTy
                   else: intTy(dim.shape)
    let stLeaf = if dim.prefix == DynamicSentinel: dim.prefixNode
                 else: IntCT(dim.prefix)
    let stLeafTy = if dim.prefix == DynamicSentinel: nnkCall.newTree(ident"typeof", dim.prefixNode)
                   else: intTy(dim.prefix)
    builder.append((shLeaf, shLeafTy), (stLeaf, stLeafTy))
  return bindSym"coalesce".newCall(builder.emitLayout())

macro left_inverse*(layout: typed): untyped =
  ## Left inverse, Li(L(i)) == i for injective layouts.
  ##
  ## Say a layout maps coordinates one-to-one and you need
  ## to undo the mapping, the left inverse computes it.
  ##
  ## Returns:
  ## - a coalesced Layout over the static-stride gaps
  ## - requires all static strides, compile-time assert
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  template leftInverseDelegate(sh2, st2) =
    ## Coalesce canonicalizes strides first, the chaining asserts require it.
    evalOnceAs(coalescedLayout, coalesce(make_layout(sh2, st2)))
    leftInverseImpl(coalescedLayout.shape, coalescedLayout.stride)
  result.add getAst(leftInverseDelegate(sh, st))

