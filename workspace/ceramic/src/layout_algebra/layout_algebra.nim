# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout algebra: coalesce, complement, compose, divide, inverses, products.

import std/macros
import std/sequtils
import std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_unsanctioned_helpers
import ./layouts_unsanctioned_helpers
import ./layouts
import ./layout_compiletime
import ./layouts_unsanctioned_helpers
import ./layout_indexing_gpu

# ═══════════════════════════════════════════════════════════════
#  getIndicesSortedByStride, sort permutation by stride
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
#  coalesce, merge contiguous dimensions where stride matches
# ═══════════════════════════════════════════════════════════════

macro coalesceImpl(originalLayout, sh, st: typed, preserveTrailing: static bool = false): untyped =
  var builder = TupleBuilderFlat.new(2)
  var chunkShape, chunkStride: NimNode
  var chunkShapeVal, chunkStrideVal: int
  var chunkVerbatim = true
  # the trailing-leaf state feeds only the preserveTrailing marker emission
  var lastShapeVal, lastStrideVal: int
  var lastStride: NimNode
  for (shapeEv, strideEv) in sh.tupleStream().zip(st.tupleStream()):
    if shapeEv.kind != kLeaf:
      continue
    if shapeEv.depth > 1:
      # a leaf below the first tuple level, the fold flattens nested profiles
      builder.markNonVerbatim()
    let shapeVal = shapeEv.leafTy.getStaticInt()
    let strideVal = strideEv.leafTy.getStaticInt()
    if preserveTrailing:
      lastShapeVal = shapeVal
      lastStrideVal = strideVal
      lastStride = strideEv.leaf
    if shapeVal == 1:
      # a dropped dimension is a stream restructure
      builder.markNonVerbatim()
      continue
    if chunkShape.isNil:
      # a chain opens on the first live leaf
      chunkShape = shapeEv.leaf
      chunkShapeVal = shapeVal
      chunkStride = strideEv.leaf
      chunkStrideVal = strideVal
      continue
    if shapeVal != DynamicSentinel and strideVal != DynamicSentinel and
        chunkShapeVal != DynamicSentinel and chunkStrideVal != DynamicSentinel and
        chunkShapeVal * chunkStrideVal == strideVal:
      # the chain's span reaches this dimension's stride, merge frontward
      chunkShapeVal *= shapeVal
      chunkShape = IntCT(chunkShapeVal)
      chunkVerbatim = false
      continue
    # the chain stops short of this dimension, flush and open the next chain
    builder.append(chunkShape, chunkStride, verbatim = chunkVerbatim)
    chunkShape = shapeEv.leaf
    chunkShapeVal = shapeVal
    chunkStride = strideEv.leaf
    chunkStrideVal = strideVal
    chunkVerbatim = true

  if chunkShape.isNil:
    # every leaf is size-1, a lone (1):(0) sentinel or the preserved marker
    builder.markNonVerbatim()
    if preserveTrailing:
      builder.append(IntCT(DynamicSentinel), lastStride)
  else:
    builder.append(chunkShape, chunkStride, verbatim = chunkVerbatim)
    if preserveTrailing and lastShapeVal == 1 and not (
        chunkShapeVal != DynamicSentinel and chunkStrideVal != DynamicSentinel and
        lastStrideVal != DynamicSentinel and
        chunkShapeVal * chunkStrideVal == lastStrideVal):
      # the trailing size-1 marker survives a chain that stops short of it
      builder.append(IntCT(DynamicSentinel), lastStride)
  let (node, verbatim) = builder.emitLayout()
  if verbatim: # Reuse the original to avoid destructuring -> restructuring temporaries
    result = originalLayout
  else:
    result = node

macro coalesce*(layout: Layout; preserveTrailing: static bool = false): untyped =
  ## Merge contiguous dimensions.
  ## Merge a layout's flat (shape, stride) leaf pairs into contiguous chains,
  ## one pure `make_layout` emission:
  ## - size-1 dimensions drop, the frontmost one opens no chain
  ## - a dimension joins the chain in front of it when the chain's span
  ##   reaches the dimension's stride, the merged chain keeps the front
  ##   dimension's own stride
  ## - preserveTrailing keeps a trailing size-1 dimension as the dynamic
  ##   `Int[DynamicSentinel]` marker
  ##
  ##   (2, 4):(1, 2)  folds to (8):(1), the (2,1) chain's span 2
  ##                  reaches the second dimension's stride 2
  ##   (4, 1):(1, 0)  folds to (4):(1), the trailing broadcast drops,
  ##                  a chain reaching its stride 0 absorbs it
  ##   (4, 1):(1, 0)  with preserveTrailing stays (4, Int[DynamicSentinel]):(1, 0)
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  # The original layout when coalesce is a no-op,
  # This avoids deconstruction -> reconstruction temporaries in the generated code
  let originalLayout = if result.len == 0: layout
                       else: result[^1][1] # returned `let`/`const` symbol
  result.add bindSym"coalesceImpl".newCall(originalLayout, sh, st, newLit(preserveTrailing))

# ═══════════════════════════════════════════════════════════════
#  complement, fill stride gaps up to the cosize bound
# ═══════════════════════════════════════════════════════════════

type ComplementRegime = enum
  crStatic, crDynMulti, crDynRank1Zero, crDynRank1

func complementRegime(shDims: seq[tuple[shape, depth: int, leaf: NimNode]],
                      stDims: seq[tuple[stride: int, leaf: NimNode]]): ComplementRegime =
  # Handles every complement limitations, the case branches stay pure processing:
  # - one stride per shape dimension, or one stride broadcasted over the shape
  # - multi-dimensional complements need compile-time strides, parity with CuTe
  # - the multi-dimension walk indexes flat leaves only
  if stDims.len != shDims.len and stDims.len != 1:
    error "complement: expected one stride per shape dimension, got " &
      $stDims.len & " strides for " & $shDims.len & " shape dimensions"
  if DynamicSentinel notin shDims.mapIt(it.shape) and
      DynamicSentinel notin stDims.mapIt(it.stride):
    return crStatic
  if shDims.len == 1:
    # rank-1 runtime, the formula or the zero collapse
    if stDims[0].stride == 0:
      return crDynRank1Zero
    return crDynRank1
  if DynamicSentinel in stDims.mapIt(it.stride):
    error "complement: multi-dimension with dynamic strides not supported"
  for i in 0 ..< shDims.len:
    if shDims[i].depth > 1:
      error "complement: non-flat shape at index " & $i
  crDynMulti

func complementFold(dims: seq[tuple[stride, shape, depth: int, leaf: NimNode]], bound: NimNode, defaultBound: bool): NimNode =
  var gapNodes, curNodes: seq[NimNode]
  var curNode = IntCT(1)
  var b = 1
  var allSkipped = true
  for idx in getIndicesSortedByStride(dims.mapIt(it.stride)):
    let dim = dims[idx]
    if dim.stride == 0 or dim.shape == 1:
      # a stride-0 dimension maps every coordinate to offset 0,
      # a size-1 dimension covers a single offset, neither joins
      # the gap-fill chain
      continue
    let s = IntCT(dim.stride)
    gapNodes.add bindSym"max".newCall(
      IntCT(1), nnkInfix.newTree(ident"div", s, curNode))
    curNodes.add curNode
    curNode =
      if dim.shape == DynamicSentinel:
        # past a dynamic shape leaf the frontier is runtime arithmetic
        s * dim.leaf
      else:
        IntCT(dim.stride * dim.shape)
    if defaultBound:
      # cosize over the live leaves, invariant under the skip
      b += (dim.shape - 1) * abs(dim.stride)
    allSkipped = false
  if allSkipped:
    # every leaf skipped, the complement collapses to (bound):(1)
    return bindSym"make_layout".newCall(
      (if defaultBound: IntCT(b) else: bound), newLit(1))
  curNodes.add curNode
  gapNodes.add bindSym"ceil_div".newCall(
    (if defaultBound: IntCT(b) else: bound), curNode)
  # coalesce is a macro and folds when the call site expands
  result = bindSym"coalesce".newCall(bindSym"make_layout".newCall(
    nnkPar.newTree(gapNodes), nnkPar.newTree(curNodes)))

macro complementImpl(sh, st: typed, bound: typed, defaultBound: static bool): untyped =
  var shDims: seq[tuple[shape, depth: int, leaf: NimNode]]
  var stDims: seq[tuple[stride: int, leaf: NimNode]]
  for shEv in sh.tupleStream():
    if shEv.kind == kLeaf:
      shDims.add (shEv.leafTy.getStaticInt(), shEv.depth, shEv.leaf)
  for stEv in st.tupleStream():
    if stEv.kind == kLeaf:
      stDims.add (stEv.leafTy.getStaticInt(), stEv.leaf)

  let boundExpr =
    if bound.getTypeInst().kind == nnkTupleConstr:
      bindSym"product".newCall(bound)
    else:
      bound
  let boundDyn = if defaultBound: bound else: boundExpr

  # one stride leaf broadcasts over the shape
  var dims: seq[tuple[stride, shape, depth: int, leaf: NimNode]]
  for i in 0 ..< shDims.len:
    let j = if stDims.len == 1: 0 else: i
    dims.add (stDims[j].stride, shDims[i].shape, shDims[i].depth, shDims[i].leaf)

  case complementRegime(shDims, stDims)
  of crStatic:
    result = complementFold(dims, boundExpr, defaultBound)
  of crDynMulti:
    # the fold emits runtime arithmetic for dynamic shape leaves
    result = complementFold(dims, boundDyn, defaultBound = false)
  of crDynRank1Zero:
    # a static zero stride, every coordinate maps to offset 0
    result = bindSym"make_layout".newCall(boundDyn, newLit(1))
  of crDynRank1:
    # rank-1, runtime gap formula
    let stLeaf = stDims[0].leaf
    let shLeaf = shDims[0].leaf
    result = quote do:
      coalesce(make_layout(
        (max(Int[1](), `stLeaf`), ceil_div(`boundDyn`, `stLeaf` * `shLeaf`)),
        (1, `stLeaf` * `shLeaf`)))

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
  ##   `make_layout(layout, result)` is a bijection
  ##   onto a contiguous range, otherwise it under-fills:
  ##   the largest ordered, disjoint layout that fits
  ##
  ## Skip semantics:
  ## - a stride-0 dimension maps every coordinate to offset 0,
  ##   a size-1 dimension covers a single offset, neither opens a gap
  ## - the default bound is cosize(layout), invariant under the skip
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
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)

  let originalLayout = if result.len == 0: layout else: result[^1][1]
  result.add bindSym"complementImpl".newCall(
    sh, st, bindSym"cosize".newCall(originalLayout), newLit(true))

macro complement*(layout: Layout, cosizeBound: static int): untyped =
  ## Complement with a compile-time int bound.
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  result.add bindSym"complementImpl".newCall(
    sh, st,
    nnkCall.newTree(nnkBracketExpr.newTree(bindSym"Int", newLit(cosizeBound))),
    newLit(false))

macro complement*(layout: Layout, cosizeBound: typed): untyped =
  ## Complement with an `Int`, a runtime `int`, or a shape-tuple bound,
  ## the tuple enters as its product.
  let bound =
    if cosizeBound.kind == nnkIntLit:
      # an int literal wraps to the static Int leaf
      nnkCall.newTree(nnkBracketExpr.newTree(bindSym"Int", cosizeBound))
    else:
      cosizeBound
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  result.add bindSym"complementImpl".newCall(sh, st, bound, newLit(false))

# ═══════════════════════════════════════════════════════════════
#  compose, apply a layout through another
# ═══════════════════════════════════════════════════════════════

macro composeImpl(aLayout, bShape, bStrides: typed): untyped =
  ## Nested walk over coalesced LHS and the destructured RHS:
  ## - level 1: zip walk over the RHS shape and stride
  ## - level 2: one dimension of B at a time is walked through A's dimensions.
  ##   Each A dimension takes as many B coordinates as its size,
  ##   and emits them as a (shape, stride) pair.
  ##   The leftover B (shape, strides) carry into the next A dimension.
  ##
  ##   A = (2,3):(2,1) composed with B = 6:(-1)
  ##     A's 2:2  takes 2 B coords -> pair (2, -2);  3 left, step -1
  ##     A's 3:1  takes 3 B coords -> pair (3, -1);  none left
  ##     result (2,3):(-2,-1)
  result = newStmtList()

  let (aShape, aStrides) = destructureLayout(result, aLayout)
  let shapeLeaves = aShape.tupleFlatten()
  let strideLeaves = aStrides.tupleFlatten()

  # level 1: the zip walk over the RHS shape and stride trees
  var builder = TupleBuilderNested.new(2)
  for (shapeEv, strideEv) in bShape.tupleStream().zip(bStrides.tupleStream()):
    if shapeEv.kind != kLeaf:
      builder.append(shapeEv, strideEv)
      continue
    if strideEv.leafTy.getStaticInt() == 0:
      # a stride-0 RHS dimension maps every coordinate to offset 0,
      # the pair is the RHS dimension itself, the LHS is untouched
      builder.append(shapeEv.leaf, strideEv.leaf)
      continue
    if shapeLeaves.len == 1:
      # a 1-leaf profile consumes nothing, the strides multiply, no lets
      builder.append(shapeEv.leaf, strideEv.leaf * strideLeaves[0].leaf)
      continue
    # level 2: fold this RHS leaf over the flat LHS profile
    var pairs: seq[tuple[shape, stride: NimNode]]
    var remShape = shapeEv.leaf
    var remStride = strideEv.leaf
    var remShapeV = shapeEv.leafTy.getStaticInt()
    var remStrideV = strideEv.leafTy.getStaticInt()
    let R = shapeLeaves.len
    for k in 0 ..< R:
      let shapeLeaf = shapeLeaves[k].leaf
      let strideLeaf = strideLeaves[k].leaf
      let shVk = shapeLeaves[k].leafTy.getStaticInt()
      let stVk = strideLeaves[k].leafTy.getStaticInt()
      if k == R - 1:
        if pairs.len == 0 or remShapeV != 1:
          # no LHS leaf was consumed, the RHS leaf passes through
          pairs.add (shape: remShape, stride: remStride * strideLeaf)
        break
      let absRemV = if remStrideV != DynamicSentinel: abs(remStrideV)
                    else: DynamicSentinel
      let absRem = result.newLetAsgn("absRem", abs(remStride))
      let clampedV = if absRemV != DynamicSentinel and shVk != DynamicSentinel and remShapeV != DynamicSentinel:
        min(ceil_div(shVk, absRemV), remShapeV)
      else:
        DynamicSentinel
      let clamped = result.newLetAsgn("clampedShape", min(ceil_div(shapeLeaf, absRem), remShape))
      if clampedV != 1 and remShapeV != 1:
        # a leaf whose consumed shape folds to 1 contributes nothing
        # dynamic leaves never fold to 1
        pairs.add (shape: clamped, stride: remStride * strideLeaf)
        let remShUpdate = remShape div clamped
        remShape = result.newLetAsgn("remainingShape", remShUpdate)
        if clampedV != DynamicSentinel:
          remShapeV = remShapeV div clampedV
      let remStUpdate = ceil_div(absRem, shapeLeaf) * sign(remStride)
      remStride = result.newLetAsgn("remainingStride", remStUpdate)
      if absRemV != DynamicSentinel and shVk != DynamicSentinel:
        remStrideV = ceil_div(absRemV, shVk) * sign(remStrideV)
    appendDimension(builder, pairs)

  result.add builder.emitLayout().resultLayout

macro compose*[A, B: Layout](a: A, b: B): untyped =
  ## Layout composition, `A ∘ B`.
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
  ## Divisibility of the consumed shape is a caller precondition.
  ## Runtime shapes are unchecked.
  ##
  ##    domain  ──── B ────▶  A's domain  ──── A ────▶  values
  ##    domain  ══════════════ R ════════════════════▶  values
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
  let (aShape, aStrides) = result.destructureLayout(a)
  let (bShape, bStrides) = result.destructureLayout(b)

  template composeDelegateCoalesced(aShape2, aStrides2, bShape2, bStrides2) =
    composeImpl(coalesce(make_layout(aShape2, aStrides2), true), bShape2, bStrides2)
  template composeDelegatePlain(aShape2, aStrides2, bShape2, bStrides2) =
    composeImpl(make_layout(aShape2, aStrides2), bShape2, bStrides2)

  let aShapeIsTuple = layoutTypeArgs(a).shapeTy.isTupleTy()
  if aShapeIsTuple:
    result.add getAst(composeDelegateCoalesced(aShape, aStrides, bShape, bStrides))
  else:
    result.add getAst(composeDelegatePlain(aShape, aStrides, bShape, bStrides))


macro compose*(layout: Layout; tiler: tuple): untyped =
  ## Layout composition
  ##
  ## Returns a layout `R` such that `R(i) = A(B(i))` for all
  ## `i` in `0 ..< cosize(B)`.
  ##
  ## Divisibility of the consumed shape is a caller precondition.
  ##
  ## Example:
  ##   compose(make_layout((32, 8), (1, 32)), (16, _))
  ##   # → (16, 8):(1, 32)
  ##
  ## Dimension 0 consumes 16 positions, dimension 1 passes through.
  let
    shTy = layoutTypeArgs(layout).shapeTy
    R = if shTy.kind == nnkTupleConstr: shTy.len else: 1
    tilerRank = tiler.getTypeInst().len
  doAssert tilerRank <= R,
    "compose: tiler has more dimensions (" & $tilerRank & ") than the layout (" & $R & ")"
  result = newStmtList()
  let (aShape, aStrides) = result.destructureLayout(layout)
  template composeTilerDim(aS2, aSt2, bElem) =
    composeImpl(make_layout(aS2, aSt2), bElem.shape, bElem.stride)
  var shapes: seq[NimNode]
  var strides: seq[NimNode]
  for k in 0 ..< tilerRank:
    # bracket nodes, aShape[k] would index the NimNode's children
    let aShapeK = if shTy.kind == nnkTupleConstr:
                    nnkBracketExpr.newTree(aShape, newLit k)
                  else:
                    aShape
    let aStrideK = if shTy.kind == nnkTupleConstr:
                     nnkBracketExpr.newTree(aStrides, newLit k)
                   else:
                     aStrides
    let elemTy = tiler.getTypeInst()[k]
    if elemTy.eqIdent("X"):
      # the profiler mark passes the dimension through whole
      shapes.add aShapeK
      strides.add aStrideK
    elif elemTy.kind == nnkBracketExpr and elemTy[0].eqIdent("Layout"):
      # a layout tiler element composes the dimension once
      let dk = result.newLetAsgn("composedDim",
        getAst(composeTilerDim(aShapeK, aStrideK, tiler.getTupleIndex(k))))
      shapes.add dk.newDotExpr(ident"shape")
      strides.add dk.newDotExpr(ident"stride")
    elif elemTy.isTupleTy():
      # a sub-tuple tiler element recurses per-sub-dimension, the rank
      # guards hold at every level of that recursion
      let dk = result.newLetAsgn("composedDim",
        ident"compose".newCall(ident"make_layout".newCall(aShapeK, aStrideK),
                               tiler.getTupleIndex(k)))
      shapes.add dk.newDotExpr(ident"shape")
      strides.add dk.newDotExpr(ident"stride")
    else:
      # an int tiler element composes the dimension with (N):(1),
      # the first N positions: the pair is (N, the dimension's stride)
      shapes.add tiler.getTupleIndex(k)
      strides.add aStrideK

  if shapes.len == 1:
    result.add bindSym"make_layout".newCall(
      nnkTupleConstr.newTree(shapes[0]), nnkTupleConstr.newTree(strides[0]))
  else:
    result.add bindSym"make_layout".newCall(nnkPar.newTree(shapes), nnkPar.newTree(strides))

# ═══════════════════════════════════════════════════════════════
#  logical_divide, tile a layout into (tile, rest)
# ═══════════════════════════════════════════════════════════════

template divideFormula(shA, stA, shB, stB) =
  compose(make_layout(shA, stA),
          make_layout(
            (shB, complement(make_layout(shB, stB), size(make_layout(shA, stA))).shape),
            (stB, complement(make_layout(shB, stB), size(make_layout(shA, stA))).stride)))

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
  ## - every element of `layout` appears exactly once, the divide
  ##   reorders elements, it drops none
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
  result.add getAst(divideFormula(shA, stA, shB, stB))

macro logical_divide*[L: Layout](layout: L, tiler: int): untyped =
  ## Logical divide by an int tiler
  getAst(divideRank1(layout, tiler))

macro logical_divide*[L: Layout, V: static int](layout: L, tiler: Int[V]): untyped =
  ## Logical divide by a static int tiler, see the Layout
  ## overload for the contract.
  getAst(divideRank1(layout, tiler))

macro logical_divide*[L: Layout](layout: L, tiler: static int): untyped =
  ## Logical divide by a static int tiler
  result = newStmtList()
  let (shA, stA) = result.destructureLayout(layout)
  result.add getAst(divideFormula(shA, stA,
    nnkCall.newTree(nnkBracketExpr.newTree(bindSym"Int", newLit(tiler))),
    nnkCall.newTree(nnkBracketExpr.newTree(bindSym"Int", newLit(1)))))

macro divideTupleImpl(sh, st, tiler: typed): untyped =
  ## Per-dimension divide over the destructured layout.
  ##
  ## - one tiler element per layout dimension
  ## - dimensions past the tiler pass through
  ## - a divided dimension carries the (tile, rest) pair.
  template divideTupleDimShape(dsh, dst, dtl) =
    logical_divide(make_layout(dsh, dst), dtl).shape
  template divideTupleDimStride(dsh, dst, dtl) =
    logical_divide(make_layout(dsh, dst), dtl).stride
  let shTy = sh.getTypeInst()
  let layoutRank = if shTy.kind == nnkTupleConstr: shTy.len else: 1
  let tilerRank = tiler.getTypeInst().len
  if tilerRank > layoutRank:
    error "logical_divide: tiler has more dimensions (" & $tilerRank &
      ") than the layout (" & $layoutRank & ")"
  var builder = TupleBuilderFlat.new(2)
  var tilerDims = tiler.tupleDimsStream()
  for (shEv, stEv) in sh.tupleDimsStream().zip(st.tupleDimsStream()):
    shEv.onLeaves():
      if not tilerDims.done():
        # the per-dim divide is emitted twice, once per projection.
        # Both copies stay pure expressions the C compiler folds and deduplicates,
        # a binding would instead materialize a wall of temporaries + temp types
        # that might be harder to optimize away on certain backends (Vulkan / WebGPU that don't use LLVM for example)
        let tilerEv = tilerDims.next()
        builder.append(getAst(divideTupleDimShape(shEv.leaf, stEv.leaf, tilerEv.leaf)),
                       getAst(divideTupleDimStride(shEv.leaf, stEv.leaf, tilerEv.leaf)))
      else:
        # a pass-through dimension, the dimension arrives whole
        builder.append(shEv.leaf, stEv.leaf)
  result = builder.emitLayout().resultLayout

macro logical_divide*(layout: Layout, tiler: tuple): untyped =
  ## Logical divide by a tuple tiler, one tiler element per layout dimension:
  ## - tiler elements matched positionally to layout dimensions,
  ##   dimensions past the tiler length pass through undivided
  ## - a divided dimension becomes the (tile, rest) pair, each
  ##   pair carries the Layout-tiler contract
  ## - an empty tiler divides nothing, every dimension passes through

  proc unwrapSLE(e: NimNode): NimNode =
    result = e
    while result.kind == nnkStmtListExpr:
      result = result[^1]
  if unwrapSLE(tiler).getTypeInst().len == 0:
    # an empty tiler divides nothing, the layout passes through verbatim
    result = newStmtList()
    result.add layout
  else:
    result = newStmtList()
    let (sh, st) = result.destructureLayout(layout)
    template divideTupleDelegate(sh2, st2, tiler2) =
      divideTupleImpl(sh2, st2, tiler2)
    result.add getAst(divideTupleDelegate(sh, st, tiler))

func zipped_divide*[LayoutT: Layout, TilerT](layout: LayoutT; tiler: TilerT): auto {.inline.} =
  ## Divide layout by tiler and zip tile/rest dimensions into rank-2 result.
  hier_unzip(logical_divide, layout, tiler)

template tiled_divide*(layout: Layout; tiler: auto): auto =
  ## Like zipped_divide but unpack the second dimension into individual dimensions.
  ## Keeps dimension-0 grouped (the tile).
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zd, zipped_divide(lyt, tlr))
    make_layout(
      groupedHead(dimension(zd, 0).shape, dimension(zd, 1).shape),
      groupedHead(dimension(zd, 0).stride, dimension(zd, 1).stride)
    )

template flat_divide*(layout: Layout; tiler: auto): auto =
  ## Like zipped_divide but unpack BOTH dimensions into a flat layout.
  ## Unlike tiled_divide the tile dimensions are also unpacked.
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zd, zipped_divide(lyt, tlr))
    make_layout(
      concatFlat(
        dimension(zd, 0).shape,
        dimension(zd, 1).shape,
      ),
      concatFlat(
        dimension(zd, 0).stride,
        dimension(zd, 1).stride,
      ),
    )

# ═══════════════════════════════════════════════════════════════
#  right_inverse, quasi-inverse sorted by stride
# ═══════════════════════════════════════════════════════════════

macro rightInverseImpl(sh, st: typed): untyped =
  ## right_inverse core, emits coalesced contiguous chains in stride order:
  ## - a dimension joins when its stride equals the chain span
  ## - a dynamic shape ends the chain and keeps its own value node
  ## - the chain stride is the product of the shapes before the dimension,
  ##   a runtime prefix expression once a dynamic leaf precedes it
  ## - an empty chain collapses to the empty layout (1, 0)
  var dims: seq[tuple[stride, shape, prefix: int, prefixNode: NimNode, leaf: NimNode]]
  var prefix = 1
  # prefixNode carries the prefix as an expression once it turns runtime,
  # so a stride after a dynamic leaf can still be emitted
  var prefixNode: NimNode = nil
  for (shEv, stEv) in sh.tupleStream().zip(st.tupleStream()):
    if shEv.kind != kLeaf:
      continue
    let shape = shEv.leafTy.getStaticInt()
    dims.add (stEv.leafTy.getStaticInt(), shape, prefix, prefixNode, shEv.leaf)
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
      let stLeaf = if dim.prefix == DynamicSentinel: dim.prefixNode
                   else: IntCT(dim.prefix)
      builder.append(shLeaf, stLeaf)
      if dim.shape == DynamicSentinel:
        break
      curr = dim.stride * dim.shape
  result = bindSym"coalesce".newCall(builder.emitLayout().resultLayout)

macro right_inverse*(layout: typed): untyped =
  ## Quasi-inverse, the largest injective R with L(R(i)) == i.
  ## Returns:
  ## - a coalesced Layout, typically lower rank than L
  ## - (1, 0) when no chain exists
  result = newStmtList()
  let (sh, st) = destructureLayout(result, layout)
  template rightInverseDelegate(sh2, st2) =
    ## Expands the inverse core on the destructured tuples at the use site.
    rightInverseImpl(sh2, st2)
  result.add getAst(rightInverseDelegate(sh, st))

# ═══════════════════════════════════════════════════════════════
#  left_inverse, left inverse (injective layouts only)
# ═══════════════════════════════════════════════════════════════

macro leftInverseImpl(sh, st: typed): untyped =
  ## Left-inverse dimensions built from stride ratios:
  ##
  ##   result_shape[i]  = stride / size_so_far
  ##   result_stride[i] = shape prefix of the previous stride-sorted dimension
  ##
  ## All strides must be static (compile-time assert).
  var dims: seq[tuple[stride, shape, prefix: int, prefixNode: NimNode, leaf: NimNode]]
  var prefix = 1
  # prefixNode carries the prefix as an expression once it turns runtime,
  # so a stride after a dynamic leaf can still be emitted
  var prefixNode: NimNode = nil
  for (shEv, stEv) in sh.tupleStream().zip(st.tupleStream()):
    if shEv.kind != kLeaf:
      continue
    let shape = shEv.leafTy.getStaticInt()
    dims.add (stEv.leafTy.getStaticInt(), shape, prefix, prefixNode, shEv.leaf)
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
      builder.append(IntCT(gap), prevStLeaf)
    sizeSoFar = dim.stride
    prevIdx = idx
    prevPrefix = dim.prefix
    prevPrefixNode = dim.prefixNode
  if prevIdx >= 0:
    # tail = the last stride-sorted dimension's own shape
    let dim = dims[prevIdx]
    let shLeaf = if dim.shape == DynamicSentinel: dim.leaf
                 else: IntCT(dim.shape)
    let stLeaf = if dim.prefix == DynamicSentinel: dim.prefixNode
                 else: IntCT(dim.prefix)
    builder.append(shLeaf, stLeaf)
  result = bindSym"coalesce".newCall(builder.emitLayout().resultLayout)

macro left_inverse*(layout: typed): untyped =
  ## Left inverse, Li(L(i)) == i for injective layouts.
  ## Returns:
  ## - a coalesced Layout over the static-stride gaps
  ## - requires all static strides, compile-time assert
  result = newStmtList()
  let (sh, st) = destructureLayout(result, layout)
  template leftInverseDelegate(sh2, st2) =
    ## Coalesce canonicalizes strides first, the chaining asserts require it.
    evalOnceAs(coalescedLayout, coalesce(make_layout(sh2, st2)))
    leftInverseImpl(coalescedLayout.shape, coalescedLayout.stride)
  result.add getAst(leftInverseDelegate(sh, st))


# ═══════════════════════════════════════════════════════════════
#  logical_product, reproduce a block over a tiler
# ═══════════════════════════════════════════════════════════════

macro logicalProductFinish(a, rest: typed): untyped =
  result = newStmtList()
  # rest evaluates to a Layout constructor nested under statement lists,
  var grid = rest
  while grid.kind == nnkStmtListExpr:
    for i in 0 ..< grid.len - 1:
      result.add(grid[i])
    grid = grid[^1]
  let (aShape, aStrides) = result.destructureLayout(a)
  let (rShape, rStrides) = result.destructureLayout(grid)
  result.add bindSym"make_layout".newCall(
    nnkTupleConstr.newTree(aShape, rShape),
    nnkTupleConstr.newTree(aStrides, rStrides)
  )

macro logical_product*[A, B: Layout](a: A, tiler: B): untyped =
  ## Logical product, `a x tiler = (a, a* ∘ tiler)`.
  ##
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

  # Implementation note
  #
  #   For direct AST->AST transformation of constructors we need to manipulate the AST *produced* by compose(complement(...), tiler)
  #   In this macro we can only see the macro call AST so we need to defer to another macro
  #   so that compose(complement(...), tiler) have the time to do their own AST->AST constructor transformation
  #
  #   Now one tricky part of this is that using an AST node or a template input in multiple plice will paste it verbatim
  #   if it's used 3 times like below, `a` expression will be evaluated 3 times. This is problematic if the expression has side-effects like 'echo "launch_missiles"'.
  #
  #   In our case, layouts are pure and only involve integer arithmetic.
  #   Furthermore, I argue that compared to the alternative (assigning expressions to temporaties)
  #   an integer expression is significantly more compiler-friendly as they can be:
  #   - constant-folded (done by nim compiler)
  #   - terms can be reorder, say we receive (2 * (3 * (dynamic_value * (5 * 6))))
  #     with temporaries dynamic_value would be an optimization barrier, so we would have `6 * dynamic_value * 30` with a naive compiler,
  #     while we would have 180 * dynamic_value with a expression with more certainty as it's easier for the compiler to reorder integers
  #   - compilers can do common sub-expression elimination more easily when only integers are dumped into an expression
  #
  #   This is particularly relevant for Vulkan and WebGPU backends which might not have
  #   optimizers as thorough as LLVM's.

  template logicalProductDelegate(a_layout, tiler_layout) =
    logicalProductFinish(a_layout, compose(complement(a_layout, size(a_layout) * cosize(tiler_layout)), tiler_layout))
  result = getAst(logicalProductDelegate(a, tiler))

# ═══════════════════════════════════════════════════════════════
#  zipped_product, tiled_product, flat_product
# ═══════════════════════════════════════════════════════════════

template zipped_product*(blk: Layout; tiler: auto): auto =
  ## Reproduce block over tiler, zipped into rank-2 result.
  ##
  ## CuTe: zipped_product = hier_unzip(logical_product, block, tiler)
  hier_unzip(logical_product, blk, tiler)

template tiled_product*(blk: Layout; tiler: auto): auto =
  ## Like zipped_product but unpack the second dimension.
  ## Keeps dimension-0 grouped (the block).
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zp, zipped_product(bk, tlr))
    make_layout(
      groupedHead(dimension(zp, 0).shape, dimension(zp, 1).shape),
      groupedHead(dimension(zp, 0).stride, dimension(zp, 1).stride),
    )

template flat_product*(blk: Layout; tiler: auto): auto =
  ## Like zipped_product but unpack BOTH dimensions into a flat layout.
  ## Unlike tiled_product the block dimensions are also unpacked.
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tlr, tiler)
    evalOnceAs(zp, zipped_product(bk, tlr))
    make_layout(
      concatFlat(
        dimension(zp, 0).shape,
        dimension(zp, 1).shape,
      ),
      concatFlat(
        dimension(zp, 0).stride,
        dimension(zp, 1).stride,
      ),
    )

# ═══════════════════════════════════════════════════════════════
#  blocked_product, blocks laid out contiguously
# ═══════════════════════════════════════════════════════════════

func blocked_product*[A, B: Layout](blk: A; tiler: B): auto =
  ## Repeat block over tiler grid, each block contiguous.
  ## Results in ((BLK_A, TILER_A), (BLK_B, TILER_B), ...).
  const mxR = max(blk.rank(), tiler.rank())
  let lp = logical_product(padRight(blk, mxR), padRight(tiler, mxR))
  let m0 = dimension(lp, 0)
  let m1 = dimension(lp, 1)
  zipDimensions(m0, m1)

# ═══════════════════════════════════════════════════════════════
#  raked_product, blocks interleaved over the tiler grid
# ═══════════════════════════════════════════════════════════════

func raked_product*[A, B: Layout](blk: A; tiler: B): auto =
  ## Repeat block over tiler grid, blocks interleaved.
  ## Results in ((TILER_A, BLK_A), (TILER_B, BLK_B), ...).
  const mxR = max(blk.rank(), tiler.rank())
  let lp = logical_product(padRight(blk, mxR), padRight(tiler, mxR))
  let m0 = dimension(lp, 0)
  let m1 = dimension(lp, 1)
  zipDimensions(m1, m0)

# ═══════════════════════════════════════════════════════════════
#  tile_to_shape, repeat a block layout to fill a target shape
# ═══════════════════════════════════════════════════════════════

template tile_to_shape*(blk: Layout; target_shape: typed; ord_shape: static StrideOrder = LayoutLeft): auto =
  ## Repeat a block layout to fill a target shape.
  ##
  ## Returns:
  ##   the block tiled over the target shape, ceil_div repeats per
  ##   target dimension, repeat order per dimension from ord_shape
  ##
  ## Example:
  ##   let tile = tile_to_shape(make_layout((2,3), (1,2)), (6, 12))
  ##   # block (2,3) repeated to fill (6,12) in 3 columns:
  ##   # ((2,3),3):((1,2),6)
  const R = static(rank(target_shape))
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(ts, target_shape)
    let padded_blk = padRight(bk, R)
    let blk_shape = product_each(padded_blk.shape)
    let trg_flat = product_each(ts)
    let product_shape = zipDimensionsWith(trg_flat, blk_shape): ceil_div(it_a, it_b)
    let tiler = make_layout(product_shape, ord_shape)
    blocked_product(padded_blk, tiler)

