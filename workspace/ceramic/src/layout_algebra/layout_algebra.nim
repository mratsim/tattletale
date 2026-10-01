# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout algebra: coalesce, filter_zeros, filter, sort.

import std/macros
import std/sequtils
import std/algorithm
import std/typetraits
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_unsanctioned_helpers
import ./layouts
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

macro coalesceBackward(layoutShape, layoutStride: typed; preserveTrailing: static bool = false): untyped =
  var shLeaves, shTypes, stLeaves, stTypes: seq[NimNode]
  for (leaf, ty) in layoutShape.tupleFlatten().reversed():
    shLeaves.add leaf
    shTypes.add ty
  for (leaf, ty) in layoutStride.tupleFlatten().reversed():
    stLeaves.add leaf
    stTypes.add ty

  if shLeaves.len == 1 and stLeaves.len == 1:
    if isStaticOne(shTypes[0]):
      result = bindSym"make_layout".newCall(newLit(1), newLit(0))
    else:
      result = bindSym"make_layout".newCall(shLeaves[0], stLeaves[0])
    return

  # chunks collect back-to-front while the walk merges frontward.
  # head is the current front chunk, the emission walks the chunk list backward
  type Chunk = tuple[shape, shapeTy, stride, strideTy: NimNode]
  var chunks: seq[Chunk]
  var head: Chunk = (shLeaves[0], shTypes[0], stLeaves[0], stTypes[0])
  if preserveTrailing and isStaticOne(shTypes[0]):
    head.shape = IntCT(low(int))
    head.shapeTy = newNimNode(nnkBracketExpr).add(ident"Int", newLit(low(int)))

  for k in 1 ..< shLeaves.len:
    if isStaticOne(shTypes[k]):
      continue
    if isStaticOne(head.shapeTy):
      head = (shLeaves[k], shTypes[k], stLeaves[k], stTypes[k])
    elif isStaticInt(shTypes[k]) and isStaticInt(stTypes[k]) and
        isStaticInt(head.shapeTy) and isStaticInt(head.strideTy) and
        shTypes[k].getStaticInt() * stTypes[k].getStaticInt() == head.strideTy.getStaticInt():
      let mergedVal = shTypes[k].getStaticInt() * head.shapeTy.getStaticInt()
      head = (IntCT(mergedVal),
              newNimNode(nnkBracketExpr).add(ident"Int", newLit(mergedVal)),
              stLeaves[k], stTypes[k])
    else:
      chunks.add head
      head = (shLeaves[k], shTypes[k], stLeaves[k], stTypes[k])
  chunks.add head

  if not preserveTrailing:
    while chunks.len > 0 and isStaticOne(chunks[0].shapeTy):
      discard chunks.pop()  # back chunks sit at the seq front

  if chunks.len == 0:
    result = bindSym"make_layout".newCall(IntCT(1), newLit(0))
    return

  var rShape = newNimNode(nnkTupleConstr)
  var rStride = newNimNode(nnkTupleConstr)
  for idx in countdown(chunks.len - 1, 0):
    rShape.add chunks[idx].shape
    rStride.add chunks[idx].stride
  if rShape.len == 1:
    rShape = rShape[0]
    rStride = rStride[0]

  result = bindSym"make_layout".newCall(rShape, rStride)

func coalesce*(layout: Layout): auto {.inline, noInit.} =
  ## Merge contiguous dimensions.
  coalesceBackward(layout.shape, layout.stride)

func coalesce_preserve_trailing(layout: Layout): auto {.inline, noInit.} =
  ## Like `coalesce` but preserves trailing size-1 dimensions (e.g. stride-0 broadcasts).
  coalesceBackward(layout.shape, layout.stride, preserveTrailing = true)

# ═══════════════════════════════════════════════════════════════
#  filter_inactive, remove stride-0 and size-1 dimensions
# ═══════════════════════════════════════════════════════════════

func filter_inactive*(layout: Layout): auto {.inline.} =
  ## Remove stride-0 and size-1 dimensions
  coalesce(filter_zeros(layout))

# ═══════════════════════════════════════════════════════════════
#  complement, fill stride gaps up to the cosize bound
# ═══════════════════════════════════════════════════════════════

proc complementFold(shNode, boundExpr: NimNode;
                    strides: seq[int]): NimNode {.compileTime.} =
  ## Multi-dimension complement fold, one (gap, stride) pair per
  ## stride-sorted dimension plus the final gap up to the bound.
  var gapNodes, curNodes: seq[NimNode]
  var curNode = IntCT(1)
  for idx in getIndicesSortedByStride(strides):
    let s = IntCT(strides[idx])
    gapNodes.add bindSym"max".newCall(
      IntCT(1), nnkInfix.newTree(ident"div", s, curNode))
    curNodes.add curNode
    curNode = nnkInfix.newTree(
      ident"*", s, nnkBracketExpr.newTree(shNode, newLit(idx)))
  gapNodes.add bindSym"ceil_div".newCall(boundExpr, curNode)
  curNodes.add curNode
  # coalesceBackward is a macro and folds when the call site expands
  result = bindSym"coalesceBackward".newCall(
    nnkPar.newTree(gapNodes), nnkPar.newTree(curNodes))

macro complementImpl(sh, st, cosizeBound: typed): untyped =
  ## Dispatch to scalar or multi-dimension complement.
  ##
  ## Contract:
  ## - shape and stride args are flat, filter_zeros and coalesce fold
  ##   before the call

  let boundExpr =
    if cosizeBound.getTypeInst().kind == nnkTupleConstr:
      bindSym"product".newCall(cosizeBound)
    else:
      cosizeBound
  if sh.getTypeInst().kind != nnkTupleConstr:
    let stTyp = st.getTypeInst()
    if stTyp.kind == nnkBracketExpr and $stTyp[0] == "Int" and stTyp[1].intVal == 0:
      # Static zero stride, every coordinate maps to offset 0
      result = bindSym"make_layout".newCall(boundExpr, newLit(1))
    else:
      result = quote do:
        coalesceBackward(
          (max(Int[1](), `st`), ceil_div(`boundExpr`, `st` * `sh`)),
          (1, `st` * `sh`))
  else:
    # Multi-dimension complement, all strides must be static Int leaves
    let shTyp = sh.getTypeInst()
    let stTyp = st.getTypeInst()
    doAssert stTyp.kind == nnkTupleConstr,
      "complement: expected tuple type for strides"
    for i in 0 ..< shTyp.len:
      doAssert shTyp[i].kind notin {nnkTupleConstr, nnkTupleTy},
        "complement: non-flat shape at index " & $i
    for i in 0 ..< stTyp.len:
      doAssert stTyp[i].kind == nnkBracketExpr and $stTyp[i][0] == "Int",
        "complement: multi-dimension with dynamic strides not supported at index " & $i
    result = complementFold(sh, boundExpr, toSeqStaticInts(stTyp))

proc filterInactiveValues(shapeVals, strideVals: seq[int]): tuple[shape, stride: seq[int]] {.compileTime.} =
  ## filter_zeros + coalesce as one compile-time fold over flat static values,
  ## coalesceBackward's chunk walk in value form, walked back-to-front:
  ## - a stride of 0 shrinks its shape to 1
  ## - size-1 shapes drop
  ## - a dimension whose span ends where the head's stride begins merges
  ##   into the head and takes the dimension's own stride, the result stays
  ##   in forward order
  var head = -1
  for i in countdown(shapeVals.len - 1, 0):
    let stI = strideVals[i]
    let shI = if stI == 0: 1 else: shapeVals[i]
    if shI == 1: continue
    if head == -1:
      head = 0
      result.shape.add shI
      result.stride.add stI
    elif shI * stI == result.stride[head]:
      result.shape[head] *= shI
      result.stride[head] = stI
    else:
      result.shape.insert(shI, 0)
      result.stride.insert(stI, 0)

macro complementEmit(lyt, cosizeBound: typed; defaultBound: static bool): untyped =
  ## complement's emission, dispatched on the layout's staticness:
  ## - a fully static layout folds filter_zeros and coalesce to constant
  ##   values at compile time and emits the complement over them alone, no
  ##   runtime filter chain
  ## - any other layout runs filter_inactive at runtime, complementImpl folds
  ##   gap arithmetic from the coalesced shape/stride
  ## defaultBound swaps cosizeBound for cosize of the filtered layout.
  let (shTy, stTy) = layoutTypeArgs(lyt)
  let shVals = toSeqStaticInts(shTy)
  let stVals = toSeqStaticInts(stTy)
  if DynamicSentinel in shVals or DynamicSentinel in stVals:
    # a runtime leaf, the bound is the caller's expression or cosize(f),
    # plain idents in spliced subtrees resolve by name at the call site
    let f = ident"f"
    let bound = if defaultBound: bindSym"cosize".newCall(f) else: cosizeBound
    result = nnkStmtListExpr.newTree(
      nnkLetSection.newTree(nnkIdentDefs.newTree(
        f, newEmptyNode(), bindSym"filter_inactive".newCall(lyt))),
      bindSym"complementImpl".newCall(
        f.newDotExpr(ident"shape"), f.newDotExpr(ident"stride"), bound))
  else:
    # scalar stride broadcasts over the shape profile (repeat_like)
    let pairedStrides =
      if stTy.kind in {nnkTupleConstr, nnkTupleTy}:
        doAssert shTy.kind in {nnkTupleConstr, nnkTupleTy},
          "complement: stride profile larger than shape profile"
        stVals
      else:
        var s = newSeq[int](shVals.len)
        for i in 0 ..< shVals.len:
          s[i] = stVals[0]
        s
    let pairs = filterInactiveValues(shVals, pairedStrides)
    var bound = if cosizeBound.getTypeInst().kind == nnkTupleConstr:
      bindSym"product".newCall(cosizeBound)
    else:
      cosizeBound
    if defaultBound:
      var b = 1
      for i in 0 ..< pairs.shape.len:
        b += (pairs.shape[i] - 1) * abs(pairs.stride[i])
      bound = IntCT(b)
    if pairs.shape.len == 0:
      # every dimension broadcasts or is size-1, the complement collapses to (bound):(1)
      result = bindSym"make_layout".newCall(bound, newLit(1))
    else:
      var shN = newNimNode(nnkTupleConstr)
      var stN = newNimNode(nnkTupleConstr)
      for i in 0 ..< pairs.shape.len:
        shN.add IntCT(pairs.shape[i])
        stN.add IntCT(pairs.stride[i])
      if shN.len == 1:
        shN = shN[0]
        stN = stN[0]
      result = bindSym"complementImpl".newCall(shN, stN, bound)

func complement*(layout: Layout; cosizeBound: Int or int): auto =
  ## Complement of the layout, filling stride gaps up to cosizeBound.
  ## Filters inactive dimensions first, the filter folds at compile time
  ## for fully static layouts.
  complementEmit(layout, cosizeBound, false)

func complement*(layout: Layout; cosizeBound: static int): auto =
  ## Compile-time int overload.
  complement(layout, Int[cosizeBound]())

func complement*(layout: Layout): auto =
  ## Compute complement with default bound = cosize(filtered layout).
  complementEmit(layout, Int[1](), true)

func complement*(layout: Layout; cosizeBound: tuple): auto =
  ## Compute complement with a shape-tuple bound (size converted to product).
  complementEmit(layout, cosizeBound, false)

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
    composeImpl(coalesceBackward(aShape2, aStrides2, true), bShape2, bStrides2)
  template composeDelegatePlain(aShape2, aStrides2, bShape2, bStrides2) =
    composeImpl(make_layout(aShape2, aStrides2), bShape2, bStrides2)

  let aShapeIsTuple = layoutTypeArgs(a).shapeTy.kind in {nnkTupleConstr, nnkTupleTy}
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
    let tilerTy = tiler.getTypeInst()[k]
    if tilerTy.eqIdent("X"):
      # the profiler mark passes the dimension through whole
      shapes.add aShapeK
      strides.add aStrideK
    elif tilerTy.kind == nnkBracketExpr and tilerTy[0].eqIdent("Layout"):
      # a layout tiler element composes the dimension once
      let dk = result.newLetAsgn("composedDim",
        getAst(composeTilerDim(aShapeK, aStrideK, tiler[k])))
      shapes.add dk.newDotExpr(ident"shape")
      strides.add dk.newDotExpr(ident"stride")
    else:
      # an int tiler element composes the dimension with (N):(1),
      # the first N positions: the pair is (N, the dimension's stride)
      shapes.add tiler[k]
      strides.add aStrideK
  if shapes.len == 1:
    result.add bindSym"make_layout".newCall(shapes[0], strides[0])
  else:
    result.add bindSym"make_layout".newCall(nnkPar.newTree(shapes), nnkPar.newTree(strides))

# ═══════════════════════════════════════════════════════════════
#  logical_divide, tile a layout into (tile, rest)
# ═══════════════════════════════════════════════════════════════

func logical_divide_impl[A, B: Layout](layout: A; tiler: B): auto =
  ## Complement the tiler up to the layout size, then compose.
  block:
    mixin comp, combined
    evalOnceAs(comp, complement(tiler, size(layout)))
    evalOnceAs(combined, make_layout((tiler.shape, comp.shape), (tiler.stride, comp.stride)))
    compose(layout, combined)

func logical_divide*[L, T: Layout](layout: L; tiler: T): auto =
  ## Logical divide by a layout tiler.
  logical_divide_impl(layout, tiler)

func logical_divide*[L: Layout](layout: L; tiler: int): auto {.inline.} =
  ## Logical divide by a dynamic int tiler.
  when layout.shape isnot tuple:
    # Rank-1 (s):(d) divides by T into (T):(d) and (ceil_div(s,T)):(d*T)
    make_layout((tiler, ceil_div(layout.shape, tiler)),
                (layout.stride, layout.stride * tiler))
  else:
    logical_divide_impl(layout, make_layout(tiler))

func logical_divide*[L: Layout; V: static int](layout: L; tiler: Int[V]): auto {.inline.} =
  ## Logical divide by a static int tiler.
  when layout.shape isnot tuple:
    # Rank-1 (s):(d) divides by T into (T):(d) and (ceil_div(s,T)):(d*T)
    make_layout((tiler, ceil_div(layout.shape, tiler)),
                (layout.stride, layout.stride * tiler))
  else:
    logical_divide_impl(layout, make_layout(tiler))

func logical_divide*[L: Layout](layout: L; tiler: static int): auto {.inline.} =
  ## Logical divide by a compile-time int tiler.
  logical_divide_impl(layout, make_layout(Int[tiler]()))

macro logical_divide*(layout: Layout; tiler: tuple): untyped =
  ## Divides the layout by the tiler, one tiler element per dimension.
  ##
  ## - a divided dimension becomes the (tile, rest) pair
  ## - dimensions past the tiler length pass through unchanged
  template logicalDivideT(l, t) =
    transform_layout(l, t):
      logical_divide(it_l, it_t)
  getAst(logicalDivideT(layout, tiler))

# ═══════════════════════════════════════════════════════════════
#  hier_unzip, split a layout dimension by dimension, gather tiles and rest
# ═══════════════════════════════════════════════════════════════

macro hier_unzip*(splitter: untyped; layout: typed; tiler: typed): untyped =
  ## Split `layout` by `tiler` through `splitter` and gather the parts into one rank-2 Layout:
  ## - dimension 0 carries the tile parts of every tiler element
  ## - dimension 1 carries the rest parts plus the leftover dimensions, PyCute hier_unzip chain semantics
  ## - a scalar (int, Int) or Layout tiler becomes `splitter(layout, tiler)` verbatim, a sub-tuple tiler element recurses
  ## Usage:
  ##   let r = hier_unzip(logical_divide, make_layout((4, 8), (1, 4)), (2, 4))
  ##   doAssert r === (((2, 4), (2, 2)), ((1, 4), (2, 16)))
  let splitterNode = splitter
  proc dimensionCall(e: NimNode; idx: int): NimNode =
    ## `e.dimension(idx)` as a method-call node.
    let dim = ident"dimension"
    result = newCall(nnkDotExpr.newTree(e, dim), newLit(idx))
  proc fieldElem(e: NimNode; f: string; idx: int): NimNode =
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
    return newCall(splitterNode, layout, tiler)
  let tilerRank = tlrTy.len
  doAssert tilerRank <= R,
    "hier_unzip: tiler has more dimensions (" & $tilerRank & ") than the layout (" & $R & ")"

  var stmts = newStmtList()
  var bindingCount = 0
  proc freshAlias(): NimNode {.compileTime.} =
    inc bindingCount
    ident("huzSplit" & $bindingCount)

  type Parts = tuple[fsh, fst, ssh, sst: seq[NimNode]]

  proc walk(e, eShapeTy, tval, ty: NimNode; needBinding: static bool): Parts {.compileTime.} =
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
      stmts.add newCall(bindSym"evalOnceAs", leafR,
        newCall(splitterNode, e, tval))
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

macro rightInverseEmit(sh, st: typed): untyped =
  ## right_inverse core, emits coalesced contiguous chains in stride order:
  ## - a dimension joins when its stride equals the chain span
  ## - a dynamic shape ends the chain and keeps its own value node
  ## - an empty chain collapses to the empty layout (1, 0)
  var dims: seq[tuple[stride, shape, prefix: int, leaf: NimNode]]
  var prefix = 1
  for (shEv, stEv) in sh.tupleStream().zip(st.tupleStream()):
    if shEv.kind != kLeaf:
      continue
    let shape = shEv.leafTy.getStaticInt()
    dims.add (stEv.leafTy.getStaticInt(), shape, prefix, shEv.leaf)
    # past a dynamic shape the prefix is unknown, no sentinel arithmetic
    prefix = if shape == DynamicSentinel: DynamicSentinel
             else: prefix * shape

  var builder = TupleBuilderFlat.new(2)
  var curr = 1
  for idx in getIndicesSortedByStride(dims.mapIt(it.stride)):
    let dim = dims[idx]
    if dim.stride == curr:
      let shLeaf = if dim.shape == DynamicSentinel: dim.leaf else: IntCT(dim.shape)
      builder.append(shLeaf, IntCT(dim.prefix))
      if dim.shape == DynamicSentinel:
        break
      curr = dim.stride * dim.shape
  result = builder.emitLayout(bindSym"coalesceBackward").resultLayout

macro right_inverse*(layout: typed): untyped =
  ## Quasi-inverse, the largest injective R with L(R(i)) == i.
  ## Returns:
  ## - a coalesced Layout, typically lower rank than L
  ## - (1, 0) when no chain exists
  var stmts = newStmtList()
  let (sh, st) = destructureLayout(stmts, layout)
  template rightInverseDelegate(sh2, st2) =
    ## Expands the inverse core on the destructured tuples at the use site.
    rightInverseEmit(sh2, st2)
  result = stmts
  result.add getAst(rightInverseDelegate(sh, st))

# ═══════════════════════════════════════════════════════════════
#  left_inverse, left inverse (injective layouts only)
# ═══════════════════════════════════════════════════════════════

macro leftInverseEmit(sh, st: typed): untyped =
  ## Left-inverse dimensions built from stride ratios:
  ##
  ##   result_shape[i]  = stride / size_so_far
  ##   result_stride[i] = shape prefix of the previous stride-sorted dimension
  ##
  ## All strides must be static (compile-time assert).
  var dims: seq[tuple[stride, shape, prefix: int, leaf: NimNode]]
  var prefix = 1
  for (shEv, stEv) in sh.tupleStream().zip(st.tupleStream()):
    if shEv.kind != kLeaf:
      continue
    let shape = shEv.leafTy.getStaticInt()
    dims.add (stEv.leafTy.getStaticInt(), shape, prefix, shEv.leaf)
    # past a dynamic shape the prefix is unknown, no sentinel arithmetic
    prefix = if shape == DynamicSentinel: DynamicSentinel else: prefix * shape

  var builder = TupleBuilderFlat.new(2)
  var sizeSoFar = 1
  var prevIdx = -1
  var prevPrefix = 0
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
      builder.append(IntCT(gap), IntCT(prevPrefix))
    sizeSoFar = dim.stride
    prevIdx = idx
    prevPrefix = dim.prefix
  if prevIdx >= 0:
    # tail = the last stride-sorted dimension's own shape
    let dim = dims[prevIdx]
    let shLeaf = if dim.shape == DynamicSentinel: dim.leaf else: IntCT(dim.shape)
    builder.append(shLeaf, IntCT(dim.prefix))
  result = builder.emitLayout(bindSym"coalesceBackward").resultLayout

macro left_inverse*(layout: typed): untyped =
  ## Left inverse, Li(L(i)) == i for injective layouts.
  ## Returns:
  ## - a coalesced Layout over the static-stride gaps
  ## - requires all static strides, compile-time assert
  var stmts = newStmtList()
  let (sh, st) = destructureLayout(stmts, layout)
  template leftInverseDelegate(sh2, st2) =
    ## Coalesce canonicalizes strides first, the chaining asserts require it.
    evalOnceAs(coalescedLayout, coalesceBackward(sh2, st2))
    leftInverseEmit(coalescedLayout.shape, coalescedLayout.stride)
  result = stmts
  result.add getAst(leftInverseDelegate(sh, st))


# ═══════════════════════════════════════════════════════════════
#  logical_product, reproduce a block over a tiler
# ═══════════════════════════════════════════════════════════════

func logical_product*[A, B: Layout](a: A; tiler: B): auto =
  ## Reproduce block over tiler: rank-2 result ((BLOCK), (TILE)).
  ## Inverse of logical_divide.
  let rest = compose(complement(a, size(a) * cosize(tiler)), tiler)
  make_layout((a.shape, rest.shape), (a.stride, rest.stride))


func nested_product*[A, B: Layout](a: A; b: B): auto =
  ## Categorical product of two layouts, preserving each argument's dimension grouping.
  ##
  ## Given:
  ##   A: (a0, a1, ...):(sa0, sa1, ...)
  ##   B: (b0, b1, ...):(sb0, sb1, ...)
  ## Returns:
  ##   ((a0, a1, ...), (b0, b1, ...)) : ((sa0, sa1, ...), (sb0, sb1, ...))
  make_layout((a.shape, b.shape), (a.stride, b.stride))


# ── zipped_product / tiled_product / flat_product ──

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

