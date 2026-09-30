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
import ./layouts
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
  for (leaf, ty) in flatLeavesRev(layoutShape):
    shLeaves.add leaf
    shTypes.add ty
  for (leaf, ty) in flatLeavesRev(layoutStride):
    stLeaves.add leaf
    stTypes.add ty

  if shLeaves.len == 1 and stLeaves.len == 1:
    if isStaticOne(shTypes[0]):
      result = newCall(bindSym"make_layout", newLit(1), newLit(0))
    else:
      result = newCall(bindSym"make_layout", shLeaves[0], stLeaves[0])
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
        getStaticInt(shTypes[k]) * getStaticInt(stTypes[k]) == getStaticInt(head.strideTy):
      let mergedVal = getStaticInt(shTypes[k]) * getStaticInt(head.shapeTy)
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
    result = newCall(bindSym"make_layout", IntCT(1), newLit(0))
    return

  var rShape = newNimNode(nnkTupleConstr)
  var rStride = newNimNode(nnkTupleConstr)
  for idx in countdown(chunks.len - 1, 0):
    rShape.add chunks[idx].shape
    rStride.add chunks[idx].stride
  if rShape.len == 1:
    rShape = rShape[0]
    rStride = rStride[0]

  result = newCall(bindSym"make_layout", rShape, rStride)

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
      result = newCall(bindSym"make_layout", boundExpr, newLit(1))
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
      result = newCall(bindSym"make_layout", bound, newLit(1))
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

proc emitLet(body: NimNode; name: string; expr: NimNode): NimNode {.compileTime.} =
  ## Bind `expr` to a fresh let symbol appended to `body`, return the symbol.
  let sym = genSym(nskLet, name)
  body.add nnkLetSection.newTree(
    nnkIdentDefs.newTree(sym, newEmptyNode(), expr))
  sym

proc staticVal(t: NimNode): int {.compileTime.} =
  ## Static Int value of a type node, DynamicSentinel when not static.
  if isStaticInt(t):
    getStaticInt(t)
  else:
    DynamicSentinel

proc composeFold(lhsShLeaves, lhsStLeaves, lhsShTys, lhsStTys: seq[NimNode];
                 remSh, remSt: NimNode; remShV, remStV: int): NimNode {.compileTime.} =
  ## Composition fold over LHS dimensions as one flat let-chain:
  ## - remaining shape and stride carry static values beside their nodes
  ## - a static-1 next shape or remaining shape skips the dimension
  ## - otherwise the clamped remainder accumulates one (shape, stride) pair
  let R = lhsShLeaves.len
  var remShN = remSh
  var remStN = remSt
  var shV = remShV
  var stV = remStV
  var accSh, accSt: seq[NimNode]
  let body = newNimNode(nnkStmtList)
  for k in 0 ..< R:
    if k == R - 1:
      let scaled = nnkInfix.newTree(ident"*", remStN, lhsStLeaves[k])
      if accSh.len == 0:
        return nnkStmtListExpr.newTree(body, bindSym"make_layout".newCall(
          bindSym"unwrap".newCall(nnkTupleConstr.newTree(remShN)),
          bindSym"unwrap".newCall(nnkTupleConstr.newTree(scaled))))
      if shV == 1:
        return nnkStmtListExpr.newTree(body, bindSym"make_layout".newCall(
          bindSym"unwrap".newCall(nnkTupleConstr.newTree(accSh)),
          bindSym"unwrap".newCall(nnkTupleConstr.newTree(accSt))))
      return nnkStmtListExpr.newTree(body, bindSym"make_layout".newCall(
        bindSym"unwrap".newCall(nnkTupleConstr.newTree(accSh & @[remShN])),
        bindSym"unwrap".newCall(nnkTupleConstr.newTree(accSt & @[scaled]))))
    let shVk = staticVal(lhsShTys[k])
    let stVk = staticVal(lhsStTys[k])
    let absRemV = if stV != DynamicSentinel: abs(stV) else: DynamicSentinel
    let nextShV = if shVk != DynamicSentinel and absRemV != DynamicSentinel:
      ceil_div(shVk, absRemV)
    else:
      DynamicSentinel
    let currShape = lhsShLeaves[k]
    let absRem = body.emitLet("absRem", bindSym"abs".newCall(remStN))
    if nextShV != 1 and shV != 1:
      let clampedV = if nextShV != DynamicSentinel and shV != DynamicSentinel:
        min(nextShV, shV)
      else:
        DynamicSentinel
      if clampedV != DynamicSentinel:
        doAssert shV mod clampedV == 0,
          "compose: shape " & $shV & " and consumed shape " & $clampedV & " are not divisible"
      let clamped = body.emitLet("clampedShape", bindSym"min".newCall(
        bindSym"ceil_div".newCall(currShape, absRem), remShN))
      remShN = body.emitLet("remainingShape", nnkInfix.newTree(ident"div", remShN, clamped))
      accSh.add clamped
      accSt.add nnkInfix.newTree(ident"*", remStN, lhsStLeaves[k])
      if clampedV != DynamicSentinel:
        shV = shV div clampedV
    remStN = body.emitLet("remainingStride", nnkInfix.newTree(
      ident"*",
      bindSym"ceil_div".newCall(absRem, currShape),
      bindSym"sign".newCall(remStN)))
    if absRemV != DynamicSentinel and shVk != DynamicSentinel and stV != DynamicSentinel:
      stV = ceil_div(absRemV, shVk) * sign(stV)

macro composeImpl(remainingShape, remainingStride: typed; lhsShapes, lhsStrides: typed): untyped =
  ## Fold over LHS dimensions with a 2-state accumulator, the remaining
  ## shape and stride, emitting one (shape, stride) dimension pair per
  ## unconsumed LHS dimension.
  ##
  ## Returns the composed layout as an untyped node.

  let remSh0 = genSym(nskLet, "remainingShape")
  let remSt0 = genSym(nskLet, "remainingStride")
  let remStV0 = staticVal(remainingStride.getTypeInst())
  let lets = nnkLetSection.newTree(
    nnkIdentDefs.newTree(remSh0, newEmptyNode(), remainingShape),
    nnkIdentDefs.newTree(remSt0, newEmptyNode(), remainingStride))
  # static stride-0 RHS dimension, every coordinate maps to offset 0
  if remStV0 == 0:
    result = nnkStmtListExpr.newTree(
      lets, bindSym"make_layout".newCall(remSh0, remSt0))
  else:
    var lhsShLeaves, lhsStLeaves, lhsShTys, lhsStTys: seq[NimNode]
    for (leaf, ty) in flatLeaves(lhsShapes):
      lhsShLeaves.add leaf
      lhsShTys.add ty
    for (leaf, ty) in flatLeaves(lhsStrides):
      lhsStLeaves.add leaf
      lhsStTys.add ty
    result = nnkStmtListExpr.newTree(lets, composeFold(
      lhsShLeaves, lhsStLeaves, lhsShTys, lhsStTys, remSh0, remSt0,
      staticVal(remainingShape.getTypeInst()), remStV0))

func composeDistribute(lhsShapes, lhsStrides: tuple; rhsShapes, rhsStrides: tuple): auto =
  ## Layer RHS dimensions one by one over the full coalesced LHS via mapDimensionsWith.
  ## Nested RHS dimensions recurse into composeDistribute.
  ## Scalar dimensions go directly to composeImpl.
  mapDimensionsWith(make_layout(rhsShapes, rhsStrides)):
    when it_l.shape is tuple:
      composeDistribute(lhsShapes, lhsStrides, it_l.shape, it_l.stride)
    else:
      composeImpl(it_l.shape, it_l.stride, lhsShapes, lhsStrides)


func compose*[A, B: Layout](a: A, b: B): auto =
  ## Layout composition.
  ##
  ## Returns a layout `R` such that `R(i) = A(B(i))` for all
  ## `i` in `0 ..< cosize(B)`.
  ##
  ## Divisibility of the consumed shape is a caller precondition.
  ## Static leaves assert at compile time.
  ## Runtime shapes are unchecked.
  when a.shape isnot tuple:
    when b.stride is tuple:
      when countLeaves(b.shape) != rank(b.shape):
        composeDistribute((a.shape,), (a.stride,), b.shape, b.stride)
      else:
        make_layout(b.shape, flatMapLeaves(b.stride, it * a.stride))
    else:
      make_layout(flatMapLeaves(b.shape, it), b.stride * a.stride)
  elif b.shape isnot tuple:
    # Coalesce the LHS first, preserving trailing stride-0 dimensions
    let flatA = coalesce_preserve_trailing(a)
    when flatA.shape isnot tuple:
      # flatA is rank-1, the result strides scale by flatA.stride
      make_layout(b.shape, b.stride.scaleBy(flatA.stride))
    else:
      composeImpl(b.shape, b.stride, flatA.shape, flatA.stride)
  else:
    # Coalesce the LHS first, preserving trailing stride-0 dimensions
    let flatA = coalesce_preserve_trailing(a)
    when flatA.shape isnot tuple:
      # flatA is rank-1, preserve B's nesting
      make_layout(b.shape, b.stride.scaleBy(flatA.stride))
    else:
      composeDistribute(flatA.shape, flatA.stride, b.shape, b.stride)

macro compose*(layout: Layout; tiler: tuple): untyped =
  ## Compose with a tiler tuple, one tiler element per dimension.
  ##
  ## Contract:
  ## - dimension i composes with `tiler[i]`
  ## - dimensions past the tiler length drop, CuTe and pycute agree
  ## - a tiler longer than the layout is rejected at expansion
  ##
  ## Tiler elements:
  ## - a Layout composes by layout algebra
  ## - an int/Int takes the dimension's first N positions
  ## - `_` passes the dimension through whole
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
  if tilerRank < R:
    template dropT(l, t) =
      ## Dimensions past the tiler length drop, slice them off first,
      ## transform_layout passes leftovers through otherwise.
      transform_layout(takeDimensions(l, 0, tupleLen(typeof(t))), t):
        when it_t is X:
          it_l
        elif it_t is (int or Int):
          compose(it_l, make_layout(it_t))
        else:
          compose(it_l, it_t)
    result = getAst(dropT(layout, tiler))
  else:
    template keepT(l, t) =
      ## Full-rank tiler, every dimension is hit.
      transform_layout(l, t):
        when it_t is X:
          it_l
        elif it_t is (int or Int):
          compose(it_l, make_layout(it_t))
        else:
          compose(it_l, it_t)
    result = getAst(keepT(layout, tiler))
  if tilerRank == 1:
    # a rank-1 tiler rebuilds flat in CuTe and pycute, unwrap the 1-tuple
    result = newCall(nnkDotExpr.newTree(result, bindSym"dimension"), newLit 0)

# ═══════════════════════════════════════════════════════════════
#  logical_divide, tile a layout into (tile, rest)
# ═══════════════════════════════════════════════════════════════

func logical_divide_impl[A, B: Layout](layout: A; tiler: B): auto =
  ## Complement the tiler up to the layout size, then compose.
  let comp = complement(tiler, size(coalesce(layout)))
  let combined = make_layout((tiler.shape, comp.shape), (tiler.stride, comp.stride))
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
#  tile_unzip, unzip a divide or product result into tiles and rest
# ═══════════════════════════════════════════════════════════════

template tile_unzip*[L: Layout, T](layout: L; tiler: T): auto =
  ## Unzip a logical_divide/logical_product result according to a tiler.
  ## Returns a rank-2 Layout, the tile dimensions and the rest dimensions.
  block:
    evalOnceAs(lyt, layout)
    evalOnceAs(tlr, tiler)
    when tiler is Layout:
      make_layout(
        zip2_by(lyt.shape, tlr.shape),
        zip2_by(lyt.stride, tlr.shape))
    else:
      make_layout(
        zip2_by(lyt.shape, tlr),
        zip2_by(lyt.stride, tlr))

# ═══════════════════════════════════════════════════════════════
func zipped_divide*[LayoutT: Layout, TilerT](layout: LayoutT; tiler: TilerT): auto {.inline.} =
  ## Divide layout by tiler and zip tile/rest dimensions into rank-2 result.
  when TilerT is Layout:
    logical_divide(layout, tiler)
  elif TilerT is int or TilerT is Int:
    logical_divide(layout, tiler)
  else:
    tile_unzip(logical_divide(layout, tiler), tiler)

macro groupedHead(head, tail: typed): untyped =
  ## Tuple (head, tail[0], tail[1], ...) with head verbatim so nesting survives,
  ## tail top-level elements unpacked one level, a scalar tail kept whole.
  ## tiled_divide/tiled_product reassembly, the CuTe `result(_, repeat<R1>(_))` slice.
  result = nnkTupleConstr.newTree(head)
  let tt = tail.getTypeInst()
  if tt.kind notin {nnkTupleTy, nnkTupleConstr} or tt.len == 0:
    result.add tail
    return
  for i in 0 ..< tt.len:
    result.add nnkBracketExpr.newTree(tail, newLit(i))

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

func emitInverse(acc: LayoutCT): NimNode {.compileTime.} =
  ## Coalesce the folded inverse dimensions of both inverses,
  ## an empty fold collapses to the empty layout (1, 0).
  if acc.shape.len == 0:
    bindSym"make_layout".newCall(IntCT(1), newLit(0))
  else:
    bindSym"coalesce".newCall(acc.emit())

type InverseChain = tuple[shape, stride, leafIdx: int]
  ## One inverse chain element as flat values:
  ## - shape, the shape value, DynamicSentinel marks a live leaf
  ## - stride, the stride value
  ## - leafIdx, the flat leaf index in the source shape,
  ##   live only for a DynamicSentinel shape

type InverseFoldFn = proc(strides, shapes, prefixProd: seq[int]): seq[InverseChain] {.nimcall.}

proc getMaxContiguous(
    strides, shapes, prefixProd: seq[int]): seq[InverseChain] =
  ## Maximal contiguous chain of the stride-sorted dimensions, empty if none found:
  ## - a dimension joins when its stride equals the chain span so far
  ## - a dynamic shape ends the chain, the next stride span is undecidable
  var curr = 1
  for idx in getIndicesSortedByStride(strides):
    if strides[idx] == curr:
      result.add((shapes[idx], prefixProd[idx], idx))
      if shapes[idx] == DynamicSentinel:
        break
      curr = strides[idx] * shapes[idx]

proc inverseFold(shTy, stTy, shNode: NimNode;
                 fold: InverseFoldFn): LayoutCT {.compileTime.} =
  ## inverse core shared by both inverses, folds arrive directly:
  ## - fold over the flat values, emit the chain as Int values
  ## - a dynamic shape leaf comes from the value node, a scalar shape is bare
  let shV = toSeqStaticInts(shTy)
  let stV = toSeqStaticInts(stTy)
  for chain in fold(stV, shV, prefixProduct(shV)):
    let leaf = if chain.shape == DynamicSentinel:
      tupleLeaf(shNode, shV.len, chain.leafIdx)
    else:
      IntCT(chain.shape)
    result.append(leaf, IntCT(chain.stride))

macro rightInverseEmit(sh, st: typed): untyped =
  emitInverse(inverseFold(sh.getTypeInst(), st.getTypeInst(), sh, getMaxContiguous))

func right_inverse*(layout: Layout): auto =
  ## Quasi-inverse, the largest injective R with L(R(i)) == i.
  ## Returns:
  ## - a coalesced Layout, typically lower rank than L
  ## - (1, 0) when no chain exists
  let c = coalesce(layout)
  rightInverseEmit(flatten(c.shape), flatten(c.stride))

# ═══════════════════════════════════════════════════════════════
#  left_inverse, left inverse (injective layouts only)
# ═══════════════════════════════════════════════════════════════

proc getGaps(
    strides, shapes, prefixProd: seq[int]): seq[InverseChain] =
  ## Returns the left-inverse dimensions as a LayoutCT, built from stride ratios:
  ##
  ##   result_shape[i]  = stride / size_so_far
  ##   result_stride[i] = prefixProd of the previous stride-sorted dimension
  ##
  ## Left-inverse dimensions as (gap, stride) values plus the tail, built
  ## from stride ratios, all strides must be static (compile-time assert)
  var sizeSoFar = 1
  var prevIdx = -1
  var prevPrefix = 0
  for idx in getIndicesSortedByStride(strides):
    if strides[idx] == 0:
      continue
    doAssert strides[idx] != DynamicSentinel,
      "left_inverse: dynamic strides are not chainable"
    doAssert strides[idx] mod sizeSoFar == 0,
      "left_inverse: stride " & $strides[idx] & " not divisible by " & $sizeSoFar
    let gap = strides[idx] div sizeSoFar
    # a unit gap marks no hole, the final coalesce would drop the shape-1 head
    if gap != 1:
      result.add((gap, prevPrefix, idx))
    sizeSoFar = strides[idx]
    prevIdx = idx
    prevPrefix = prefixProd[idx]
  if prevIdx >= 0:
    # tail = the last stride-sorted dimension's own shape
    result.add((shapes[prevIdx], prevPrefix, prevIdx))

macro leftInverseEmit(sh, st: typed): untyped =
  emitInverse(inverseFold(sh.getTypeInst(), st.getTypeInst(), sh, getGaps))

func left_inverse*(layout: Layout): auto =
  ## Left inverse, Li(L(i)) == i for injective layouts.
  ## Returns:
  ## - a coalesced Layout over the static-stride gaps
  ## - requires all static strides, compile-time assert
  let c = coalesce(layout)
  leftInverseEmit(flatten(c.shape), flatten(c.stride))


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
  ## CuTe: zipped_product = tile_unzip(logical_product(block, tiler), tiler)
  block:
    evalOnceAs(bk, blk)
    evalOnceAs(tlr, tiler)
    when tiler is Layout:
      logical_product(bk, tlr)
    else:
      tile_unzip(logical_product(bk, tlr), tlr)

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
  const mxR = max(rank(type(blk)), rank(type(tiler)))
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
  const mxR = max(rank(type(blk)), rank(type(tiler)))
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

