# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/macros/replace_nodes
import ./layouts_datatypes
import ./layout_constructors
import ./layouts_unsanctioned_helpers
import ./layout_compiletime
import ./layouts_unsanctioned_helpers

export layouts_datatypes
export layout_constructors

# ═══════════════════════════════════════════════════════════════
#  dimension, extract dimension as rank-1 Layout
# ═══════════════════════════════════════════════════════════════

macro dimensionImpl(l, sh, st: typed, idx: static int): untyped =
  if sh.isTupleTy():
    nnkCall.newTree(bindSym"make_layout", getTupleIndex(sh, idx), getTupleIndex(st, idx))
  else:
    doAssert idx == 0, "dimension: scalar layout only has dimension 0"
    l

macro dimension*(layout: Layout, idx: static int): untyped =
  ## Extract dimension `idx` as a standalone rank-1 Layout.
  ## For scalar layouts (rank-1), only idx=0 is valid.
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  let originalLayout = if result.len == 0: layout
                      else: result[^1][1]
  template dimensionDelegate(l2, sh2, st2, idx2) =
    dimensionImpl(l2, sh2, st2, idx2)
  result.add getAst(dimensionDelegate(originalLayout, sh, st, newLit(idx)))


# ═══════════════════════════════════════════════════════════════
#  isCompact, check if strides match canonical col-major ordering
# ═══════════════════════════════════════════════════════════════

func isCompact*(layout: Layout): bool =
  ## True when strides match canonical column-major ordering.
  ## Does not coalesce first, size-1 dimensions may cause false negatives.
  layout === (layout.shape, col_major_strides(layout.shape))

func isCompact*(layout: static Layout): static bool {.inline.} =
  ## True when strides match canonical column-major ordering.
  ## Does not coalesce first, size-1 dimensions may cause false negatives.
  layout === (layout.shape, col_major_strides(layout.shape))

# ═══════════════════════════════════════════════════════════════
#  Padding
# ═══════════════════════════════════════════════════════════════


macro padRightImpl(originalLayout, sh, st: typed, rank: static int): untyped =
  ## Pass every outer dimension through whole, append `(1, 0)` pads after.
  var builder = TupleBuilderFlat.new(2)
  var dimCount = 0
  for (shapeEvent, strideEvent) in sh.tupleDimsStream().zip(st.tupleDimsStream()):
    builder.append(shapeEvent.leaf, strideEvent.leaf, verbatim = true)
    inc dimCount
  if dimCount >= rank:
    result = originalLayout
    return
  for i in dimCount ..< rank:
    builder.append(IntCT(1), IntCT(0))
  let (node, verbatim) = builder.emitLayout()
  if verbatim:
    result = originalLayout
  else:
    result = node

macro padRight*(layout: Layout; rank: static int): untyped =
  ## Extend layout to target rank by padding with identity dimensions (1, 0).
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  let originalLayout = if result.len == 0: layout
                       else: result[^1][1] # returned `let`/`const` symbol
  result.add bindSym"padRightImpl".newCall(originalLayout, sh, st, newLit(rank))

macro padLeftImpl(originalLayout, sh, st: typed, rank: static int): untyped =
  ## Prepend `(1, 0)` pads, pass every outer dimension through whole after them.
  var builder = TupleBuilderFlat.new(2)
  var dimCount = 0
  for (shapeEvent, strideEvent) in sh.tupleDimsStream().zip(st.tupleDimsStream()):
    builder.append(shapeEvent.leaf, strideEvent.leaf, verbatim = true)
    inc dimCount
  if dimCount >= rank:
    result = originalLayout
    return
  var padShape, padStride: seq[NimNode]
  for i in dimCount ..< rank:
    padShape.add IntCT(1)
    padStride.add IntCT(0)
  builder.prependBatch(padShape, padStride)
  let (node, verbatim) = builder.emitLayout()
  if verbatim:
    result = originalLayout
  else:
    result = node

macro padLeft*(layout: Layout; rank: static int): untyped =
  ## Extend layout to target rank by prepending identity dimensions (1, 0).
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  let originalLayout = if result.len == 0: layout
                       else: result[^1][1] # returned `let`/`const` symbol
  result.add bindSym"padLeftImpl".newCall(originalLayout, sh, st, newLit(rank))

# ═══════════════════════════════════════════════════════════════
#  mapLeavesWith, apply body to each leaf (shape, stride) pair
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
    let blockExpr = nnkBlockExpr.newTree(
      newEmptyNode(), replaceNodes(body, ("it_sh", shExpr), ("it_st", stExpr)))
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
  stmts.add bindSym"make_layout".newCall(outSh, outSt)
  return nnkBlockExpr.newTree(newEmptyNode(), stmts)


# ═══════════════════════════════════════════════════════════════
#  upcast / downcast, reinterpret layout at coarser/finer granularity
# ═══════════════════════════════════════════════════════════════

template upcast*(layout: Layout; N: static int): auto =
  ## Reinterpret layout from finer to coarser granularity.
  ##
  ## Every N consecutive elements become one coarse element.
  ## shape shrinks, stride adjusts (not simply shape/N, stride*N).
  ##
  ## Examples:
  ##   upcast<4>(make_layout(32, 1))  # → (8, 1)  32 int8 → 8 int32
  ##   upcast<4>(make_layout(8, 2))   # → (4, 1)  strided int8 → int32

  mapLeavesWith(layout):
    when it_st is Int:
      when it_st.V == 0:
        (it_sh, it_st)
      else:
        # The stride is a multiple of N or N is a multiple of the stride
        static:
          doAssert abs(it_st.V) mod N == 0 or N mod abs(it_st.V) == 0,
            "upcast: stride " & $it_st.V & " and granularity " & $N & " are not divisible"
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
#  zipDimensions, interleave corresponding dimensions of two layouts
# ═══════════════════════════════════════════════════════════════

macro zipDimensionsImpl(ash, ast, bsh, bst: typed): untyped =
  proc zipPair(a, b: NimNode): NimNode =
    # The pair tree, one (a_i, b_i) tuple leaf per dimension pair.
    var builder = TupleBuilderNested.new(1)
    var zs = zip(a.tupleStream(), b.tupleStream())
    while not zs.done():
      let (ea, eb) = zs.next()
      case ea.kind
      of kOpen, kClose:
        builder.append(ea)
      of kLeaf:
        builder.append(nnkTupleConstr.newTree(ea.leaf, eb.leaf))
    builder.emit(0).resultTuple
  # a scalar broadcasts, its pair is bare, the stream's size-1 wrapper stays out
  let zShape = if ash.isTupleTy(): zipPair(ash, bsh)
               else: nnkTupleConstr.newTree(ash, bsh)
  let zStride = if ast.isTupleTy(): zipPair(ast, bst)
                else: nnkTupleConstr.newTree(ast, bst)
  return bindSym"make_layout".newCall(zShape, zStride)

macro zipDimensions*[A, B: Layout](a: A, b: B): untyped =
  ## Zip dimensions of two layouts: interleave corresponding dimensions pairwise.
  ##
  ##   Given layouts A with dimensions (a0, a1, ..., aN) and
  ##   B with dimensions (b0, b1, ..., bN), zipDimensions produces a layout
  ##   with dimensions ((a0,b0), (a1,b1), ..., (aN,bN)).
  ##
  ##   For rank-1 inputs: (a:b, x:y) → ((a,x):(b,y))

  result = newStmtList()
  let (aSh, aSt) = result.destructureLayout(a)
  let (bSh, bSt) = result.destructureLayout(b)
  template zipDelegate(ash2, ast2, bsh2, bst2) =
    zipDimensionsImpl(ash2, ast2, bsh2, bst2)
  result.add getAst(zipDelegate(aSh, aSt, bSh, bSt))


# ═══════════════════════════════════════════════════════════════
#  selection macros, group/take/select/replace dimensions
# ═══════════════════════════════════════════════════════════════

macro groupDimensionsImpl(originalLayout, sh, st: typed, B, E: static int): untyped =
  ## Wrap outer dimensions `[B, E)` into one nested sub-tuple, pass the rest whole.
  let (shapeLeaves, strideLeaves, dimCount) = streamDims(sh, st)
  doAssert B >= 0, "groupDimensions: B must be a valid dimension index"
  let endIdx = min(E, dimCount)
  if B == 0 and endIdx == dimCount:
    result = originalLayout
    return
  var builder = TupleBuilderNested.new(2)
  let openEvent = TupleStreamEvent(verbatim: true, path: @[], kind: kOpen)
  let closeEvent = TupleStreamEvent(verbatim: true, path: @[], kind: kClose)
  builder.append(openEvent, openEvent)
  for i in 0 ..< B:
    builder.append(shapeLeaves[i], strideLeaves[i], verbatim = true)
  var grouped = TupleBuilderFlat.new(2)
  for i in B ..< endIdx:
    grouped.append(shapeLeaves[i], strideLeaves[i], verbatim = true)
  builder.append(grouped.emit(0).resultTuple, grouped.emit(1).resultTuple)
  for i in endIdx ..< dimCount:
    builder.append(shapeLeaves[i], strideLeaves[i], verbatim = true)
  builder.append(closeEvent, closeEvent)
  result = builder.emitLayout().resultLayout

macro groupDimensions*(layout: Layout; B, E: static int): untyped =
  ## Wraps dimensions at indices `[B, E)` into a nested sub-tuple in both
  ## shape and stride, producing a higher-rank Layout.
  ##
  ## Examples:
  ##   groupDimensions(make_layout((2, 3, 5, 7)), 0, 2)
  ##   # → ((2, 3), 5, 7):((1, 2), 6, 30)
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  let originalLayout = if result.len == 0: layout
                       else: result[^1][1] # returned `let`/`const` symbol
  result.add bindSym"groupDimensionsImpl".newCall(originalLayout, sh, st, newLit(B), newLit(E))

macro takeDimensions*(layout: Layout; B, E: static int): untyped =
  ## Extract dimensions in range `[B, E)` into a new Layout.
  ## Returns a scalar Layout if only one dimension is extracted.
  ##
  ## Examples:
  ##   takeDimensions(make_layout((2, 3, 5, 7)), 1, 3)
  ##   # → (3, 5):(2, 6)
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  let originalLayout = if result.len == 0: layout
                       else: result[^1][1] # returned `let`/`const` symbol
  result.add bindSym"takeDimensionsImpl".newCall(originalLayout, sh, st, newLit(B), newLit(E))

macro selectDimensionsImpl(originalLayout, sh, st: typed, Is: varargs[int]{lit|`const`}): untyped =
  ## Append the indexed outer dimensions whole into the extracted layout.
  var dimCount = 0
  var counter = sh.tupleDimsStream()
  while not counter.done():
    discard counter.next()
    inc dimCount
  var whole = Is.len == dimCount
  for i in 0 ..< Is.len:
    if Is[i].intVal != i:
      whole = false
  if whole:
    result = originalLayout
    return
  var builder = TupleBuilderFlat.new(2)
  for i in 0 ..< Is.len:
    builder.append(getTupleIndex(sh, Is[i].intVal), getTupleIndex(st, Is[i].intVal))
  result = builder.emitLayout().resultLayout

macro selectDimensions*(layout: Layout, Is: varargs[int]{lit|`const`}): untyped =
  ## Extract specific dimension indices into a new Layout.
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  let originalLayout = if result.len == 0: layout
                       else: result[^1][1] # returned `let`/`const` symbol
  var call = bindSym"selectDimensionsImpl".newCall(originalLayout, sh, st)
  for i in 0 ..< Is.len:
    call.add newLit(int(Is[i].intVal))
  result.add call

macro replaceDimensionImpl(sh, st, xShape, xStride: typed, N: static int): untyped =
  ## Slot `N` takes the replacement's dimension, every other slot passes whole.
  let (shapeLeaves, strideLeaves, dimCount) = streamDims(sh, st)
  doAssert N < dimCount, "replaceDimension: slot out of range"
  var builder = TupleBuilderFlat.new(2)
  for i in 0 ..< dimCount:
    if i == N:
      builder.append(xShape, xStride, verbatim = true)
    else:
      builder.append(shapeLeaves[i], strideLeaves[i], verbatim = true)
  return builder.emitLayout().resultLayout

macro replaceDimension*(layout: Layout; x: typed; N: static int): untyped =
  ## Replace dimension N of layout with Layout x.
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  let (xSh, xSt) = result.destructureLayout(x)
  result.add bindSym"replaceDimensionImpl".newCall(sh, st, xSh, xSt, newLit(N))
