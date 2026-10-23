## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/algorithm
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/macros/static_for
import ./layouts_datatypes
import ./layout_compiletime
import ./layouts_unsanctioned_helpers
import ./ism_coord_strides
export ism_coord_strides

# ═══════════════════════════════════════════════════════════════
#  col_major_strides, canonical column-major strides
# ═══════════════════════════════════════════════════════════════

func col_major_strides*(shape: IntOrIntTuple): auto =
  ## Canonical column-major strides: prefix_product(shape).
  ## For shape (2,4): strides (1,4).
  prefix_product(shape)

# ═══════════════════════════════════════════════════════════════
#  make_layout
# ═══════════════════════════════════════════════════════════════

# Implementation note for direct AST->AST transformation of constructors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Why no evalOnceAs here?
#
#   For direct AST->AST transformation of constructors we need to manipulate the AST *produced* by compose(complement(...), tiler)
#   In this macro we can only see the macro call AST so we need to defer to another macro
#   so that compose(complement(...), tiler) have the time to do their own AST->AST constructor transformation
#
#   Now one tricky part of this is that using an AST node or a template input in multiple place will paste it verbatim
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

proc make_layout*(shape, stride: NimNode): NimNode {.compileTime.} =
  ## make_layout(shape, stride) as a compile-time node
  ident"make_layout".newCall(shape, stride)

macro make_layout*(shape: typed, order: static StrideOrder = LayoutLeft): untyped =
  ## (doc kept from the template)
  let (shapeVal, shapeTy) = makeIntTuple(shape)
  var strideVal, strideTy: NimNode
  if order == LayoutLeft:
    (strideVal, strideTy) = prefixProductScan(shape, IntCT(1), IntCT(1)[0])
  else:
    (strideVal, strideTy) = suffixProductScan(shape, IntCT(1), IntCT(1)[0])
  result = emitLayoutLit(shapeVal, shapeTy, strideVal, strideTy)

macro make_layout*(shape, stride: typed): untyped =
  ## (doc kept from the template)
  let strideTyInst = stride.getTypeOf()
  if strideTyInst != nil and strideTyInst.isCoordStride():
    let (shapeVal, shapeTy) = makeIntTuple(shape)
    result = quote do:
      Layout[`shapeTy`, `strideTyInst`](shape: `shapeVal`, stride: `stride`)
    return
  let (shapeVal, shapeTy) = makeIntTuple(shape)
  let (strideVal, strideTy) = makeIntTuple(stride)
  let (strideFolded, strideFoldedTy) =
    prefixProductScan(shape, strideVal, strideTy)
  result = emitLayoutLit(shapeVal, shapeTy, strideFolded, strideFoldedTy)

proc compactOrderStridesImpl(shVals, ordVals: seq[int]): seq[int] =
  ## Assign compact strides in ascending order-value order, each stride the prefix product so far. Ties keep left-to-right position.
  result = newSeq[int](shVals.len)
  var pairs = newSeq[(int, int)](shVals.len)
  for i in 0 ..< shVals.len:
    pairs[i] = (ordVals[i], i)
  var current = 1
  for (_, i) in pairs.sorted(system.cmp):
    result[i] = current
    current *= shVals[i]

proc compactOrderDynamicSubstitution(ordVals: openArray[int]): seq[int] =
  ## Dynamic order entries rank after every static entry, each resolving to a unique
  ## value above the largest static one, in left-to-right position.
  let n = ordVals.len
  var maxStatic = -1
  for v in ordVals:
    if v != DynamicSentinel and v > maxStatic:
      maxStatic = v
  var nextVal = maxStatic + 1
  result = newSeq[int](n)
  for i in 0 ..< n:
    if ordVals[i] == DynamicSentinel:
      result[i] = nextVal
      inc nextVal
    else:
      result[i] = ordVals[i]

proc compactOrderPermPos(ordVals: openArray[int]): seq[int] =
  ## Position of each dimension in the (order value, position) ascending sort, ties keeping left-to-right position.
  var pairs = newSeq[(int, int)](ordVals.len)
  for i in 0 ..< ordVals.len:
    pairs[i] = (ordVals[i], i)
  let sortedPairs = pairs.sorted(system.cmp)
  result = newSeq[int](ordVals.len)
  for pos, p in sortedPairs:
    result[p[1]] = pos

proc compactLikeStrides(sh, st: seq[int], scale = 1): seq[int] {.compileTime.} =
  ## Strides of the compact layout preserving an (shape, stride) pair's element-access order.
  ## - stride-0 dimensions collapse to shape 1 and keep stride 0
  ## - dynamic strides take the slowest free positions
  ## - remaining strides scale by `scale`, the number of positions before them, 1 when none
  var fsh = sh
  for i in 0 ..< sh.len:
    if st[i] == 0:
      fsh[i] = 1
  result = compactOrderStridesImpl(fsh, compactOrderDynamicSubstitution(st))
  for i in 0 ..< sh.len:
    if st[i] == 0:
      result[i] = 0
    else:
      result[i] *= scale

func orderLeafAt(order: int, i: static int): int =
  order

func orderLeafAt[V: static int](order: Int[V], i: static int): int =
  V

func orderLeafAt[T: tuple](order: T, i: static int): auto =
  order[i]

func orderLeafAt[I: Ordinal, T](order: array[I, T], i: static int): int =
  order[i]

func orderLeafRank(t: typedesc[int]): static int =
  ## Rank of one order leaf from its type alone.
  ## A runtime leaf is a dynamic order entry, reporting the dynamic sentinel.
  DynamicSentinel

func orderLeafRank[V: static int](t: typedesc[Int[V]]): static int =
  V

func leafAt[T: tuple](t: T, i: static int): auto =
  t[i]

func leafAt[V: static int](t: Int[V], i: static int): Int[V] =
  t

func leafAt(t: int, i: static int): int =
  t

proc productExpr(shape: NimNode, permPos: seq[int], i: int): NimNode {.compileTime.} =
  ## Product of the shape leaves ranked before dimension i, over leaf range [0, i),
  ## an ordinary expression chain: the `Int` overload matrix decides per product
  ## whether it folds to an `Int[...]()` node or computes at runtime.
  result = quote do: Int[1]()
  for j in 0 ..< permPos.len:
    if permPos[j] < permPos[i]:
      let leaf = nnkCall.newTree(bindSym"leafAt", shape, newLit(j))
      result = nnkInfix.newTree(bindSym"*", result, leaf)

func orderedPermPos[O](n: static int, order: O): seq[int] =
  ## Ranks of the order leaves:
  ##
  ## - dynamic entries substituted
  ## - dimension positions sorted ascending
  ## - evaluated at compile time, feeding the scheduling macro as a static argument
  var ranks: array[n, int]
  staticFor i, 0, n:
    ranks[i] = orderLeafAt(order, i)
  compactOrderPermPos(compactOrderDynamicSubstitution(ranks))

macro stridesTuple(shape: typed, permPos: static seq[int]): untyped =
  ## Stride leaves in original dimension order, dimension i taking the product
  ## of the shape leaves ranked before it.
  ##
  ## - the scheduling is compile-time macro work and vanishes with the expansion
  ## - the products are ordinary expressions for the `Int` overload matrix
  ## - the tuple constructor is emitted directly, a fold through `concat` lowers as a call over the callee's parameter symbols
  ##   and dies in device code generation
  result = nnkPar.newTree(productExpr(shape, permPos, 0))
  for i in 1 ..< permPos.len:
    result.add(productExpr(shape, permPos, i))

func compactOrderStrides*[Sh: IntOrIntTuple](shape: Sh, order: static auto): auto =
  ## Compact strides ranked by `order`, one body over IntOrIntTuple operands.
  ##
  ## Contract:
  ## - `order[i]` ranks dimension i: smaller value = faster-varying (smaller stride)
  ## - ties keep left-to-right position
  ## - dynamic entries rank after every static entry
  ## - the order is compile-time per contract
  ##
  ## - static shape leaves are `Int[...]` nodes, their products fold at compile time
  ## - any runtime leaf turns the strides it touches into plain runtime arithmetic
  ## - operands flow through the `Int` overload matrix, no per-leaf dispatch
  ## - order ranks and product scheduling are compile-time, leaf products are
  ##   ordinary expressions evaluated by overload resolution
  const n: int = Sh.rank()
  stridesTuple(shape, orderedPermPos(n, order))

func make_ordered_layout_impl[Sh: IntOrIntTuple](shape: Sh, order: static auto): auto =
  ## Compile-time order: the order values rank the shape leaves directly.
  make_layout(shape, compactOrderStrides(shape, order))

func make_ordered_layout_impl[Sh: IntOrIntTuple, O: tuple](shape: Sh, order: O): auto =
  ## Runtime order leaves. Each int leaf is a dynamic order entry.
  ## Ranks derive from the leaf types, after every static `Int[...]` leaf.
  const n: int = Sh.rank()
  const ranks = block:
    var r: array[n, int]
    staticFor i, 0, n:
      r[i] = orderLeafRank(typeof(order[i]))
    r
  make_layout(shape, compactOrderStrides(shape, ranks))

template make_ordered_layout*(shape, order: untyped): auto =
  ## Construct a compact layout whose strides are ranked by `order`.
  ##
  ## Returns: the layout with the input shape and strides ranked by `order`.
  ##
  ## `order[i]` specifies the position of dimension `i` in the stride ordering:
  ## smaller value = faster-varying (smaller stride).
  ##
  ## - the dimension with `order[i] = 0` gets stride 1, the next stride = shape[fastest], and so forth
  ## - dynamic order entries rank after every static entry, in left-to-right position,
  ##   compacting column-major behind every static rank
  ##
  ## Example, 2D permutations:
  ##   make_ordered_layout((2,3), (0,1))  →  (2,3):(1,2)   # col-major (dimension 0 fastest)
  ##   make_ordered_layout((2,3), (1,0))  →  (2,3):(3,1)   # row-major (dimension 1 fastest)
  ##
  ## Example, tied orders keep left-to-right position:
  ##   make_ordered_layout((2,3), (0,0))  →  (2,3):(1,2)
  ##
  ## Example, 3D custom permutation:
  ##
  ##   make_ordered_layout((2,3,4), (0,2,1))
  ##
  ## - dimension 0 fastest → stride 1
  ## - dimension 2 next → stride 1*2 = 2
  ## - dimension 1 slowest → stride 1*2*4 = 8
  ##   and the result is (2,3,4):(1,8,2)
  ##
  ## Static shape leaves fold to phantom `Int[...]` strides, a runtime leaf
  ## computes its strides at runtime through the same body:
  ##
  ##   make_ordered_layout((rows, 32), (0,1))  # rows: int
  ##   # → (rows, 32):(Int[1], int)
  ##
  ## Expansion happens at the call site. Static promotion rewrites literal leaves
  ## into `Int[...]` nodes before the algebra runs. A runtime leaf stays verbatim.
  ## The two impl overloads differ in order regime only, the stride algebra is one body.
  make_ordered_layout_impl(makeIntTuple(shape), order)

# ═══════════════════════════════════════════════════════════════
#  make_layout_like
# ═══════════════════════════════════════════════════════════════

macro make_layout_likeImpl(sh, st: typed): untyped =
  ## Compact strides preserving the (shape, stride) pair's element-access order.
  ##
  ## Contract:
  ## - the shape passes through verbatim, only the stride is rebuilt
  ## - broadcast dimensions, statically Int[0], keep stride 0
  ## - dynamic strides take the slowest free positions
  ## - a scalar shape keeps a scalar stride, a 1-tuple shape keeps a 1-tuple stride
  let shapeIsTuple = sh.getTypeInst().isTupleTy()
  var shVals, stVals: seq[int] = @[]
  for (leaf, leafTy) in sh.tupleStream().leaves():
    shVals.add leafTy.getStaticInt()
  for (leaf, leafTy) in st.tupleStream().leaves():
    stVals.add leafTy.getStaticInt()
  doAssert shVals.len == stVals.len,
    "make_layout_like: shape/stride rank mismatch"
  let strides = compactLikeStrides(shVals, stVals)
  var builder = TupleBuilderNested.new(1, dropEmpty = false)
  var i = 0
  for ev in sh.tupleStream():
    if ev.kind == kLeaf:
      builder.append((IntCT(strides[i]), intTy(strides[i])))
      i.inc
    else:
      builder.append(ev)
  let (shVal, shTy) = makeIntTuple(sh)
  return emitLayoutLit(shVal, shTy,
    builder.emit(0, emitScalarForSize1 = not shapeIsTuple),
    builder.emitTy(0, emitScalarForSize1 = not shapeIsTuple))

macro make_layout_like*(layout: Layout): untyped =
  ## Create a compact layout with the same shape and element-access order.
  ##
  ## Produce a compact layout that accesses elements in the same logical
  ## order as the input, compaction order comes from the input strides.
  ## Broadcast dimensions (statically Int[0]) keep stride 0.
  ##
  ## Example, non-compact (2,1) gives compact row-major (3,1):
  ##
  ##   make_layout_like(make_layout((2,3), (2,1)))  # → (2,3):(3,1)
  ##   # dimension 1 has the smaller stride (1), so it becomes fastest
  ##   # and dimension 0 gets stride shape[1] = 3
  ##
  ## Example, broadcast dimension preserved:
  ##   make_layout_like(make_layout((2,3), (0,1)))  # → (2,3):(0,1)
  ##
  ## Example, 3D reordering, (2,3,4):(3,6,1) gives (2,3,4):(4,8,1):
  ## - dimension 2 (stride 1) fastest → stride 1
  ## - dimension 0 (stride 3) middle → stride 1*4 = 4
  ## - dimension 1 (stride 6) slowest → stride 1*4*2 = 8

  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  template likeDelegate(sh2, st2) =
    make_layout_likeImpl(sh2, st2)
  result.add getAst(likeDelegate(sh, st))

# ═══════════════════════════════════════════════════════════════
#  make_identity_layout, coordinate strides as strides
# ═══════════════════════════════════════════════════════════════

template make_identity_layout*(shape: IntOrIntTuple): Layout =
  make_layout(shape, make_basis_like(shape))

# ═══════════════════════════════════════════════════════════════
#  make_fragment_like
# ═══════════════════════════════════════════════════════════════

macro make_fragment_like*(layout: Layout): untyped =
  ## Register-buffer layout for a partition view.
  ##
  ## Contract:
  ## - dimension 0 = the registers each thread owns, packed dense col-major (stride-1 chain)
  ## - broadcast registers (cosize 1, all strides 0) keep the zero strides verbatim
  ## - dimensions 1.. keep the view's stride order, compacted, scaled after the registers,
  ##   same size and flat access order as the view so view and fragment copies match
  ##
  ## Precondition, static shape and stride, the register part compact col-major or all-zero
  result = newStmtList()
  let (sh, st) = result.destructureLayout(layout)
  let originalLayout = if result.len == 0: layout else: result[^1][1]
  let dim0Shape = getTupleIndex(sh, 0)
  let dim0Stride = getTupleIndex(st, 0)
  result.add quote do:
    when `sh`.rank() == 1:
      make_layout(`sh`)
    else:
      when cosize(typeof(make_layout(`dim0Shape`, `dim0Stride`))) == 1:
        tiled_product(make_layout(`dim0Shape`, `dim0Stride`),
          make_layout_like(takeDimensionsImpl(`originalLayout`, `sh`, `st`, 1, `sh`.rank())))
      else:
        tiled_product(make_layout(`dim0Shape`),
          make_layout_like(takeDimensionsImpl(`originalLayout`, `sh`, `st`, 1, `sh`.rank())))
