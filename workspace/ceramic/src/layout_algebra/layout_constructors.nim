## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/algorithm
import workspace/ceramic/src/int_tuples
import ./layouts_datatypes
import ./layout_compiletime
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

template make_layout*(shapeArg: IntOrIntTuple; order: static StrideOrder = LayoutLeft): auto =
  # Implementation note:

  when order == LayoutLeft:
    Layout[typeof(makeIntTuple(shapeArg)), typeof(prefix_product(convShape))](
      shape: makeIntTuple(shapeArg),
      stride: prefix_product(convShape)
    )
  else:
    Layout[typeof(convShape), typeof(suffix_product(convShape))](
      shape: convShape,
      stride: suffix_product(convShape)
    )

template make_layout*[ShT, StT: IntOrIntTuple or CoordStride](shapeArg: ShT, strideArg: StT): auto =
  when StT is CoordStride:
    Layout[typeof(makeIntTuple(shapeArg)), StT](
      shape: makeIntTuple(shapeArg),
      stride: strideArg
    )
  elif StT is int or StT is Int:
    Layout[typeof(makeIntTuple(shapeArg)),
           typeof(prefix_scanIt(makeIntTuple(shapeArg),
                                makeIntTuple(strideArg), acc * it))](
      shape: makeIntTuple(shapeArg),
      stride: prefix_scanIt(makeIntTuple(shapeArg),
                            makeIntTuple(strideArg), acc * it)
    )
  else:
    Layout[typeof(makeIntTuple(shapeArg)), typeof(makeIntTuple(strideArg))](
      shape: makeIntTuple(shapeArg),
      stride: makeIntTuple(strideArg)
    )

# ═══════════════════════════════════════════════════════════════
#  make_ordered_layout, strides following a dimension ordering
# ═══════════════════════════════════════════════════════════════

proc compactOrderStridesImpl(shVals, ordVals: seq[int]): seq[int] {.compileTime.} =
  ## Assign compact strides in ascending order-value order, ties keep
  ## left-to-right position, each stride the prefix product so far.
  result = newSeq[int](shVals.len)
  var pairs = newSeq[(int, int)](shVals.len)
  for i in 0 ..< shVals.len:
    pairs[i] = (ordVals[i], i)
  var current = 1
  for (_, i) in pairs.sorted(system.cmp):
    result[i] = current
    current *= shVals[i]

proc compactOrderDynamicSubstitution(ordVals: seq[int]): seq[int] {.compileTime.} =
  ## Resolve dynamic order entries to unique values larger than any static
  ## order value, preserving their relative position.
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

proc compactLikeStrides(sh, st: seq[int]; scale = 1): seq[int] {.compileTime.} =
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

proc orderedStaticInt(leaf, leafTy: NimNode): int {.compileTime.} =
  ## Returns: the static value of one shape/order leaf,
  ## DynamicSentinel when the leaf is a runtime value.
  ##
  ## Leaf value sources, in order:
  ## - literals and Int[N] nodes, the value sits in the node
  ## - the leaf's type, a tuple variable's Int[N] leaf is an indexing
  ##   expression and carries its static value in the type
  ## - a const definition, `const order = (1, 0)`, each leaf becomes
  ##   an indexing of a const symbol whose impl carries the value
  result = leaf.getStaticInt()
  if result != DynamicSentinel:
    return
  result = leafTy.getStaticInt()
  if result != DynamicSentinel:
    return
  if leaf.kind in {nnkBracketExpr, nnkCall} and leaf.len == 2 and
      leaf[0].kind == nnkSym and leaf[0].symKind == nskConst and
      leaf[1].kind == nnkIntLit:
    let impl = leaf[0].getImpl()
    if impl.kind == nnkConstDef and impl[2].kind in {nnkTupleConstr, nnkPar}:
      let val = impl[2][int(leaf[1].intVal)]
      if val.kind == nnkIntLit:
        return int(val.intVal)
  result = DynamicSentinel

macro make_ordered_layout*(shape, order: typed): untyped =
  ## Construct a compact layout whose strides are ranked by `order`.
  ##
  ## Returns: the layout with the input shape and strides ranked by `order`.
  ##
  ## `order[i]` specifies the position of dimension `i` in the stride ordering:
  ## smaller value = faster-varying (smaller stride).
  ## The dimension with `order[i] = 0` gets stride 1, the next gets
  ## stride = shape[fastest], and so on.
  ## Dynamic order entries rank after every static entry, in left-to-right position, so they compact column-major behind every static rank.
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
  var shVals, ordVals: seq[int] = @[]
  for (leaf, leafTy) in shape.tupleStream().leaves():
    let v = orderedStaticInt(leaf, leafTy)
    doAssert v != DynamicSentinel,
      "make_ordered_layout: shape leaves must be statically known, " &
      "a dynamic shape leaf cannot produce a compile-time compact stride"
    shVals.add v
  for (leaf, leafTy) in order.tupleStream().leaves():
    # dynamic order leaves rank after every static entry,
    # the substitution keeps their relative position
    ordVals.add orderedStaticInt(leaf, leafTy)
  doAssert shVals.len == ordVals.len,
    "make_ordered_layout: shape and order of equal flat rank required"
  let strides = compactOrderStridesImpl(shVals, compactOrderDynamicSubstitution(ordVals))
  var builder = TupleBuilderFlat.new(1)
  for s in strides:
    builder.append(newLit(s))
  result = bindSym"make_layout".newCall(
    shape,
    builder.emit(0, emitScalarForSize1 = not shape.getTypeInst().isTupleTy()))

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
  var builder = TupleBuilderFlat.new(1)
  for s in strides:
    builder.append(newLit(s))
  result = bindSym"make_layout".newCall(
    sh,
    builder.emit(0, emitScalarForSize1 = not shapeIsTuple))

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
