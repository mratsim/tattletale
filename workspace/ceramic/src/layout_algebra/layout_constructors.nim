## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import workspace/ceramic/src/int_tuples
import ./layouts_datatypes
import ./layout_compiletime
import ./layouts_unsanctioned_helpers

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

template make_layout*(shapeArg: IntOrIntTuple; order: static StrideOrder = LayoutLeft): auto =
  ## Create a compact Layout from a shape, computing strides automatically.
  block:
    evalOnceAs(convShape, makeIntTuple(shapeArg))
    when order == LayoutLeft:
      evalOnceAs(strideVal, prefix_product(convShape))
      Layout[typeof(convShape), typeof(strideVal)](
        shape: convShape,
        stride: strideVal
      )
    else:
      evalOnceAs(strideVal, suffix_product(convShape))
      Layout[typeof(convShape), typeof(strideVal)](
        shape: convShape,
        stride: strideVal
      )

template make_layout*[ShT, StT: IntOrIntTuple](shapeArg: ShT; strideArg: StT): auto =
  ## Make a Layout from explicit shape and stride.
  Layout[typeof(makeIntTuple(shapeArg)), typeof(makeIntTuple(strideArg))](
    shape: makeIntTuple(shapeArg),
    stride: makeIntTuple(strideArg)
  )

# ═══════════════════════════════════════════════════════════════
#  compact_order
# ═══════════════════════════════════════════════════════════════

proc compactOrderStridesImpl(shVals, ordVals: seq[int]): seq[int] {.compileTime.} =
  ## Compute stride for each dimension m as product of shapes of dimensions
  ## whose order value is smaller than order[m].
  let n = shVals.len
  result = newSeq[int](n)
  for m in 0 ..< n:
    var strideStart = 1
    for k in 0 ..< n:
      if ordVals[k] < ordVals[m]:
        strideStart *= shVals[k]
    result[m] = strideStart

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

macro compact_order*(shape, order): untyped =
  ## Produce compact strides for a given dimension permutation.
  ##
  ## `order[i]` specifies the position of dimension `i` in the stride ordering:
  ## smaller value = faster-varying (smaller stride).
  ## Returns a tuple of strides where the dimension with `order[i] = 0` gets
  ## stride 1, the next gets stride = shape[fastest], and so on.
  ##
  ## Example, 2D permutations:
  ##   compact_order((2,3), (0,1))  → (1, 2)   # col-major (dimension 0 fastest)
  ##   compact_order((2,3), (1,0))  → (3, 1)   # row-major (dimension 1 fastest)
  ##
  ## Example, 3D custom permutation:
  ##
  ##   compact_order((2,3,4), (0,2,1))
  ##
  ## - dimension 0 fastest → stride 1
  ## - dimension 2 next → stride 1*2 = 2
  ## - dimension 1 slowest → stride 1*2*4 = 8
  ##   and the result is (1, 8, 2)

  let shLeaves = flattenAst(shape)
  let ordLeaves = flattenAst(order)

  if shLeaves.len == 0 or ordLeaves.len != shLeaves.len:
    error "compact_order: compile-time known shape and order of " &
          "equal flat rank required"

  let n = shLeaves.len
  var shVals = newSeq[int](n)
  var ordVals = newSeq[int](n)

  for i in 0 ..< n:
    shVals[i] = shLeaves[i].getStaticInt()
    ordVals[i] = ordLeaves[i].getStaticInt()

  # Apply max-order substitution for dynamic entries
  let resolvedOrder = compactOrderDynamicSubstitution(ordVals)

  # Compute strides
  let strides = compactOrderStridesImpl(shVals, resolvedOrder)

  # Emit result tuple
  if n == 1:
    result = newLit(strides[0])
  else:
    result = nnkPar.newTree()
    for s in strides:
      result.add newLit(s)

# ═══════════════════════════════════════════════════════════════
#  make_layout_like
# ═══════════════════════════════════════════════════════════════

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

  let (shTyp, stTyp) = layoutTypeArgs(layout)
  let shVals = typeIntVals(shTyp)
  let stVals = typeIntVals(stTyp)

  if shVals.len != stVals.len:
    error "make_layout_like: shape/stride rank mismatch"

  let strides = compactLikeStrides(shVals, stVals)
  # Rank-1 compaction stays a 1-tuple so the like of a rank-1 layout
  # keeps the same shape and stride tuple rank.
  let outSt = if strides.len == 1:
                nnkTupleConstr.newTree(newLit(strides[0]))
              else:
                litTuple(strides)
  result = quote do:
    make_layout(`layout`.shape, `outSt`)

# ═══════════════════════════════════════════════════════════════
#  make_fragment_like
# ═══════════════════════════════════════════════════════════════

template make_fragment_like*(layout: Layout): auto =
  ## Register-buffer layout for a partition view.
  ##
  ## Contract:
  ## - dimension 0 = the registers each thread owns, packed dense col-major (stride-1 chain)
  ## - broadcast registers (cosize 1, all strides 0) keep the zero strides verbatim
  ## - dimensions 1.. keep the view's stride order, compacted, scaled after the registers,
  ##   same size and flat access order as the view so view and fragment copies match
  ##
  ## Precondition, static shape and stride, the register part compact col-major or all-zero
  block:
    evalOnceAs(lyt, layout)
    when rank(lyt) == 1:
      make_layout(lyt.shape)
    else:
      evalOnceAs(v, dimension(lyt, 0))
      evalOnceAs(rest, takeDimensions(lyt, 1, rank(lyt)))
      when cosize(typeof(v)) == 1:
        tiled_product(v, make_layout_like(rest))
      else:
        tiled_product(make_layout(v.shape), make_layout_like(rest))
