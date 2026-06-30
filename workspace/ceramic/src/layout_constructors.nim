## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout construction primitives: make_layout, col_major_strides, LayoutCT.
##
## These primitives construct Layout values from shapes and strides.
## The `Layout` type itself lives in `layouts_datatypes.nim`.

import std/macros
import ./int_tuples
import ./layouts_datatypes

# ═══════════════════════════════════════════════════════════════
#  col_major_strides — canonical column-major strides
# ═══════════════════════════════════════════════════════════════

func col_major_strides*(shape: IntOrIntTuple): auto =
  ## Canonical column-major strides: prefix_product(shape).
  ## For shape (2,4): strides (1,4).
  prefix_product(shape)

# ═══════════════════════════════════════════════════════════════
#  make_layout — construct Layout values
# ═══════════════════════════════════════════════════════════════

template make_layout*(shapeArg: IntOrIntTuple; order: static StrideOrder = LayoutLeft): auto =
  ## Create a compact Layout from a shape, computing strides automatically.
  ## Encode compile-time integers into a Int[V] type for constant folding
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
  ## Encode compile-time integers into a Int[V] type for constant folding
  ## NOTE: inline makeIntTuple to avoid C++ temp-name collision
  Layout[typeof(makeIntTuple(shapeArg)), typeof(makeIntTuple(strideArg))](
    shape: makeIntTuple(shapeArg),
    stride: makeIntTuple(strideArg)
  )

# ═══════════════════════════════════════════════════════════════
#  LayoutCT — compile-time Layout accumulator for macros
# ═══════════════════════════════════════════════════════════════

type LayoutCT* = object
  shape*, stride*: seq[NimNode]

proc append*(ct: var LayoutCT; sh, st: NimNode) {.compileTime.} =
  ct.shape.add sh
  ct.stride.add st

func emit*(ct: LayoutCT): NimNode {.compileTime.} =
  ## Build make_layout from accumulated modes (no coalesce).
  ## This auto-constant-folds expressions that can be computed at compile-time.
  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  var outSh = newNimNode(nnkPar)
  var outSt = newNimNode(nnkPar)
  for i in 0 ..< ct.shape.len:
    outSh.add ct.shape[i]; outSt.add ct.stride[i]
  if ct.shape.len == 0:
    result = newCall(bindSym"make_layout", newLit(1), newLit(0))
  else:
    result = newCall(bindSym"make_layout", outSh, outSt)


proc compactOrderStridesImpl(shVals, ordVals: seq[int]): seq[int] {.compileTime.} =
  ## Compute stride for each mode m as product of shapes of modes
  ## whose order value is smaller than order[m].
  ## For each mode m: stride_start[m] = product of shapes of modes
  ## whose order value < order[m].
  let n = shVals.len
  result = newSeq[int](n)
  for m in 0 ..< n:
    var strideStart = 1
    for k in 0 ..< n:
      if ordVals[k] < ordVals[m]:
        strideStart *= shVals[k]
    result[m] = strideStart

proc compactOrderDynamicSubstitution(ordVals: seq[int]): seq[int] {.compileTime.} =
  ## Resolve dynamic order entries to unique values
  ## larger than any static order value, preserving their relative position.
  ## finds max STATIC order value, replaces dynamic entries (sentinel)
  ## with unique values > max_static, preserving their relative position.
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

# ── AST-level helpers (compile-time value extraction) ──

proc flattenAst(n: NimNode): seq[NimNode] {.compileTime.} =
  case n.kind
  of nnkIntLit, nnkUIntLit:
    result.add n
  of nnkCall, nnkBracketExpr:
    if n.len >= 1 and $n[0] == "Int" and n[1].kind == nnkIntLit:
      result.add n  # Int[N]()
    else:
      discard
  of nnkPar, nnkTupleConstr, nnkArgList:
    for child in n:
      for leaf in flattenAst(child):
        result.add leaf
  else:
    discard

proc leafIntVal(n: NimNode): int {.compileTime.} =
  case n.kind
  of nnkIntLit, nnkUIntLit:
    n.intVal
  of nnkCall, nnkBracketExpr:
    if n.len >= 1 and $n[0] == "Int" and n[1].kind == nnkIntLit:
      n[1].intVal
    else:
      DynamicSentinel
  else:
    DynamicSentinel

proc flattenType(t: NimNode): seq[NimNode] {.compileTime.} =
  case t.kind
  of nnkTupleConstr:
    for child in t:
      for leaf in flattenType(child):
        result.add leaf
  else:
    result.add t

proc typeIntVal(t: NimNode): int {.compileTime.} =
  if t.kind == nnkBracketExpr and $t[0] == "Int" and t[1].kind == nnkIntLit:
    t[1].intVal
  else:
    DynamicSentinel

macro compact_order*(shape, order): untyped =
  ## Produce compact strides for a given mode permutation.
  ##
  ## `order[i]` specifies the position of mode `i` in the stride ordering:
  ## smaller value = faster-varying (smaller stride).
  ## Returns a tuple of strides where the mode with `order[i] = 0` gets
  ## stride 1, the next gets stride = shape[fastest], and so on.
  ##
  ## Example — 2D permutations:
  ##   compact_order((2,3), (0,1))  → (1, 2)   # col-major (mode 0 fastest)
  ##   compact_order((2,3), (1,0))  → (3, 1)   # row-major (mode 1 fastest)
  ##
  ## Example — 3D custom permutation:
  ##   compact_order((2,3,4), (0,2,1))
  ##   # mode 0 fastest → stride 1
  ##   # mode 2 next    → stride 1*2   = 2
  ##   # mode 1 slowest → stride 1*2*4 = 8
  ##   # result: (1, 8, 2)

  let shLeaves = flattenAst(shape)
  let ordLeaves = flattenAst(order)

  if shLeaves.len == 0 or ordLeaves.len != shLeaves.len:
    error "compact_order: compile-time known shape and order of " &
          "equal flat rank required"

  let n = shLeaves.len
  var shVals = newSeq[int](n)
  var ordVals = newSeq[int](n)

  for i in 0 ..< n:
    shVals[i] = leafIntVal(shLeaves[i])
    ordVals[i] = leafIntVal(ordLeaves[i])

  # Apply CuTe max-order-substitution for dynamic entries
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

macro make_layout_like*(layout: Layout): untyped =
  ## Create a compact layout with the same shape and element-access order.
  ##
  ## Given a layout (possibly with non-compact strides), produces a new
  ## layout with compact strides that accesses elements in the same
  ## logical order. The input's stride values signal the desired ordering
  ## — the mode with the smallest stride gets stride 1 in the output,
  ## the next gets stride = product of faster modes' shapes, etc.
  ## Broadcast modes (statically Int[0]) keep stride 0.
  ##
  ## Example — non-compact (2,1) gives compact row-major (3,1):
  ##   make_layout_like(make_layout((2,3), (2,1)))  → (2,3):(3,1)
  ##   # mode 1 has the smaller stride (1), so it becomes fastest
  ##   # mode 0 stride becomes shape[1] = 3
  ##
  ## Example — broadcast mode preserved:
  ##   make_layout_like(make_layout((2,3), (0,1)))  → (2,3):(0,1)
  ##
  ## Example — 3D reordering:
  ##   make_layout_like(make_layout((2,3,4), (3,6,1)))  → (2,3,4):(4,8,1)
  ##   # mode 2 (stride 1) fastest  → stride 1
  ##   # mode 0 (stride 3) middle   → stride 1*4   = 4
  ##   # mode 1 (stride 6) slowest  → stride 1*4*2 = 8

  let lTyp = layout.getTypeInst()

  if lTyp.kind != nnkBracketExpr or $lTyp[0] != "Layout":
    error "make_layout_like: compile-time Layout expression required"

  let shTyp = lTyp[1]
  let stTyp = lTyp[2]

  let shLeaves = flattenType(shTyp)
  let stLeaves = flattenType(stTyp)
  let n = shLeaves.len

  if stLeaves.len != n:
    error "make_layout_like: shape/stride rank mismatch"

  var shVals = newSeq[int](n)
  var stVals = newSeq[int](n)

  for i in 0 ..< n:
    shVals[i] = typeIntVal(shLeaves[i])
    stVals[i] = typeIntVal(stLeaves[i])

  # Step 1: filter_zeros — replace stride-0 shapes with 1
  var fsh = shVals
  for i in 0 ..< n:
    if stVals[i] != DynamicSentinel and stVals[i] == 0:
      fsh[i] = 1

  # Step 2: apply CuTe max-order-substitution for dynamic strides
  let resolvedOrder = compactOrderDynamicSubstitution(stVals)

  # Step 3: compact_order(filtered_shape, resolved_order)
  var strides = compactOrderStridesImpl(fsh, resolvedOrder)

  # Step 4: restore broadcast strides
  for i in 0 ..< n:
    if stVals[i] != DynamicSentinel and stVals[i] == 0:
      strides[i] = 0

  # Emit result
  var strideTuple = nnkPar.newTree()
  for s in strides:
    strideTuple.add newLit(s)

  result = quote do:
    make_layout(`layout`.shape, `strideTuple`)
