## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout construction primitives: make_layout, col_major_strides, LayoutCT.
##
## These primitives construct Layout values from shapes and strides.

import std/macros
import workspace/ceramic/src/int_tuples
import ./layouts_datatypes

# ═══════════════════════════════════════════════════════════════
#  col_major_strides, canonical column-major strides
# ═══════════════════════════════════════════════════════════════

func col_major_strides*(shape: IntOrIntTuple): auto =
  ## Canonical column-major strides: prefix_product(shape).
  ## For shape (2,4): strides (1,4).
  prefix_product(shape)

# ═══════════════════════════════════════════════════════════════
#  make_layout, construct Layout values
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
#  LayoutCT, compile-time Layout accumulator for macros
# ═══════════════════════════════════════════════════════════════

type LayoutCT* = object
  shape*, stride*: seq[NimNode]

proc append*(ct: var LayoutCT; sh, st: NimNode) {.compileTime.} =
  ct.shape.add sh
  ct.stride.add st

func emit*(ct: LayoutCT): NimNode {.compileTime.} =
  ## Build make_layout from accumulated dimensions (no coalesce).
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

proc fragmentVPart(sh, st, vShape: seq[int]): tuple[va, vStride, vCosize: int] {.compileTime.} =
  ## The V block of a fragment layout, the first vShape.len leaves flattened
  ## to one (VA):(1|0) dimension.
  ## - broadcast V (all strides 0) keeps stride 0
  ## - otherwise the V block is stride-1, broadcast shapes count 1 toward the V cosize
  let vc = vShape.len
  doAssert vc >= 1 and vc <= sh.len,
    "make_fragment_like: V leaf count (" & $vc & ") out of range for rank " & $sh.len
  result.va = 1
  result.vCosize = 1
  var allZero = true
  var allNonZero = true
  for i in 0 ..< vc:
    doAssert vShape[i] == DynamicSentinel or vShape[i] == sh[i],
      "make_fragment_like: vShape leaf " & $i & " value " & $vShape[i] &
      " != layout V leaf " & $sh[i]
    result.va *= sh[i]
    if st[i] == 0:
      allNonZero = false
    else:
      allZero = false
      result.vCosize *= sh[i]
  doAssert allZero or allNonZero,
    "make_fragment_like: mixed broadcast/non-broadcast V leaves unsupported —" &
    " a flattened (VA,):(1,) V block cannot represent a partially broadcast" &
    " register group without stride collisions"
  result.vStride = if allZero: 0 else: 1

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

proc flattenType*(t: NimNode): seq[NimNode] {.compileTime.} =
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

proc typeIntVals(t: NimNode): seq[int] {.compileTime.} =
  ## Flattened leaf values of an Int tuple type, DynamicSentinel where a leaf is not a static Int.
  for leaf in flattenType(t):
    result.add typeIntVal(leaf)

proc typedLeafVals(expr: NimNode): seq[int] {.compileTime.} =
  ## Int leaf values of a typed expression's static type, one per flat element.
  let ty = expr.getTypeInst()
  let inner = if ty.kind == nnkBracketExpr and $ty[0] == "typeDesc": ty[1] else: ty
  for leaf in flattenType(inner):
    result.add typeIntVal(leaf)

proc litTuple(vals: seq[int]): NimNode {.compileTime.} =
  ## Int literal tuple expression, scalar when single-valued.
  result = nnkPar.newNimNode()
  for v in vals:
    result.add newLit(v)

func layoutTypeArgs*(layout: NimNode): tuple[shapeTy, strideTy: NimNode] {.compileTime.} =
  ## Extract the Layout type's shape and stride type nodes from a typed expression, resolving type aliases.
  let typ = layout.getTypeInst()
  if typ.kind == nnkBracketExpr and typ[0].eqIdent("Layout"):
    return (typ[1], typ[2])
  if typ.kind == nnkSym:
    let objTy = typ.getTypeImpl()
    if objTy.kind == nnkObjectTy:
      for field in objTy[2]:
        if field.kind == nnkIdentDefs and field[0].eqIdent("shape"):
          result.shapeTy = field[1]
        elif field.kind == nnkIdentDefs and field[0].eqIdent("stride"):
          result.strideTy = field[1]
      if result.shapeTy != nil and result.strideTy != nil:
        return
  error("layoutTypeArgs: cannot recover Layout type args from " & typ.repr)

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
    shVals[i] = leafIntVal(shLeaves[i])
    ordVals[i] = leafIntVal(ordLeaves[i])

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
  let outSt = litTuple(strides)
  result = quote do:
    make_layout(`layout`.shape, `outSt`)

macro make_fragment_like*(layout: Layout; vShape: typed): untyped =
  ## Build a fragment layout from a partition view.
  ##
  ## Contract:
  ## - The V dimensions, the first `flattenType(typeof(vShape))` shape leaves,
  ##   flatten to one `(VA,):(1|0,)` dimension
  ## - The V dimension carries stride-1, broadcast V keeps stride-0
  ## - The remaining leaves keep the view's order, compacted by stride value and scaled after the V dimension
  let (shTyp, stTyp) = layoutTypeArgs(layout)
  let shVals = typeIntVals(shTyp)
  let stVals = typeIntVals(stTyp)

  if shVals.len != stVals.len:
    error "make_fragment_like: shape/stride rank mismatch"

  for v in shVals:
    if v == DynamicSentinel:
      error "make_fragment_like: dynamic shapes unsupported — static layout required"

  let vShapeVals = typedLeafVals(vShape)
  let (va, vStride, vCosize) = fragmentVPart(shVals, stVals, vShapeVals)
  let vc = vShapeVals.len
  let restStrides = compactLikeStrides(shVals[vc ..< shVals.len], stVals[vc ..< stVals.len], vCosize)

  let outSh = litTuple(@[va] & shVals[vc ..< shVals.len])
  let outSt = litTuple(@[vStride] & restStrides)
  result = quote do:
    make_layout(`outSh`, `outSt`)
