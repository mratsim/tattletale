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

  # Shape/stride TYPE extraction with aliased-type support.
  # layoutTypeArgs alone is not enough here: its aliased branch returns
  # the RAW literal arg types (plain int tuples) — fine for structure-
  # only consumers (compose, padRight) but wrong for typeIntVal, which
  # needs the makeIntTuple'd Int[N] leaves. Recover the make_layout
  # OUTPUT type (Int[N]-ified) from the alias's typedef RHS instead;
  # in this macro's context the RHS is always typed (const or typedef).
  let lTyp = layout.getTypeInst()
  var shTyp, stTyp: NimNode
  if lTyp.kind == nnkBracketExpr and $lTyp[0] == "Layout":
    shTyp = lTyp[1]
    stTyp = lTyp[2]
  elif lTyp.kind == nnkSym:
    # Aliased layout type (module-scope `typeof(make_layout(...))`).
    # getTypeInst normalizes back to the alias symbol; getTypeImpl yields
    # the full Layout object definition with makeIntTuple'd Int[N] args.
    let objTy = lTyp.getTypeImpl()
    if objTy.kind == nnkObjectTy:
      for field in objTy[2]:
        if field.kind == nnkIdentDefs and field[0].eqIdent("shape"):
          shTyp = field[2]
        elif field.kind == nnkIdentDefs and field[0].eqIdent("stride"):
          stTyp = field[2]
      if shTyp == nil or stTyp == nil:
        error "make_layout_like: Layout object type missing shape/stride fields"
    else:
      error "make_layout_like: aliased layout did not yield an object type"
  else:
    error "make_layout_like: compile-time Layout expression required"

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
  var strideTuple = nnkTupleConstr.newTree()
  for s in strides:
    strideTuple.add newLit(s)

  result = quote do:
    make_layout(`layout`.shape, `strideTuple`)

macro make_fragment_like*(layout: Layout; vShape: typed): untyped =
  ## Build a fragment layout from a partition view: the V leaves (the
  ## register-enumeration modes, the first `flattenType(typeof(vShape))`
  ## leaves of the shape) flatten to a single `(VA,):(1|0,)` mode — stride-1
  ## (hardware register order), stride-0 kept for broadcast V — regardless
  ## of the operand's strides. The remaining leaves keep the view's order,
  ## compacted by stride value (CuTe make_ordered_layout) and scaled after
  ## the V registers so the rest block does not collide with them.
  ##
  ## This is CuTe's make_fragment_like (layout.hpp). The point of the
  ## function: make_layout_like compacts by stride value across all modes,
  ## so a row-major operand view would reorder the V modes away from the
  ## mma hardware register order (a1/a2 swap). make_fragment_like pins the
  ## V modes to the hardware V enumeration regardless of the operand
  ## strides, and only the remaining modes follow the view's order.
  ##
  ## vShape: the atom's V shape value (getLayoutA().shape[1]). Its flat
  ## leaf count tells the macro how many leading leaves are V — tattletale
  ## partitions flatten the atom's (T,V) V part into consecutive modes, so
  ## the boundary must be stated (CuTe's make_fragment_like needs no such
  ## argument because its V mode is a single nested mode-0).
  ##
  ## The output keeps the input's leaf structure, so the fragment is
  ## coordinate-compatible with the partition view (same shape, copyFrom
  ## flat-index alignment preserved). The V block is flattened (CuTe keeps
  ## the nested structure) because gemm_atom reads the fragment data
  ## array in flat V-enumeration order — the flat enumeration is identical
  ## to the nested col-major one (v = v0 + V0·v1 + …), so copyFrom's
  ## coordinate alignment is unaffected.
  ##
  ## Examples:
  ##   (V0,V1,RepeatM,RepeatK) view  →  V flattened stride-1, remainder compact
  ##   row-major operand          →  same V order (make_layout_like would
  ##     reorder V after a fast rest mode and scramble the registers)
  ##   broadcast V (stride-0)     →  (VA,):(0,)
  # Shape/stride type extraction with aliased-type support — same
  # getTypeInst → nnkObjectTy path as make_layout_like (layoutTypeArgs'
  # aliased branch returns raw literal arg types, wrong for typeIntVal).
  let lTyp = layout.getTypeInst()
  var shTyp, stTyp: NimNode
  if lTyp.kind == nnkBracketExpr and $lTyp[0] == "Layout":
    shTyp = lTyp[1]
    stTyp = lTyp[2]
  elif lTyp.kind == nnkSym:
    let objTy = lTyp.getTypeImpl()
    if objTy.kind == nnkObjectTy:
      for field in objTy[2]:
        if field.kind == nnkIdentDefs and field[0].eqIdent("shape"):
          shTyp = field[2]
        elif field.kind == nnkIdentDefs and field[0].eqIdent("stride"):
          stTyp = field[2]
      if shTyp == nil or stTyp == nil:
        error "make_fragment_like: Layout object type missing shape/stride fields"
    else:
      error "make_fragment_like: aliased layout did not yield an object type"
  else:
    error "make_fragment_like: compile-time Layout expression required"

  let shLeaves = flattenType(shTyp)
  let stLeaves = flattenType(stTyp)
  let n = shLeaves.len

  if stLeaves.len != n:
    error "make_fragment_like: shape/stride rank mismatch"

  for i in 0 ..< n:
    if typeIntVal(shLeaves[i]) == DynamicSentinel:
      error "make_fragment_like: dynamic shapes unsupported — static layout required"

  if n == 1:
    # CuTe: rank-1 → plain compact (stride-1); broadcast (stride-0) keeps
    # stride-0. Note: emit as Int[N]() — a single-element tuple literal
    # (4,) flattens to the scalar 4 in typed argument position, breaking
    # makeIntTuple.
    let shNode = newLit(typeIntVal(shLeaves[0]))
    if typeIntVal(stLeaves[0]) == 0:
      result = quote do:
        make_layout(Int[`shNode`](), Int[0]())
    else:
      result = quote do:
        make_layout(Int[`shNode`]())
    return

  # ── V part: first vLeafCount leaves — flattened to (VA,):(1|0,) ──
  # vShape is the V shape value (e.g. getLayoutA().shape[1]) — its static
  # type tells the macro how many leading leaves are V. The argument is
  # `typed`: only its static type is read, the macro expands at compile
  # time, so crucible only ever sees the emitted layout.
  let vShapeTy = vShape.getTypeInst()
  let vShapeInner = if vShapeTy.kind == nnkBracketExpr and $vShapeTy[0] == "typeDesc":
                      vShapeTy[1]
                    else:
                      vShapeTy
  let vLeafCount = flattenType(vShapeInner).len
  doAssert vLeafCount >= 1 and vLeafCount <= n,
    "make_fragment_like: V leaf count (" & $vLeafCount & ") out of range for rank " & $n
  var vShapeVals = newSeq[int](vLeafCount)
  var vStrideVals = newSeq[int](vLeafCount)
  var va = 1
  var vAllZero = true
  var vAllNonZero = true
  for i in 0 ..< vLeafCount:
    vShapeVals[i] = typeIntVal(shLeaves[i])
    vStrideVals[i] = typeIntVal(stLeaves[i])
    va *= vShapeVals[i]
    if vStrideVals[i] == 0:
      vAllNonZero = false
    else:
      vAllZero = false
  # vShape value check: the vShape argument's leaf values must match the
  # layout's leading V leaves — a wrong-but-in-range vShape (e.g. a sibling
  # operand's V shape) would otherwise silently misbuild the fragment.
  let vShapeLeafTys = flattenType(vShapeInner)
  for i in 0 ..< vLeafCount:
    let vsv = typeIntVal(vShapeLeafTys[i])
    if vsv != DynamicSentinel:
      doAssert vsv == vShapeVals[i],
        "make_fragment_like: vShape leaf " & $i & " value " & $vsv &
        " != layout V leaf " & $vShapeVals[i]
  doAssert vAllZero or vAllNonZero,
    "make_fragment_like: mixed broadcast/non-broadcast V leaves unsupported —" &
    " a flattened (VA,):(1,) V block cannot represent a partially broadcast" &
    " register group without stride collisions"
  let vStride = if vAllZero: 0 else: 1
  # V cosize (broadcast shapes count 1) — scales the rest strides so the
  # rest block starts after the V registers (no stride collision).
  var vCosize = 1
  for i in 0 ..< vLeafCount:
    if vStrideVals[i] != 0:
      vCosize *= vShapeVals[i]

  # ── remaining leaves: compact by stride value (CuTe make_ordered_layout) ──
  var rsh = newSeq[int](n - vLeafCount)
  var rst = newSeq[int](n - vLeafCount)
  for i in vLeafCount ..< n:
    rsh[i - vLeafCount] = typeIntVal(shLeaves[i])
    rst[i - vLeafCount] = typeIntVal(stLeaves[i])

  # Step 1: filter_zeros — replace stride-0 shapes with 1
  var fsh = rsh
  for i in 0 ..< rsh.len:
    if rst[i] != DynamicSentinel and rst[i] == 0:
      fsh[i] = 1

  # Step 2: CuTe max-order-substitution for dynamic strides
  let resolvedOrder = compactOrderDynamicSubstitution(rst)

  # Step 3: compact_order(filtered_shape, resolved_order)
  var restStrides = compactOrderStridesImpl(fsh, resolvedOrder)

  # Step 4: restore broadcast strides; scale the rest after the V registers
  for i in 0 ..< rsh.len:
    if rst[i] != DynamicSentinel and rst[i] == 0:
      restStrides[i] = 0
    else:
      restStrides[i] *= vCosize

  # ── emit: (VA, rest…) : (1|0, restStrides…) — V flattened to one mode ──
  var outSh = nnkTupleConstr.newTree()
  outSh.add newLit(va)
  for i in vLeafCount ..< n:
    outSh.add newLit(typeIntVal(shLeaves[i]))

  var outSt = nnkTupleConstr.newTree()
  outSt.add newLit(vStride)
  for s in restStrides:
    outSt.add newLit(s)

  result = quote do:
    make_layout(`outSh`, `outSt`)
