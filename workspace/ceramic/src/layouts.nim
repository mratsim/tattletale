# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CuTe-compatible Layout construction: make_layout.
##
## Reference:
##   - CuTe C++: layout.hpp
##   - POC: poc_coalesce.nim
##
## Code style:
##   - make_int_tuple_rec (in int_tuples.nim) handles const->Int[N] wrapping

import std/macros
import std/typetraits
import ./macros/static_for
import ./int_tuples

# ═══════════════════════════════════════════════════════════════
#  Layout[Sh, St] — typed shape + stride pair
# ═══════════════════════════════════════════════════════════════

type Layout*[Sh, St] = object
  ## A compile-time-typed layout: `Layout[Shape, Stride]`.
  ## Both Sh and St can be int, Int[N], or tuples thereof.
  shape*: Sh
  stride*: St

func `===`*(a: Layout; b: tuple): bool =
  ## Deep comparison against a (shape, stride) tuple.
  ## This handles static Int checks against int checks
  ## and also size-1 tuples against int/Int
  (a.shape === b[0]) and (a.stride === b[1])

func `$`*(layout: Layout): string =
  ## CuTe-style representation: "(shape):(stride)".
  ##   ($make_layout(4,1))  →  "(4):(1)"
  ##   ($make_layout((4,8),(1,4)))  →  "((4,8)):((1,4))"
  $layout.shape & ":" & $layout.stride

func rank*(layout: Layout): static int =
  ## Number of modes in layout (compile-time constant).
  when layout.shape is int or layout.shape is Int:
    1
  else:
    layout.shape.tupleLen()

func mode*(layout: Layout; idx: static int): auto =
  ## Extract mode `idx` as a standalone rank-1 Layout.
  ## For scalar layouts (rank-1), only idx=0 is valid.
  when layout.shape is tuple:
    make_layout(layout.shape[idx], layout.stride[idx])
  else:
    static: doAssert idx == 0
    layout

func col_major_strides*(shape: IntOrIntTuple): auto =
  ## Canonical column-major strides: prefix_product(shape).
  ## For shape (2,4): strides (1,4).
  prefix_product(shape)

func isCompact*(layout: Layout): bool =
  ## True when strides match canonical column-major ordering.
  ## Note: does NOT coalesce first — size-1 modes may cause false negatives.
  layout === (layout.shape, col_major_strides(layout.shape))

func isCompact*(layout: static Layout): static bool =
  ## True when strides match canonical column-major ordering.
  ## Note: does NOT coalesce first — size-1 modes may cause false negatives.
  layout === (layout.shape, col_major_strides(layout.shape))

func size*(layout: Layout): auto =
  ## Number of logical elements: fold over all shape leaves.
  ## Returns Int[N] for all-static shapes, int otherwise.
  fold(flatten(layout.shape), Int[1](), acc * it)

# ═══════════════════════════════════════════════════════════════
#  cosize — max offset + 1 of a layout
# ═══════════════════════════════════════════════════════════════
#
#  CuTe: cosize(L) = size(coshape(L))
#  coshape[i] = (sh[i]-1)*|st[i]| + 1, then size(product).
#  For a compact layout: cosize = size = product(shape).
#  For a gapped layout: cosize > size.
#
#  Returns Int[N] when all-static, int otherwise.
# ═══════════════════════════════════════════════════════════════

#  ⚠ Known discrepancies between implementations of cosize on
#  COMPOSED layouts (make_layout(l1, l2)):
#
#  1. CuTe C++ — uses hierarchical (nested) cosize. For a composed
#     layout Layout<A,B>, cosize ≈ cosize(A) * cosize(B) effectively,
#     which is incorrect when the outer layout has non-trivial stride.
#
#  2. Meta tensor-layouts (Python) — enumerates ALL offsets to compute
#     max(L(i)) + 1.  This is O(size(L)) but is the only correct
#     definition for composed layouts.  CuTe's cosize(ComposedLayout)
#     bug is explicitly documented in the Python source:
#       "CuTe C++'s cosize(ComposedLayout) = cosize(layout_b()) is
#        wrong (it ignores the outer and the offset)."
#
#  3. Our Nim (flat affine) — uses the closed-form
#     1 + sum((sh_i - 1) * |st_i|) for pure affine layouts, which
#     matches Python's affine fast-path and CuTe's rank-1 cosize.
#     We DO NOT support ComposedLayout / Swizzle — our layouts are
#     always flat/affine, so the sum formula is correct.
#
#  Example cosize values for composed layouts:
#
#   Layout                    Affine sum   Cute hier   Python enum (correct)
#   ───────────────────────   ──────────   ──────────   ─────────────────────
#   make_layout(4:1,          (4-1)*1 +    cosize(4:1)  enumerate:
#               (2,2):(1,2))   (2-1)*1 +    ×             0+0=0, 2+0=2,
#                              (2-1)*2 +    cosize(       4+0=4, 6+0=6,
#                              1 = 6        (2,2):(1,2)   0+1=1, 2+1=3,
#                                          = 4 * 4 = 16   4+1=5, 6+1=7,
#                                                         0+2=2, ...
#                                                         → max=9, cosize=10
#
#   The sum formula (ours and Python's affine) gives cosize=6,
#   CuTe hierarchical product gives 16, Python enumeration gives 10.
#   All three disagree.  CuTe's product is WRONG per the Python docs;
#   enumeration is the only universally correct method.
#
#  For our pure affine layouts the sum formula IS correct —
#  we never create ComposedLayout.  The complement post-condition
#  check (1) from CuTe test_complement cannot be replicated without
#  either hierarchical product (wrong) or enumeration (expensive),
#  so we only check "doesn't crash" for complement.
# ═══════════════════════════════════════════════════════════════

func cosize*(layout: Layout): auto =
  ## Compute cosize = sum_i ((sh_i - 1) * |st_i|) + 1.
  ## CuTe: cosize(L) = size(coshape(L)).
  macro cosizeFlat(sh, st: typed): untyped =
    let shT = sh.getTypeInst()
    let one = IntCT(1)
    if shT.kind != nnkTupleConstr:
      # Scalar: (sh-1)*|st| + 1
      result = newCall(bindSym"+",
        newCall(bindSym"*",
          newCall(bindSym"-", sh, one),
          newCall(bindSym"abs", st)),
        one)
    else:
      # Flat tuple: sum over elements
      result = one
      for i in 0 ..< shT.len:
        let s = newTree(nnkBracketExpr, sh, newLit(i))
        let d = newTree(nnkBracketExpr, st, newLit(i))
        let term = newCall(bindSym"*",
          newCall(bindSym"-", s, one),
          newCall(bindSym"abs", d))
        result = newCall(bindSym"+", result, term)
  cosizeFlat(flatten(layout.shape), flatten(layout.stride))

# ═══════════════════════════════════════════════════════════════
#  Public templates
# ═══════════════════════════════════════════════════════════════

type StrideOrder* = enum
  LayoutLeft
    ## Leftmost mode is contiguous (stride 1).
    ##
    ## `LayoutLeft` means the **first** (index 0) mode of the shape tuple
    ## has stride 1. This is CuTe's **column-major** convention when
    ## the first mode represents rows and the second columns.
    ##
    ## The name refers to which end of the shape tuple gets stride 1:
    ## the "left" (first / index 0) element. Equivalent to `prefix_product`.
    ##
    ## Example:
    ##   make_layout((M, N), LayoutLeft) -> (M, N) : (1, M)
    ##   make_layout((3, 4, 5), LayoutLeft) -> (3, 4, 5) : (1, 3, 12)

  LayoutRight
    ## Rightmost mode is contiguous (stride 1).
    ##
    ## `LayoutRight` means the **last** (highest-index) mode of the shape
    ## tuple has stride 1. This is CuTe's **row-major** convention when
    ## the first mode represents rows and the second columns.
    ##
    ## The name refers to which end of the shape tuple gets stride 1:
    ## the "right" (last / highest-index) element. Equivalent to `suffix_product`.
    ##
    ## Example:
    ##   make_layout((M, N), LayoutRight) -> (M, N) : (N, 1)
    ##   make_layout((3, 4, 5), LayoutRight) -> (3, 4, 5) : (20, 5, 1)

template make_layout*(shapeArg: IntOrIntTuple; order: static StrideOrder = LayoutLeft): auto =
  ## Create a compact Layout from a shape, computing strides automatically.
  ## Encode compile-time integers into a Int[V] type for constant folding
  ## NOTE: inline makeIntTupleRec to avoid C++ temp-name collision
  let convShape = makeIntTupleRec(shapeArg)
  let strideVal = when order == LayoutLeft:
    prefix_product(convShape)
  else:
    suffix_product(convShape)
  Layout[typeof(convShape), typeof(strideVal)](
    shape: convShape,
    stride: strideVal
  )

template make_layout*[ShT, StT: IntOrIntTuple](shapeArg: ShT; strideArg: StT): auto =
  ## Make a Layout from explicit shape and stride.
  ## Encode compile-time integers into a Int[V] type for constant folding
  ## NOTE: inline makeIntTupleRec to avoid C++ temp-name collision
  Layout[typeof(makeIntTupleRec(shapeArg)), typeof(makeIntTupleRec(strideArg))](
    shape: makeIntTupleRec(shapeArg),
    stride: makeIntTupleRec(strideArg)
  )

# ═══════════════════════════════════════════════════════════════
#  crd2idx / layout[] — coordinate to linear index / layout indexing
# ═══════════════════════════════════════════════════════════════
#
#  Two forms:
#    crd2idx(coord, shape)         — flat col-major index (crd2flat)
#    crd2idx(coord, shape, stride) — memory offset (inner product)
#
#  When coord is an int and shape/stride are tuples, the int is
#  decomposed across modes (col-major ordering).

# Scalar overloads
func crd2idx(coord, shape: int): int = coord
func crd2idx[V: static int](coord: Int[V]; shape: int): int = V
func crd2idx(coord, shape, stride: int): int = coord * stride
func crd2idx[V: static int](coord: Int[V]; shape, stride: int): int = V * stride

# 3-arg: macro handles tuple coord + tuple stride → inner product
#         or int coord + tuple shape + tuple stride → decompose
macro crd2IdxImpl(coord, shape, stride: typed): untyped =
  let cT = coord.getTypeInst(); let sT = shape.getTypeInst(); let dT = stride.getTypeInst()
  if sT.kind != nnkTupleConstr:
    # Scalar: 1 * (coord * stride) to force runtime int return
    result = newCall(bindSym"*", newLit(1), newCall(bindSym"*", coord, stride))
    return
  if cT.kind == nnkTupleConstr:
    # Tuple coord: inner product with stride
    result = newLit(0)
    for i in 0 ..< sT.len:
      let t = newCall(bindSym"*",
        newTree(nnkBracketExpr, coord, newLit(i)),
        newTree(nnkBracketExpr, stride, newLit(i)))
      result = newCall(bindSym"+", result, t)
  else:
    # Int coord: decompose across modes — single expression, no intermediate vars
    var sum = newLit(0)
    for i in 0 ..< sT.len:
      let shI = newTree(nnkBracketExpr, shape, newLit(i))
      let stI = newTree(nnkBracketExpr, stride, newLit(i))
      # Build: coord div s0 div s1 ... div si-1
      var cur = coord
      for j in 0 ..< i:
        cur = newCall(bindSym"div", cur, newTree(nnkBracketExpr, shape, newLit(j)))
      if i < sT.len - 1:
        sum = newCall(bindSym"+", sum,
          newCall(bindSym"*", newCall(bindSym"mod", cur, shI), stI))
      else:
        sum = newCall(bindSym"+", sum, newCall(bindSym"*", cur, stI))
    result = sum

# 3-arg: the core overload (template so typed macro captures AST)
func crd2idx*[C, Sh, St: IntOrIntTuple](coord: C, shape: Sh, stride: St): int =
  crd2IdxImpl(coord, shape, stride)

# 2-arg: col-major flat index (macro for tuple; scalar is identity)
func crd2idx[C, Sh: IntOrIntTuple](coord: C; shape: Sh): int =
  macro impl(): untyped =
    let c = bindSym"coord"; let s = bindSym"shape"
    let cT = c.getTypeInst(); let sT = s.getTypeInst()
    if cT.kind != nnkTupleConstr or sT.kind != nnkTupleConstr:
      result = c  # scalar: identity
      return
    result = newLit(0)
    var stride = 1
    for i in 0 ..< sT.len:
      let ci = newTree(nnkBracketExpr, c, newLit(i))
      let term = newCall(bindSym"*", ci, newLit(stride))
      result = newCall(bindSym"+", result, term)
      let siT = sT[i]
      if siT.kind == nnkBracketExpr and $siT[0] == "Int":
        stride *= int(siT[1].intVal)
      # Runtime shapes: stride stays at 1 (approximate)
  impl()

# Layout indexing: `layout[i]` and `layout[coord_tuple]`
macro `[]`*[Sh, St](layout: Layout[Sh, St]; idx: typed): int =
  let sh = newCall(bindSym"flatten", newTree(nnkDotExpr, layout, ident"shape"))
  let st = newCall(bindSym"flatten", newTree(nnkDotExpr, layout, ident"stride"))
  result = newCall(bindSym"crd2idx", idx, sh, st)

# Convenience: crd2idx on a Layout
func crd2idx*(coord: IntOrIntTuple; layout: Layout): int =
  layout[coord]

# ═══════════════════════════════════════════════════════════════
#  filter_zeros — replace stride-0 shapes with Int[1]
# ═══════════════════════════════════════════════════════════════

macro filterZerosFlat(sh, st: typed): untyped =
  ## Stride-0 mode shapes → Int[1](), everything else as-is.
  let stT = st.getTypeInst(); let shT = sh.getTypeInst()
  if shT.kind != nnkTupleConstr:
    if stT.kind == nnkBracketExpr and $stT[0] == "Int" and stT[1].intVal == 0:
      result = IntCT(1)
    else:
      result = sh
    return
  result = newNimNode(nnkTupleConstr)
  for i in 0 ..< shT.len:
    let stN = stT[i]
    if stN.kind == nnkBracketExpr and $stN[0] == "Int" and stN[1].intVal == 0:
      result.add IntCT(1)
    else:
      result.add newTree(nnkBracketExpr, sh, newLit(i))

func filter_zeros*(layout: Layout): auto =
  ## Replace stride-0 shapes with 1; returns flat (both shape and stride flattened).
  let st = flatten(layout.stride)
  let sh = filterZerosFlat(flatten(layout.shape), st)
  make_layout(sh, st)

# ═══════════════════════════════════════════════════════════════
#  Shape-structure predicates (operate on IntOrIntTuple)
# ═══════════════════════════════════════════════════════════════
#
#  Reference:
#    - CuTe C++: layout_algebra.hpp (compatible, congruent)
#    - Meta tensor-layouts: core.py type predicates
# ═══════════════════════════════════════════════════════════════

func congruent*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if `a` and `b` have the same rank and nesting structure.
  a is typeof(b)

func weakly_congruent*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if A's nesting is contained in B's structure.
  ## Scalar matches anything; tuple must have at least as much structure.
  when a is int or a is Int:
    true
  elif b is int or b is Int:
    false
  else:
    when tupleLen(a) != tupleLen(b):
      false
    else:
      block:
        var ok = true
        staticFor i, 0, tupleLen(a):
          if not weakly_congruent(a[i], b[i]):
            ok = false
        ok

func can_group_a_into_b_impl[A, B](a: A; aStartIdx: int; b: B): int =
  ## Find consecutive modes in `a` from `aStartIdx` whose product equals `b`.
  static: doAssert a isnot int, "scalar a should be handled by caller"
  let bVal = fold(b, 1, acc * it)
  var acc = 1
  var aIdx = aStartIdx
  block accLoop:
    staticFor i, 0, tupleLen(a):
      if i >= aStartIdx:
        if acc < bVal:
          acc *= fold(a[i], 1, acc * it)
          aIdx = i + 1
        else:
          aIdx = i
          break accLoop
  if acc == bVal: aIdx else: -1

func can_group_a_into_b*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## Check if shape `a` (flat) can be grouped into shape `b` (nested).
  static: doAssert a isnot int, "scalar a should be handled by caller"
  when b is int or b is Int:
    can_group_a_into_b_impl(a, 0, b) != -1
  else:
    var aIdx = 0
    staticFor j, 0, tupleLen(b):
      aIdx = can_group_a_into_b_impl(a, aIdx, b[j])
      if aIdx == -1:
        return false
    aIdx == tupleLen(a)

func compatible*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if `a` is structurally compatible with `b`: same total size, and
  ## a's nesting can address into b's structure.
  ## Supports grouping: (2,2,3) is compatible with (4,3).
  let aSize = fold(a, 1, acc * it)
  let bSize = fold(b, 1, acc * it)
  if aSize != bSize:
    return false
  when a is int or a is Int:
    true
  elif b is int or b is Int:
    false
  elif tupleLen(a) == tupleLen(b):
    block:
      var ok = true
      staticFor i, 0, tupleLen(a):
        if not compatible(a[i], b[i]):
          ok = false
      ok
  else:
    can_group_a_into_b(a, b)

# ═══════════════════════════════════════════════════════════════
#  LayoutCT — compile-time Layout accumulator for macros
# ═══════════════════════════════════════════════════════════════

type LayoutCT* = object
  shape*, stride*: seq[NimNode]

proc append*(ct: var LayoutCT; sh, st: NimNode) {.compileTime.} =
  ct.shape.add sh; ct.stride.add st

func emit*(ct: LayoutCT): NimNode {.compileTime.} =
  ## Build make_layout from accumulated modes (no coalesce).
  ## This auto-constant-folds expressions that can be computed at compile-time.
  var outSh = newNimNode(nnkTupleConstr)
  var outSt = newNimNode(nnkTupleConstr)
  for i in 0 ..< ct.shape.len:
    outSh.add ct.shape[i]; outSt.add ct.stride[i]
  if ct.shape.len == 0:
    result = newCall(bindSym"make_layout", newLit(1), newLit(0))
  elif ct.shape.len == 1:
    outSh = outSh[0]; outSt = outSt[0]
    result = newCall(bindSym"make_layout", outSh, outSt)
  else:
    result = newCall(bindSym"make_layout", outSh, outSt)


# ═══════════════════════════════════════════════════════════════
#  getIndicesSortedByStride — sort permutation by stride
# ═══════════════════════════════════════════════════════════════

proc getIndicesSortedByStride*(strides: seq[int]): seq[int] {.compileTime.} =
  ## Return indices sorted by stride ascending.
  ## Data stays in original arrays — just iterate this permutation.
  result = newSeq[int](strides.len)
  for i in 0 ..< result.len:
    result[i] = i
  for i in 0 ..< result.len:
    for j in i + 1 ..< result.len:
      if strides[result[i]] > strides[result[j]]:
        swap result[i], result[j]

# ═══════════════════════════════════════════════════════════════
#  zip — interleave corresponding modes of two layouts
# ═══════════════════════════════════════════════════════════════
#
#  Given layouts A with modes (a0, a1, ..., aN) and
#  B with modes (b0, b1, ..., bN), zip produces a layout
#  with modes ((a0,b0), (a1,b1), ..., (aN,bN)).
#
#  For rank-1 inputs: (a:b, x:y) → ((a,x):(b,y))

macro zip*[A, B: Layout](a: A, b: B): untyped =
  ## Zip two layouts: interleave corresponding modes pairwise.
  let aTyp = a.getTypeInst()
  let bTyp = b.getTypeInst()
  let aShape = newTree(nnkDotExpr, a, ident"shape")
  let bShape = newTree(nnkDotExpr, b, ident"shape")
  let aStride = newTree(nnkDotExpr, a, ident"stride")
  let bStride = newTree(nnkDotExpr, b, ident"stride")
  let aShT = aTyp[1]
  let bShT = bTyp[1]
  let aStT = aTyp[2]
  let bStT = bTyp[2]

  proc zipElems(valA, valB, typA, typB: NimNode): NimNode =
    let aIsTuple = typA.kind == nnkTupleConstr
    let bIsTuple = typB.kind == nnkTupleConstr
    if not aIsTuple and not bIsTuple:
      result = newTree(nnkTupleConstr, valA, valB)
    elif aIsTuple and bIsTuple:
      result = newNimNode(nnkTupleConstr)
      for i in 0 ..< typA.len:
        let ai = newTree(nnkBracketExpr, valA, newLit i)
        let bi = newTree(nnkBracketExpr, valB, newLit i)
        let subA = typA[i].getTypeInst()
        let subB = typB[i].getTypeInst()
        result.add zipElems(ai, bi, subA, subB)
    else:
      error "zip: mismatched rank"

  let zShape = zipElems(aShape, bShape, aShT, bShT)
  let zStride = zipElems(aStride, bStride, aStT, bStT)
  result = newCall(bindSym"make_layout", zShape, zStride)