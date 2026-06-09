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

func `===`*[A, B: Layout](a: A; b: B): bool =
  ## Deep comparison between two Layouts.
  a.shape === b.shape and a.stride === b.stride

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

func rank*[Sh, St](_: typedesc[Layout[Sh, St]]): static int =
  ## Number of modes in a layout type (compile-time constant).
  when Sh is int or Sh is Int:
    1
  else:
    tupleLen(Sh)

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
# 3-arg: tuple coord → inner product (a·b)
func crd2idx*[C, Sh, St: tuple](coord: C; shape: Sh; stride: St): auto =
  ## Inner product: sum coord[i] * stride[i]
  foldZipWith(coord, stride, 0): acc + it_a * it_b

# 3-arg: int coord → decompose across modes
func crd2idx*[C: int or Int; Sh, St: tuple](coord: C; shape: Sh; stride: St): auto =
  ## Decompose coord across shape modes with strides.
  ## Sequential: result += (cur mod s) * d; cur = cur div s
  var sum = 0
  var cur = int(coord)
  staticFor i, 0, tupleLen(Sh):
    let s = int(shape[i])
    let d = int(stride[i])
    when i < Sh.tupleLen - 1:
      sum += (cur mod s) * d
    else:
      sum += cur * d
    cur = cur div s
  sum

# Layout indexing: `layout[i]` and `layout[coord_tuple]`
macro `[]`*[Sh, St](layout: Layout[Sh, St]; idx: typed): int =
  let sh = newCall(bindSym"flatten", newTree(nnkDotExpr, layout, ident"shape"))
  let st = newCall(bindSym"flatten", newTree(nnkDotExpr, layout, ident"stride"))
  result = newCall(bindSym"crd2idx", idx, sh, st)

# Convenience: crd2idx on a Layout
func crd2idx*(coord: IntOrIntTuple; layout: Layout): int =
  layout[coord]

# idx2crd on a Layout — index → coordinate tuple
#  idx2crd — index to coordinate (stride-based)
# ═══════════════════════════════════════════════════════════════
##
## Convert a linear index into a hierarchical coordinate
## compatible with the given shape and stride.
## Each mode: (idx / stride[i]) % shape[i].
##
## MoYe: index_to_coord(idx, shape, stride)
## CuTe: idx2crd(idx, shape)

macro idx2crd*[Sh, St: IntOrIntTuple](idx: int or Int; shape: Sh; stride: St): untyped =
  ## Convert linear index to coordinate with explicit strides.
  let sh = shape.getTypeInst()
  if sh.kind != nnkTupleConstr:
    result = newCall(bindSym"div", idx, stride)
  else:
    var parts: seq[NimNode] = @[]
    for i in 0 ..< sh.len:
      let s = newCall(bindSym"[]", stride, newLit(i))
      let shI = newCall(bindSym"[]", shape, newLit(i))
      # Each mode: (idx / stride[i]) % shape[i] — independent per-mode
      parts.add newCall(bindSym"mod",
        newCall(bindSym"div", idx, s), shI)
    result = nnkPar.newTree(parts)

# ═══════════════════════════════════════════════════════════════
#  filter_zeros — replace stride-0 shapes with Int[1]
# ═══════════════════════════════════════════════════════════════

macro filterZerosFlat(sh, st: typed): untyped =
  ## Stride-0 mode shapes → Int[1](), everything else as-is.
  let stT = st.getTypeInst()
  let shT = sh.getTypeInst()
  # ── scalar path: single mode ──
  if shT.kind != nnkTupleConstr:
    if stT.kind == nnkBracketExpr and $stT[0] == "Int" and stT[1].intVal == 0:
      result = IntCT(1)
    else:
      result = sh
    return
  # ── tuple path: iterate over modes ──
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
#  padRight — extend layout rank by padding with identity modes
# ═══════════════════════════════════════════════════════════════
#
#  CuTe: append<R>(layout) pads to rank R with (1, 0) modes.
#  Used by blocked_product / raked_product to equalize ranks.
#  Pads on the RIGHT (appends identity modes at the end).

macro padRight*(layout: Layout; rank: static int): untyped =
  ## Extend layout to target rank by padding with identity modes (1, 0).
  ## Identity modes are appended on the right.
  ## Zero-cost if layout is at least the target rank. The input AST is passed as-is
  ## No intermediate value is materialized.
  let lTyp = layout.getTypeInst()
  let shTyp = lTyp[1]
  let curRank = if shTyp.kind == nnkTupleConstr: shTyp.len else: 1

  if curRank >= rank:
    result = layout
    return

  var ct = LayoutCT()
  if shTyp.kind == nnkTupleConstr:
    for i in 0 ..< shTyp.len:
      ct.shape.add newTree(nnkBracketExpr, newTree(nnkDotExpr, layout, ident"shape"), newLit i)
      ct.stride.add newTree(nnkBracketExpr, newTree(nnkDotExpr, layout, ident"stride"), newLit i)
  else:
    ct.shape.add newTree(nnkDotExpr, layout, ident"shape")
    ct.stride.add newTree(nnkDotExpr, layout, ident"stride")
  for i in curRank ..< rank:
    ct.shape.add IntCT(1)
    ct.stride.add IntCT(0)
  result = ct.emit()

#  padLeft — extend layout rank by prepending identity modes
# ═══════════════════════════════════════════════════════════════
##
##  CuTe: prepend<R>(layout) pads to rank R with (1, 0) modes.
##  Used by gemm.hpp to lift 2D operands to 3D.
##  Pads on the LEFT (prepends identity modes at the front).

macro padLeft*(layout: Layout; rank: static int): untyped =
  ## Extend layout to target rank by prepending identity modes (1, 0).
  ## Identity modes are prepended on the left.
  ## Zero-cost if layout is at least the target rank.
  let lTyp = layout.getTypeInst()
  let shTyp = lTyp[1]
  let curRank = if shTyp.kind == nnkTupleConstr: shTyp.len else: 1

  if curRank >= rank:
    result = layout
    return

  var ct = LayoutCT()
  for i in 0 ..< (rank - curRank):
    ct.shape.add IntCT(1)
    ct.stride.add IntCT(0)
  if shTyp.kind == nnkTupleConstr:
    for i in 0 ..< shTyp.len:
      ct.shape.add nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit i)
      ct.stride.add nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit i)
  else:
    ct.shape.add nnkDotExpr.newTree(layout, ident"shape")
    ct.stride.add nnkDotExpr.newTree(layout, ident"stride")
  result = ct.emit()

# ═══════════════════════════════════════════════════════════════
#  mapLeavesWith — apply body to each leaf (shape, stride) pair
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
    proc subst(n: NimNode): NimNode =
      if n.kind in {nnkIdent, nnkSym} and n.eqIdent("it_sh"):
        result = shExpr
      elif n.kind in {nnkIdent, nnkSym} and n.eqIdent("it_st"):
        result = stExpr
      else:
        result = n.copyNimTree()
        for j in 0 ..< n.len:
          result[j] = subst(n[j])
    let blockExpr = nnkBlockExpr.newTree(newEmptyNode(), subst(body))
    let tmp = ident("pairLeaves_" & $(stmts.len+1))
    stmts.add quote do:
      evalOnceAs(`tmp`, `blockExpr`)
    return (shape: nnkBracketExpr.newTree(tmp, newLit(0)),
            stride: nnkBracketExpr.newTree(tmp, newLit(1)))

macro mapLeavesWith*(layout: Layout; body: untyped): untyped =
  ## Apply `body` to each leaf (shape, stride) pair.
  ## Body receives `it_sh` and `it_st`, must return (new_shape, new_stride).
  let bodyExpr = if body.kind == nnkStmtList and body.len == 1: body[0] else: body
  let typ = getTypeInst(layout)
  let shTyp = typ[1]
  let stTyp = typ[2]
  let shExpr = newTree(nnkDotExpr, layout, ident"shape")
  let stExpr = newTree(nnkDotExpr, layout, ident"stride")
  var stmts = newStmtList()
  let (outSh, outSt) = mapLeavesRec(stmts, shExpr, shTyp, stExpr, stTyp, bodyExpr)
  stmts.add nnkCall.newTree(bindSym"make_layout", outSh, outSt)
  result = nnkBlockExpr.newTree(newEmptyNode(), stmts)


# ═══════════════════════════════════════════════════════════════
#  zip — interleave corresponding modes of two layouts
# ═══════════════════════════════════════════════════════════════

macro zip*[A, B: Layout](a: A, b: B): untyped =
  ## Zip two layouts: interleave corresponding modes pairwise.
  ##
  ##   Given layouts A with modes (a0, a1, ..., aN) and
  ##   B with modes (b0, b1, ..., bN), zip produces a layout
  ##   with modes ((a0,b0), (a1,b1), ..., (aN,bN)).
  ##
  ##   For rank-1 inputs: (a:b, x:y) → ((a,x):(b,y))

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

# ═══════════════════════════════════════════════════════════════
#  groupModes — wrap modes [B, E) into a nested sub-Layout
# ═══════════════════════════════════════════════════════════════

macro groupModes*(layout: Layout; B, E: static int): untyped =
  ## Wraps modes at indices `[B, E)` into a nested sub-tuple in both
  ## shape and stride, producing a higher-rank Layout.
  ##
  ## CuTe: group<B,E>(layout) — layout.hpp:1011
  ## Python: group(layout, B, E) — algebra.py:319
  ##
  ## Examples:
  ##   groupModes(make_layout((2, 3, 5, 7)), 0, 2)
  ##   # → ((2, 3), 5, 7):((1, 2), 6, 30)
  var ct = LayoutCT()
  let lTyp = layout.getTypeInst()
  let shTyp = lTyp[1]
  let R =
    if shTyp.kind == nnkTupleConstr:
      shTyp.len
    else: 1
  for i in 0 ..< B:
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit i),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit i))
  var gSh = nnkPar.newNimNode()
  var gSt = nnkPar.newNimNode()
  for i in B ..< E:
    gSh.add nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit i)
    gSt.add nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit i)
  ct.append(gSh, gSt)
  for i in E ..< R:
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit i),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit i))
  result = ct.emit()

#  takeModes — extract modes [B, E) into a new Layout
# ═══════════════════════════════════════════════════════════════

macro takeModes*(layout: Layout; B, E: static int): untyped =
  ## Extract modes in range `[B, E)` into a new Layout.
  ## Returns a scalar Layout if only one mode is extracted.
  ##
  ## Examples:
  ##   takeModes(make_layout((2, 3, 5, 7)), 1, 3)
  ##   # → (3, 5):(2, 6)
  var ct = LayoutCT()
  let lTyp = layout.getTypeInst()
  let shTyp = lTyp[1]
  let R = if shTyp.kind == nnkTupleConstr: shTyp.len else: 1
  for i in B ..< min(E, R):
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit(i)),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit(i)))
  result = ct.emit()

#  selectModes — extract specific mode indices into a new Layout
# ═══════════════════════════════════════════════════════════════
##
## CuTe: select<Is...>(layout) — layout.hpp:521
##
## Examples:
##   selectModes(make_layout((2, 3, 5, 7)), 0, 3)
##   # → (2, 7):(1, 30)

macro selectModes*(layout: Layout, Is: varargs[int]{lit|`const`}): untyped =
  ## Extract specific mode indices into a new Layout.
  var ct = LayoutCT()
  for i in 0 ..< Is.len:
    let idx = Is[i].intVal
    ct.append(nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"shape"), newLit(idx)),
               nnkBracketExpr.newTree(nnkDotExpr.newTree(layout, ident"stride"), newLit(idx)))
  result = ct.emit()

# ═══════════════════════════════════════════════════════════════
#  map — apply fn to each mode independently
#  zipWith — pairwise fn over modes of two Layouts
# ═══════════════════════════════════════════════════════════════

macro mapModesWith*[L: Layout](arg: L; body: untyped): untyped =
  ## Apply `body` to each mode of Layout `arg`. Within body, `it` is the current mode.
  ## `body` must evaluate to a Layout.
  ##
  ## Example:
  ##   mapModesWith(make_layout((2, 4), (1, 2))):
  ##     make_layout(it.shape, it.stride * 2)
  ##   # → (2, 4):(2, 4)
  let typ = getTypeInst(arg)
  let shTy = typ[1]
  let R = if shTy.kind == nnkTupleConstr: shTy.len else: 1

  result = newStmtList()
  proc subst(n: NimNode; i: int; la: NimNode): NimNode =
    if n.kind in {nnkIdent, nnkSym} and n.eqIdent("it"):
      result = newCall(bindSym"mode", la, newLit(i))
    else:
      result = n.copyNimTree()
      for j in 0 ..< n.len:
        result[j] = subst(n[j], i, la)

  var ct = LayoutCT()
  for i in 0 ..< R:
    let bodyExpr = subst(body, i, arg)
    let resName = ident("r" & $i)
    result.add newLetStmt(resName, bodyExpr)
    ct.append(newTree(nnkDotExpr, resName, ident"shape"),
               newTree(nnkDotExpr, resName, ident"stride"))
  result.add ct.emit()

macro zipModesWith*[A: Layout, B: Layout](a: A; b: B; body: untyped): untyped =
  ## Zip modes of `a` and `b` pairwise via body, append leftovers.
  ## Within body, `it_a` / `it_b` are the current modes.
  ##
  ## Example:
  ##   let r = zipModesWith(a, b):
  ##     make_layout(it_a.shape, it_b.stride)
  ##   # zips first min(rank(a), rank(b)) modes pairwise,
  ##   # appends any leftover modes unchanged.
  let ta = getTypeInst(a)
  let tb = getTypeInst(b)
  let shA = ta[1]
  let shB = tb[1]
  let RA = if shA.kind == nnkTupleConstr: shA.len else: 1
  let RB = if shB.kind == nnkTupleConstr: shB.len else: 1
  let rMin = min(RA, RB)
  let rMax = max(RA, RB)

  proc subst(n: NimNode; i: int; la, lb: NimNode): NimNode =
    if n.kind in {nnkIdent, nnkSym} and n.eqIdent("it_a"):
      result = newCall(bindSym"mode", la, newLit(i))
    elif n.kind in {nnkIdent, nnkSym} and n.eqIdent("it_b"):
      result = newCall(bindSym"mode", lb, newLit(i))
    else:
      result = n.copyNimTree()
      for j in 0 ..< n.len:
        result[j] = subst(n[j], i, la, lb)

  var ct = LayoutCT()
  result = newStmtList()
  for i in 0 ..< rMax:
    if i < rMin:
      let bodyExpr = subst(body, i, a, b)
      let resName = ident("r" & $i)
      result.add newLetStmt(resName, bodyExpr)
      ct.append(newTree(nnkDotExpr, resName, ident"shape"),
                 newTree(nnkDotExpr, resName, ident"stride"))
    elif i < RA:
      let mName = ident("m" & $i)
      result.add newLetStmt(mName, newCall(bindSym"mode", a, newLit(i)))
      ct.append(newTree(nnkDotExpr, mName, ident"shape"),
                 newTree(nnkDotExpr, mName, ident"stride"))
    else:
      let mName = ident("m" & $i)
      result.add newLetStmt(mName, newCall(bindSym"mode", b, newLit(i)))
      ct.append(newTree(nnkDotExpr, mName, ident"shape"),
                 newTree(nnkDotExpr, mName, ident"stride"))
  result.add ct.emit()

# ═══════════════════════════════════════════════════════════════
#  slice/dice/slice_and_offset on Layout [CUTE layout.hpp]
# ═══════════════════════════════════════════════════════════════
##
## These wrap int_tuples.slice/dice for Layout objects.
## Templates (not macros) so they participate in normal overload resolution.
## The `layout: Layout` param ensures precedence over the tuple macros.

template slice*(coord: CoordType; layout: Layout): untyped =
  ## Slice a layout by coordinate: keep modes paired with joker/`_`.
  make_layout(slice(coord, layout.shape), slice(coord, layout.stride))

template dice*(coord: CoordType; layout: Layout): untyped =
  ## Dice a layout by coordinate: keep modes paired with ints.
  make_layout(dice(coord, layout.shape), dice(coord, layout.stride))

template slice_and_offset*(coord: CoordType; layout: Layout): untyped =
  ## Slice a layout and compute the offset from fixed dims.
  ## Returns (sublayout, offset).
  let sub = slice(coord, layout)
  let off = crd2idx(coord, layout.shape, layout.stride)
  (sub, off)

# idx2crd on a Layout — index → coordinate tuple
macro idx2crd*(idx: int or Int; layout: Layout): untyped =
  result = newCall(bindSym"idx2crd", idx,
    newTree(nnkDotExpr, layout, ident"shape"),
    newTree(nnkDotExpr, layout, ident"stride"))
