## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout data types: Layout[Sh, St], basic accessors, and type-level predicates.

import std/macros
import std/typetraits
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/layout_algebra/layout_compiletime

# ═══════════════════════════════════════════════════════════════
#  Layout[Sh, St], typed shape + stride pair
# ═══════════════════════════════════════════════════════════════

type Layout*[Sh, St] = object
  ## A compile-time-typed layout: `Layout[Shape, Stride]`.
  ## Both Sh and St can be int, Int[N], or tuples thereof.
  shape*: Sh
  stride*: St

func `===`*(a: Layout, b: tuple): bool {.inline.} =
  ## Deep comparison against a (shape, stride) tuple.
  ## This handles static Int checks against int checks
  ## and also size-1 tuples against int/Int
  (a.shape === b[0]) and (a.stride === b[1])

func `===`*[A, B: Layout](a: A, b: B): bool {.inline.} =
  ## Deep comparison between two Layouts.
  a.shape === b.shape and a.stride === b.stride

func `$`*(layout: Layout): string =
  ## CuTe-style representation: "(shape):(stride)".
  ##   ($make_layout(4,1))  →  "(4):(1)"
  ##   ($make_layout((4,8),(1,4)))  →  "((4,8)):((1,4))"
  $layout.shape & ":" & $layout.stride

template rank*(layout: Layout): static int =
  ## Number of dimensions in layout (compile-time constant).
  rank(layout.shape)

template rank*[Sh, St](_: typedesc[Layout[Sh, St]]): static int =
  ## Number of dimensions in a layout type (compile-time constant).
  rank(Sh)

func size*(layout: Layout): auto {.inline.} =
  ## Number of logical elements: fold over all shape leaves.
  ## Returns Int[N] for all-static shapes, int otherwise.
  fold(flatten(layout.shape), Int[1](), acc * it)

# ═══════════════════════════════════════════════════════════════
#  Codomain properties
# ═══════════════════════════════════════════════════════════════

macro coshapeImpl*(sh, st: typed): untyped =
  result = IntCT(1)
  for (shEv, stEv) in sh.tupleStream().zip(st.tupleStream()):
    if shEv.kind == kLeaf:
      result = result + (shEv.leaf - IntCT(1)) * abs(stEv.leaf)

macro coshape*(layout: Layout): untyped =
  ## Coshape of the layout, the size of the layout's codomain:
  ## `coshape = Σᵢ (shᵢ - 1)·|stᵢ| + 1`
  result = newStmtList()
  let (shape, strides) = result.destructureLayout(layout)
  result.add quote do:
    coshapeImpl(`shape`, `strides`)

macro coprofileImpl*(bStrides: typed): untyped =
  result = IntCT(0)
  for stEv in bStrides.tupleStream():
    if stEv.kind == kLeaf:
      result = result + stEv.leaf

macro coprofile*(layout: Layout): untyped =
  ## Codomain profile of the layout:
  ## `profile = Σ leaves(layout.stride)`
  result = newStmtList()
  let (_, strides) = result.destructureLayout(layout)
  result.add quote do:
    coprofileImpl(`strides`)

# ═══════════════════════════════════════════════════════════════
#  StrideOrder
# ═══════════════════════════════════════════════════════════════

type StrideOrder* = enum
  LayoutLeft
    ## Leftmost dimension is contiguous (stride 1),
    ## equivalent to col-major / prefix_product.
    ##
    ## Example:
    ##   make_layout((M, N), LayoutLeft) -> (M, N) : (1, M)
    ##   make_layout((3, 4, 5), LayoutLeft) -> (3, 4, 5) : (1, 3, 12)

  LayoutRight
    ## Rightmost dimension is contiguous (stride 1),
    ## equivalent to row-major / suffix_product.
    ##
    ## Example:
    ##   make_layout((M, N), LayoutRight) -> (M, N) : (N, 1)
    ##   make_layout((3, 4, 5), LayoutRight) -> (3, 4, 5) : (20, 5, 1)

# ═══════════════════════════════════════════════════════════════
#  shape-structure predicates over IntOrIntTuple
# ═══════════════════════════════════════════════════════════════

template congruent*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if `a` and `b` have the same hierarchical rank structure.
  ## Returns a bool typedesc, usable in both `static` and runtime contexts.
  when a is (int or Int) or b is (int or Int):
    a is (int or Int) and b is (int or Int)
  elif a is tuple and b is tuple:
    when rank(a) != rank(b):
      false
    else:
      block:
        var ok = true
        staticFor i, 0, rank(a):
          if not congruent(a[i], b[i]):
            ok = false
        ok
  else:
    false

func weakly_congruent*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if A's nesting is contained in B's structure.
  ## Scalar matches anything; tuple must have at least as much structure.
  when a is (int or Int):
    true
  elif b is (int or Int):
    false
  elif a is tuple and b is tuple:
    when rank(a) != rank(b):
      false
    else:
      block:
        var ok = true
        staticFor i, 0, rank(a):
          if not weakly_congruent(a[i], b[i]):
            ok = false
        ok
  else:
    false

func can_group_a_into_b_impl[A, B](a: A; aStartIdx: int; b: B): int =
  ## Find consecutive dimensions in `a` from `aStartIdx` whose product equals `b`.
  static: doAssert a isnot int, "scalar a should be handled by caller"
  let bVal = fold(b, 1, acc * it)
  var acc = 1
  var aIdx = aStartIdx
  block accLoop:
    staticFor i, 0, rank(a):
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
    staticFor j, 0, rank(b):
      aIdx = can_group_a_into_b_impl(a, aIdx, b[j])
      if aIdx == -1:
        return false
    aIdx == rank(a)

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
  elif rank(a) == rank(b):
    block:
      var ok = true
      staticFor i, 0, rank(a):
        if not compatible(a[i], b[i]):
          ok = false
      ok
  else:
    can_group_a_into_b(a, b)
