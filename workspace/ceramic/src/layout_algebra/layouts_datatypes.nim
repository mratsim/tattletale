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
#  cosize, max offset + 1 of a layout
# ═══════════════════════════════════════════════════════════════

func cosize*(layout: Layout): auto =
  ## Compute cosize = sum_i ((sh_i - 1) * |st_i|) + 1.
  macro cosizeFlat(sh, st: typed): untyped =
    ## Cosize emit, one term per shape leaf, a scalar stride broadcasts.
    let one = IntCT(1)
    var shDims, stDims: seq[NimNode] = @[]
    for shEv in sh.tupleStream():
      if shEv.kind == kLeaf:
        shDims.add shEv.leaf
    for stEv in st.tupleStream():
      if stEv.kind == kLeaf:
        stDims.add stEv.leaf
    result = one
    for i in 0 ..< shDims.len:
      # a scalar stride is one leaf, it broadcasts over every shape leaf
      let d = if stDims.len == 1: stDims[0] else: stDims[i]
      let term = (shDims[i] - one) * abs(d)
      result = result + term
  cosizeFlat(flatten(layout.shape), flatten(layout.stride))

func cosize*[A, B](_: typedesc[Layout[A, B]]): static int {.inline.} =
  ## Compile-time cosize from the Layout type alone.
  ## Precondition, the shape and stride are all-static Int[N] leaves.
  ## Dynamic layouts produce a compile error.
  var tmp {.noInit.}: Layout[A, B]
  cosize(tmp).toIntVal()

# ═══════════════════════════════════════════════════════════════
#  StrideOrder, layout-left (col-major) or layout-right (row-major)
# ═══════════════════════════════════════════════════════════════

type StrideOrder* = enum
  LayoutLeft
    ## Leftmost dimension is contiguous (stride 1), equivalent to prefix_product.
    ##
    ## Example:
    ##   make_layout((M, N), LayoutLeft) -> (M, N) : (1, M)
    ##   make_layout((3, 4, 5), LayoutLeft) -> (3, 4, 5) : (1, 3, 12)

  LayoutRight
    ## Rightmost dimension is contiguous (stride 1), equivalent to suffix_product.
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
