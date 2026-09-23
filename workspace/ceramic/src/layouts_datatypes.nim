## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layout data types: Layout[Sh, St], basic accessors, and type-level predicates.
##
## Reference:
##   - CuTe C++: layout.hpp
##
## This file contains only the core types and operations that do NOT require
## `make_layout`. Construction primitives live in `layout_constructors.nim`.

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

template rank*(layout: Layout): static int =
  ## Number of modes in layout (compile-time constant).
  rank(layout.shape)

template rank*[Sh, St](_: typedesc[Layout[Sh, St]]): static int =
  ## Number of modes in a layout type (compile-time constant).
  rank(Sh)

func size*(layout: Layout): auto =
  ## Number of logical elements: fold over all shape leaves.
  ## Returns Int[N] for all-static shapes, int otherwise.
  fold(flatten(layout.shape), Int[1](), acc * it)

# ═══════════════════════════════════════════════════════════════
#  cosize — max offset + 1 of a layout
# ═══════════════════════════════════════════════════════════════

#  CuTe: cosize(L) = size(coshape(L))
#  coshape[i] = (sh[i]-1)*|st[i]| + 1, then size(product).
#  For a compact layout: cosize = size = product(shape).
#  For a gapped layout: cosize > size.
#
#  Returns Int[N] when all-static, int otherwise.

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
#                                                           0+2=2, ...
#                                                           → max=9, cosize=10
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

func cosize*[A, B](_: typedesc[Layout[A, B]]): static int =
  ## Compile-time cosize from Layout type alone.
  ## Requires static shape/stride (all Int[N], no runtime int).
  ## Dynamic layouts produce a compile error — matching CuTe's
  ## `static_assert("Dynamic owning tensors not supported")`.
  var tmp {.noInit.}: Layout[A, B]
  cosize(tmp).toIntVal()

# ═══════════════════════════════════════════════════════════════
#  StrideOrder — layout-left (col-major) / layout-right (row-major)
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

# ═══════════════════════════════════════════════════════════════
#  Shape-structure predicates (operate on IntOrIntTuple)
# ═══════════════════════════════════════════════════════════════

#  Reference:
#    - CuTe C++: layout_algebra.hpp (compatible, congruent)
#    - Meta tensor-layouts: core.py type predicates

template congruent*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if `a` and `b` have the same hierarchical rank structure.
  ##
  ## CuTe: `repeat_like(shape(a), _0{})` same type as `repeat_like(shape(b), _0{}})
  ## Returns a bool typedesc, usable in both `static` and runtime contexts.
  when a is (int or Int):
    when b is (int or Int):
      true
    else:
      false
  elif a is tuple:
    when b is tuple:
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
  else:
    false

func weakly_congruent*[A, B: IntOrIntTuple](a: A; b: B): bool =
  ## True if A's nesting is contained in B's structure.
  ## Scalar matches anything; tuple must have at least as much structure.
  when a is int or a is Int:
    true
  elif b is int or b is Int:
    false
  else:
    when rank(a) != rank(b):
      false
    else:
      block:
        var ok = true
        staticFor i, 0, rank(a):
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
