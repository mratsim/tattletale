# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Anti-regressions for layout_algebra: complex cases surfaced by
## integration (MMA atoms, partition algebra) that the unit tests did
## not cover. Each section pins the behavior that MUST hold. Asserts use
## layouts_testutils.check, which verifies BOTH the value (===) and that
## the shape/stride elements are Int[N] (constant-folded) — a value-only
## === would not catch a type-level regression (e.g. plain ints leaking
## out of the static paths).
##
## Section 1 — compose nested-RHS under module-scope typeof-alias
## fixtures:
##   atoms_nvidia.nim declares its SM80_* fragment layout types as
##   module-scope `typeof(make_layout(...))` aliases. Any module that
##   declares such aliases (this one does, below) breaks the nested-RHS
##   path of `compose` — mapDimensionsWith's getTypeInst(make_layout(
##   rhsShapes, rhsStrides)) resolves to an nnkSym → "cannot get child
##   of node kind: nnkSym". The flat-RHS and coalescable-LHS paths are
##   unaffected.
##
##   The layoutTypeArgs helper (nnkSym-safe shape/stride type
##   extraction) keeps this path working. The file's compose output is
##   CuTe-flat (single-dimension results unwrapped to scalars), matching the
##   unwrap in CuTe's composition_impl.
##
## Section 2 — inline coalesce(make_layout(...)) with a constant layout:
##   the inline form must type-check and stay Int[N]. NOTE: compiles()
##   does NOT capture the failure (returns true) — only a real compile
##   surfaces it, so the guarded case is a direct assert, not a
##   compiles() check.
## Section 3 — complement, multi-dimension layout + compile-time bound:
##   complement with a compile-time bound must produce the coalesced
##   result: a statically-1 remainder dimension is dropped by coalesce's
##   trailing size-1 discard, and a lone size-1 result is the library's
##   (1):(0) sentinel. Both coalesced values cross-checked against CuTe
##   host-side output.
##
## Section 4 — complement with a runtime shape must produce the same
##   layout as the identical layout spelled with constants.
##

{.experimental: "callOperator".}

import std/macros
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/tests/layouts_testutils

# ── integration fixture: module-scope typeof(make_layout(...)) aliases ──
# Identical in shape to atoms_nvidia.nim's SM80_16x8x8_A_TF32 / _B_TF32 /
# SM80_16x8_Row declarations. The asserts below must hold WITH these
# present — that is the integration condition. Do not remove them.
type
  LayoutAliasA = typeof(make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64))))
  LayoutAliasB = typeof(make_layout(((4, 8), 2), ((8, 1), 32)))
  LayoutAliasC = typeof(make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8))))

const nested = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
const flat   = make_layout((4, 8), (1, 32))

# ═══════════════════════════════════════════════════════════════
#  Section 1 — compose must work in the fixture module
# ═══════════════════════════════════════════════════════════════
proc runComposeFixtureTests =
  block:
    ## rank-1 LHS × flat RHS → flatten path — identity (a.stride = 1)
    let r = compose(make_layout(32, 1), flat)
    check r.shape, (4, 8), (Int[4], Int[8])
    check r.stride, (1, 32), (Int[1], Int[32])
  block:
    ## coalescable rank-2 LHS × nested RHS → coalesce→rank-1→make_layout
    ## (a = (8,8):(1,8) coalesces to (64):(1); result = b unchanged)
    let r = compose(make_layout((8, 8), (1, 8)), nested)
    check r.shape, ((4, 8), (2, 2)), ((Int[4], Int[8]), (Int[2], Int[2]))
    check r.stride, ((16, 1), (8, 64)), ((Int[16], Int[1]), (Int[8], Int[64]))
  block:
    ## non-coalescable rank-2 LHS × flat RHS → composeImpl path.
    ## R(i) = A(B(i)): B(i) = r + 32c, A at flat r + 32c = r + 64c ✓
    let r = compose(make_layout((16, 8), (1, 32)), flat)
    check r.shape, (4, 8), (Int[4], Int[8])
    check r.stride, (1, 64), (Int[1], Int[64])
  block:
    ## rank-1 LHS × nested RHS → composeDistribute path — fixed by
    ## layoutTypeArgs (nnkSym-safe type extraction); must stay working.
    let r = compose(make_layout(32, 1), nested)
    check r.shape, ((4, 8), (2, 2)), ((Int[4], Int[8]), (Int[2], Int[2]))
    check r.stride, ((16, 1), (8, 64)), ((Int[16], Int[1]), (Int[8], Int[64]))
  block:
    ## non-coalescable rank-2 LHS × nested RHS → composeDistribute path
    let r = compose(make_layout((16, 8), (1, 32)), nested)
    check r.shape, ((4, 8), (2, 2)), ((Int[4], Int[8]), (Int[2], Int[2]))
    check r.stride, ((32, 1), (8, 128)), ((Int[32], Int[1]), (Int[8], Int[128]))

  echo "    compose fixture: 5 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Section 2 — inline coalesce(make_layout(...)) with a constant layout
# ═══════════════════════════════════════════════════════════════
proc runCoalesceConstantFixtureTests =
  # Guarded case — inline coalesce over a constant layout: the result
  # must type-check and stay Int[N].
  block:
    let r = coalesce(make_layout((2, ceil_div(16, 8)), (2, 8)))
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (2, 8), (Int[2], Int[8])
  # Guarded case (const form) — const-context evaluation must work too.
  block:
    const c = coalesce(make_layout((2, ceil_div(16, 8)), (2, 8)))
    check c.shape, (2, 2), (Int[2], Int[2])
    check c.stride, (2, 8), (Int[2], Int[8])

  echo "    coalesce constant fixture: 2 guarded cases OK"

# ═══════════════════════════════════════════════════════════════
#  Section 3 — complement must work for multi-dimension layouts
# ═══════════════════════════════════════════════════════════════
proc runComplementFixtureTests =
  # Guarded cases — complement with a compile-time bound must produce
  # the coalesced result: the rem dimension (ceil_div(460, 512) = 1) is
  # statically 1, so coalesce's trailing size-1 discard removes it; a
  # lone size-1 result is the library's (1):(0) sentinel.
  block:
    let r = complement(make_layout((2, 2), (1, 4)), 16)
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (2, 8), (Int[2], Int[8])
  block:
    let r = complement(make_layout((2, 4, 8), (8, 1, 64)), 460)
    check r.shape, (2, 4), (Int[2], Int[4])
    check r.stride, (4, 16), (Int[4], Int[16])
  block:
    let r = complement(make_layout((2, (3, 4)), (3, (1, 6))))
    check r.shape, 1, Int[1]
    check r.stride, 0, Int[0]

  echo "    complement fixture: 3 guarded cases OK"

# ═══════════════════════════════════════════════════════════════
#  Section 4 — complement with a runtime shape matches the static twin
# ═══════════════════════════════════════════════════════════════
proc runComplementDynamicShapeTests =
  # Guarded cases — a runtime shape must give the same result as the
  # identical layout spelled with constants.
  block:
    let n = 2
    let r = complement(make_layout((2, n, 4), (1, 3, 12)), 200)
    check r.shape, (2, 5), (int, Int[5])
    check r.stride, (6, 48), (int, Int[48])
    let rs = complement(make_layout((2, 2, 4), (1, 3, 12)), 200)
    doAssert r.shape === rs.shape
    doAssert r.stride === rs.stride
  block:
    let m = 4
    let r = complement(make_layout((m, 4, 8), (8, 1, 64)), 460)
    check r.shape, (2, 2), (Int[2], int)
    check r.stride, (4, 32), (Int[4], int)
    let rs = complement(make_layout((4, 4, 8), (8, 1, 64)), 460)
    doAssert r.shape === rs.shape
    doAssert r.stride === rs.stride

  echo "    complement dynamic-shape fixture: 2 guarded cases OK"

# ═══════════════════════════════════════════════════════════════
#  Section 5 — make_layout_like / make_tensor_like under the alias
#  fixture (RID CONS-B-008): the alias path would crash with
#  "cannot get child of node kind: nnkSym" before layoutTypeArgs
#  migration — getTypeInst on an aliased typeof(make_layout(...))
#  returns an nnkSym with no children.
# ═══════════════════════════════════════════════════════════════

proc runMakeLayoutLikeAliasTests =
  block:
    ## alias fixture present → getTypeInst resolves direct make_layout
    ## calls to the aliased symbol (nnkSym); must agree with the const
    ## path and must not crash
    let viaAlias  = make_layout_like(make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64))))
    let viaDirect = make_layout_like(nested)
    doAssert viaAlias.shape === viaDirect.shape,
      "make_layout_like: aliased-module path shape != const path shape"
    doAssert viaAlias.stride === viaDirect.stride,
      "make_layout_like: aliased-module path stride != const path stride"
  block:
    ## the remaining alias fixtures must also survive make_layout_like
    let b = make_layout_like(make_layout(((4, 8), 2), ((8, 1), 32)))
    let c = make_layout_like(make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8))))
    doAssert toIntVal(size(b)) > 0
    doAssert toIntVal(size(c)) > 0
  echo "    make_layout_like under alias fixture: 2 guarded cases OK"


# ═══════════════════════════════════════════════════════════════
#  Section 6 — compose with a static stride-0 RHS dimension
# ═══════════════════════════════════════════════════════════════
#
#  CuTe's composition_impl shortcuts a static stride-0 RHS dimension.
#  Every coordinate maps to offset 0, so the composed dimension is the RHS
#  dimension itself and the LHS is never touched.
#
#  This arises when composing with a logical_divide whose complement
#  filler is (1):(0), i.e. tiler cosize == cosize bound.
#  For example, take max_alignment of a rank-2 layout whose strides are
#  fully contiguous after sorting by stride.
#  coalesce cannot merge them in the original dimension order, but
#  right_inverse covers the whole cosize of make_layout((2, 3), (3, 1)).

proc runComposeZeroStrideTests =
  block:
    ## Isolated shortcut. Composing a rank-2 LHS with a (1):(0) RHS dim yields (1):(0).
    ## No division by a zero stride at compile time.
    let lhs = make_layout((2, 3), (3, 1))
    check(compose(lhs, make_layout(1, 0)), make_layout(1, 0), Layout)

  block:
    ## The (1):(0) filler inside a logical_divide of make_layout((2, 3), (3, 1)).
    ## This is the pipeline consumed by max_alignment (k_layout_copy_gpu),
    ## CuTe formula gcd(size<0>, stride<1>).
    ##
    ## size<0> = the sorted-by-stride contiguous run = 6.
    ## The filler's stride<1> is 0 (every coordinate maps to offset 0), and gcd(6, 0) = 6.
    let lhs = make_layout((2, 3), (3, 1))
    let permuted = logical_divide(lhs, right_inverse(lhs))
    check(toIntVal(size(make_layout(permuted.shape[0], permuted.stride[0]))), 6, int)
    check(permuted.stride[1], Int[0](), Int[0])

#  Section 7. Nested-shape indexing and Layout-tiler unzip

proc runNestedShapeIntegrationTests =
  ## crd2idx with nested shape, layout[coord] on (2, (3,4)):((1,6),3)
  block:
    let l = make_layout((2, (3, 4)), (3, (1, 6)))
    doAssert l(0) === 0
    doAssert l(6) === 6
    doAssert l(1) === 3
    doAssert l(23) === 23
  ## zipped_divide with Layout tiler
  block:
    let A = make_layout((8, 8), (1, 8))
    let T = make_layout((2, 2), (1, 4))
    let zd = zipped_divide(A, T)
    doAssert zd === (((2, 2), (2, 8)), ((1, 4), (2, 8)))
  echo "    PASS"

#  Section 8. coalesce over a trailing size-1 dimension, the compose LHS path

proc runCoalesceTrailingSizeOneTests =
  ## A chain that reaches a trailing size-1 dimension's stride absorbs it,
  ## a chain that stops short keeps the dynamic `Int[DynamicSentinel]`
  ## marker and the marker must never enter the chain's arithmetic.
  block:
    ## (4,1):(1,4) ∘ 4:1: the (4,1) chain's span 4 reaches the trailing
    ## dimension's stride 4, the size-1 dimension is absorbed
    let r = compose(make_layout((4, 1), (1, 4)), make_layout(4, 1))
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  block:
    ## (2,1):(1,3) ∘ 2:1: the (2,1) chain's span 2 falls short of stride 3,
    ## the marker stays in the coalesced layout, the compose fold drops it
    let r = compose(make_layout((2, 1), (1, 3)), make_layout(2, 1))
    check r.shape, 2, Int[2]
    check r.stride, 1, Int[1]
  echo "    PASS"

#  Section 9. complement skips inactive dimensions in its own walk

proc runComplementInlineSkipTests =
  ## The complement walk itself skips stride-0 and size-1 dimensions:
  ## a stride-0 dimension maps every coordinate to offset 0 and a size-1
  ## dimension covers a single offset, neither joins the gap-fill chain.
  block:
    ## dynamic shape, static stride-0 dimension: (n, m):(1, 0) behaves
    ## as (n):(1), the complement is (ceil_div(64, n)):(n).
    ## The stride-0 dimension's shape never enters the arithmetic
    let n = 6
    let m = 4
    let r = complement(make_layout((n, m), (1, 0)), 64)
    let rs = complement(make_layout((6, 4), (1, 0)), 64)
    check rs.shape, 11, Int[11]
    check rs.stride, 6, Int[6]
    doAssert r.shape === rs.shape
    doAssert r.stride === rs.stride
  block:
    ## scalar stride-0 broadcast over a dynamic shape profile: every
    ## coordinate maps to offset 0, the complement is the full bound,
    ## and the default bound (cosize = 1) collapses it to the (1):(1) mark
    let n = 6
    let m = 4
    let r = complement(make_layout((n, m), 0), 32)
    doAssert r.shape === 32
    doAssert r.stride === 1
    let rd = complement(make_layout((n, m), 0))
    doAssert rd.shape === 1
    doAssert rd.stride === 1
  block:
    ## static size-1 dimension under a dynamic shape: (1, n):(8, 3)
    ## behaves as (n):(3), the complement is (3, ceil_div(32, 3n)):(1, 3n)
    let n = 6
    let r = complement(make_layout((1, n), (8, 3)), 32)
    let rs = complement(make_layout((1, 6), (8, 3)), 32)
    check rs.shape, (3, 2), (Int[3], Int[2])
    check rs.stride, (1, 18), (Int[1], Int[18])
    doAssert r.shape === rs.shape
    doAssert r.stride === rs.stride
  echo "    PASS"

# ═══════════════════════════════════════════════════════════════
#  Section 10. coalesce over runtime layouts, double-eval safety
# ═══════════════════════════════════════════════════════════════
proc runCoalesceRuntimeLayoutTests =
  # A runtime leaf is not visible at compile time: it never merges,
  # it passes through with its own value between the static folds.
  # The layout argument must be evaluated exactly once, a layout-valued
  # call argument materializes through a single binding.
  block:
    var buildCount = 0
    proc countedLayout(n, m: int): auto =
      inc buildCount
      make_layout((n, m), (1, n))
    let r = coalesce(countedLayout(4, 8))
    doAssert buildCount == 1,
      "coalesce must evaluate a layout-valued argument exactly once"
    doAssert r.shape === (4, 8)
    doAssert r.stride === (1, 4)
  block:
    # a runtime scalar layout passes through unchanged
    let n = 4
    let s = 8
    let r = coalesce(make_layout(n, s))
    doAssert r.shape === 4
    doAssert r.stride === 8
    doAssert r(3) == 24
  block:
    # a runtime leaf blocks a merge the paired static layout performs,
    # the static leaves around it still fold on their own values
    let n = 2
    let r = coalesce(make_layout((4, n, 8), (1, 4, 8)))
    doAssert r.shape === (4, 2, 8)
    doAssert r.stride === (1, 4, 8)
    let rs = coalesce(make_layout((4, 2, 8), (1, 4, 8)))
    check rs.shape, 64, Int[64]
    check rs.stride, 1, Int[1]
  echo "    PASS"

# ═══════════════════════════════════════════════════════════════
#  Section 11. coalesce preserveTrailing, the marker survives the fold
# ═══════════════════════════════════════════════════════════════
proc runCoalescePreserveTrailingTests =
  # With preserveTrailing the trailing size-1 dimension survives
  # as an Int[DynamicSentinel] marker when the chain stops short of its stride,
  # and is absorbed when the chain reaches it.
  block:
    ## the chain's span 2 stops short of stride 3, the marker stays
    let r = coalesce(make_layout((2, 1), (1, 3)), true)
    check r.shape, (2, DynamicSentinel), (Int[2], Int[DynamicSentinel])
    check r.stride, (1, 3), (Int[1], Int[3])
  block:
    ## the chain's span 4 reaches stride 4, the size-1 dimension is absorbed
    let r = coalesce(make_layout((4, 1), (1, 4)), true)
    check r.shape, 4, Int[4]
    check r.stride, 1, Int[1]
  block:
    ## every dimension is size-1, the whole layout collapses to the marker
    let r = coalesce(make_layout((1, 1), (0, 0)), true)
    check r.shape, DynamicSentinel, Int[DynamicSentinel]
    check r.stride, 0, Int[0]
  echo "    PASS"

# ═══════════════════════════════════════════════════════════════
#  Section 12. coalesce degenerate 1-leaf inputs fold without an early-out
# ═══════════════════════════════════════════════════════════════
proc runCoalesceOneLeafTests =
  # These cases match what a pre-flight 1-leaf early-out would emit.
  # A lone non-one leaf flushes as one chunk. An empty-builder tail
  # collapses an all-size-1 layout, one leaf or many, to the (1):(0)
  # sentinel. No pre-flight early-out exists, the fold itself covers
  # every degenerate shape.
  block:
    ## the lone leaf opens and flushes one chunk, identity result
    let r = coalesce(make_layout(64, 1))
    check r.shape, 64, Int[64]
    check r.stride, 1, Int[1]
  block:
    ## a lone size-1 leaf is skipped, the empty tail is the sentinel
    let r = coalesce(make_layout(1, 0))
    check r.shape, 1, Int[1]
    check r.stride, 0, Int[0]
  block:
    ## a lone size-1 leaf with a nonzero stride is skipped the same way
    let r = coalesce(make_layout(1, 4))
    check r.shape, 1, Int[1]
    check r.stride, 0, Int[0]
  block:
    ## every leaf is size-1, no chunk ever opens, the sentinel again
    let r = coalesce(make_layout((1, 1), (0, 0)))
    check r.shape, 1, Int[1]
    check r.stride, 0, Int[0]
  echo "    PASS"

func stripGensymCounters(ast: string): string {.compileTime.} =
  ## Strip the gensym counter suffixes the semmed AST carries: a decimal
  ## run of 8+ digits right after an underscore that ends the identifier.
  ##
  ## Mangled base62 hash tails pass through byte for byte.
  const digits = {'0' .. '9'}
  result = newStringOfCap(ast.len)
  var i = 0
  while i < ast.len:
    if ast[i] == '_':
      var j = i + 1
      while j < ast.len and ast[j] in digits:
        inc j
      if j - (i + 1) >= 8 and (j == ast.len or ast[j] notin {'a' .. 'z', 'A' .. 'Z', '0' .. '9', '_'}):
        i = j
        continue
    result.add ast[i]
    inc i

macro astRepr(x: typed): string =
  ## The semmed expansion of x as a string. An identity coalesce pastes
  ## its input, so the expansion of `coalesce(X)` equals the expansion
  ## of X itself; a rebuild carries extra lets and a make_layout call.
  result = newLit(repr(x))

# ═══════════════════════════════════════════════════════════════
#  Section 13. coalesce verbatim passthrough: identity folds paste
#
#  Identity fold: every original leaf passed through in order,
#  flat profile, no drop/merge/marker.
#
#  - an identity fold re-pastes the layout expression itself
#  - an identity site's emitted MSL is byte-identical to the same kernel
#    with the coalesce call peeled off, no `make_layout` reconstruction
#  - any restructure (merge, size-1 drop, preserveTrailing marker)
#    keeps the coalesced emission
# ═══════════════════════════════════════════════════════════════

proc runCoalesceVerbatimPassthroughTests =
  block:
    ## static identity, no chain merges: the values and the Int[N]
    ## folding stay correct through the paste
    let r = coalesce(make_layout((2, 3, 4), (1, 8, 64)))
    check r.shape, (2, 3, 4), (Int[2], Int[3], Int[4])
    check r.stride, (1, 8, 64), (Int[1], Int[8], Int[64])
  block:
    ## static identity paste: `coalesce(L)` expands to L itself,
    ## a rebuild would emit lets and a make_layout call
    doAssert astRepr(coalesce(make_layout((2, 3, 4), (1, 8, 64)))) ==
      astRepr(make_layout((2, 3, 4), (1, 8, 64)))
  block:
    ## runtime identity: a layout carried by a symbol pastes the symbol,
    ## no field reads, no make_layout
    let m = 4
    let n = 8
    let s = 2
    let L = make_layout((m, n), (1, s))
    doAssert astRepr(coalesce(L)) == astRepr(L)
    doAssert astRepr(coalesce(make_layout((2, 3), (1, 4)))) ==
      astRepr(make_layout((2, 3), (1, 4)))
  block:
    ## runtime identity values survive the paste
    let m = 4
    let n = 8
    let s = 2
    let r = coalesce(make_layout((m, n), (1, s)))
    doAssert r.shape === (4, 8)
    doAssert r.stride === (1, 2)
    doAssert toIntVal(size(r)) == 32
  block:
    ## preserveTrailing identity, the trailing leaf is not size-1 so no
    ## marker is appended: still a paste
    doAssert astRepr(coalesce(make_layout((2, 3, 4), (1, 8, 64)), true)) ==
      astRepr(make_layout((2, 3, 4), (1, 8, 64)))
  block:
    ## call-valued chain site, right_inverse + coalesce(compose(A, R)):
    ## - the compose call expands once, coalesce pastes its result,
    ##   identical to the peeled composition, with no evalOnce lets
    ##   and no make_layout reconstruction
    ## - the expansion carries macro-gensym'd temporaries, each
    ##   expansion advances the global gensym counter independently,
    ##   the comparison strips the `_<10-digit counter>` suffixes
    ## - the raw equality checks above carry no such temporaries
    let m = 4
    let n = 8
    let sd = 2
    let ss = 16
    let R = right_inverse(make_layout((int(m), int(n)), (1, int(ss))))
    doAssert stripGensymCounters(astRepr(coalesce(compose(
        make_layout((int(m), int(n)), (1, int(sd))), R)))) ==
      stripGensymCounters(astRepr(compose(
        make_layout((int(m), int(n)), (1, int(sd))), R)))
  block:
    ## a restructure never pastes: the merged fold keeps its emission
    let r = coalesce(make_layout((2, 4), (1, 2)))
    check r.shape, 8, Int[8]
    check r.stride, 1, Int[1]
  echo "    PASS"

# ═══════════════════════════════════════════════════════════════
#  Section 14. logical_divide over a tuple tiler, the per-dimension stream rewrite.
#
#  One tiler element per layout dimension, the per-dimension divide
#  is emitted twice, once per shape and once per stride projection,
#  both copies pure expressions, no evalOnceAs and no lets introduced
#  by the macro itself. The elements come off a tuple dims stream,
#  the layout dimensions off a shape/stride stream zip, symbols
#  and calls work exactly like tuple constructors.
#  ═══════════════════════════════════════════════════════════════

proc runDivideTupleTilerTests =
  block:
    ## static tiler, the fully folded emission
    let r = logical_divide(make_layout((10, 8), (2, 1)), (4, 4))
    check r.shape, ((4, 3), (4, 2)), ((Int[4], Int[3]), (Int[4], Int[2]))
    check r.stride, ((2, 8), (1, 4)), ((Int[2], Int[8]), (Int[1], Int[4]))
  block:
    ## a tiler carried by a symbol: the elements are bracket reads,
    ## the runtime elements stay runtime, the static parts still fold,
    ## and the result equals the fully static tiler result
    let t = (4, 4)
    let r = logical_divide(make_layout((10, 8), (2, 1)), t)
    let rs = logical_divide(make_layout((10, 8), (2, 1)), (4, 4))
    doAssert r === rs
  block:
    ## an empty tiler divides nothing, every dimension passes through,
    ## the result is the layout itself
    let r = logical_divide(make_layout((10, 8), (2, 1)), ())
    check r.shape, (10, 8), (Int[10], Int[8])
    check r.stride, (2, 1), (Int[2], Int[1])
    doAssert r === make_layout((10, 8), (2, 1))
  block:
    ## a runtime layout through the same path, the runtime leaves stay
    ## runtime and the static folds still fold
    let dM = 4
    let dN = 5
    let r = logical_divide(make_layout((dM, dN), (1, 16)), (4, 4))
    check r.shape, ((4, 1), (4, 2)), ((Int[4], int), (Int[4], int))
    check r.stride, ((1, 4), (16, 64)), ((Int[1], Int[4]), (Int[16], Int[64]))
  block:
    ## a tiler longer than the layout is a compile-time error
    static:
      doAssert not compiles(logical_divide(make_layout((2, 4), (1, 2)), (4, 4, 2)))
  echo "    divide tuple-tiler rewrite: 5 cases OK"

#  Section 15. hier_unzip zipped gather over the stream walk.
#
#  One tiler element per layout dimension, the splitter call per terminal
#  element binds once, the four shape/stride projections read the binding.
#  A single-element gather collapses to a scalar,
#  so a 1-element tiler and a 1-element sub-tiler carry no tuple wrapper
#  and a leftover dimension splices verbatim into the rest part.
#  ═══════════════════════════════════════════════════════════════

proc runZippedGatherTests =
  block:
    ## int vs 1-element tuple tiler agree on a rank-1 layout
    let L = make_layout(6, 1)
    doAssert zipped_divide(L, 2) === zipped_divide(L, (2,)),
      "scalar tiler guard: " & $zipped_divide(L, (2,))
  block:
    ## 1-element tiler: the tile part collapses to a scalar, the leftover
    ## dimension splices verbatim into the rest part
    let zd = zipped_divide(make_layout((4, 8), (1, 4)), (2,))
    doAssert zd === ((2, (2, 8)), (1, (2, 4))), "1-element tiler: " & $zd
  block:
    ## empty tiler: nothing divides, every dimension gathers into the rest
    ## part behind an empty tile part
    let z = zipped_divide(make_layout((10, 8), (2, 1)), ())
    doAssert z === make_layout(((), (10, 8)), ((), (2, 1))), "empty tiler: " & $z
  block:
    ## sub-tuple tiler element recurses, the gathered pair contributes one
    ## element per part, the nesting mirrors the tiler
    let r = zipped_divide(make_layout((8, (4, 2)), (16, (1, 2))), (2, (4, 2)))
    doAssert r === (((2, (4, 2)), (4, (1, 1))), ((16, (1, 2)), (32, (4, 4)))),
      "sub-tuple tiler: " & $r
  block:
    ## a 1-element sub-tiler collapses like the flat tiler
    let L = make_layout((4, 8), (1, 4))
    doAssert zipped_divide(L, (2, (4,))) === zipped_divide(L, (2, 4)),
      "sub-tuple collapse: " & $zipped_divide(L, (2, (4,)))
  echo "    zipped gather rewrite: 5 cases OK"


#  Section 16. right_inverse / left_inverse over a runtime shape leaf.
#
#  A runtime shape leaf makes the stride prefix runtime: dimensions
#  after it in original order carry the prefix as a runtime expression.
#  A runtime layout must match the identical layout spelled with constants.
#  ═══════════════════════════════════════════════════════════════

proc runInverseDynamicShapeTests =
  block:
    ## runtime shape = static counterpart, the dynamic-shape dimension
    ## precedes a smaller-stride dimension in original order
    let n = 5
    let r = right_inverse(make_layout((n, 4), (4, 1)))
    check r.shape, (4, n), (Int[4], int)
    doAssert r.stride[0] === n, "right_inverse stride: " & $r.stride
    doAssert r.stride[1] === 1, "right_inverse stride: " & $r.stride
    let rs = right_inverse(make_layout((5, 4), (4, 1)))
    doAssert r === rs, "right_inverse runtime shape: " & $r & " != " & $rs
  block:
    let n = 5
    let r = left_inverse(make_layout((n, 4), (4, 1)))
    doAssert r.stride[0] === n, "left_inverse stride: " & $r.stride
    let rs = left_inverse(make_layout((5, 4), (4, 1)))
    doAssert r === rs, "left_inverse runtime shape: " & $r & " != " & $rs

  echo "    inverse dynamic-shape fixture: 2 guarded cases OK"



#  Section 17. compose with a symbol-bound tiler.
#
#  A tiler bound to a symbol is not readable by child index at macro
#  time. The tiler binds once to a fresh let and the loop reads every
#  element by index, literal and symbol-bound tiler nodes alike.
#  ═══════════════════════════════════════════════════════════════

proc runComposeSymbolTilerTests =
  block:
    ## int tiler: the symbol-bound tuple matches the literal tuple
    let t = (16, 2)
    let lit = compose(make_layout((32, 8), (1, 32)), (16, 2))
    let viaSym = compose(make_layout((32, 8), (1, 32)), t)
    doAssert viaSym === lit, "int tiler: " & $viaSym & " != " & $lit
  block:
    ## Layout tiler element: the symbol-bound tuple matches the literal form
    let t = (make_layout(16, 1), 2)
    let lit = compose(make_layout((32, 8), (1, 32)), (make_layout(16, 1), 2))
    let viaSym = compose(make_layout((32, 8), (1, 32)), t)
    doAssert viaSym === lit, "layout tiler: " & $viaSym & " != " & $lit

  echo "    compose symbol-tiler fixture: 2 guarded cases OK"



#  Section 18. make_layout_like over a scalar shape.
#
#  A scalar-shape layout keeps a scalar stride, the like of a scalar
#  shape must stay scalar so make_tensor_like's strict === holds.
#  A rank-1 layout keeps its 1-tuple stride, the 1-tuple case is locked.
#  ═══════════════════════════════════════════════════════════════

proc runMakeLayoutLikeScalarShapeTests =
  block:
    ## scalar shape = scalar stride, rank-1 keeps the 1-tuple stride
    let like1 = make_layout_like(make_layout(4, 1))
    doAssert like1 === make_layout(4, 1), "scalar shape: " & $like1
    doAssert typeof(like1.stride) is typeof(make_layout(4, 1).stride),
      "make_layout_like: scalar-shape stride must stay a scalar, got " & $(typeof(like1.stride))
  block:
    ## the fragment path: the like of a scalar-shape takeDimensions extract
    let td = takeDimensions(make_layout((4, 8), (1, 4)), 1, 2)
    let likeT = make_layout_like(td)
    doAssert likeT === make_layout(8, 1), "scalar extract: " & $likeT
    doAssert typeof(likeT.stride) is typeof(make_layout(8, 1).stride),
      "make_layout_like: scalar-extract stride must stay a scalar, got " & $(typeof(likeT.stride))

  echo "    make_layout_like scalar-shape fixture: 2 guarded cases OK"


proc runTests =
  echo "\n── layout_algebra anti-regressions (integration) ──"
  echo "── Section 1: compose under module-scope typeof-alias fixture ──"
  runComposeFixtureTests()
  echo "── Section 2: inline coalesce + constant layout ──"
  runCoalesceConstantFixtureTests()
  echo "── Section 3: complement multi-dimension + compile-time bound ──"
  runComplementFixtureTests()
  echo "── Section 4: complement runtime shape = static twin ──"
  runComplementDynamicShapeTests()
  echo "── Section 5: make_layout_like under typeof-alias fixture ──"
  runMakeLayoutLikeAliasTests()
  echo "── Section 6: compose with static stride-0 RHS dim ──"
  runComposeZeroStrideTests()
  echo "── Section 7. Nested-shape indexing and Layout-tiler unzip ──"
  runNestedShapeIntegrationTests()
  echo "── Section 8. coalesce over a trailing size-1 dimension ──"
  runCoalesceTrailingSizeOneTests()
  echo "── Section 9. complement skips inactive dimensions ──"
  runComplementInlineSkipTests()
  echo "── Section 10. coalesce over runtime layouts ──"
  runCoalesceRuntimeLayoutTests()
  echo "── Section 11. coalesce preserveTrailing marker ──"
  runCoalescePreserveTrailingTests()
  echo "── Section 12. coalesce degenerate 1-leaf inputs ──"
  runCoalesceOneLeafTests()
  echo "── Section 13. coalesce verbatim passthrough ──"
  runCoalesceVerbatimPassthroughTests()
  echo "── Section 14. divide tuple-tiler stream rewrite ──"
  runDivideTupleTilerTests()
  echo "── Section 15. hier_unzip zipped gather stream rewrite ──"
  runZippedGatherTests()
  echo "── Section 16. inverse strides after a runtime shape leaf ──"
  runInverseDynamicShapeTests()
  echo "── Section 17. compose with a symbol-bound tiler ──"
  runComposeSymbolTilerTests()
  echo "── Section 18. make_layout_like over a scalar shape ──"
  runMakeLayoutLikeScalarShapeTests()
  echo "  All tests passed."

when isMainModule:
  runTests()
