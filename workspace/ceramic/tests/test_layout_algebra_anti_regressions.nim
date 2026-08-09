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
##   path of `compose` — mapModesWith's getTypeInst(make_layout(
##   rhsShapes, rhsStrides)) resolves to an nnkSym → "cannot get child
##   of node kind: nnkSym". The flat-RHS and coalescable-LHS paths are
##   unaffected.
##
##   The layoutTypeArgs helper (nnkSym-safe shape/stride type
##   extraction) keeps this path working. The file's compose output is
##   CuTe-flat (single-mode results unwrapped to scalars), matching the
##   unwrap in CuTe's composition_impl.
##
## Section 2 — inline coalesce(make_layout(...)) with a constant layout:
##   the inline form must type-check and stay Int[N]. NOTE: compiles()
##   does NOT capture the failure (returns true) — only a real compile
##   surfaces it, so the guarded case is a direct assert, not a
##   compiles() check.
## Section 3 — complement, multi-mode layout + compile-time bound:
##   complement with a compile-time bound must produce the coalesced
##   result: a statically-1 remainder mode is dropped by coalesce's
##   trailing size-1 discard, and a lone size-1 result is the library's
##   (1):(0) sentinel. Both coalesced values cross-checked against CuTe
##   host-side output.
##
## Section 4 — complement with a runtime shape must produce the same
##   layout as the identical layout spelled with constants.
##

{.experimental: "callOperator".}

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_indexing
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
#  Section 3 — complement must work for multi-mode layouts
# ═══════════════════════════════════════════════════════════════
proc runComplementFixtureTests =
  # Guarded cases — complement with a compile-time bound must produce
  # the coalesced result: the rem mode (ceil_div(460, 512) = 1) is
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

proc runTests =
  echo "\n── layout_algebra anti-regressions (integration) ──"
  echo "── Section 1: compose under module-scope typeof-alias fixture ──"
  runComposeFixtureTests()
  echo "── Section 2: inline coalesce + constant layout ──"
  runCoalesceConstantFixtureTests()
  echo "── Section 3: complement multi-mode + compile-time bound ──"
  runComplementFixtureTests()
  echo "── Section 4: complement runtime shape = static twin ──"
  runComplementDynamicShapeTests()
  echo "  All tests passed."

when isMainModule:
  runTests()
