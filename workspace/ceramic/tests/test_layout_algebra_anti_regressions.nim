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
## fixtures (MMA_LOG 23/24):
##   atoms_nvidia.nim declares its SM80_* fragment layout types as
##   module-scope `typeof(make_layout(...))` aliases. Any module that
##   declares such aliases (this one does, below) breaks the nested-RHS
##   path of `compose` — mapModesWith's getTypeInst(make_layout(
##   rhsShapes, rhsStrides)) resolves to an nnkSym → "cannot get child
##   of node kind: nnkSym". The flat-RHS and coalescable-LHS paths are
##   unaffected.
##
##   Fixed by the layoutTypeArgs helper (cc1f9d8) — nnkSym-safe type
##   extraction — and the asserts below are GREEN. The FILE itself is
##   still red on Sections 2-3 (inline-coalesce/complement, see below).
##   Minimal repro: build/wip/repro_compose_minimal.nim.
##
## Section 2 — inline coalesce(make_layout(...)) with a constant layout:
##   coalesce over an inline make_layout whose shape/stride are plain
##   compile-time constants fails to type-check — the make_layout type
##   stays unresolved in coalesce's argument position → "has no type (or
##   is ambiguous)". The two-step (let + coalesce) works — asserted as
##   controls. Section 3 (complement) hits this same failure: 6ad11fb's
##   template folding made complement's emissions constant. NOTE:
##   compiles() does NOT capture this failure (returns true) — only a
##   real compile surfaces it, so the canary is a direct assert. Isolated
##   repro: build/wip/repro_coalesce_inline.nim.
#### Section 3 — complement, multi-mode layout + compile-time bound
## (introduced by 6ad11fb on ceramic-gpu-gemm, squashed into 0e64309;
## parent 0246087 green, 6ad11fb red — a deliberate codegen change,
## "Fix Int[0] * Int[V] not being folded into 0 at Nim level", that
## accidentally regressed this test, which that commit never compiled):
##   complement emits a ceil_div(bound, Int[cur]()) expression. With a
##   compile-time bound, the emitted coalesce(make_layout(...)) becomes
##   constant (Section 2) and fails to type-check → "no type (or is
##   ambiguous)". The func form (explicit return type) compiles fine.
##   Affects every complement whose layout stays multi-mode after
##   filter_inactive with a compile-time bound; runtime-int and tuple
##   bounds are unaffected (asserted as controls).
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
    check r.shape, ((4,), (8,)), ((Int[4],), (Int[8],))
    check r.stride, ((1,), (64,)), ((Int[1],), (Int[64],))
  block:
    ## rank-1 LHS × nested RHS → composeDistribute path — fixed by
    ## layoutTypeArgs (nnkSym-safe type extraction); must stay working.
    let r = compose(make_layout(32, 1), nested)
    check r.shape, (((4,), (8,)), ((2,), (2,))), (((Int[4],), (Int[8],)), ((Int[2],), (Int[2],)))
    check r.stride, (((16,), (1,)), ((8,), (64,))), (((Int[16],), (Int[1],)), ((Int[8],), (Int[64],)))
  block:
    ## non-coalescable rank-2 LHS × nested RHS → composeDistribute path
    let r = compose(make_layout((16, 8), (1, 32)), nested)
    check r.shape, (((4,), (8,)), ((2,), (2,))), (((Int[4],), (Int[8],)), ((Int[2],), (Int[2],)))
    check r.stride, (((32,), (1,)), ((8,), (128,))), (((Int[32],), (Int[1],)), ((Int[8],), (Int[128],)))

  echo "    compose fixture: 5 cases OK"

# ═══════════════════════════════════════════════════════════════
#  Section 2 — inline coalesce(make_layout(...)) with a constant layout
# ═══════════════════════════════════════════════════════════════
proc runCoalesceConstantFixtureTests =
  # Canary — broken today (inline coalesce over a constant layout):
  # assert the DESIRED result; the file stays red until the
  # inline-coalesce type resolution is fixed (expected value from the
  # two-step form, which takes the working path).
  block:
    let r = coalesce(make_layout((2, ceil_div(16, 8)), (2, 8)))
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (2, 8), (Int[2], Int[8])

  # Controls — must stay working.
  block:
    ## two-step (let + coalesce) — the working path
    let l2 = make_layout((2, ceil_div(16, 8)), (2, 8))
    let r = coalesce(l2)
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (2, 8), (Int[2], Int[8])
  block:
    ## make_layout alone
    let r = make_layout((2, ceil_div(16, 8)), (2, 8))
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (2, 8), (Int[2], Int[8])
  block:
    ## the ceil_div call evaluates to the expected value
    check ceil_div(16, 8), 2, int

  echo "    coalesce constant fixture: 1 canary + 3 controls OK"

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  Section 3 — complement must work for multi-mode layouts
# ═══════════════════════════════════════════════════════════════
proc runComplementFixtureTests =
  # Canaries — broken today (compile-time bound + multi-mode layout):
  # the emitted coalesce(make_layout(...)) becomes constant (Section 2's
  # bug) and fails to type-check; assert the DESIRED result. Expected
  # values from runtime-bound probes, which take the working path; the
  # bound-derived modes must be Int[N] once the static path compiles.
  block:
    let r = complement(make_layout((2, 2), (1, 4)), 16)
    check r.shape, (2, 2), (Int[2], Int[2])
    check r.stride, (2, 8), (Int[2], Int[8])
  block:
    let r = complement(make_layout((2, 4, 8), (8, 1, 64)), 460)
    check r.shape[0], 2, Int
    check r.shape[1], 4, Int
    check r.shape[2], 1, Int
    check r.stride[0], 4, Int
    check r.stride[1], 16, Int
    check r.stride[2], 512, Int
  block:
    let r = complement(make_layout((2, (3, 4)), (3, (1, 6))))
    check r.shape, 1, Int
    check r.stride, 24, Int

  # Controls — must stay working (guards against a different regression
  # in the complement paths that DO compile today).
  block:
    ## runtime-int bound takes the dynamic path — the bound-derived mode
    ## carries a runtime int (not Int[N]), so only the values are
    ## asserted here; the static canary above pins the Int types.
    let b = 16
    let r = complement(make_layout((2, 2), (1, 4)), b)
    doAssert r === ((2, 2), (2, 8))
  block:
    ## tuple bound — same mixed Int/int situation as the runtime bound
    ## (value-only assert)
    let r = complement(make_layout((2, 2), (1, 4)), (4, 4))
    doAssert r === ((2, 2), (2, 8))
  block:
    ## contiguous (coalescable) layout + static bound — unaffected
    let r = complement(make_layout((2, 4), (1, 2)), 16)
    check r.shape, 2, Int
    check r.stride, 8, Int

  echo "    complement fixture: 3 canaries + 3 controls OK"

proc runTests =
  echo "\n── layout_algebra anti-regressions (integration) ──"
  echo "── Section 1: compose under module-scope typeof-alias fixture ──"
  runComposeFixtureTests()
  echo "── Section 2: inline coalesce + constant layout ──"
  runCoalesceConstantFixtureTests()
  echo "── Section 3: complement multi-mode + compile-time bound ──"
  runComplementFixtureTests()
  echo "  All tests passed."

when isMainModule:
  runTests()
