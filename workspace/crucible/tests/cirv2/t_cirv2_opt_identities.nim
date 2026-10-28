# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Optimization-stage fixtures: arithmetic identity folding.
##
## A hand-built kernel exercises each identity shape and its non-folding controls,
## the registered optimization pipeline (folding, then dead-local elimination) runs over it,
## the assertions read the IR delta through the pretty renderer, never printer output.
##
## Folding contract:
## - `mul` by the literal 1 on either side and `div` by it fold to the other operand, int and float
## - `add`/`sub` with the integer literal 0 on the identity side folds to the other operand
## - float `add` with 0 never folds, `-0.0 + 0.0` is `+0.0` and the bit pattern changes
## - `mul` by 0, `mod` by 1 and `sub 0 x` never fold, none is the identity
##
## - Run: `nim c -r --hints:off --warnings:off --outdir:build/tests/cirv2 \
##     --nimcache:nimcache/tests/cirv2 workspace/crucible/tests/cirv2/t_cirv2_opt_identities.nim`

import std/strutils
import
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_build,
  workspace/crucible/src/codegen/ir/cirv2_verify,
  workspace/crucible/src/codegen/ir/cirv2_pretty,
  workspace/crucible/src/codegen/passes/pass_registry,
  workspace/crucible/src/codegen/passes/passes_legalizations,
  workspace/crucible/src/codegen/passes/passes_normalizations,
  workspace/crucible/src/codegen/passes/passes_optimizations,
  workspace/crucible/src/codegen/passes/passes_validations

proc optPipeline(m: CirModule): string =
  ## Shared stages every backend registers, the optimization stage included.
  ## Returns "" when clean, else the first refusal.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerValidationPostPasses()
  result = runPasses(m, reg)

proc identityKernel(): CirProc =
  ## One kernel per identity shape, every folded result stored into
  ## the `dst` aptr so the surviving expression stays observable in the render.
  let
    i32 = tInt(32)
    f32 = tFloat(32)
    aptrD = tAptr(caDevice, f32)
    vi = sym("v", i32)
    vf = sym("w", f32)
  result = procD("kern_identities", @[param("dst", aptrD)], tVoid(), @[
    varD("v", i32, intLit(5, i32)),
    varD("w", f32, fltLit(2.5, f32)),
    # mul by 1, literal on the left
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      convE(f32, mulE(i32, intLit(1, i32), vi))),
    # mul by 1, literal on the right
    asgnS(patE(sym("dst", aptrD), intLit(1, i32)),
      convE(f32, mulE(i32, vi, intLit(1, i32)))),
    # div by 1
    asgnS(patE(sym("dst", aptrD), intLit(2, i32)),
      convE(f32, divE(i32, vi, intLit(1, i32)))),
    # add with 0, both sides
    asgnS(patE(sym("dst", aptrD), intLit(3, i32)),
      convE(f32, addE(i32, addE(i32, vi, intLit(0, i32)), intLit(0, i32)))),
    # sub with 0 on the right
    asgnS(patE(sym("dst", aptrD), intLit(4, i32)),
      convE(f32, subE(i32, vi, intLit(0, i32)))),
    # float mul by 1 folds, float lit form
    asgnS(patE(sym("dst", aptrD), intLit(5, i32)),
      mulE(f32, vf, fltLit(1.0, f32))),
    # nested: mul(1, add(v, 0)) collapses to v in one walk
    asgnS(patE(sym("dst", aptrD), intLit(6, i32)),
      convE(f32, mulE(i32, intLit(1, i32), addE(i32, vi, intLit(0, i32))))),
    # non-folding controls stay
    asgnS(patE(sym("dst", aptrD), intLit(7, i32)),
      convE(f32, mulE(i32, intLit(2, i32), vi))),
    asgnS(patE(sym("dst", aptrD), intLit(8, i32)),
      convE(f32, mulE(i32, vi, intLit(0, i32)))),
    asgnS(patE(sym("dst", aptrD), intLit(9, i32)),
      convE(f32, subE(i32, intLit(0, i32), vi))),
    asgnS(patE(sym("dst", aptrD), intLit(10, i32)),
      convE(f32, modE(i32, vi, intLit(1, i32)))),
    # float add with 0 never folds, -0.0 + 0.0 is +0.0
    asgnS(patE(sym("dst", aptrD), intLit(11, i32)),
      addE(f32, vf, fltLit(0.0, f32))),
    retS(),
  ])
  result.isKernel = true

static:
  let kern = identityKernel()
  verify(kern)
  let m = moduleD("opt_identities", @[], @[kern])
  let err = m.optPipeline()
  doAssert err.len == 0, "optimization pipeline refused the identity kernel: " & err
  let text = renderProc(kern)

  # every identity folded to its surviving operand
  doAssert "(mul (i +32) 1 " notin text, "int mul by 1 survived:\n" & text
  doAssert "(div (i +32) v 1)" notin text, "int div by 1 survived:\n" & text
  doAssert "(add (i +32) v 0)" notin text, "int add 0 on the right survived:\n" & text
  doAssert "(add (i +32) 0 v)" notin text, "int add 0 on the left survived:\n" & text
  doAssert "(sub (i +32) v 0)" notin text, "int sub 0 survived:\n" & text
  doAssert "(mul (f +32) w 1)" notin text, "float mul by 1 survived:\n" & text
  # folded shapes the renders must carry: the sinks read v and w directly
  doAssert "(conv (f +32) v)" in text, "mul/add folds did not collapse to v:\n" & text
  doAssert "(asgn (pat dst 5) w)" in text, "the float mul-by-1 sink did not collapse to w:\n" & text
  # non-folding controls survive verbatim
  doAssert "(mul (i +32) 2 v)" in text, "mul by 2 was folded away:\n" & text
  doAssert "(mul (i +32) v 0)" in text, "mul by 0 was folded away:\n" & text
  doAssert "(sub (i +32) 0 v)" in text, "sub from 0 was folded away:\n" & text
  doAssert "(mod (i +32) v 1)" in text, "mod by 1 was folded away:\n" & text
  doAssert "(add (f +32) w 0.0)" in text, "float add 0 was folded away:\n" & text
