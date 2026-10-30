## GLSL emission: the `not` operator through the buffer-reference pre-pass
##
## Test that a hand-built kernel whose body applies `not` to a bool
## expression prints a GLSL module: the pre-pass that registers
## buffer-reference blocks reads the operand field of `eNot` and must
## not crash on it
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/app/t_vulkan_glsl_not_expr.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/glsl_lang

proc buildNotModule(): CirModule =
  ## One kernel with a bool local bound to `not (a == 0)`
  let i32 = tInt(32)
  let kernel = procD("notKernel", @[param("a", i32)], tVoid(), @[
    varD("nb", tBool(), notE(eqE(sym("a", i32), intLit(0, i32)))),
    retS(),
  ])
  kernel.isKernel = true
  moduleD("notModule", @[], @[kernel])

const emission = static:
  printGlslModule(buildNotModule())

proc runTest() =
  doAssert emission.err.len == 0,
    "GLSL emission refused the `not` expression: " & emission.err
  doAssert "!" in emission.src,
    "GLSL source misses the `not` operator:\n" & emission.src

when isMainModule:
  runTest()
  echo "t_vulkan_glsl_not_expr: GLSL prints a `not` expression"
