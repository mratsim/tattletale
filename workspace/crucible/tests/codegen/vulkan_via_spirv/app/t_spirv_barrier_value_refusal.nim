## SPIR-V value-position barrier refusal, IR level.
##
## Test that a barrier call in value position refuses with no words emitted
## and a refusal naming the barrier as a statement. The operand is a logical
## and over a hand-built CIRv2 module.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_barrier_value_refusal.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/spirv_shader_lang

proc runTest() =
  let outTy = tAptr(caDevice, tUint(32))
  let barrierSym = sym("threadgroup_barrier", tVoid(), true)
  barrierSym.synchroBuiltin = cbkThreadgroupBarrier
  let barrier = callE(barrierSym)
  let kernel = procD("barrierValueKernel", @[param("output", outTy)], tVoid(), @[
    asgnS(atE(sym("output", outTy), intLit(0, tInt(32))),
      andE(falseE(), barrier)),
    retS()
  ])
  kernel.isKernel = true
  let m = moduleD("barrier_value_refusal", @[], @[kernel])
  let (words, err) = printSpirvModule(m)
  doAssert err.len > 0, "a value-position barrier emitted " & $words.len &
    " words instead of refusing"
  # refusal contract: the message names the barrier as a statement.
  # no words emit, result id 0 never enters an operand list
  let namesStatement = "barrier is a statement" in err
  doAssert namesStatement, "refusal does not name the barrier: " & err
  doAssert words.len == 0, "a refused emission still produced " & $words.len & " words"
  echo "  refusal: " & err

when isMainModule:
  runTest()
  echo "t_spirv_barrier_value_refusal: all checks passed"
