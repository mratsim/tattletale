## SPIR-V direct: a loop-variant step expression re-evaluates per iteration
##
## Test that a countup step carrying a ternary over a body-mutated variable re-evaluates on MoltenVK
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_loop_variant_step.nim

import workspace/crucible

const stepSpv = spirv:
  proc stepKernel(output: ptr UncheckedArray[uint32], n: int32) {.global.} =
    # the step ternary reads the body-mutated variable k
    # so the loop counts five iterations with per-iteration steps
    var count: int32 = 0
    var k: int32 = 0
    for i in countup(0, n, if k < 2: 1 else: 3):
      count = count + 1
      k = k + 1
    output[0] = uint32(count)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(stepSpv)
  let outDev = engine.allocDevice(uint32, 1)
  engine.launchKernel("stepKernel", outDev, (10'i32,))
  var o: array[1, uint32]
  engine.copyToHostFromDevice(addr o[0], outDev, 1)
  # frozen-at-entry the step stays 1 and the loop counts 11
  doAssert o[0] == 5, "loop count got " & $o[0] & ", want 5 (per-iteration step)"
  echo "  loop-variant step: [", o[0], "] exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_loop_variant_step: all checks passed"
