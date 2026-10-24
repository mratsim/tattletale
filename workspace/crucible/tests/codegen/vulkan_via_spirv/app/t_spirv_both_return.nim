## SPIR-V direct: if-statements whose every branch returns
##
## Test that a guard-style kernel with returns in both if branches compiles and reads back exact values on MoltenVK
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_both_return.nim

import workspace/crucible

const bothRetSpv = spirv:
  proc bothRetKernel(output: ptr UncheckedArray[uint32], x: int32) {.global.} =
    # both branches return: the selection merge block ends up
    # predecessor-less, an unreachable merge block is legal SPIR-V
    if x == 0:
      output[0] = 1
      return
    else:
      output[0] = 2
      return

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(bothRetSpv)
  let outDev = engine.allocDevice(uint32, 1)

  engine.launchKernel("bothRetKernel", outDev, (0'i32,))
  var o: array[1, uint32]
  engine.copyToHostFromDevice(addr o[0], outDev, 1)
  doAssert o[0] == 1, "x=0 got " & $o[0] & ", want 1"

  engine.launchKernel("bothRetKernel", outDev, (5'i32,))
  engine.copyToHostFromDevice(addr o[0], outDev, 1)
  doAssert o[0] == 2, "x=5 got " & $o[0] & ", want 2"
  echo "  both-branches-return: [", o[0], "] exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_both_return: all checks passed"
