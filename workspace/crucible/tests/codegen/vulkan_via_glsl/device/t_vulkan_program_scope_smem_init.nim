## Vulkan (GLSL): a program-scope `{.smem.}` var with an initializer reads
## back the actual values.
##
## Test that the program-scope path lowers to a bare `shared` declaration plus
## entry stores. Every kernel of the module assigns the initial values at entry.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_program_scope_smem_init.nim

import std/unittest
import workspace/crucible

const programScopeSmemGlsl = glsl:
  var scratch {.smem.}: array[4, uint32] = [1'u32, 2, 3, 4]
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = scratch[0]
    output[1] = scratch[1]
    output[2] = scratch[2]
    output[3] = scratch[3]

proc runTest() =
  suite "Vulkan - program-scope {.smem.} initializer rewrite":
    test "shared program scope reads back the initialized values":
      var engine = newEngine(runVulkan)
      engine.ingest(programScopeSmemGlsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("k", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 4 - 1), oDev)
      check res == [1'u32, 2, 3, 4]

when isMainModule:
  runTest()
