## Vulkan (GLSL): `{.smem.}` local with an initializer reads back the actual values.
##
## Test that the initializer lowers to explicit `shared` stores.
## GLSL takes only a null initializer and honors runtime writes.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_smem_init.nim

import std/unittest
import workspace/crucible

const smemInitGlsl = glsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[4, uint32] = [1'u32, 2, 3, 4]
    output[0] = scratch[0]
    output[1] = scratch[1]
    output[2] = scratch[2]
    output[3] = scratch[3]

proc runTest() =
  suite "Vulkan - {.smem.} initializer rewrite":
    test "shared storage reads back the initialized values":
      var engine = newEngine(runVulkan)
      engine.ingest(smemInitGlsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("k", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 4 - 1), oDev)
      check res == [1'u32, 2, 3, 4]

when isMainModule:
  runTest()
