## Vulkan (GLSL): address-space emission for locals.
## Unannotated locals emit no qualifier, `{.smem.}` keeps `shared` (module scope, GLSL allows `shared` only there).
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/serialization/t_vulkan_address_space.nim

import std/[strutils, unittest]
import workspace/crucible

const addressSpaceVk = vulkan:
  var scratch {.smem.}: array[8, uint32]
  proc addressSpaceKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    var x: uint32 = 7
    scratch[0] = x
    output[0] = scratch[0]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - address-space emission for locals":
    test "unannotated locals emit no qualifier, smem stays shared, and run":
      # The GLSL text is the `vulkan:` macro output; `getArtifact()` is SPIR-V.
      check "private uint" notin addressSpaceVk
      check "uint x = 7;" in addressSpaceVk
      check "shared uint scratch[8];" in addressSpaceVk
      var engine = bkVulkan.newEngine()
      engine.ingest(addressSpaceVk)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel<<(grid: (1, 1), blk: (4, 2))>>("addressSpaceKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 7'u32

when isMainModule:
  runTest()
