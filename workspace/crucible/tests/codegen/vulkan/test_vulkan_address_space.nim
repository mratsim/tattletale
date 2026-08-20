## Vulkan (GLSL): address-space emission for locals.
##
## An unannotated local is an automatic variable: the printer emits no
## qualifier (`private` is not a valid GLSL storage qualifier). `{.smem.}`
## keeps `shared`, at module scope because GLSL allows `shared` only there.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/vulkan --nimcache:nimcache/tests/vulkan \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_address_space.nim

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
      var engine = bkVulkan.init()
      engine.ingest(addressSpaceVk)
      var res: array[1, uint32]
      engine.run<<(grid: (1, 1), blk: (4, 2))>>("addressSpaceKernel", res, ())
      check res[0] == 7'u32

when isMainModule:
  runTest()
