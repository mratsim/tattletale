## Vulkan: GLSL codegen for structs with dummy/empty fields.
## Object constructors of field-less structs (padded with `char _`) must emit a valid initializer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_dummy_init.nim

import std/[unittest, strformat]
import workspace/crucible

type
  FixMe*[V: static int] = object

const kernelCode = glsl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = glsl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Vulkan - dummy-field initializers":
    test "single dummy struct const":
      var engine = newEngine(runVulkan)
      engine.ingest(kernelCode)
      echo kernelCode
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      var buf: array[1, uint32]
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      check buf[0] == 1

    test "tuple of dummy structs const":
      var engine = newEngine(runVulkan)
      engine.ingest(kernelCode2)
      echo kernelCode2
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      var buf: array[1, uint32]
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      check buf[0] == 1

when isMainModule:
  runTest()
