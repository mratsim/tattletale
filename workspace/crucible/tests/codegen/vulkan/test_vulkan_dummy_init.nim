## Test: Vulkan GLSL codegen for structs with dummy/empty fields
##
## When a struct has no fields (crucible pads it with `char _`),
## the object constructor must emit a valid initializer, not empty braces.
##
## To compile and run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_dummy_init.nim

import std/[unittest, strformat]
import workspace/crucible

type
  FixMe*[V: static int] = object

const kernelCode = vulkan:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = vulkan:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Vulkan - dummy-field initializers":
    test "single dummy struct const":
      var engine = bkVulkan.init()
      engine.ingest(kernelCode)
      echo kernelCode
      var buf: array[1, uint32]
      engine.run("dummyKernel", buf, ())
      check buf[0] == 1

    test "tuple of dummy structs const":
      var engine = bkVulkan.init()
      engine.ingest(kernelCode2)
      echo kernelCode2
      var buf: array[1, uint32]
      engine.run("dummyKernel", buf, ())
      check buf[0] == 1

when isMainModule:
  runTest()
