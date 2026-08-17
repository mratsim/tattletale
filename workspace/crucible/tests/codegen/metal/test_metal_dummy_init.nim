## Metal: structs with dummy/empty fields. When a struct has no fields,
## crucible pads it with a `char _` member. The object constructor
## must emit a valid MSL initializer, not empty braces. Two constexpr shapes
## execute on the device:
##   - a single dummy struct const (FixMe[8])
##   - a tuple of two dummy struct consts (FixMe[1], FixMe[8])
## Each kernel writes a constant that is asserted byte-exact
## after `engine.run()`. The printed initializer carries the type name,
## never bare braces.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_dummy_init.nim

import std/[strutils, unittest]
import workspace/crucible

type
  FixMe*[V: static int] = object

const kernelCode = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - dummy-field initializers":
    test "single dummy struct const":
      var engine = bkMetal.init()
      engine.ingest(kernelCode)
      echo kernelCode
      check kernelCode.contains("FixMe8")
      check not kernelCode.contains("= {}")
      var res: array[1, uint32]
      engine.run("dummyKernel", res, ())
      check res[0] == 1

    test "tuple of dummy structs const":
      var engine = bkMetal.init()
      engine.ingest(kernelCode2)
      echo kernelCode2
      check kernelCode2.contains("FixMe1") and kernelCode2.contains("FixMe8")
      check not kernelCode2.contains("= {")
      var res: array[1, uint32]
      engine.run("dummyKernel", res, ())
      check res[0] == 1

when isMainModule:
  runTest()
