## Type resolution edge case: an ObjConstr with an Empty type child, a generic
## type instance in const context losing the BracketExpr type node.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/types/t_nvrtc_type_resolution.nim

import std/[unittest, strformat]
import workspace/crucible

type
  MyBox*[V: static int] = object
    val: uint32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Crucible - type resolution edge cases":
    test "ObjConstr empty child (generic type in const)":
      const kernelCode = cuda:
        proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
          const x {.genSym.} = MyBox[1]()
          C[0] = x.val + 1'u32

      var buf: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(kernelCode)
      let oDev = engine.allocDevice(sizeof(buf))
      engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
      engine.freeDevice(oDev)
      check buf[0] == 1

when isMainModule:
  runTest()
