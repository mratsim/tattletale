## nnkMixinStmt and nnkBindStmt, compile-time resolution hints, are skipped by
## the `cuda:` codegen.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_mixin_bind.nim

import std/[unittest]
import workspace/crucible

# A template that uses `mixin` to bring a proc into scope
template callMixin(body: untyped): untyped =
  mixin dummyProc
  block:
    body

type
  Int*[V: static int] = object
    discard

# Use the template inside a cuda: block — this generates a mixin stmt
const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    callMixin:
      let x = Int[8]()
    C[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Crucible - mixin/bind statements":
    test "template with mixin inside cuda: compiles":
      var output: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(kernel)
      let oDev = engine.allocDevice(uint32, output.len)
      engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, output.len)
      check output[0] == 1

when isMainModule:
  runTest()
