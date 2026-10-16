## {.inject.} let-bindings from recursive template expansion stay unique
## after block flattening, duplicates rename to `acc_4`, `it_a_5`, etc.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_inject_dedup.nim

import std/[unittest]
import workspace/crucible

type
  Wrapper* = object
    val*: int32

template foldThrice(body: untyped): untyped =
  block:
    let it_a {.inject.} = Wrapper(val: 10)
    let it_b {.inject.} = Wrapper(val: 1)
    body
  block:
    let it_a {.inject.} = Wrapper(val: 20)
    let it_b {.inject.} = Wrapper(val: 2)
    body
  block:
    let it_a {.inject.} = Wrapper(val: 30)
    let it_b {.inject.} = Wrapper(val: 3)
    body

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[int32]) {.global.} =
    var acc: int32 = 0
    foldThrice:
      acc = acc + it_a.val * it_b.val
    C[0] = acc

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Inject variable dedup":
    test "no duplicate declarations after block flattening":
      var output: array[1, int32]
      var engine = bkCuda.init()
      engine.ingest(kernel)
      let oDev = engine.allocDevice(sizeof(output))
      engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, sizeof(output))
      engine.freeDevice(oDev)
      # 10*1 + 20*2 + 30*3 = 10 + 40 + 90 = 140
      check output[0] == 140

when isMainModule:
  runTest()
