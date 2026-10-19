## duplicate struct definitions in generated CUDA (tuple constructor)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/types/t_nvrtc_dup_typedef.nim
import std/[unittest]
import workspace/crucible

type Int*[V: static int] = object
  discard

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let t = (Int[8](), Int[16]())
    C[0] = 1.0

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Crucible - duplicate type definitions":
    test "tuple constructor generates single definition":
      let code = kernel
      var output: array[1, float32]
      var engine = bkCuda.newEngine()
      engine.ingest(code)
      let oDev = engine.allocDevice(float32, output.len)
      engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, output.len)
      check output[0] == 1.0

when isMainModule:
  runTest()
