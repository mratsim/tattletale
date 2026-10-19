## NVRTC: tuple with generic types (key CuTe pattern)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/types/t_nvrtc_bracket_in_tuple.nim
import std/strformat
import workspace/crucible

type
  MyInt[V: static int] = object
    data: array[V, uint32]

  Vec2[T] = object
    x, y: T

const kernelCode = cuda:
  proc tupleGenericKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Tuple with generic type construction (nnkObjConstr inside tuple)
    let t = (MyInt[4](data: [10'u32, 20'u32, 30'u32, 40'u32]), 99'u32)
    output[0] = t[0].data[0]
    output[1] = t[0].data[3]
    output[2] = t[1]
    # Vec2[uint32] — generic with primitive type arg
    let v = Vec2[uint32](x: 5'u32, y: 6'u32)
    output[3] = v.x + v.y

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[4, uint32]
  var engine = bkCuda.newEngine()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("tupleGenericKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 10, &"tuple[0].data[0]: got {buf[0]}"
  doAssert buf[1] == 40, &"tuple[0].data[3]: got {buf[1]}"
  doAssert buf[2] == 99, &"tuple[1]: got {buf[2]}"
  doAssert buf[3] == 11, &"Vec2[u32]: got {buf[3]}"
  echo "  OK (test_nvrtc_bracket_in_tuple)"

when isMainModule:
  runTest()
