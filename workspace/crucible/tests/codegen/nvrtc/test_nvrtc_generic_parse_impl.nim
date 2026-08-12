## NVRTC: parseGenericImpl — ObjectTy, StaticTy, distinct code paths
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_generic_parse_impl.nim
##
## Coverage: nim_to_gpu.nim:100-115
import std/strformat
import workspace/crucible

type
  ## ObjectTy — generic object with fields
  Vec2[T] = object
    x, y: T

  ## StaticTy — static int generic param
  Sized[N: static int] = object
    val: uint32

const kernelCode = cuda:
  proc genericImplKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # ObjectTy generic
    let v = Vec2[uint32](x: 10'u32, y: 20'u32)
    output[0] = v.x + v.y
    # StaticTy generic
    let s = Sized[64](val: 42'u32)
    output[1] = s.val

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run("genericImplKernel", buf, ())
  doAssert buf[0] == 30, &"objectTy: got {buf[0]}"
  doAssert buf[1] == 42, &"staticTy: got {buf[1]}"
  echo "  OK (test_nvrtc_generic_parse_impl)"

when isMainModule:
  runTest()
