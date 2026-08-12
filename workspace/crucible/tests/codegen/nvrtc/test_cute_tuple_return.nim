## CuTe: generic proc returning a composed generic type (B10)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_tuple_return.nim
##
## In CuTe, factory procs like make_layout() return composed types.
## This tests that a generic proc returning another generic type works.
import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

const kernelCode = cuda:
  type
    Tile[N: static int] = object
      data: array[N, uint32]

  proc makeTile[N: static int](val: uint32): Tile[N] =
    for i in 0 .. N-1:
      result.data[i] = val

  proc pairReturnKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let a = makeTile[4](42'u32)
    output[0] = a.data[0]
    output[1] = a.data[3]

var buf: array[2, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("pairReturnKernel", buf, ())
doAssert buf[0] == 42, &"a[0]: {buf[0]}"
doAssert buf[1] == 42, &"a[3]: {buf[1]}"
echo "  OK — generic return type (B10)"
