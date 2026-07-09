## CuTe: factory functions returning composed types (B14)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_factory.nim
##
## CuTe composes layouts through generic factory functions.
## Note: for-loop bounds use 0 .. M-1 / 0 .. N-1 (Nim inclusive range)
##       which generates i<M / i<N in C codegen.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  Layout[M, N: static int] = object
    data: array[M * N, uint32]

proc makeLayout[M, N: static int](val: uint32): Layout[M, N] {.device.} =
  # CuTe-style factory: fill entire layout with val.
  # Loop bound = M-1 (Nim inclusive .. generates i<M in C).
  for i in 0 .. M-1:
    for j in 0 .. N-1:
      result.data[i * N + j] = val

const kernelCode = cuda:
  proc factoryKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let l = makeLayout[2, 3](42'u32)
    output[0] = l.data[0]
    output[1] = l.data[5]

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("factoryKernel", buf, ())
doAssert buf[0] == 42, &"factory[0]: {buf[0]}"
doAssert buf[1] == 42, &"factory[5]: {buf[1]}"
echo "  OK — factory pattern"
