## CuTe scaling: static-int arithmetic (+ *, complex)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_static_arith.nim
##
## CuTe types are parameterized by compile-time integers:
##   Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>
## The array sizes involve compile-time arithmetic on those ints.
import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

type
  # B01: static addition
  Summed[M, N: static int] = object
    data: array[M + N, uint32]

  # B02: static multiplication
  Multiplied[M, N: static int] = object
    data: array[M * N, uint32]

  # B03: complex (multiplication + addition)
  Complex[M, N, K: static int] = object
    data: array[M * N + K, uint32]

const kernelCode = cuda:
  proc staticArithKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # B01: M + N = 3 + 2 = 5 elements
    let a = Summed[3, 2](data: [1'u32, 2'u32, 3'u32, 4'u32, 5'u32])
    output[0] = a.data[0]
    output[1] = a.data[4]

    # B02: M * N = 2 * 3 = 6 elements
    let b = Multiplied[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    output[2] = b.data[0]
    output[3] = b.data[5]

    # B03: M * N + K = 2*3 + 1 = 7 elements
    let c = Complex[2, 3, 1](data: [100'u32, 200'u32, 300'u32, 400'u32, 500'u32, 600'u32, 700'u32])
    output[4] = c.data[0]
    output[5] = c.data[6]

var buf: array[6, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("staticArithKernel", buf, ())
doAssert buf[0] == 1,   &"summed[0]: {buf[0]}"
doAssert buf[1] == 5,   &"summed[4]: {buf[1]}"
doAssert buf[2] == 10,  &"mult[0]: {buf[2]}"
doAssert buf[3] == 60,  &"mult[5]: {buf[3]}"
doAssert buf[4] == 100, &"complex[0]: {buf[4]}"
doAssert buf[5] == 700, &"complex[6]: {buf[5]}"
echo "  OK — static-int arithmetic (+, *, +*)"
