## CuTe: pointer stride + computed indexing (B15, B20)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_ptr_stride.nim
##
## CuTe kernels compute offsets from thread/block IDs and strides.
import std/strformat
import workspace/crucible

type
  Strided[M, N: static int] = object
    data: array[M * N, uint32]

const kernelCode = cuda:
  proc stridedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x

    # B15: pointer-like stride indexing
    let row: uint32 = 1
    let col: uint32 = 2
    let stride: uint32 = 4
    let offset = row * stride + col  # 1*4 + 2 = 6

    # B20: computed index via row*N + col
    let s = Strided[2, 4](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32, 70'u32, 80'u32])
    output[0] = s.data[0]        # first element
    output[5] = s.data[1 * 4 + 2]  # computed: 70
    output[6] = offset            # verify offset calc
    output[7] = s.data[offset]    # indexed via computed offset: s.data[6] = 70

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[8, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run<<(1, 1)>>("stridedKernel", buf, ())
  doAssert buf[0] == 10,  &"s.data[0]: {buf[0]}"
  doAssert buf[5] == 70,  &"s.data[1*4+2]: {buf[5]}"
  doAssert buf[6] == 6,   &"offset calc: {buf[6]}"
  doAssert buf[7] == 70,  &"s.data[offset]: {buf[7]}"
  echo "  OK — pointer stride patterns"

when isMainModule:
  runTest()
