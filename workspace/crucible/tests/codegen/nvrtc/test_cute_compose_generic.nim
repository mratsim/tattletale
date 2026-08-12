## CuTe scaling: generic composition (B08-B14)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_compose_generic.nim
##
## CuTe composes layouts through generic functions that return
## complex types. This tests generic proc chains, nested generics,
## type aliases, and many-parameter types.
import std/strformat
import workspace/crucible

type
  # Basic shape type (like CuTe's Shape)
  Shape[M, N: static int] = object
    rows: uint32
    cols: uint32

  # Layout type (like CuTe's Layout)
  Layout[M, N: static int] = object
    data: array[M * N, uint32]

  # GEMM tile config — many parameters
  GemmTile[M, N, K, Warps, Stages: static int] = object
    a: array[M * K, uint32]
    b: array[K * N, uint32]
    c: array[M * N, uint32]

# Generic factory proc returning composed type
proc makeLayout[M, N: static int](s: Shape[M, N]): Layout[M, N] {.device.} =
  discard

# Generic composition chain (2 levels)
proc gemm[M, N, K: static int](a: array[M * K, uint32]; b: array[K * N, uint32]): array[M * N, uint32] {.device.} =
  var c: array[M * N, uint32]
  for i in 0 .. M * N - 1:
    c[i] = a[i] + b[i]
  result = c

const kernelCode = cuda:
  proc composeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Layout with static dims
    let l = Layout[2, 3](data: [1'u32, 2'u32, 3'u32, 4'u32, 5'u32, 6'u32])
    output[0] = l.data[0]
    output[1] = l.data[5]

    # Multi-param tile config
    let t = GemmTile[2, 2, 2, 1, 1]()
    output[2] = 1'u32

var buf: array[3, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("composeKernel", buf, ())
doAssert buf[0] == 1, &"layout[0]: {buf[0]}"
doAssert buf[1] == 6, &"layout[5]: {buf[1]}"
echo "  OK — generic composition"
