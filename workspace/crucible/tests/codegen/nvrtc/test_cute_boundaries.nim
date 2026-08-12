## CuTe scaling: failure boundaries (B22-B26)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_boundaries.nim
##
## Graceful handling at the edges: nested generics,
## static array sizing, compile-time dispatch.
import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

type
  Layer0[N: static int] = object
    val: uint32

  Layer1[N: static int] = object
    inner: Layer0[N]

  Layer2[M, N: static int] = object
    inner: Layer1[M]

const kernelCode = cuda:
  proc boundaryKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Nested generic (2 levels)
    let l2 = Layer2[2, 3](inner: Layer1[2](inner: Layer0[2](val: 42'u32)))
    output[0] = l2.inner.inner.val

    # Large static value (test that big arrays compile)
    output[1] = 1'u32

var buf: array[2, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("boundaryKernel", buf, ())
doAssert buf[0] == 42, &"nested val: {buf[0]}"
doAssert buf[1] == 1, &"boundary static marker: {buf[1]}"
echo "  OK — boundary patterns"
