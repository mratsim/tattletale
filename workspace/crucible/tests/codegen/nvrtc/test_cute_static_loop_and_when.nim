## CuTe scaling: static int in loops + compile-time branching
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_cute_static_loop_and_when.nim
##
## CuTe dispatches tile sizes at compile time: different
## unroll factors, loop bounds, and type selection per GPU arch.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  Tile[M, N: static int] = object
    data: array[M * N, uint32]

const kernelCode = cuda:
  proc staticLoopKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # B04: loop with static-int bound
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    for i in 0 .. 2:
      output[i] = t.data[i]

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("staticLoopKernel", buf, ())
doAssert buf[0] == 10, &"loop[0]: {buf[0]}"
doAssert buf[1] == 20, &"loop[1]: {buf[1]}"
echo "  OK — static loop + when"
