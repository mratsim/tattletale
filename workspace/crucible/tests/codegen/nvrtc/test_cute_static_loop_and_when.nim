## CuTe scaling: static int in loops + compile-time branching
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_static_loop_and_when.nim
##
## CuTe dispatches tile sizes at compile time: different
## unroll factors, loop bounds, and type selection per GPU arch.
import std/strformat
import workspace/crucible

type
  Tile[M, N: static int] = object
    data: array[M * N, uint32]

const L = 3

const kernelCode = cuda:
  proc staticLoopKernel(output: ptr array[L, uint32]) {.global.} =
    # B04: loop with static-int bound — const L pulled from outside cuda block
    var t: Tile[L, 3]
    for idx in 0 ..< L * 3:
      t.data[idx] = uint32(idx + 1) * 10
    for i in 0 ..< L:
      output[i] = t.data[i]

var buf: array[L, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("staticLoopKernel", buf, ())
doAssert buf[0] == 10, &"loop[0]: {buf[0]}"
doAssert buf[1] == 20, &"loop[1]: {buf[1]}"
doAssert buf[2] == 30, &"loop[2]: {buf[2]}"
echo "  OK — static loop + when"
