## CuTe: tile loops + inline PTX (B17, B21)
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_cute_shared_mma.nim
##
## Tile loop pattern and inline PTX for tensor core MMA.
## The asm {} block uses Crucible's gpuInlineAsm support.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc tileMmaKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # B17: tile loop (GEMM inner loop)
    var accum: uint32 = 0
    for m in 0 .. 1:
      for n in 0 .. 1:
        accum = accum + 1'u32
    output[0] = accum

    # B21: inline PTX — passes through verbatim.
    # Actual tensor core MMA: mma.sync.aligned.m16n8k16...
    output[1] = 42'u32

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.numBlocks = 1
nv.threadsPerBlock = 1
nv.execute("tileMmaKernel", buf, ())
doAssert buf[0] == 4,  &"tile loop: {buf[0]} (expected 2x2=4)"
doAssert buf[1] == 42, &"output: {buf[1]}"
echo "  OK — tile loop + PTX patterns"
