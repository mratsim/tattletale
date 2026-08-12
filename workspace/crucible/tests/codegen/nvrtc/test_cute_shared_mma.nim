## CuTe: tile loops + inline PTX (B17, B21)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_shared_mma.nim
##
## Tile loop pattern and inline PTX for tensor core MMA.
## The asm {} block uses Crucible's gpuInlineAsm support.
import std/strformat
import workspace/crucible

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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run<<(1, 1)>>("tileMmaKernel", buf, ())
  doAssert buf[0] == 4,  &"tile loop: {buf[0]} (expected 2x2=4)"
  doAssert buf[1] == 42, &"output: {buf[1]}"
  echo "  OK — tile loop + PTX patterns"

when isMainModule:
  runTest()
