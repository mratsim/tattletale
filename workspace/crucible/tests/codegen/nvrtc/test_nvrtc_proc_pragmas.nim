## NVRTC: proc pragma handlers (collectProcAttributes)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_proc_pragmas.nim
##
## Coverage: nim_to_gpu.nim:571-579
import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

proc withPragmas(x: uint32): uint32 {.noinit, noSideEffect, inline.} =
  x * 2

const kernelCode = cuda:
  proc pragmaKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = withPragmas(21'u32)

var buf: array[1, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("pragmaKernel", buf, ())
doAssert buf[0] == 42, &"proc pragmas: got {buf[0]}"
echo "  OK (test_nvrtc_proc_pragmas)"
