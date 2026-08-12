## NVRTC: nnkBindStmt (bind statement)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_bind_stmt.nim
##
## Coverage: nim_to_gpu.nim:1489-1491
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc bindStmtKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 42'u32

var buf: array[1, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("bindStmtKernel", buf, ())
doAssert buf[0] == 42
echo "  OK (test_nvrtc_bind_stmt)"
