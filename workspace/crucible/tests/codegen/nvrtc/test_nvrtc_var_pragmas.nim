## NVRTC: var pragma handlers (collectAddressSpace - inject/gensym)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_var_pragmas.nim
##
## Coverage: nim_to_gpu.nim:611-612
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc varPragmaKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 42'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run("varPragmaKernel", buf, ())
  doAssert buf[0] == 42
  echo "  OK (test_nvrtc_var_pragmas)"

when isMainModule:
  runTest()
