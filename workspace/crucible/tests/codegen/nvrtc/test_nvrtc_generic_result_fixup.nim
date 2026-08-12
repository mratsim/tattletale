## NVRTC: maybeInsertResult generic fixup loop
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_generic_result_fixup.nim
##
## Coverage: nim_to_gpu.nim:898-912
##
## Generic proc with single-expression body (no explicit `result =`)
## to trigger the fixup loop that converts `x * 2` into `result = x * 2`.
import std/strformat
import workspace/crucible

proc double[T](x: T): T {.device.} =
  x * 2

const kernelCode = cuda:
  proc resultFixupKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = double(21'u32)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run("resultFixupKernel", buf, ())
  doAssert buf[0] == 42, &"result fixup: got {buf[0]}"
  echo "  OK (test_nvrtc_generic_result_fixup)"

when isMainModule:
  runTest()
