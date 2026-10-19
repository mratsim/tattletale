## NVRTC: maybeInsertResult converts a generic device proc single-expression
## body into a `result =` assignment.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/generics/t_nvrtc_generic_result_fixup.nim
import std/strformat
import workspace/crucible

proc double[T](x: T): T {.device.} =
  x * 2

const kernelCode = cuda:
  proc resultFixupKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = double(21'u32)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.newEngine()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("resultFixupKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 42, &"result fixup: got {buf[0]}"
  echo "  OK (test_nvrtc_generic_result_fixup)"

when isMainModule:
  runTest()
