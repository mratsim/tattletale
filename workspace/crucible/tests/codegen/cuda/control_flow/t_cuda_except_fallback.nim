## NVRTC: try/except fallback in addProcToGenericInsts
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/control_flow/t_cuda_except_fallback.nim
import std/strformat
import workspace/crucible

# A function that Crucible can translate (happy path)
proc addEm[T](a, b: T): T {.device.} =
  a + b

# This generic should translate fine — verifies the except: path is NOT
# triggered for well-formed generic instantiations.
const kernelCode = cuda:
  proc exceptFallbackKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = addEm(10'u32, 20'u32)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = newEngine(runCuda)
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("exceptFallbackKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 30, &"addEm fallback: got {buf[0]}, expected 30"
  echo "  OK (test_cuda_except_fallback)"

when isMainModule:
  runTest()
