## NVRTC: var pragma handlers (collectAddressSpace, inject/gensym).
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_var_pragmas.nim
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
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("varPragmaKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  engine.freeDevice(oDev)
  doAssert buf[0] == 42
  echo "  OK (test_nvrtc_var_pragmas)"

when isMainModule:
  runTest()
