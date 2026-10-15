## NVRTC: proc pragma handlers (collectProcAttributes).
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_proc_pragmas.nim
import std/strformat
import workspace/crucible

proc withPragmas(x: uint32): uint32 {.noinit, noSideEffect, inline.} =
  x * 2

const kernelCode = cuda:
  proc pragmaKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = withPragmas(21'u32)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(sizeof(buf))
  engine.run("pragmaKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 42, &"proc pragmas: got {buf[0]}"
  echo "  OK (test_nvrtc_proc_pragmas)"

when isMainModule:
  runTest()
