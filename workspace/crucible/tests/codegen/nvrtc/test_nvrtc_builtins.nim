## NVRTC: cross-platform GPU builtins (min, max, abs)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_builtins.nim
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc builtinKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = min(10'u32, 20'u32)
    output[1] = max(30'u32, 7'u32)
    output[2] = abs(int32(-5)).uint32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[3, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel<<(1, 1)>>("builtinKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 10, &"min: {buf[0]}"
  doAssert buf[1] == 30, &"max: {buf[1]}"
  doAssert buf[2] == 5, &"abs: {buf[2]}"
  echo "  OK — builtins"

when isMainModule:
  runTest()
