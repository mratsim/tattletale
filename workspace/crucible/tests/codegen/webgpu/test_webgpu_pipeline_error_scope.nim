## WebGPU: pipeline creation pops a validation error scope around wgpuDeviceCreateComputePipeline, a cache miss must not hang the callback.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_pipeline_error_scope.nim

import workspace/crucible

const kernelDouble = webgpu:
  proc doubleKernel(output: ptr UncheckedArray[uint32], input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 2'u32
    output[1] = input[1] * 2'u32
    output[2] = input[2] * 2'u32
    output[3] = input[3] * 2'u32

proc runTest() =
  # First pipeline creation is a cache miss: the error scope wraps pipeline creation and the callback must fire.
  var engine = bkWGSL.init()
  engine.ingest(kernelDouble)
  var res: array[4, uint32]
  var a = [1'u32, 2'u32, 3'u32, 4'u32]
  let aDev = engine.allocDevice(a.len * sizeof(uint32))
  let oDev = engine.allocDevice(res.len * sizeof(uint32))
  engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
  engine.launchKernel("doubleKernel", oDev, (aDev,))
  engine.copyToHostFromDevice(addr res[0], oDev, res.len * sizeof(uint32))
  engine.freeDevice(aDev)
  engine.freeDevice(oDev)
  doAssert res == [2'u32, 4, 6, 8], $res

when isMainModule:
  runTest()
