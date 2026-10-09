## WebGPU pipeline creation pops a validation error scope around
## wgpuDeviceCreateComputePipeline:
##
## - the pop callback info must carry a valid WGPUCallbackMode
## - mode 0 is wgpuCallbackModeInvalid, a conforming wgpu-native build can
##   then never call the callback, hanging every pipeline cache miss
## - the dispatch scope already set the mode, the pipeline scope matches it
##
## Run:
##   nim c -r --hints:off --warnings:off workspace/crucible/tests/codegen/webgpu/test_webgpu_pipeline_error_scope.nim (from tattletale)

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
  engine.run("doubleKernel", res, (a,))
  doAssert res == [2'u32, 4, 6, 8], $res

when isMainModule:
  runTest()
