## WebGPU launches reject raw pointer args at compile time: kernel pointer
## params bind as storage buffers, only WGPUBuffer values cross the serializer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/manual_webgpu_ptr_arg_rejected.nim

import workspace/crucible

const code = webgpu:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 1'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runWebGPU)
  engine.ingest(code)
  let oDev = engine.allocDevice(uint32, 1)
  # a raw pointer arg must fail to compile: it would pack garbage bits where
  # the kernel expects a storage-buffer binding
  engine.launchKernel("k", oDev, (cast[ptr UncheckedArray[uint32]](0),))
  echo "UNREACHABLE — raw pointer args must be rejected at compile time"

when isMainModule:
  runTest()
