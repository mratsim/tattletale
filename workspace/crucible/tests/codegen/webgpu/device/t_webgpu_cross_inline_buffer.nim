## WebGPU: an inlined crossing call writes through the kernel's buffer binding.
##
## Test that a device function taking the kernel's buffer pointer inlines into the kernel
## and the buffer argument substitutes directly, with no fresh pointer-typed local.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/t_webgpu_cross_inline_buffer.nim

import workspace/crucible

const inlineWgsl = webgpu:
  proc writeAt(buf: ptr UncheckedArray[uint32], i: uint32, v: uint32) {.device.} =
    buf[i] = v
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    writeAt(output, 0'u32, 42'u32)
    writeAt(output, 1'u32, 7'u32)

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runWebGPU)
  engine.ingest(inlineWgsl)
  let oDev = engine.allocDevice(uint32, 2)
  engine.launchKernel("kernelMain", oDev, ())
  var out32: array[2, uint32]
  engine.copyToHostFromDevice(addr out32[0], oDev, 2)
  doAssert out32[0] == 42'u32,
    "first write through the inlined buffer binding got " & $out32[0] & ", want 42"
  doAssert out32[1] == 7'u32,
    "second write through the inlined buffer binding got " & $out32[1] & ", want 7"

when isMainModule:
  runTest()
  echo "t_webgpu_cross_inline_buffer: all checks passed"
