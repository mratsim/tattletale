## WebGPU: the bit-reinterpret cast reinterprets the operand's bits.
##
## Test that `cast[uint32](float32)` emits WGSL's `bitcast`, moving the bit pattern.
## The readback equals the host Nim `cast` bit for bit.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/t_webgpu_bit_cast.nim

import workspace/crucible

const bitsToUintWgsl = webgpu:
  proc bitsToUintKernel(output: ptr UncheckedArray[uint32], inBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    output[lane] = cast[uint32](inBuf[lane])

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runWebGPU)
  engine.ingest(bitsToUintWgsl)
  var inp: array[8, float32]
  var expected: array[8, uint32]
  for lane in 0 ..< 8:
    inp[lane] = float32(-2.0) - float32(lane)
    expected[lane] = cast[uint32](inp[lane])
  let iDev = engine.allocDevice(float32, 8)
  let oDev = engine.allocDevice(uint32, 8)
  engine.copyToDeviceFromHost(iDev, addr inp[0], 8)
  engine.launchKernel("bitsToUintKernel", oDev, (iDev,))
  var res: array[8, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 8)
  for lane in 0 ..< 8:
    doAssert res[lane] == expected[lane],
      "bitcast readback " & $res[lane] & " != host cast " & $expected[lane]
  echo "  OK: cast[uint32](float32) moves the bit pattern over 8 lanes"

when isMainModule:
  runTest()
