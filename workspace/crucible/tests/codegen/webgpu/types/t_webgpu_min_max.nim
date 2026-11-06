## WebGPU: max/min over f32 compute through the native builtins.
##
## Test that the folded max/min compute the expected results over both operand orders.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/t_webgpu_min_max.nim

import workspace/crucible

const minMaxWgsl = webgpu:
  proc minMax(output: ptr UncheckedArray[float32], input: ptr UncheckedArray[float32]) {.global.} =
    let x = input[0]
    let y = input[1]
    output[0] = max(x, y)
    output[1] = min(x, y)
    output[2] = max(y, x)
    output[3] = max(3.5'f32, 7.25'f32)
    output[4] = min(3.5'f32, 7.25'f32)

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(minMaxWgsl)
  let oDev = engine.allocDevice(float32, 5)
  let iDev = engine.allocDevice(float32, 2)
  var input: array[2, float32] = [4.5'f32, 1.25'f32]
  engine.copyToDeviceFromHost(iDev, addr input[0], 2)
  engine.launchKernel("minMax", oDev, (iDev,))
  var res: array[5, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 5)

  doAssert res[0] == 4.5'f32, "max(x, y): got " & $res[0] & ", want 4.5"
  doAssert res[1] == 1.25'f32, "min(x, y): got " & $res[1] & ", want 1.25"
  doAssert res[2] == 4.5'f32, "max(y, x): got " & $res[2] & ", want 4.5"
  doAssert res[3] == 7.25'f32, "max(literal, literal): got " & $res[3] & ", want 7.25"
  doAssert res[4] == 3.5'f32, "min(literal, literal): got " & $res[4] & ", want 3.5"

  echo "[OK] max/min fold to the native builtins on WebGPU, both operand orders and literals"

when isMainModule:
  runTest()
