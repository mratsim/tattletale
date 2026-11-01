## Metal: scalar float `min`/`max`/`abs`/`exp2` over device values
##
## Test that a kernel computing min, max, abs, and exp2 over float32 device
## values matches the host results
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_scalar_min_max_abs.nim

import workspace/crucible

const code = metal:
  proc scalarMinMaxAbs(output: ptr UncheckedArray[float32]) {.global.} =
    let x = output[0]
    let y = output[1]
    output[2] = min(x, y)
    output[3] = max(x, y)
    output[4] = abs(x)
    output[5] = exp2(x)

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(code)
  var inputs: array[2, float32] = [-2.5'f32, 1.5'f32]
  let oDev = engine.allocDevice(float32, 6)
  engine.copyToDeviceFromHost(oDev, addr inputs[0], 2)
  engine.launchKernel("scalarMinMaxAbs", oDev, ())
  var res: array[6, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 6)
  doAssert res[2] == -2.5'f32, "min = " & $res[2]
  doAssert res[3] == 1.5'f32, "max = " & $res[3]
  doAssert res[4] == 2.5'f32, "abs = " & $res[4]
  # exp2 is a bare-name pass-through on MSL, its device result must match
  # host `pow` within the float32 rounding of 2^(-2.5)
  doAssert abs(res[5] - 0.1767767'f32) <= 1e-6'f32, "exp2 = " & $res[5]
  echo "  OK - t_metal_scalar_math_builtins"

when isMainModule:
  runTest()
