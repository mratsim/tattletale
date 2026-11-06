## WebGPU: a pointer-holding struct kernel parameter dissolves to leaves.
##
## Test that the struct's pointer field binds as its own buffer.
## The dissolved reads compute the expected results.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/t_webgpu_ptr_struct_dissolved.nim

import workspace/crucible

type AxpyView = object
  p: ptr UncheckedArray[float32]
  n: int32

const dissolveWgsl = webgpu:
  proc axpyView(output: ptr UncheckedArray[float32], v: AxpyView) {.global.} =
    for i in 0 ..< v.n:
      output[i] = v.p[i] + 1.0'f32

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(dissolveWgsl)
  let oDev = engine.allocDevice(float32, 4)
  let pDev = engine.allocDevice(float32, 4)
  var pData: array[4, float32] = [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32]
  engine.copyToDeviceFromHost(pDev, addr pData[0], 4)
  engine.launchKernel("axpyView", oDev, (pDev, 4'i32))
  var res: array[4, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 4)

  doAssert res[0] == 11.0'f32, "leaf p[0]: got " & $res[0] & ", want 11.0"
  doAssert res[1] == 21.0'f32, "leaf p[1]: got " & $res[1] & ", want 21.0"
  doAssert res[2] == 31.0'f32, "leaf p[2]: got " & $res[2] & ", want 31.0"
  doAssert res[3] == 41.0'f32, "leaf p[3]: got " & $res[3] & ", want 41.0"

  echo "[OK] pointer-holding struct dissolves to leaf bindings on WebGPU"

when isMainModule:
  runTest()
