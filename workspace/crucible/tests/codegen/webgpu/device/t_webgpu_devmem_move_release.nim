## WebGPU: a DeviceMem moved into a launch releases exactly once.
##
## launchDeviceFn moves the DeviceMem kernel arguments into the launch block.
## The moved-from binding is still destroyed at scope exit and must not release
## the buffer a second time, wgpu aborts on releasing such a handle.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/t_webgpu_devmem_move_release.nim

import workspace/crucible

proc incKernel(dst: ptr UncheckedArray[uint32], src: ptr UncheckedArray[uint32], n: int) {.inline.} =
  for i in 0 ..< n:
    dst[i] = src[i] + 1'u32

proc runTest() =
  var engine = newEngine(runWebGPU)
  var y = [10'u32, 20, 30, 40]
  var x = [1'u32, 2, 3, 4]
  let yDev = engine.allocDevice(uint32, y.len)
  let xDev = engine.allocDevice(uint32, x.len)
  engine.copyToDeviceFromHost(yDev, addr y[0], y.len)
  engine.copyToDeviceFromHost(xDev, addr x[0], x.len)
  engine.launchDeviceFn incKernel(yDev, xDev, 4) do:
    grid = (1, 1)
    blk = (32, 1)
    lang = "spirv"
  engine.copyToHostFromDevice(addr y[0], yDev, y.len)
  doAssert y == [2'u32, 3, 4, 5], "y = " & $y

when isMainModule:
  runTest()
  echo "OK: DeviceMem move-release survived the launch"
