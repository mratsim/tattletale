## WebGPU: a kernel launch with an input buffer argument exits cleanly.
##
## Test that a launch with an input buffer argument plus a readback
## returns with exit code 0: the launch moves the argument DeviceMem
## into the launch tuple, the caller's DeviceMem keeps the nil handle,
## and destroying that nil handle must not abort.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/app/t_webgpu_input_arg_launch.nim

import workspace/crucible

const wgsl = webgpu:
  proc addOneKernel(o, a: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    o[0] = a[0] + 1

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(wgsl)
  var input: array[1, uint32] = [41'u32]
  let inDev = engine.allocDevice(uint32, 1)
  engine.copyToDeviceFromHost(inDev, addr input[0], 1)
  let outDev = engine.allocDevice(uint32, 1)
  engine.launchKernel("addOneKernel", outDev, (inDev,))
  var res: array[1, uint32]
  engine.copyToHostFromDevice(addr res[0], outDev, 1)
  doAssert res[0] == 42, "readback = " & $res[0] & ", expected 42"
  echo "  OK - t_webgpu_input_arg_launch"
  flushFile(stdout)

when isMainModule:
  runTest()
