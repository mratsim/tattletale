## Metal launch reusing buffers across two launches.
##
## Test that d = c + b reads the buffer written by c = a + b
## and returns a + b + b.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/c17d42 \
##     workspace/crucible/tests/codegen/metal/device/t_metal_launch_buffer_reuse_across_launches.nim

import workspace/crucible

const addMsl = metal:
  proc addK(output: ptr UncheckedArray[uint32], x: ptr UncheckedArray[uint32], y: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = x[0] + y[0]

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(addMsl)
  let a = engine.allocDevice(uint32, 1)
  let b = engine.allocDevice(uint32, 1)
  let c = engine.allocDevice(uint32, 1)
  let d = engine.allocDevice(uint32, 1)
  let av = 40'u32
  let bv = 2'u32
  engine.copyToDeviceFromHost(a, [av])
  engine.copyToDeviceFromHost(b, [bv])
  engine.launchKernel("addK", c, (a, b))
  engine.launchKernel("addK", d, (c, b))
  var res: array[1, uint32]
  engine.copyToHostFromDevice(res.toOpenArray(0, 1 - 1), d)
  doAssert res[0] == av + 2 * bv, "addK relaunch chain: got " & $res[0]

when isMainModule:
  runTest()
