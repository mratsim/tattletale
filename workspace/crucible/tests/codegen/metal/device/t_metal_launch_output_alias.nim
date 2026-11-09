## Metal launch aliasing the output with an argument.
##
## Test that an aliased launch computes in place and reads the result
## back through the same buffer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/c17d42 \
##     workspace/crucible/tests/codegen/metal/device/t_metal_launch_output_alias.nim

import workspace/crucible

const scaleMsl = metal:
  proc scaleBy(output: ptr UncheckedArray[uint32], input: ptr UncheckedArray[uint32], factor: uint32) {.global.} =
    output[0] = input[0] * factor

let one = 3'u32

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(scaleMsl)
  let dev = engine.allocDevice(uint32, 1)
  engine.copyToDeviceFromHost(dev, [one])
  engine.launchKernel("scaleBy", dev, (dev, 2'u32))
  var res: array[1, uint32]
  engine.copyToHostFromDevice(res.toOpenArray(0, 1 - 1), dev)
  doAssert res[0] == 2 * one, "scaleBy in-place: got " & $res[0]

when isMainModule:
  runTest()
