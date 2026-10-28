## Metal: `{.unroll.}` marks a device loop for full unrolling
##
## Test that a kernel with an `{.unroll.}`-marked loop emits the clang
## full-unroll directive and computes the right result
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_unroll_loop.nim

import std/strutils
import workspace/crucible

const unrollMsl = metal:
  proc unrollKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 4:
      {.unroll.}
      output[i] = uint32(i)

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(unrollMsl)
  # the directive locks the construct to the artifact
  doAssert "#pragma clang loop unroll(full)" in engine.getArtifact(),
    "the unroll directive is missing:\n" & engine.getArtifact()
  let oDev = engine.allocDevice(uint32, 4)
  engine.launchKernel("unrollKernel", oDev, ())
  var res: array[4, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 4)
  for i in 0 ..< 4:
    doAssert res[i] == uint32(i), "output[" & $i & "] = " & $res[i]
  echo "  OK - t_metal_unroll_loop"

when isMainModule:
  runTest()
