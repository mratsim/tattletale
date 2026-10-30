## WebGPU (WGSL): `{.unroll.}` surfaces in the artifact
##
## Test that a kernel with an `{.unroll.}`-marked loop emits the visible
## marker comment (WGSL has no unroll directive) and the loop computes
## the right result
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/control_flow/t_webgpu_unroll_loop.nim

import std/strutils
import workspace/crucible

const unrollWgsl = webgpu:
  proc unrollKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    for i in 0 ..< 4:
      {.unroll.}
      output[i] = uint32(i)

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(unrollWgsl)
  const unrollComment = "// {.unroll.} requested but not honored on WebGPU"
  doAssert unrollComment in engine.getArtifact(),
    "unroll drop marker missing in the generated WGSL"
  let oDev = engine.allocDevice(uint32, 4)
  engine.launchKernel("unrollKernel", oDev, ())
  var res: array[4, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 4)
  for i in 0 ..< 4:
    doAssert res[i] == uint32(i), "output[" & $i & "] = " & $res[i]
  echo "  OK - t_webgpu_unroll_loop"
  flushFile(stdout)

when isMainModule:
  runTest()
