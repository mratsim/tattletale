## WebGPU (WGSL): a user proc named `rsqrt` stays a user call
##
## Test that a user-defined proc named `rsqrt` is emitted and called.
## The readback value is the user proc's result, never the WGSL
## `inverseSqrt` builtin's
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/control_flow/t_webgpu_user_rsqrt.nim

import std/strutils
import workspace/crucible

const userRsqrtWgsl = webgpu:
  proc rsqrt(x: float32): float32 =
    # deliberately not the inverse square root, its value betrays a reroute
    0.5'f32 / x

  proc rsqrtKernel(output: ptr UncheckedArray[float32]) {.global, workgroup: (1, 1, 1).} =
    output[0] = rsqrt(4.0'f32)

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(userRsqrtWgsl)
  let artifact = engine.getArtifact()
  doAssert "inverseSqrt" notin artifact,
    "the user proc named rsqrt was rerouted to the inverseSqrt builtin"
  doAssert "fn rsqrt" in artifact, "the user proc rsqrt is not in the module"
  doAssert "rsqrt(4" in artifact, "the call site does not name the user proc"
  let oDev = engine.allocDevice(float32, 1)
  engine.launchKernel("rsqrtKernel", oDev, ())
  var res: array[1, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 1)
  doAssert res[0] == 0.125'f32, "output[0] = " & $res[0] & ", expected the user proc's 0.5/4"
  echo "  OK - t_webgpu_user_rsqrt"
  flushFile(stdout)

when isMainModule:
  runTest()
