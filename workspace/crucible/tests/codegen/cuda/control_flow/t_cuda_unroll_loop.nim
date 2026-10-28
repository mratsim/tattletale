## CUDA: `{.unroll.}` marks a device loop for full unrolling
##
## Test that a kernel with an `{.unroll.}`-marked loop compiles through
## NVRTC and the loop computes the right result
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/control_flow/t_cuda_unroll_loop.nim

import workspace/crucible

const unrollCuda = cuda:
  proc unrollKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 4:
      {.unroll.}
      output[i] = uint32(i)

proc runTest() =
  var engine = newEngine(runCuda)
  engine.ingest(unrollCuda)
  doAssert "#pragma unroll" in engine.getArtifact(), "unroll pragma missing in the generated code"
  let oDev = engine.allocDevice(uint32, 4)
  engine.launchKernel("unrollKernel", oDev, ())
  var res: array[4, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 4)
  for i in 0 ..< 4:
    doAssert res[i] == uint32(i), "output[" & $i & "] = " & $res[i]
  echo "  OK - t_cuda_unroll_loop"

when isMainModule:
  runTest()
