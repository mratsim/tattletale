## CUDA: a recursive device function refuses at compile time.
##
## Test that a self-recursive `cuda:` device function refuses at compile time
##   with the pass refusal naming
##   "recursive device function `down` would not terminate, recursion is rejected"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/device/manual_cuda_device_fn_recursion_refused.nim

import workspace/crucible

const code = cuda:
  proc down(n: int32): int32 {.device.} =
    if n <= 0:
      0
    else:
      down(n - 1)
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = down(3).uint32

proc runTest() =
  echo "UNREACHABLE: the recursive device function must refuse at compile time"

when isMainModule:
  runTest()
