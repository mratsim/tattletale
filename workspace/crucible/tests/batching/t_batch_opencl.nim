## Batch suite, opencl backend.
##
## Test that every batching scenario through the public batch API matches
## a `launchDeviceFn` launch of the same kernel.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/t_batch_opencl.nim

import workspace/crucible
import ./batch_suite

proc runTest() =
  runBatchSuite(newEngine(runOpenCL))

when isMainModule:
  runTest()
