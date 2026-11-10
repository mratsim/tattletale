## Batch suite, metal backend.
##
## Test that every batching scenario through the public batch API matches
## a `launchDeviceFn` launch of the same kernel.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/t_batch_metal.nim

import workspace/crucible
import ./batch_suite

proc runTest() =
  runBatchSuite(newEngine(runMetal))

when isMainModule:
  runTest()
