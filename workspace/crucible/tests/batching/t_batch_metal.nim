## Batch suite, metal runtime.
##
## Test that every batching scenario through the public batch API matches
## an eager launch on the same engine.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/t_batch_metal.nim

import workspace/crucible
import ./batch_suite

when isMainModule:
  runBatchSuite(newEngine(runMetal))
