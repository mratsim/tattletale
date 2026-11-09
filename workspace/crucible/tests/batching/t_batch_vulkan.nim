## Batch suite, vulkan runtime.
##
## Test that every batching scenario through the tier-2 public API matches
## an eager launch on the same engine.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/t_batch_vulkan.nim

import workspace/crucible
import ./batch_suite

when isMainModule:
  runBatchSuite(newEngine(runVulkan))
