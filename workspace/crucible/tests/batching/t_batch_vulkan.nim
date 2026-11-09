## Batch suite, vulkan backend.
##
## Test that every batching scenario through the public batch API matches
## a `launchDeviceFn` launch of the same kernel.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/crucible/tests/batching/t_batch_vulkan.nim

import workspace/crucible
import ./batch_suite

when isMainModule:
  runBatchSuite(newEngine(runVulkan))
