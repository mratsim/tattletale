## Metal: a labeled `block` with a `break label` refuses at compile time.
##
## Test that a `metal:` kernel breaking out of a labeled block fails to compile
##   with the frontend refusal "labeled block: break out of blocks is outside the supported set"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/manual_metal_labeled_block_refused.nim

import workspace/crucible

const code = metal:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    block done:
      output[0] = 1
      break done

proc runTest() =
  echo "UNREACHABLE: a `metal:` kernel breaking out of a labeled block fails to compile with the frontend refusal 'labeled block: break out of blocks is outside the supported set'"

when isMainModule:
  runTest()
