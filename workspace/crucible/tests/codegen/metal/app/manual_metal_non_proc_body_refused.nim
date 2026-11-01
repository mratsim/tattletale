## Metal: a `metal:` body that is not a proc refuses at compile time.
##
## Test that a `metal:` macro applied to a non-proc symbol fails to compile with the frontend refusal "gv does not name a proc"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/app/manual_metal_non_proc_body_refused.nim

import workspace/crucible

var gv: uint32 = 7

const code = metal:
  gv

proc runTest() =
  echo "UNREACHABLE: a `metal:` macro applied to a non-proc symbol fails to compile with the frontend refusal 'gv does not name a proc'"

when isMainModule:
  runTest()
