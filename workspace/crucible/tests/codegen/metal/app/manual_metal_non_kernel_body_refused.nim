## Metal: a `metal:` body that is not a kernel proc refuses at compile time.
##
## Test that a `metal:` macro applied to a bare literal fails to compile with the frontend
##   refusal "expected a block of kernel proc definitions or a single kernel proc, got nnkIntLit"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/app/manual_metal_non_kernel_body_refused.nim

import workspace/crucible

const code = metal:
  5

proc runTest() =
  echo "UNREACHABLE: a `metal:` macro applied to a bare literal fails to compile with the frontend refusal 'expected a block of kernel proc definitions or a single kernel proc, got nnkIntLit'"

when isMainModule:
  runTest()
