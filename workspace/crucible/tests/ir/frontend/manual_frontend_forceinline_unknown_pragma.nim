## Nim sem: `{.forceinline.}` is gone, `{.inline.}` is the one inline mark.
##
## Test that a proc marked `{.forceinline.}` fails Nim semantic analysis:
## the unknown pragma has no template, Nim rejects the name before crucible code runs.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/frontend/manual_frontend_forceinline_unknown_pragma.nim

import workspace/crucible

proc spliceMe(a, b: int32): int32 {.forceinline.} =
  result = a + b
