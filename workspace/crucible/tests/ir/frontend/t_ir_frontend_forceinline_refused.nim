## Frontend refusal: `{.forceinline.}` is gone, `inline` is the one splice mark.
##
## Test that a proc marked `{.forceinline.}` refuses at the frontend pragma walk
## with the message naming `inline` as the replacement.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/frontend/t_ir_frontend_forceinline_refused.nim

import std/macros
import std/strutils
import workspace/crucible
import workspace/crucible/src/codegen/frontend/nim_to_crucible

const RefusalMessage = "forceinline was merged into inline; use inline"

macro probe(body: typed): untyped =
  ## Lowers `body` and asserts the forceinline refusal surfaces through
  ## the zero-tolerance channel.
  let (_, err) = lowerCirv2Unverified(body)
  doAssert err.contains(RefusalMessage), err
  result = newEmptyNode()

probe:
  proc spliceMe(a, b: int32): int32 {.forceinline.} =
    result = a + b
