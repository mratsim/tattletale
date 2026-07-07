## Minimal reproducer for post-fix assertion:
##   `argTyp.kind == gtPtr` in `getType(gpuDeref)`
##
## Root cause: `getType(gpuBlock)` can't determine the type when the block's
## trailing expression is a `gpuCall` (getType returns dfl()), or when it
## scans backward and picks a variable whose type doesn't have a `.data` field.
## Then `getFieldType` returns `gtInvalid`, and `gpuDeref` asserts.
##
## Pattern: `block:` with local temps, trailing CALL (not ident/objconstr).
## The gpuCall's return type is unknown to getType → dfl() → chain breaks.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/repro_block_call_trail.nim 2>&1

import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  MyView = object
    data: ptr UncheckedArray[float32]
    len: int32

# A function whose return type is known to crucible (registered as device fn)
proc makeView(ptrVal: ptr UncheckedArray[float32]; n: int32): MyView =
  MyView(data: ptrVal, len: n)

# Template: block: with temps + trailing CALL (not ident/objconstr)
# getType(gpuCall) → dfl() → getType can't determine block's result type
template wrapView(ptrVal: ptr UncheckedArray[float32]; n: int32): MyView =
  block:
    let a = ptrVal
    let b = n
    makeView(a, b)

# Device function: inline `()` + .data[0] triggers the chain:
proc deviceFn(view: MyView) =
  # Use wrapView inline: block: with temps + trailing CALL
  # The GPU AST: gpuIndex(gpuDeref(gpuDot(gpuBlock(isExpr, [let a, let b, gpuCall(makeView)]), "data")), 0)
  # getType(gpuBlock) → getType(gpuCall) → dfl() → gpuDeref asserts
  discard wrapView(view.data, view.len).data[0]

const kernel = cuda:
  proc reproKernel(
    output: ptr UncheckedArray[float32],
    input: ptr UncheckedArray[float32],
    n: int32,
  ) {.global.} =
    let v = makeView(input, n)
    deviceFn(v)

when isMainModule:
  echo "Testing block with trailing call..."
  var nv = initNvrtc(kernel)
  try:
    nv.compile()
    echo "  OK — compiled"
  except:
    echo "  FAIL"
    quit(1)
