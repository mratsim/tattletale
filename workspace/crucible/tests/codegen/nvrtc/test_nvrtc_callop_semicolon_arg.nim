## A `()` call-operator template expands to a nnkStmtListExpr (let binding +
## final expression), which Crucible converts to gpuBlock(isExpr: true).
## When such a block appears as a function-call argument, the backend
## appends `;` after every statement in the block — including the final
## value-producing one — leaking the semicolon inside the call parens.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_callop_semicolon_arg.nim

import std/[unittest, macros]
import workspace/crucible

{.experimental: "callOperator".}

type MyObj* = object
  data*: ptr UncheckedArray[float32]

# Call-operator template producing nnkStmtListExpr → gpuBlock(isExpr: true)
# Same pattern as TensorView's `()` operator:
#   let tmp = idx
#   obj.data[tmp]
template `()`*(obj: var MyObj; idx: int): var float32 =
  let tmp = idx
  obj.data[tmp]

# Byref function — gpuBlock(isExpr: true) passed as byref arg
proc passthrough(x: var float32): float32 = x

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    var obj = MyObj(data: C)
    # obj(0) produces gpuBlock(isExpr: true).
    # Passing it as a call arg leaks `;` into the parens.
    let a = passthrough(obj(0))

suite "Call-op argument semicolon":
  test "compiles via NVRTC":
    var buf: array[8, float32]
    var engine = bkCuda.init()
    engine.ingest(kernel)
    engine.run<<(1, 1)>>("kernel", buf, ())
