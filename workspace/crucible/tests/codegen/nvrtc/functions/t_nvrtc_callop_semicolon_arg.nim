## NVRTC: `()` call-operator template as a call argument (block expression)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_callop_semicolon_arg.nim
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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Call-op argument semicolon":
    test "compiles via NVRTC":
      var buf: array[8, float32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernel)
      let oDev = engine.allocDevice(float32, buf.len)
      engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)

when isMainModule:
  runTest()
