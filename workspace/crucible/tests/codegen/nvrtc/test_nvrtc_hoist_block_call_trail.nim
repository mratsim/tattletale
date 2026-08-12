## gpuBlock(isExpr) with trailing gpuCall — assigned to output
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_block_call_trail.nim 2>&1

import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

type
  MyView = object
    data: ptr UncheckedArray[float32]
    len: int32

proc makeView(ptrVal: ptr UncheckedArray[float32]; n: int32): MyView =
  MyView(data: ptrVal, len: n)

template wrapView(ptrVal: ptr UncheckedArray[float32]; n: int32): MyView =
  block:
    let a = ptrVal
    let b = n
    makeView(a, b)

proc deviceFn(view: MyView, output: ptr UncheckedArray[float32]) =
  output[0] = wrapView(view.data, view.len).data[0]

const kernel = cuda:
  proc reproKernel(
    output: ptr UncheckedArray[float32],
    input: ptr UncheckedArray[float32],
    n: int32,
  ) {.global.} =
    let v = makeView(input, n)
    deviceFn(v, output)

when isMainModule:
  echo "════════ kernel ═══════════════════════════════════════════════════════"
  echo kernel
  echo "═══════════════════════════════════════════════════════════════════════"

  var data = [1.0'f32, 2.0, 3.0, 4.0]
  var outBuf: array[1, float32]
  var engine = bkCuda.init()
  engine.ingest(kernel)
  engine.run<<(1, 1)>>("reproKernel", outBuf, (data, 4'i32))
  doAssert abs(outBuf[0] - data[0]) < 1e-5, "output[0] = " & $outBuf[0] & " (expected " & $data[0] & ")"
  echo "  OK (test_nvrtc_hoist_block_call_trail)"
