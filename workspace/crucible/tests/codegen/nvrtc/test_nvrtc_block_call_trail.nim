## gpuBlock(isExpr) with trailing gpuCall — assigned to output
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_block_call_trail.nim 2>&1

import std/strformat
import workspace/crucible/src/codegen/nvrtc

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
  echo kernel

  var data = [1.0'f32, 2.0, 3.0, 4.0]
  var outBuf: array[1, float32]
  var nv = initNvrtc(kernel)
  nv.numBlocks = 1
  nv.threadsPerBlock = 1
  try:
    nv.compile()
    nv.getPtx()
    nv.execute("reproKernel", outBuf, (data, 4'i32))
    echo &"output[0] = {outBuf[0]} (expected {data[0]})"
    if abs(outBuf[0] - data[0]) < 1e-5:
      echo "  OK"
      quit(0)
    else:
      echo "  FAIL"
      quit(1)
  except:
    echo "  FAIL"
    quit(1)
