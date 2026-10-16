## gpuBlock(isExpr) with a trailing gpuCall assigned to the output.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_hoist_block_call_trail.nim

import std/strformat
import workspace/crucible

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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo "════════ kernel ═══════════════════════════════════════════════════════"
  echo kernel
  echo "═══════════════════════════════════════════════════════════════════════"

  var data = [1.0'f32, 2.0, 3.0, 4.0]
  var outBuf: array[1, float32]
  var engine = bkCuda.init()
  engine.ingest(kernel)
  let iDev = engine.allocDevice(sizeof(data))
  engine.copyToDeviceFromHost(iDev, addr data[0], sizeof(data))
  let oDev = engine.allocDevice(sizeof(outBuf))
  engine.launchKernel<<(1, 1)>>("reproKernel", oDev, (iDev, 4'i32))
  engine.copyToHostFromDevice(addr outBuf[0], oDev, sizeof(outBuf))
  engine.freeDevice(iDev)
  engine.freeDevice(oDev)
  doAssert abs(outBuf[0] - data[0]) < 1e-5, "output[0] = " & $outBuf[0] & " (expected " & $data[0] & ")"
  echo "  OK (test_nvrtc_hoist_block_call_trail)"

when isMainModule:
  runTest()
