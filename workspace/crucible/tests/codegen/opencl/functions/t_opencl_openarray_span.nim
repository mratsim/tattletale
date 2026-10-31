## OpenCL: openArray spans split to pointer + length (v1 port).
##
## Test that a device function taking an openArray[float32] parameter compiles, runs, and reads back exactly:
##
## - its call sites pass toOpenArray views, sentinel-initialized input,
##   launch + byte-exact readback
## - len reads fold to the split length
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/functions/t_opencl_openarray_span.nim

import workspace/crucible

proc readSpan(s: openArray[float32], i: int32): float32 {.device.} =
  s[i]

proc spanLen(s: openArray[float32]): int32 {.device.} =
  int32(len(s))

const spanCl = opencl:
  proc kern(outp: ptr UncheckedArray[float32]) {.global.} =
    # view read: readSpan(outp[1..4], 2) = outp[3]
    outp[8] = readSpan(toOpenArray(outp, 1, 4), 2)
    # len(outp[2..6]) folds to 5
    outp[9] = float32(spanLen(toOpenArray(outp, 2, 6)))

proc runTest() =
  var engine = newEngine(runOpenCL)
  engine.ingest(spanCl)
  let dev = engine.allocDevice(float32, 16)
  # upload count: 16 elements, the 8 sentinels plus 8 zero elements
  var host: array[16, float32]
  for i in 0 ..< 8:
    host[i] = 1000.0'f32 + float32(i)
  engine.copyToDeviceFromHost(dev, addr host[0], 16)
  engine.launchKernel("kern", dev, ())
  var got: array[16, float32]
  engine.copyToHostFromDevice(addr got[0], dev, 16)
  # elements 0..7 hold the input sentinels, 8..9 the two span results,
  # 10..15 stay at the buffer's zero init
  let expected: array[16, float32] = [
    1000.0'f32, 1001.0'f32, 1002.0'f32, 1003.0'f32, 1004.0'f32, 1005.0'f32,
    1006.0'f32, 1007.0'f32,
    1003.0'f32,   # device view: readSpan(outp[1..4], 2) = outp[3]
    5.0'f32,      # len fold: spanLen(outp[2..6]) = 5
    0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
  for i in 0 ..< 16:
    doAssert got[i] == expected[i],
      "kern readback: element " & $i & " got " & $got[i] &
      " want " & $expected[i]
  echo "[OK] OpenCL: openArray device fn spans split to pointer + length" &
    " and len folds"

when isMainModule:
  runTest()
