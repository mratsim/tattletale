## OpenCL: per-address-space callee instantiation.
##
## Test that a device function with a pointer parameter compiles and runs
## when its call sites pass pointers derived from different address spaces,
## each call site reading the storage its argument names:
##
## - global buffer, local scratch, private local call sites
## - sentinel-initialized input, launch + byte-exact readback
##
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/functions/t_opencl_space_generics.nim

import workspace/crucible

proc readAt(p: ptr UncheckedArray[float32], i: int32): float32 {.device.} =
  p[i]

const spaceGenCl = opencl:
  proc kern(outp: ptr UncheckedArray[float32]) {.global.} =
    var scratch {.smem.}: array[8, float32]
    var priv: array[8, float32]
    let t = thread_position_in_threadgroup.x
    if t == 0:
      for i in 0 ..< 8:
        scratch[i] = outp[i] + 100.0'f32
        priv[i] = outp[i] + 200.0'f32
    threadgroup_barrier()
    if t == 0:
      outp[8] = readAt(outp, 3)
      outp[9] = readAt(cast[ptr UncheckedArray[float32]](addr scratch[0]), 5)
      outp[10] = readAt(cast[ptr UncheckedArray[float32]](addr priv[0]), 7)

proc runTest() =
  var engine = newEngine(runOpenCL)
  engine.ingest(spaceGenCl)
  let dev = engine.allocDevice(float32, 16)
  var host: array[16, float32]
  for i in 0 ..< 8:
    host[i] = 1000.0'f32 + float32(i)
  engine.copyToDeviceFromHost(dev, addr host[0], 16)
  engine.launchKernel<<(grid: (1, 1), blk: (8, 1))>>("kern", dev, ())
  var got: array[16, float32]
  engine.copyToHostFromDevice(addr got[0], dev, 16)
  # elements 0..7 hold the input sentinels, 8..10 the three call-site reads,
  # 11..15 stay at the uploaded zero init
  let expected: array[16, float32] = [
    1000.0'f32, 1001.0'f32, 1002.0'f32, 1003.0'f32, 1004.0'f32, 1005.0'f32,
    1006.0'f32, 1007.0'f32,
    1003.0'f32,   # global-space argument: outp[3]
    1105.0'f32,   # local argument: scratch[5] = outp[5] + 100
    1207.0'f32,   # private argument: priv[7] = outp[7] + 200
    0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
  for i in 0 ..< 16:
    doAssert got[i] == expected[i],
      "kern readback: element " & $i & " got " & $got[i] &
      " want " & $expected[i]
  echo "[OK] OpenCL: device fn instantiates per argument address space," &
    " all three call sites read their own storage"

when isMainModule:
  runTest()
