## CPU engine: types and functions through the CIRv2-to-C printer.
##
## Test that bit-reinterpret casts preserve the bit pattern in both directions.
## Test that a device callee runs from the kernel body and mixed-width integer division, modulo and conversions are exact.
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc bitsToUintKernel(outBuf: ptr UncheckedArray[uint32]) =
  let lane = thread_index_in_threadgroup
  let f = float32(-2.0) - float32(lane)
  outBuf[lane] = cast[uint32](f)

proc uintToBitsKernel(outBuf: ptr UncheckedArray[float32]) =
  let lane = thread_index_in_threadgroup
  let u = 0xC0000000'u32 + uint32(lane)
  outBuf[lane] = cast[float32](u)

proc doubleIt(x: int32): int32 =
  return x * 2

proc calleeKernel(outBuf: ptr UncheckedArray[int32]) =
  let lane = thread_index_in_threadgroup
  outBuf[lane] = doubleIt(int32(lane) + 1)

proc divModKernel(outBuf: ptr UncheckedArray[uint32]) =
  let lane = thread_index_in_threadgroup
  outBuf[lane] = uint32(lane div 2) * 10'u32 + uint32(lane mod 3)

proc widthKernel(outBuf: ptr UncheckedArray[int64]) =
  let lane = thread_index_in_threadgroup
  let small: int32 = int32(lane) * 3
  outBuf[lane] = int64(small) * 1000'i64

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runCpu)

  block:   # cast[uint32](float32) preserves the bit pattern
    var buf: array[4, uint32]
    engine.launchCpuFn bitsToUintKernel(cast[ptr UncheckedArray[uint32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (4, 1)
    doAssert buf[0] == 0xC0000000'u32, "bitsToUint lane 0: got " & $buf[0]
    doAssert buf[1] == 0xC0400000'u32, "bitsToUint lane 1 (-3.0): got " & $buf[1]
    doAssert buf[2] == 0xC0800000'u32, "bitsToUint lane 2 (-4.0): got " & $buf[2]
    echo "[OK] cast[uint32](float32) preserves the bit pattern"

  block:   # cast[float32](uint32) preserves the bit pattern
    var buf: array[4, float32]
    engine.launchCpuFn uintToBitsKernel(cast[ptr UncheckedArray[float32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (4, 1)
    doAssert cast[uint32](buf[0]) == 0xC0000000'u32, "uintToBits lane 0"
    doAssert cast[uint32](buf[1]) == 0xC0000001'u32, "uintToBits lane 1"
    doAssert buf[0] == -2.0'f32, "uintToBits value: got " & $buf[0]
    echo "[OK] cast[float32](uint32) preserves the bit pattern"

  block:   # .device. callee from the kernel body
    var buf: array[4, int32]
    engine.launchCpuFn calleeKernel(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (4, 1)
    for lane in 0 ..< 4:
      doAssert buf[lane] == int32(lane + 1) * 2,
        "callee lane " & $lane & ": got " & $buf[lane]
    echo "[OK] .device. callee from the kernel body"

  block:   # integer division and modulo
    var buf: array[6, uint32]
    engine.launchCpuFn divModKernel(cast[ptr UncheckedArray[uint32]](addr buf[0])) do:
      grid = (1, 1)
      blk = (6, 1)
    for lane in 0 ..< 6:
      let want = uint32(lane div 2) * 10 + uint32(lane mod 3)
      doAssert buf[lane] == want, "divMod lane " & $lane & ": got " & $buf[lane] & " want " & $want
    echo "[OK] integer division and modulo"

  block:   # mixed integer widths through conversions
    var buf: array[4, int64]
    engine.launchCpuFn widthKernel(cast[ptr UncheckedArray[int64]](addr buf[0])) do:
      grid = (1, 1)
      blk = (4, 1)
    for lane in 0 ..< 4:
      doAssert buf[lane] == int64(lane * 3) * 1000,
        "width lane " & $lane & ": got " & $buf[lane]
    echo "[OK] mixed integer widths through conversions"

runTest()
echo "t_cpu_types_functions: all checks passed"
