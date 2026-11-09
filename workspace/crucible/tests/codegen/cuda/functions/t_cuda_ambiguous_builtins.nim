## NVRTC: ambiguous builtins (min, max, abs) resolve on host code and inside a kernel.
##
## Test that min, max and abs resolve inside a cuda: kernel.
## Test that the user-defined clamp resolves into the kernel body.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/functions/t_cuda_ambiguous_builtins.nim
import std/strformat
import workspace/crucible

proc clampU32(x, lo, hi: uint32): uint32 =
  max(lo, min(x, hi))

proc runTest() =
  block:
    let a = clampU32(5'u32, 10'u32, 20'u32)
    doAssert a == 10, &"host clamp(5): {a}"
    let b = clampU32(25'u32, 10'u32, 20'u32)
    doAssert b == 20, &"host clamp(25): {b}"
    let c = clampU32(15'u32, 10'u32, 20'u32)
    doAssert c == 15, &"host clamp(15): {c}"
    echo "  OK — host"

  const kernelCode = cuda:
    proc builtinKernel(output: ptr UncheckedArray[uint32]) {.global.} =
      output[0] = min(10'u32, 20'u32)
      output[1] = max(30'u32, 7'u32)
      output[2] = abs(int32(-5)).uint32
      # clampU32 is defined outside the cuda block, the pass pulls it into the kernel body
      output[3] = clampU32(5'u32, 10'u32, 20'u32)
      output[4] = clampU32(25'u32, 10'u32, 20'u32)
      output[5] = clampU32(15'u32, 10'u32, 20'u32)

  var buf: array[6, uint32]
  var engine = newEngine(runCuda)
  engine.ingest(kernelCode)
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel<<(1, 1)>>("builtinKernel", oDev, ())
  engine.copyToHostFromDevice(buf, oDev)
  doAssert buf[0] == 10, &"direct min: {buf[0]}"
  doAssert buf[1] == 30, &"direct max: {buf[1]}"
  doAssert buf[2] == 5, &"direct abs: {buf[2]}"
  doAssert buf[3] == 10, &"shared clamp(5): {buf[3]}"
  doAssert buf[4] == 20, &"shared clamp(25): {buf[4]}"
  doAssert buf[5] == 15, &"shared clamp(15): {buf[5]}"
  echo "  OK — direct + shared fn in kernel"
  echo "  OK — all scenarios pass"

when isMainModule:
  runTest()
