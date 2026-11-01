## CUDA: `{.builtin.}` scalar-math over float operands, launched and read back
##
## `abs` and `exp2` must emit the device math names on CUDA:
##
## - `fabsf` and `fabs` for `abs`, `exp2f` for `exp2`
## - bare `abs` resolves to NVRTC's integer `abs(int)`, a silent
##   float-to-int miscompile
## - bare `exp2` is undeclared
##
## The kernel launches on a CUDA device and the results are read back.
## Without a CUDA device the test refuses loudly:
## that miscompile is invisible at compile time.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/functions/t_cuda_abs_exp2.nim

import workspace/crucible

const absExp2Cuda = cuda:
  proc mathKernel(output: ptr UncheckedArray[float32]) {.global.} =
    let x = output[0]
    output[1] = abs(x)
    output[2] = exp2(x)

const abs64Cuda = cuda:
  proc abs64Kernel(output: ptr UncheckedArray[float64]) {.global.} =
    let x = output[0]
    output[1] = abs(x)

proc runTest() =   # private, tests run in a proc so engines are destroyed at return
  var bufF32: array[3, float32]
  var engine = newEngine(runCuda)
  engine.ingest(absExp2Cuda)

  bufF32[0] = -2.5'f32
  let o32 = engine.allocDevice(float32, bufF32.len)
  engine.copyToDeviceFromHost(o32, addr bufF32[0], bufF32.len)
  engine.launchKernel("mathKernel", o32, ())
  engine.copyToHostFromDevice(addr bufF32[0], o32, bufF32.len)
  echo "  abs(-2.5'f32)   = ", bufF32[1]
  echo "  exp2(-2.5'f32)  = ", bufF32[2]
  doAssert bufF32[1] == 2.5'f32
  doAssert abs(bufF32[2] - 0.1767767'f32) < 1e-6'f32

  var bufF64: array[2, float64]
  engine.ingest(abs64Cuda)
  bufF64[0] = -2.5'f64
  let o64 = engine.allocDevice(float64, bufF64.len)
  engine.copyToDeviceFromHost(o64, addr bufF64[0], bufF64.len)
  engine.launchKernel("abs64Kernel", o64, ())
  engine.copyToHostFromDevice(addr bufF64[0], o64, bufF64.len)
  echo "  abs(-2.5'f64)   = ", bufF64[1]
  doAssert bufF64[1] == 2.5'f64

runTest()
echo "CUDA abs/exp2 builtin names: GREEN"
