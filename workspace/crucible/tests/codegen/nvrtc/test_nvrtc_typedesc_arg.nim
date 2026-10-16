## NVRTC: typedesc-argument erasure, a type passed as a value to a generic
## helper erases the argument and the matching callee param from the signature.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_typedesc_arg.nim
import std/strformat
import workspace/crucible

# Generic helper mirroring the make_tensor typedesc-param shape:
#   func make_tensor*[Sh, St, T](_: typedesc[T]; L: Layout[Sh, St]) = ...
# The caller passes the type as a value; the callee never reads it.
proc scaleType*[T](_: typedesc[T]; v, k: T): T =
  ## Scales v by k. The typedesc param `_` is compile-time only — it carries
  ## no runtime value and must be erased from the emitted signature.
  v * k

# Same shape with the typedesc param in the MIDDLE, and non-typedesc args
# both before and after it — verifies non-typedesc args keep their order.
proc mixArgs*[T](a: T; _: typedesc[T]; v, b: T): T =
  v * b + a

const kernelCode = cuda:
  proc typedescKernel(output: ptr UncheckedArray[float32]) {.global.} =
    output[0] = scaleType(float32, 2.5'f32, 4.0'f32)
    output[1] = scaleType(float32, 3.0'f32, 2.0'f32)
    output[2] = mixArgs(2.0'f32, float32, 2.5'f32, 1.5'f32)
    output[3] = mixArgs(0.0'f32, float32, 3.0'f32, 2.0'f32)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[4, float32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel<<(1, 1)>>("typedescKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 10.0, &"scaleType(float32, 2.5, 4.0): {buf[0]}"
  doAssert buf[1] == 6.0, &"scaleType(float32, 3.0, 2.0): {buf[1]}"
  doAssert buf[2] == 5.75, &"mixArgs(2.0, float32, 2.5, 1.5): {buf[2]}"
  doAssert buf[3] == 6.0, &"mixArgs(0.0, float32, 3.0, 2.0): {buf[3]}"
  echo "  OK — typedesc-arg erasure compiles and executes correctly"

when isMainModule:
  runTest()
