## NVRTC: external functions + external types + kernel chaining
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/types/t_cuda_external_fn_type.nim
import workspace/crucible

# ── External types ──────────────────────────────────────────────────────────

type
  Triplet = object
    a, b, c: uint32

# ── External functions ──────────────────────────────────────────────────────

proc initTriplet(x, y, z: uint32): Triplet {.device.} =
  result.a = x
  result.b = y
  result.c = z

proc tripletSum(t: Triplet): uint32 {.device.} =
  result = t.a + t.b + t.c

proc reduceMax(a, b: uint32): uint32 {.device.} =
  if a > b:
    result = a
  else:
    result = b

proc composeValues(a, b, c: uint32): uint32 {.device.} =
  let t = initTriplet(a, b, c)
  result = tripletSum(t)

# ── GPU code (two kernels sharing the same external code) ───────────────────

const kernelCode = cuda:
  proc sumKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let t = initTriplet(100'u32, 200'u32, 300'u32)
    output[0] = tripletSum(t)

  proc maxKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # External functions chaining: composeValues calls initTriplet + tripletSum
    output[0] = composeValues(1'u32, 2'u32, 3'u32)
    # External function using external type Triplet
    let t = initTriplet(10'u32, 20'u32, 30'u32)
    output[1] = t.a + t.b + t.c
    output[2] = reduceMax(42, 17)

# ── Host code ───────────────────────────────────────────────────────────────

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var engine = bkCuda.newEngine()

  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"

  # Kernel 1: tripletSum
  var buf1: array[1, uint32]
  let oDev1 = engine.allocDevice(uint32, buf1.len)
  engine.launchKernel("sumKernel", oDev1, ())
  engine.copyToHostFromDevice(addr buf1[0], oDev1, buf1.len)
  echo "  sumKernel (100+200+300) = ", buf1[0]
  doAssert buf1[0] == 600

  # Kernel 2: compose + reduce
  var buf2: array[3, uint32]
  let oDev2 = engine.allocDevice(uint32, buf2.len)
  engine.launchKernel("maxKernel", oDev2, ())
  engine.copyToHostFromDevice(addr buf2[0], oDev2, buf2.len)
  echo "  maxKernel compose(1,2,3) = ", buf2[0]
  echo "  maxKernel t.a+b+c        = ", buf2[1]
  echo "  maxKernel reduceMax      = ", buf2[2]
  doAssert buf2[0] == 6    # 1+2+3
  doAssert buf2[1] == 60   # 10+20+30
  doAssert buf2[2] == 42   # max(42, 17)

  echo "  OK (test_cuda_external_fn_type)"

when isMainModule:
  runTest()
