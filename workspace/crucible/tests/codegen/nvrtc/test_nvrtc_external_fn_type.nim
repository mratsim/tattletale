## NVRTC: external functions + external types + kernel chaining
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_external_fn_type.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
##
## Tests the PR #565 feature end-to-end: functions AND types defined
## entirely outside the `cuda` block, only called/used from within.
## Also tests that multiple kernels sharing the same external code work.
import workspace/crucible/src/codegen/nvrtc

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

var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"

# Kernel 1: tripletSum
var buf1: array[1, uint32]
nv.execute("sumKernel", buf1, ())
echo "  sumKernel (100+200+300) = ", buf1[0]
doAssert buf1[0] == 600

# Kernel 2: compose + reduce
var buf2: array[3, uint32]
nv.execute("maxKernel", buf2, ())
echo "  maxKernel compose(1,2,3) = ", buf2[0]
echo "  maxKernel t.a+b+c        = ", buf2[1]
echo "  maxKernel reduceMax      = ", buf2[2]
doAssert buf2[0] == 6    # 1+2+3
doAssert buf2[1] == 60   # 10+20+30
doAssert buf2[2] == 42   # max(42, 17)

echo "  OK"
