## NVRTC: external functions pulled from outside the `cuda` block
## Run with: nim cpp -d:cuda -r workspace/positron/tests/nvrtc/test_nvrtc_external_fn.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
##
## Tests the PR #565 feature: functions defined outside the `cuda` block
## are pulled in automatically when called from GPU code.
import workspace/positron/src/codegen/nvrtc

# ── External device functions (defined outside the `cuda` block) ────────────

proc addThree(a, b, c: uint32): uint32 {.device.} =
  result = a + b + c

proc scaleAndAdd(a, b: uint32; factor: uint32): uint32 {.device.} =
  result = a * factor + b

proc isEven(x: uint32): bool {.device.} =
  result = (x and 1) == 0

proc selectValue(cond: bool; t, f: uint32): uint32 {.device.} =
  if cond:
    result = t
  else:
    result = f

# ── GPU code ────────────────────────────────────────────────────────────────

const kernelCode = cuda:
  proc externalFnKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Calls to functions defined OUTSIDE the `cuda` block
    output[0] = addThree(10'u32, 20'u32, 30'u32)
    output[1] = scaleAndAdd(5'u32, 2'u32, 3'u32)
    let even = isEven(42'u32)
    output[2] = selectValue(even, 100'u32, 200'u32)

# ── Host code ───────────────────────────────────────────────────────────────

var buf: array[3, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"

nv.execute("externalFnKernel", buf, ())
echo "  addThree(10,20,30)  = ", buf[0]
echo "  scaleAndAdd(5,2,3)  = ", buf[1]
echo "  selectValue(isEven(42), 100, 200) = ", buf[2]

doAssert buf[0] == 60  # 10 + 20 + 30
doAssert buf[1] == 17   # 5*3 + 2
doAssert buf[2] == 100  # 42 is even, so true branch

echo "  OK"
