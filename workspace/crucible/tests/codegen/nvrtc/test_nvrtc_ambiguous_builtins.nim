## NVRTC: ambiguous builtins (min, max, abs) work in 3 scenarios:
##   1. Called directly within a CUDA kernel
##   2. Called from a pure Nim host function (outside cuda block)
##   3. Called from the SAME function used in both host and device contexts
##
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_ambiguous_builtins.nim
import std/strformat
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════
# Shared function using min/max
# ═════════════════════════════════════════════════════════════════════
proc clampU32(x, lo, hi: uint32): uint32 =
  max(lo, min(x, hi))

# ═════════════════════════════════════════════════════════════════════
# Scenario 2: Host-only usage (pure Nim, no GPU)
# Ensure no ambiguous call warnings (i.e. if we define min/max as {.builtin.})
# ═════════════════════════════════════════════════════════════════════
block:
  let a = clampU32(5'u32, 10'u32, 20'u32)
  doAssert a == 10, &"host clamp(5): {a}"
  let b = clampU32(25'u32, 10'u32, 20'u32)
  doAssert b == 20, &"host clamp(25): {b}"
  let c = clampU32(15'u32, 10'u32, 20'u32)
  doAssert c == 15, &"host clamp(15): {c}"
  echo "  OK — scenario 2 (host)"

# ═════════════════════════════════════════════════════════════════════
# Scenario 1 + 3: Direct calls + same shared function in CUDA kernel
# The function is auto-pulled by Cuda, and calls the Cuda builtin
# ═════════════════════════════════════════════════════════════════════
const kernelCode = cuda:
  proc builtinKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Scenario 1: direct builtin calls
    output[0] = min(10'u32, 20'u32)
    output[1] = max(30'u32, 7'u32)
    output[2] = abs(int32(-5)).uint32
    # Scenario 3: same function used by host — pulled in from outside the cuda block
    output[3] = clampU32(5'u32, 10'u32, 20'u32)
    output[4] = clampU32(25'u32, 10'u32, 20'u32)
    output[5] = clampU32(15'u32, 10'u32, 20'u32)

var buf: array[6, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
engine.run<<(1, 1)>>("builtinKernel", buf, ())
doAssert buf[0] == 10, &"direct min: {buf[0]}"
doAssert buf[1] == 30, &"direct max: {buf[1]}"
doAssert buf[2] == 5, &"direct abs: {buf[2]}"
doAssert buf[3] == 10, &"shared clamp(5): {buf[3]}"
doAssert buf[4] == 20, &"shared clamp(25): {buf[4]}"
doAssert buf[5] == 15, &"shared clamp(15): {buf[5]}"
echo "  OK — scenario 1+3 (direct + shared fn in kernel)"
echo "  OK — all scenarios pass"
