## NVRTC: `var result` inside if/else branches (local shadows)
##
## `var result` inside if/else branches creates LOCAL shadows that
## go out of scope. The outer `return result` refers to the implicit
## `result` (zero-initialized), expected to return 0.
##
## NOTE: `return result` currently generates `return ;` (no value) due
## to a pre-existing codegen bug. The `result` variable is never declared
## at the outer scope. Results happen to be 0 on Blackwell but are
## technically UB and may differ on other hardware.
##
## Run with: nim c -r workspace/crucible/tests/codegen/nvrtc/manual_nvrtc_reject_result_ifelse_shadow.nim
import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

const kernelCode = cuda:
  proc shadowIfElse(cond: bool; a, b: uint32): uint32 {.device.} =
    if cond:
      var result = a # The outer implicit result is not modified here and stays 0
    else:
      var result = b # The outer implicit result is not modified here and stays 0
    return result

  proc shadowKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = shadowIfElse(true, 10'u32, 20'u32)
    output[1] = shadowIfElse(false, 10'u32, 20'u32)

var buf: array[2, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
engine.run<<(1, 1)>>("shadowKernel", buf, ())
# Expected: 0 (outer implicit result, never assigned).
# Currently works on Blackwell but relies on `return ;` UB.
doAssert buf[0] == 0, &"shadowIfElse(true): {buf[0]} (expected 0)"
doAssert buf[1] == 0, &"shadowIfElse(false): {buf[1]} (expected 0)"
echo "  OK — shadowIfElse test"
