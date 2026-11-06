## CUDA: known bug, `return result` under if/else shadowing emits `return ;`.
##
## A `var result` inside if/else branches creates LOCAL shadows that go out of scope.
## The outer `return result` reads the implicit outer `result`, never assigned, expected 0.
##
## Known bug, recorded until fixed:
## - `return result` generates `return ;` (no value): the outer `result` is
##   never declared at the outer scope
## - the readback is 0 on Blackwell, UB by the C standard, may differ on other
##   hardware
## - a known-bug record, NOT refusal coverage
##
## Run with: nim c -r workspace/crucible/tests/codegen/cuda/control_flow/manual_cuda_result_shadow_ub_known_bug.nim
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc shadowIfElse(cond: bool; a, b: uint32): uint32 {.device.} =
    if cond:
      var result = a # shadows the implicit outer result
    else:
      var result = b
    return result

  proc shadowKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = shadowIfElse(true, 10'u32, 20'u32)
    output[1] = shadowIfElse(false, 10'u32, 20'u32)

var buf: array[2, uint32]
var engine = newEngine(runCuda)
engine.ingest(kernelCode)
engine.launchKernel<<(1, 1)>>("shadowKernel", buf, ())
doAssert buf[0] == 0, &"shadowIfElse(true): {buf[0]} (expected 0)"
doAssert buf[1] == 0, &"shadowIfElse(false): {buf[1]} (expected 0)"
echo "  OK — shadowIfElse test"
