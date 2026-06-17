## NVRTC: result variable insertion in control flow contexts — implicit return
##
## Tests procs that rely on the implicit `result` variable (no explicit `var result`
## or `return result`). maybeInsertResult must insert the declaration.
##
## NOTE: `result` is declared but NOT zero-initialized. Patterns that read `result`
## before writing (e.g. `result = result + expr`) depend on undefined behavior.
## Use explicit `result = 0` or assign before reading.
##
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_result_flow.nim
import std/strformat
import workspace/crucible/src/codegen/nvrtc

# All procs assign to `result` before reading — safe without zero-init.
proc plain(x: uint32): uint32 =
  result = x * 2

proc ifElse(cond: bool; a, b: uint32): uint32 =
  if cond: result = a
  else:    result = b

proc forLoop(start: uint32): uint32 =
  result = 0
  for i in 0 ..< 5:
    result = result + start + uint32(i)

proc forWithIf(n: uint32): uint32 =
  result = 0
  for i in 0 ..< n:
    if i mod 2 == 0: result = result + i
    else:            result = result + (i * 2)

proc ifWithFor(cond: bool; n: uint32): uint32 =
  result = 0
  if cond:
    for i in 0 ..< n: result = result + i
  else:
    result = n * n

# Host verification
block:
  doAssert plain(5) == 10
  doAssert ifElse(true, 10, 20) == 10
  doAssert ifElse(false, 10, 20) == 20
  doAssert forLoop(10) == 60
  doAssert forWithIf(5) == 14
  doAssert ifWithFor(true, 3) == 3
  doAssert ifWithFor(false, 9) == 81
  echo "  OK — host"

# GPU verification (same procs, called from device kernel)
const kernelCode = cuda:
  proc resultFlowKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = plain(5'u32)
    output[1] = ifElse(true, 10'u32, 20'u32)
    output[2] = ifElse(false, 10'u32, 20'u32)
    output[3] = forLoop(10'u32)
    output[4] = forWithIf(5'u32)
    output[5] = ifWithFor(true, 3'u32)
    output[6] = ifWithFor(false, 9'u32)

var buf: array[7, uint32]
var nv = initNvrtc(kernelCode)
nv.numBlocks = 1
nv.threadsPerBlock = 1
nv.compile()
nv.getPtx()
nv.execute("resultFlowKernel", buf, ())
doAssert buf[0] == 10,  &"plain(5): {buf[0]}"
doAssert buf[1] == 10,  &"ifElse(true): {buf[1]}"
doAssert buf[2] == 20,  &"ifElse(false): {buf[2]}"
doAssert buf[3] == 60,  &"forLoop(10): {buf[3]}"
doAssert buf[4] == 14,  &"forWithIf(5): {buf[4]}"
doAssert buf[5] == 3,   &"ifWithFor(true,3): {buf[5]}"
doAssert buf[6] == 81,  &"ifWithFor(false,9): {buf[6]}"
echo "  OK — device"
echo "  OK — all result flow tests pass"
