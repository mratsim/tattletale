## NVRTC: ClosedSymChoice overload resolution via bind
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_sym_choice.nim
##
## Tests overload resolution with `bind` creating a ClosedSymChoice.
## The symchoice handler in initGpuGenericInst is currently dead code
## but this test ensures it stays harmless if the code path activates.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc addOne(x: uint32): uint32 {.device.} = x + 1
  proc addOne(x: uint64): uint64 {.device.} = x + 2

  bind addOne

  proc symChoiceKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let a: uint32 = 10
    let b: uint64 = 20
    output[0] = uint32(addOne(a))
    output[1] = uint32(addOne(b))

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.numBlocks = 1
nv.threadsPerBlock = 1
nv.compile()
nv.getPtx()
nv.execute("symChoiceKernel", buf, ())
echo "  output: [", buf[0], ", ", buf[1], "]"
doAssert buf[0] == 11, &"uint32 addOne: {buf[0]} != 11"
doAssert buf[1] == 22, &"uint64 addOne: {buf[1]} != 22"
echo "  OK — SymChoice resolution"
