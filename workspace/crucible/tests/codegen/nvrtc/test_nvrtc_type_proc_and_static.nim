## NVRTC: ntyProc / ntyStatic branches in nimToGpuType
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_type_proc_and_static.nim
##
## Coverage: nim_to_gpu.nim:431-449
import std/strformat
import workspace/crucible

type
  ## ntyStatic: array sized by static int
  Buffer[N: static int] = object
    data: array[N, uint32]

  ## Proc type — ntyProc
  FnPtr = proc(x: uint32): uint32 {.cdecl.}

proc applyTwice(fn: FnPtr; x: uint32): uint32 {.device.} =
  result = fn(fn(x))

const kernelCode = cuda:
  proc typeBranchesKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # ntyStatic: Buffer[N] with N as a static parameter
    let b = Buffer[4](data: [1'u32, 2'u32, 3'u32, 4'u32])
    output[0] = b.data[0]
    output[1] = b.data[3]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run("typeBranchesKernel", buf, ())
  doAssert buf[0] == 1, &"ntyStatic buf[0]: got {buf[0]}"
  doAssert buf[1] == 4, &"ntyStatic buf[1]: got {buf[1]}"
  echo "  OK (test_nvrtc_type_proc_and_static)"

when isMainModule:
  runTest()
