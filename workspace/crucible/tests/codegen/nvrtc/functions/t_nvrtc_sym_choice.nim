## NVRTC: ClosedSymChoice overload resolution via `bind`.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_sym_choice.nim
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc addOne(x: uint32): uint32 {.device.} = x + 1
  proc addOne(x: uint64): uint64 {.device.} = x + 2

  bind addOne

  proc symChoiceKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let a: uint32 = 10
    let b: uint64 = 20
    output[0] = uint32(addOne(a))
    output[1] = uint32(addOne(b))

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel<<(1, 1)>>("symChoiceKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  echo "  output: [", buf[0], ", ", buf[1], "]"
  doAssert buf[0] == 11, &"uint32 addOne: {buf[0]} != 11"
  doAssert buf[1] == 22, &"uint64 addOne: {buf[1]} != 22"
  echo "  OK — SymChoice resolution"

when isMainModule:
  runTest()
