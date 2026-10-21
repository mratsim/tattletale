## NVRTC: `var T` byref params on device procs via the codegen pipeline.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/serialization/t_cuda_var_param.nim
import workspace/crucible

const kernelCode = cuda:
  type Pair = object
    x: uint32
    y: uint32
  proc setPair(p: var Pair; vx, vy: uint32) {.device.} =
    p.x = vx
    p.y = vy
  proc swap(a, b: var uint32) {.device.} =
    let tmp = a
    a = b
    b = tmp
  proc varParamKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var p: Pair
    setPair(p, 10'u32, 20'u32)
    output[0] = p.x
    output[1] = p.y
    var a, b: uint32 = 1
    b = 2
    swap(a, b)
    output[2] = a
    output[3] = b

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[4, uint32]
  var engine = bkCuda.newEngine()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("varParamKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  echo "  p.x=", buf[0], " p.y=", buf[1], " a=", buf[2], " b=", buf[3]
  doAssert buf[0] == 10
  doAssert buf[1] == 20
  doAssert buf[2] == 2
  doAssert buf[3] == 1
  echo "  OK (test_cuda_var_param)"

when isMainModule:
  runTest()
