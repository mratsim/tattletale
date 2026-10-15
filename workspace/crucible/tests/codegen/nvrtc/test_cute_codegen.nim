## CuTe scaling: codegen patterns (B15-B21)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_cute_codegen.nim
import std/strformat
import workspace/crucible

type
  Tensor[M, N: static int] = object
    data: array[M * N, uint32]

proc idx(row: uint32; col: uint32; stride: uint32): uint32 {.device.} =
  let tmp = row * stride
  result = tmp + col

const kernelCode = cuda:
  proc codegenKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Grid-stride loop with thread id
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid < 4:
      output[tid] = uint32(tid) + 100'u32

    # Computed index via device function
    let index = idx(1'u32, 2'u32, 4'u32)
    output[4] = index

    # 2D tensor access
    let t = Tensor[2, 2](data: [1'u32, 2'u32, 3'u32, 4'u32])
    output[5] = t.data[0]
    output[6] = t.data[3]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[7, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(sizeof(buf))
  engine.run<<(1, 4)>>("codegenKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 100, &"tid0: {buf[0]}"
  doAssert buf[3] == 103, &"tid3: {buf[3]}"
  doAssert buf[4] == 6,   &"idx(1,2,4): {buf[4]} (expected 1*4+2=6)"
  doAssert buf[5] == 1,   &"tensor[0]: {buf[5]}"
  doAssert buf[6] == 4,   &"tensor[3]: {buf[6]}"
  echo "  OK — codegen patterns"

when isMainModule:
  runTest()
