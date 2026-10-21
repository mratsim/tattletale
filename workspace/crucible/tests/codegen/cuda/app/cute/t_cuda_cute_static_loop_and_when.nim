## CuTe scaling: static int in loops + compile-time branching
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/app/cute/t_cuda_cute_static_loop_and_when.nim
import std/strformat
import workspace/crucible

type
  Tile[M, N: static int] = object
    data: array[M * N, uint32]

const L = 3

const kernelCode = cuda:
  proc staticLoopKernel(output: ptr array[L, uint32]) {.global.} =
    # B04: loop with static-int bound — const L pulled from outside cuda block
    var t: Tile[L, 3]
    for idx in 0 ..< L * 3:
      t.data[idx] = uint32(idx + 1) * 10
    for i in 0 ..< L:
      output[i] = t.data[i]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[L, uint32]
  var engine = bkCuda.newEngine()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("staticLoopKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 10, &"loop[0]: {buf[0]}"
  doAssert buf[1] == 20, &"loop[1]: {buf[1]}"
  doAssert buf[2] == 30, &"loop[2]: {buf[2]}"
  echo "  OK — static loop + when"

when isMainModule:
  runTest()
