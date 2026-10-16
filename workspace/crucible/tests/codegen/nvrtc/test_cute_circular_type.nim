## CuTe: forward-referenced types (B23)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_cute_circular_type.nim
import std/strformat
import workspace/crucible

# Forward-referenced types (defined at Nim level, adjacent in scope)
type
  Node[N: static int] = object
    val: uint32

const kernelCode = cuda:
  proc fwdRefKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Node[4](val: 42'u32)
    output[0] = x.val

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel("fwdRefKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 42, &"fwd ref: {buf[0]}"
  echo "  OK — forward-referenced types (B23)"

when isMainModule:
  runTest()
