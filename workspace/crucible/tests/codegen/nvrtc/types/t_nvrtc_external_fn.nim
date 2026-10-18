## NVRTC: external functions pulled from outside the `cuda` block
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/types/t_nvrtc_external_fn.nim
import workspace/crucible

# ── External device functions (defined outside the `cuda` block) ────────────

proc addThree(a, b, c: uint32): uint32 {.device.} =
  result = a + b + c

proc scaleAndAdd(a, b: uint32; factor: uint32): uint32 {.device.} =
  result = a * factor + b

proc isEven(x: uint32): bool {.device.} =
  result = (x and 1) == 0

proc selectValue(cond: bool; t, f: uint32): uint32 {.device.} =
  if cond:
    result = t
  else:
    result = f

# ── GPU code ────────────────────────────────────────────────────────────────

const kernelCode = cuda:
  proc externalFnKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Calls to functions defined OUTSIDE the `cuda` block
    output[0] = addThree(10'u32, 20'u32, 30'u32)
    output[1] = scaleAndAdd(5'u32, 2'u32, 3'u32)
    let even = isEven(42'u32)
    output[2] = selectValue(even, 100'u32, 200'u32)

# ── Host code ───────────────────────────────────────────────────────────────

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[3, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"

  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("externalFnKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  engine.freeDevice(oDev)
  echo "  addThree(10,20,30)  = ", buf[0]
  echo "  scaleAndAdd(5,2,3)  = ", buf[1]
  echo "  selectValue(isEven(42), 100, 200) = ", buf[2]

  doAssert buf[0] == 60  # 10 + 20 + 30
  doAssert buf[1] == 17   # 5*3 + 2
  doAssert buf[2] == 100  # 42 is even, so true branch

  echo "  OK (test_nvrtc_external_fn)"

when isMainModule:
  runTest()
