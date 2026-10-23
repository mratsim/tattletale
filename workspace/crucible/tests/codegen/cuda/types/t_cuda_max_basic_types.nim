## NVRTC: `max` on float32, uint32, and int, an ambiguous builtin resolved
## per overload: non-magic overloads body-parse into device fns, the int magic
## overload forwards to the native max.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/types/t_cuda_max_basic_types.nim

import std/[unittest]
import workspace/crucible

# ── kernel ───────────────────────────────────────────────────────────────
# Output buffer MUST be the first kernel param (the harness prepends res).
const kernelCode = cuda:
  proc maxBasic(res: ptr UncheckedArray[float32];
                dyn: ptr UncheckedArray[float32]) {.global.} =
    let x = dyn[0]
    let y = dyn[1]
    # float32 — inline (non-magic) overload -> body-parsed device fn
    res[0] = max(x, y)              # dynamic operands
    res[1] = max(3.5'f32, 7.25'f32) # literals
    res[2] = max(y, x)              # swapped order
    # uint32 — generic not-SomeFloat overload -> body-parsed device fn
    res[3] = float32(max(uint32(x), uint32(y)))
    res[4] = float32(max(3'u32, 7'u32))
    # int — magic MaxI -> plain native max call
    res[5] = float32(max(int(x), int(y)))

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode   # show the emitted CUDA

  suite "NVRTC — max on basic types":

    test "float32/uint32/int max compile, run, and produce the right values":
      var buf: array[6, float32]
      var dynArr: array[2, float32] = [2.5'f32, 9.0'f32]
      var engine = newEngine(runCuda)
      engine.ingest(kernelCode)
      let iDev = engine.allocDevice(float32, dynArr.len)
      engine.copyToDeviceFromHost(iDev, addr dynArr[0], dynArr.len)
      let oDev = engine.allocDevice(float32, buf.len)
      engine.launchKernel<<(1, 1)>>("maxBasic", oDev, (iDev,))
      engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
      check buf[0] == 9.0    # max(2.5, 9.0)
      check buf[1] == 7.25   # max(3.5, 7.25)
      check buf[2] == 9.0    # max(9.0, 2.5)
      check buf[3] == 9.0    # max(2, 9)
      check buf[4] == 7.0    # max(3, 7)
      check buf[5] == 9.0    # max(2, 9)

when isMainModule:
  runTest()
