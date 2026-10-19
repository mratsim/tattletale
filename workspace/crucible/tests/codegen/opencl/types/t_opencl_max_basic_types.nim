## OpenCL: `max`/`min` on float32, uint32, and int.
## Float max/min lower to `fmax`/`fmin` (OpenCL C's max/min are integer-only, §6.12.3).
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/types/t_opencl_max_basic_types.nim

import std/[unittest, strutils]
import workspace/crucible

# ── kernel ───────────────────────────────────────────────────────────────
# OpenCL binds kernel args in order: output first (binding 0), then inputs.
const kernelCode = opencl:
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
    # float32 min — same body-parsed path as max (fmin on OpenCL)
    res[6] = min(x, y)              # dynamic operands
    res[7] = min(3.5'f32, 7.25'f32) # literals
    # uint32 min — generic not-SomeFloat overload
    res[8] = float32(min(uint32(x), uint32(y)))

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode   # show the emitted OpenCL C

  suite "OpenCL — max on basic types":

    test "float32/uint32/int max compile, run, and produce the right values":
      # The fold must emit fmax/fmin for OpenCL floats — max/min are
      # integer-only there and silently compute wrong values on float args.
      check kernelCode.contains("fmax")
      check kernelCode.contains("fmin")

      var dynArr: array[2, float32] = [2.5'f32, 9.0'f32]
      var engine = bkOpenCL.newEngine()
      engine.ingest(kernelCode)
      var out32: array[9, float32]
      let dDev = engine.allocDevice(float32, dynArr.len)
      engine.copyToDeviceFromHost(dDev, addr dynArr[0], dynArr.len)
      let oDev = engine.allocDevice(float32, 9)
      engine.launchKernel("maxBasic", oDev, (dDev,))
      engine.copyToHostFromDevice(addr out32[0], oDev, 9)
      check out32[0] == 9.0    # max(2.5, 9.0)
      check out32[1] == 7.25   # max(3.5, 7.25)
      check out32[2] == 9.0    # max(9.0, 2.5)
      check out32[3] == 9.0    # max(2, 9)
      check out32[4] == 7.0    # max(3, 7)
      check out32[5] == 9.0    # max(2, 9)
      check out32[6] == 2.5    # min(2.5, 9.0)
      check out32[7] == 3.5    # min(3.5, 7.25)
      check out32[8] == 2.0    # min(2, 9)

when isMainModule:
  runTest()
