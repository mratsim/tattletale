## OpenCL: a device function's `{.smem.}` local refuses, `__local` is kernel storage.
##
## - v1 printed `__local` in device-function bodies verbatim
## - `__local` is kernel storage in OpenCL C: v1's emission is invalid OpenCL C
##
## Test that a device function declaring a `{.smem.}` local refuses at codegen.
## OpenCL has no device-function `__local` storage, only kernel bodies print `__local` legally.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/device/manual_opencl_devfn_smem_rejected.nim

import workspace/crucible

const code = opencl:
  proc f(output: ptr UncheckedArray[uint32]) {.inline.} =
    var scratch {.smem.}: array[4, uint32]
    scratch[0] = output[0]
    output[1] = scratch[0]
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    f(output)

proc runTest() =
  echo code
  echo "UNREACHABLE — a device function {.smem.} local must be refused at compile time"

when isMainModule:
  runTest()
