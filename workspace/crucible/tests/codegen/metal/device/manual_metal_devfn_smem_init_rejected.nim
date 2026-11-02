## Metal: a device function's `{.smem.}` var refuses an initializer, a hoisted `{.smem.}` arena takes no initializer.
##
## - v1 printed `threadgroup T x = init` verbatim in any proc
## - Metal rejects an initialized `threadgroup` variable: v1's emission is invalid MSL
##
## Test that a device function declaring a `{.smem.}` local with an initializer refuses in `mslHoistDeviceFnSmem`.
## The arena hoists to a threadgroup reference parameter, an initializer on it has no target form.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/manual_metal_devfn_smem_init_rejected.nim

import workspace/crucible

const code = metal:
  proc fold(input: ptr UncheckedArray[float32], output: ptr UncheckedArray[float32]) {.inline.} =
    var scratch {.smem.}: array[4, float32] = [1'f32, 2, 3, 4]
    let t = thread_position_in_threadgroup.x
    scratch[t] = input[t]
    output[t] = scratch[t]
  proc k(output: ptr UncheckedArray[float32], input: ptr UncheckedArray[float32]) {.global.} =
    var scratch {.smem.}: array[4, float32]
    fold(input, output)

proc runTest() =
  echo code
  echo "UNREACHABLE — a device function {.smem.} local with an initializer must be refused at compile time"

when isMainModule:
  runTest()
