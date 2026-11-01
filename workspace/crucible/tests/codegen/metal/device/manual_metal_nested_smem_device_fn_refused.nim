## Metal: a `{.smem.}` local nested in a device function refuses at compile time.
##
## Test that a `metal:` device function with a `{.smem.}` local nested inside
##   two `if` bodies fails to compile with the pass refusal naming
##   "only a kernel allocates threadgroup storage"
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/manual_metal_nested_smem_device_fn_refused.nim

import workspace/crucible

const code = metal:
  proc helper(flag: bool) {.device.} =
    if flag:
      if true:
        var scratch {.smem.}: array[4, uint32]
        scratch[0] = 1'u32
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    helper(true)
    output[0] = 0'u32

proc runTest() =
  echo "UNREACHABLE: the nested device-function smem local must be refused at compile time"

when isMainModule:
  runTest()
