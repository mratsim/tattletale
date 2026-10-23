## Direct SPIR-V: a bool kernel parameter refuses, the push-constant block cannot carry OpTypeBool.
##
## Test that the spirv macro rejects a kernel whose push-constant block would contain a bool member.
##
## Expected compile error:
##   SPIR-V backend does not support: push-constant member bool, OpTypeBool cannot appear in a push-constant block
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/manual_spirv_bool_param_rejected.nim

import workspace/crucible

proc runTest() =
  let k = spirv:
    proc flagSet(output: ptr uint32, flag: bool) {.global.} =
      if flag:
        output[] = 1'u32
      else:
        output[] = 0'u32
  echo k

when isMainModule:
  runTest()
