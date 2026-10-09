## Manual test: Vulkan rejects by-value kernel params carrying a 64-bit
## scalar:
##
## - the push-constant block packs 4-byte aligned
## - std430 aligns a 64-bit member to 8 bytes
## - an int32 member before a 64-bit member puts it at a 4-mod-8 offset
## - the shader then misreads the input
##
# Run (expected FAIL, compile only):
#   nim c --hints:off --warnings:off workspace/crucible/tests/codegen/vulkan/manual_vulkan_pushconst_align8.nim
# Expect: "Vulkan: by-value param 'x' carries a 64-bit type, the push-constant
# block packs 4-byte aligned and cannot honor the std430 8-byte alignment"

import workspace/crucible

const code = vulkan:
  proc k(output: ptr UncheckedArray[uint32], x: float64) {.global.} =
    output[0] = 1'u32
