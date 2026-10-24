## SPIR-V direct: an object param with a narrower-than-32-bit leaf refuses loudly
##
## Manual: expected compile failure, run to verify the refusal message
##
## Test that an object kernel param carrying a uint8 field refuses at emission naming the member
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/manual_spirv_narrow_leaf_refusal.nim

import workspace/crucible

type
  Stats = object
    count: uint32
    flag: uint8

const statsSpv = spirv:
  proc statsKernel(output: ptr UncheckedArray[uint32], s: Stats) {.global.} =
    output[0] = s.count + uint32(s.flag)

when isMainModule:
  echo "manual test: expect an emission-time refusal naming the uint8 push member"
