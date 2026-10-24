## SPIR-V direct: a bool leaf in the push-constant interface refuses loudly
##
## Manual: expected compile failure, run to verify the refusal message
##
## Test that an object kernel param carrying a bool field refuses at emission naming the member
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/manual_spirv_bool_leaf_refusal.nim

import workspace/crucible

type
  Flags = object
    count: uint32
    on: bool

const flagsSpv = spirv:
  proc flagsKernel(output: ptr UncheckedArray[uint32], f: Flags) {.global.} =
    if f.on:
      output[0] = f.count

when isMainModule:
  echo "manual test: expect an emission-time refusal naming the bool push member"
