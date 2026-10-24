## SPIR-V direct: a body local named like an exploded leaf refuses loudly
##
## Manual: expected compile failure, run to verify the refusal message
##
## Test that a kernel whose body local collides with a leaf param name fails at compile time
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/manual_spirv_leaf_name_collision.nim

import workspace/crucible

type
  Config = object
    data: uint32

const cfgSpv = spirv:
  proc cfgKernel(output: ptr UncheckedArray[uint32], c: Config) {.global.} =
    # the body local shares the leaf param's name `c_data`, so the explode
    # route's collision check refuses loudly at pass time and the emitter
    # never sees a surviving name collision
    var c_data: uint32 = 7
    output[0] = c.data + c_data

when isMainModule:
  echo "manual test: expect a compile-time collision refusal naming `c_data`"
