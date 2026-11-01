## Vulkan (GLSL): a program-scope `{.smem.}` var refuses an initializer, GLSL shared variables take only a null initializer.
##
## Test that a top-level `{.smem.}` declaration with an initializer refuses at codegen.
## The frontend copies a program-scope var into `CirModule.globals`, so the printer's hoisted-globals guard is the refusal site.
## The kernel-body path refuses earlier in `glslHoistProgramScopeLocals`.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/manual_vulkan_program_scope_smem_init_rejected.nim

import workspace/crucible

const code = glsl:
  var scratch {.smem.}: array[4, uint32] = [1'u32, 2, 3, 4]
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    scratch[0] = output[0]
    output[1] = scratch[0]

proc runTest() =
  echo code
  echo "UNREACHABLE — a program-scope {.smem.} var with an initializer must be refused at compile time"

when isMainModule:
  runTest()
