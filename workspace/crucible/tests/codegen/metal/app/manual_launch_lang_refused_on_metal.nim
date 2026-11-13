## launchDeviceFn `langVulkan =` extra refuses on non-Vulkan engines at expansion.
##
## Test that langVulkan = glsl on a Metal engine is a compile-time error naming
## the engine. The launch never reaches codegen.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/app/manual_launch_lang_refused_on_metal.nim

import workspace/crucible

proc addKernel(output: ptr UncheckedArray[uint32]) =
  output[0] = 1'u32

proc runTest() =
  var engine = newEngine(runMetal)
  let oDev = engine.allocDevice(uint32, 1)
  # on MetalEngine the expansion must fail loudly naming the engine
  engine.launchDeviceFn addKernel(oDev) do:
    grid = (1, 1)
    blk = (1, 1)
    langVulkan = glsl
  echo "UNREACHABLE — langVulkan = glsl must be refused on a Metal engine"

when isMainModule:
  runTest()
