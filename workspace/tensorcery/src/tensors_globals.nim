# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/crucible
import workspace/ceramic
import workspace/iface
import ./tensors_internals
import ./tensors_internal_types
import ./compute_graph/eval

# ═══════════════════════════════════════════════════════════════════════
#  Engine singletons
# ═══════════════════════════════════════════════════════════════════════

proc gMetalEngine*(): MetalEngine =
  let g {.global.} = newEngine(runMetal)
  g

proc gOpenCLEngine*(): OpenCLEngine =
  let g {.global.} = newEngine(runOpenCL)
  g

proc gVulkanEngine*(): VulkanEngine =
  let g {.global.} = newEngine(runVulkan)
  g

proc gWebgpuEngine*(): WebgpuEngine =
  let g {.global.} = newEngine(runWebGPU)
  g

proc gCudaEngine*(): CudaEngine =
  let g {.global.} = newEngine(runCuda)
  g

# ═══════════════════════════════════════════════════════════════════════
#  Erased construction surface
# ═══════════════════════════════════════════════════════════════════════

proc gMetalEngineErased(_: static Runtime = runMetal): AnyEngine =
  gMetalEngine().to(AnyEngine)

proc gOpenCLEngineErased(_: static Runtime = runOpenCL): AnyEngine =
  gOpenCLEngine().to(AnyEngine)

proc gVulkanEngineErased(_: static Runtime = runVulkan): AnyEngine =
  gVulkanEngine().to(AnyEngine)

proc gWebgpuEngineErased(_: static Runtime = runWebGPU): AnyEngine =
  gWebgpuEngine().to(AnyEngine)

proc gCudaEngineErased(_: static Runtime = runCuda): AnyEngine =
  gCudaEngine().to(AnyEngine)

## Eval dispatch instantiates a backend's walk only when this record contains the backend.
var ctgBackendSelection* {.compileTime.}: set[Runtime]

proc gEngine*(run: static Runtime): AnyEngine =
  static: ctgBackendSelection.incl(run)
  when run == runMetal:
    gMetalEngineErased()
  elif run == runOpenCL:
    gOpenCLEngineErased()
  elif run == runVulkan:
    gVulkanEngineErased()
  elif run == runWebGPU:
    gWebgpuEngineErased()
  elif run == runCuda:
    gCudaEngineErased()
  else:
    {.error: "gEngine: unsupported run kind".}

template eval*(run: static Runtime, roots: var openArray[TensorGraph]) =
  ## - the erased engine handle must never reach the dispatch: it would
  ##   instantiate the whole walk over the erased vtable type
  when run == runMetal:
    evalRoots(gMetalEngine(), roots)
  elif run == runOpenCL:
    evalRoots(gOpenCLEngine(), roots)
  elif run == runVulkan:
    evalRoots(gVulkanEngine(), roots)
  elif run == runWebGPU:
    evalRoots(gWebgpuEngine(), roots)
  elif run == runCuda:
    {.error: "eval: CUDA is not on the batch stream yet".}
  else:
    {.error: "eval: unsupported run kind".}
