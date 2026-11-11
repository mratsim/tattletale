# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/ceramic
import workspace/iface
import ./tensors_internals
import ./tensors_internal_types
import ./compute_graph/eval
import ./compute_graph/ops_2i1o/ops_add
import ./compute_graph/ops_1iNo/ops_split

proc gMetalEngineErased(_: static Runtime = runMetal): AnyEngine =
  let g {.global.} = newEngine(runMetal)
  g.to(AnyEngine)

proc gOpenCLEngineErased(_: static Runtime = runOpenCL): AnyEngine =
  let g {.global.} = newEngine(runOpenCL)
  g.to(AnyEngine)

proc gVulkanEngineErased(_: static Runtime = runVulkan): AnyEngine =
  let g {.global.} = newEngine(runVulkan)
  g.to(AnyEngine)

proc gWebgpuEngineErased(_: static Runtime = runWebGPU): AnyEngine =
  let g {.global.} = newEngine(runWebGPU)
  g.to(AnyEngine)

proc gCudaEngineErased(_: static Runtime = runCuda): AnyEngine =
  let g {.global.} = newEngine(runCuda)
  g.to(AnyEngine)

proc gEngine*(run: static Runtime): AnyEngine =
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
