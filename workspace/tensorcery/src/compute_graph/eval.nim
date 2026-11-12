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
import workspace/math
import ../tensors_internals
import ../compute_graph/batching

proc eval*[E: HwEngine](engine: E, roots: var openArray[TensorNode]) =
  if roots.len == 0:
    return
  var batch = engine.recordBatch()
  for i in 0 ..< roots.len:
    engine.enqueueNode(batch, roots[i])
  if batch.entries.len == 0:
    return
  batch.submit()

template evalErased*(run: static Runtime, engine: AnyEngine, roots: var openArray[TensorNode]) =
  ## Evaluates the graph rooted at `roots` on the engine behind the type-erased `AnyEngine` handle.
  ##
  ## Args:
  ## - `run`: the runtime behind `engine`, resolved at compile time
  ## - `engine`: the erased engine handle held by `Tensor`
  ## - `roots`: graph roots, replaced in place by the evaluated nodes
  ##
  ## `eval` instantiates for exactly the selected engine type, once per engine type per compilation.
  ##
  ## The body is checked at call sites only: `eval` instantiates at the first genuine call, never at import time.
  ##
  ## `eval` is absent from the `AnyEngine` interface on purpose: the interface vtable instantiates every method lambda
  ## at vtable creation, so an `eval` entry would compile the op kernels and run the artifact pipeline whenever any
  ## engine is erased, even in programs that never evaluate.
  when run == runMetal:
    engine.to(MetalEngine).eval(roots)
  elif run == runCuda:
    engine.to(CudaEngine).eval(roots)
  elif run == runOpenCL:
    engine.to(OpenCLEngine).eval(roots)
  elif run == runVulkan:
    engine.to(VulkanEngine).eval(roots)
  elif run == runWebGPU:
    engine.to(WebgpuEngine).eval(roots)
  else:
    {.error: "evalErased: no eval for this runtime".}
