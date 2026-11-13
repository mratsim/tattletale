# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/crucible
import workspace/iface
import ../tensors_internals
import ./batching

proc eval*[E: HwEngine](engine: E, roots: var openArray[TensorNode]) =
  ## Eval slot of the erased AnyEngine handle, vtable-bound at construction.
  ##
  ## - the body stays free of the op dispatch: a dispatch expansion here
  ##   bakes an empty mark set into the binary
  ## - real dispatch goes through evalRoots on the concrete engine singletons
  raiseAssert("eval: the erased engine handle does not dispatch, use eval(tensors) or t.item(T)")

template evalRoots*(engine: HwEngine, roots: var openArray[TensorNode]) =
  ## Enqueues and submits the graph rooted at `roots` on `engine`.
  ##
  ## - a template: the dispatch macros expand per eval call site and read
  ##   the marks recorded so far
  ## - the walk runs over an explicit stack, a compile-time recursive walk
  ##   through expanded handlers does not terminate
  ## - kernels enqueue in reverse visit order: every kernel launches after
  ##   the kernels producing its operands
  if roots.len != 0:
    var batch = engine.recordBatch()
    var stack: seq[WalkFrame]
    for i in 0 ..< roots.len:
      stack.add WalkFrame(n: roots[i], kernelPending: false)
    while stack.len > 0:
      let frame = stack.pop()
      var n = frame.n
      if not frame.kernelPending:
        if n.isPreAllocated():
          continue
        stack.add WalkFrame(n: n, kernelPending: true)
        let v = n.to(ViewNode)
        if v.isNil():
          engine.dispatchCollect(n, stack)
        else:
          stack.add WalkFrame(n: v.src, kernelPending: false)
      else:
        # Views are metadata, not ops: the walk wires them outside the marked registry.
        if not engine.dispatchViewKernel(n, batch):
          engine.dispatchKernel(n, batch)
    # A view-only graph stages no kernels.
    when ctgHasStagedKernels():
      if batch.entries.len != 0:
        batch.submit()
