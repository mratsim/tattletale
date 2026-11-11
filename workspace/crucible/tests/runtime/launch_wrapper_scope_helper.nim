# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Launch surface without crucible: what a downstream wrapper module presents
## to its own users. t_launch_wrapper_scope expands the launch from a module
## that imports this helper only, never crucible.

import workspace/crucible

export allocDevice, copyToDeviceFromHost, copyToHostFromDevice

proc newTestEngine*(): auto {.inline.} =
  newEngine(runMetal)

proc axpyKernel*(outP, x, y: ptr UncheckedArray[float32], n: int, alpha: float32) =
  let i = int(thread_position_in_grid.x)
  if i < n:
    outP[i] = alpha * (x[i] + y[i])

template enqueueAxpy*(engine, outD, xD, yD: untyped, n: int, alpha: float32) =
  var batch = engine.recordBatch()
  batch.enqueue<<(1, 64)>> axpyKernel(outD, xD, yD, n, alpha)
  batch.submit()
