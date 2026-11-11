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
    # a leaf-only graph, the values are already materialized
    return
  batch.submit()
