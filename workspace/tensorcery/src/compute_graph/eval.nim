# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/iface
import workspace/math
import ../tensors_internals
import ../compute_graph/dispatch
import ../compute_graph/ops_2i1o/ops_add
import ../compute_graph/ops_2i1o/ops_matmul
import ../compute_graph/ops_2i1o/ops_silu_mul
import ../compute_graph/ops_1i1o/ops_rmsnorm
import ../compute_graph/ops_Ni1o/ops_rope
import ../compute_graph/ops_Ni1o/ops_pool_write
import ../compute_graph/ops_Ni1o/ops_attention
import ../compute_graph/ops_1iNo/ops_split
import ../compute_graph/views

proc eval*[E: HwEngine](engine: E, roots: var Span[TensorNode]) =
  if roots.len == 0:
    return
  var batch = engine.recordBatch()
  var visited: VisitedNodes
  for i in 0 ..< roots.len:
    engine.evalNode(batch, roots[i], visited)
  if batch.entries.len == 0:
    # a leaf-only graph, the values are already materialized
    return
  batch.submit()
