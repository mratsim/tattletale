# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Eval walk: one batched submit over the graph
##
## - op nodes enqueue through the registry: the walk looks the node's
##   opName up and calls the running engine's handler
## - a new op is a new ops_{arity} file registering its handlers, the op
##   set opens by registration with the freeze below the op imports
## - a materialized tensor is a node with no op, the walk leaves it alone

import std/tables
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/iface
import workspace/math
import ../tensors_internals
import ../compute_graph/dispatch
import ../compute_graph/ops_2i1o/ops_add
import ../compute_graph/ops_1iNo/ops_split

# freeze below the op imports, registration precedes the freeze by construction
const frozenOpRegistry* = static(OpRegistry)

proc evalWalk*[E: HwEngine, Sh: tuple, Mem](
    engine: E, batch: Batch[E], n: TensorNode) =
  if n.opName() == "":
    return
  let handlers = frozenOpRegistry.getOrDefault(n.opName())
  let handler = handlers.handler(E)
  if handler.isNil:
    raise ValueError.newException(
      "eval: op '" & n.opName() & "' has no " & $E & " handler")
  if not handler(engine, batch, n, evalWalk[E, Sh, Mem]):
    raise ValueError.newException("eval: unsupported node in the graph")

proc eval*[E: HwEngine](engine: E, roots: Span[TensorNode]) =
  if roots.len == 0:
    return
  let impl = roots[0].getImpl()
  var batch = engine.recordBatch()
  case impl.dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    case impl.shape().len
    of 2:
      for i in 0 ..< roots.len:
        evalWalk[E, (int, int), Mem](engine, batch, roots[i])
    else:
      raise ValueError.newException("eval: rank " & $impl.shape().len & " unsupported")
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    case impl.shape().len
    of 2:
      for i in 0 ..< roots.len:
        evalWalk[E, (int, int), Mem](engine, batch, roots[i])
    else:
      raise ValueError.newException("eval: rank " & $impl.shape().len & " unsupported")
  else:
    raise ValueError.newException("eval: dtype unsupported")
  if batch.entries.len == 0:
    # a leaf-only graph, the values are already materialized
    return
  batch.submit()
