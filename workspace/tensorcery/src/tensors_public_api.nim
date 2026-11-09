# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/iface
import tensors_internals
import tensors_public_types
import compute_graph/ops_2i1o/ops_add
import compute_graph/ops_1iNo/ops_split
import tensors_globals
import workspace/crucible/src/runtime/engines/engines_api
import workspace/design_by_contract

export tensors_public_types.Dtype
export engines_api.Runtime

# #######################################################################
#
#                            Core Types
#
# #######################################################################

type Tensor* = object
  engine: AnyEngine
  node: TensorNode

# #######################################################################
#
#                            Construction
#
# #######################################################################

proc toTensor*[T: float32 | bfloat16](data: openArray[T], shape: openArray[int],
    run: static Runtime = runMetal, dtype: typedesc = T): Tensor {.contracts.} =
  ## One tensor from host data, the upload is the construction
  ## (the mx.array face), run picks the backend and dtype converts
  ## elements that differ from the element type
  {.precondition: shape.len == 2.}
  {.precondition: data.len == shape[0] * shape[1].}
  when T is dtype:
    let engine = gEngine(run)
    result = Tensor(engine: engine,
                    node: engine.newTensorNode(toDtype(T), shape.toSpan()))
    if data.len > 0:
      engine.upload(result.node, data)
  else:
    var converted: seq[dtype]
    converted.newSeq(data.len)
    for i in 0 ..< data.len:
      converted[i] = data[i].to(dtype)
    result = toTensor(converted, shape, run, dtype)

# #######################################################################
#
#                            Evaluation
#
# #######################################################################

proc eval*(ts: varargs[Tensor]) =
  var roots = newSeq[TensorNode](ts.len)
  for i in 0 ..< ts.len:
    roots[i] = ts[i].node
  ts[0].engine.eval(roots.toSpan())

# #######################################################################
#
#                            Materialization
#
# #######################################################################

proc toHost*[T: float32 | bfloat16](t: Tensor, _: typedesc[T]): seq[T] {.contracts.} =
  ## Schedules the graph and reads the tensor's values back as host data
  ## (the np.array face), the typedesc picks T and must match the dtype
  let impl = t.node.getImpl()
  {.precondition: impl.dtype() == toDtype(T).}
  var roots = newSeq[TensorNode](1)
  roots[0] = t.node
  t.engine.eval(roots.toSpan())
  result = newSeq[T](impl.shape().getNumElems())
  if result.len > 0:
    t.engine.download(result, t.node)

# #######################################################################
#
#                            Ops
#
# #######################################################################

proc `+`*(a, b: Tensor): Tensor =
  Tensor(engine: a.engine, node: tn_add(a.engine, a.node, b.node))

proc split*(t: Tensor, axis: int, k: int): seq[Tensor] =
  for n in t.engine.tn_split(t.node, axis, k):
    result.add(Tensor(engine: t.engine, node: n))
