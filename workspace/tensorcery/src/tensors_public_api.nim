# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/crucible
import workspace/iface
import tensors_internals
import tensors_public_types
import compute_graph/ops_2i1o/ops_add
import compute_graph/ops_1iNo/ops_split
import tensors_globals
import workspace/design_by_contract

export tensors_public_types
export bfloat16

# #######################################################################
#
#                            Core Types
#
# #######################################################################

type Tensor* = ref object
  engine: AnyEngine
  node: TensorNode
  impl: AnyTensorImpl

# #######################################################################
#
#                            Construction
#
# #######################################################################

proc toTensor*[T: float32 | bfloat16](data: openArray[T], shape: openArray[int],
    run: static Runtime = runMetal, dtype: typedesc = T): Tensor {.contracts.} =
  {.precondition: shape.len == 2.}
  {.precondition: data.len == shape[0] * shape[1].}
  when T is dtype:
    let engine = gEngine(run)
    let node = engine.newTensorNode(toDtype(T), shape)
    result = Tensor(engine: engine, node: node, impl: node.getImpl())
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
  ts[0].engine.eval(roots)
  # op nodes leave the walk materialized
  for i in 0 ..< ts.len:
    ts[i].node = roots[i]
    ts[i].impl = roots[i].getImpl()

proc shape*(t: Tensor): seq[int] =
  let impl = t.node.getImpl()
  result = newSeq[int](impl.shape().len)
  for i in 0 ..< result.len:
    result[i] = impl.shape()[i]

proc strides*(t: Tensor): seq[int] =
  let impl = t.node.getImpl()
  result = newSeq[int](impl.strides().len)
  for i in 0 ..< result.len:
    result[i] = impl.strides()[i]

proc dtype*(t: Tensor): Dtype =
  t.node.getImpl().dtype()

proc item*[T: float32 | bfloat16](t: Tensor, i, j: int, _: typedesc[T]): T {.contracts.} =
  let impl = t.node.getImpl()
  {.precondition: impl.dtype() == toDtype(T).}
  var roots = newSeq[TensorNode](1)
  roots[0] = t.node
  t.engine.eval(roots)
  t.node = roots[0]
  var one: array[1, T]
  t.engine.download(one, t.node, i * impl.strides()[0] + j * impl.strides()[1])
  one[0]

# #######################################################################
#
#                            Ops
#
# #######################################################################

proc `+`*(a, b: Tensor): Tensor =
  let node = tn_add(a.engine, a.node, b.node)
  Tensor(engine: a.engine, node: node, impl: node.getImpl())

proc split*(t: Tensor, axis: int, k: int): seq[Tensor] =
  for n in t.engine.tn_split(t.node, axis, k):
    result.add(Tensor(engine: t.engine, node: n, impl: n.getImpl()))
