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
import compute_graph/ops_metadata/ops_swapaxes
import compute_graph/ops_reductions/ops_allClose
import tensors_globals
import workspace/design_by_contract
from std/math import sqrt

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

# #######################################################################
#
#                            Construction
#
# #######################################################################

proc toTensor*[T](data: openArray[T], shape: openArray[int], run: static Runtime, dtype: typedesc = T): Tensor {.contracts.} =
  {.precondition: shape.len == 2.}
  {.precondition: data.len == shape[0] * shape[1].}
  when T is dtype:
    let engine = gEngine(run)
    result = Tensor(engine: engine,
                    node: engine.newTensorNode(toDtype(T), shape))
    if data.len > 0:
      engine.upload(result.node, data)
  else:
    var converted: seq[dtype]
    converted.newSeq(data.len)
    for i in 0 ..< data.len:
      converted[i] = data[i].to(dtype)
    result = converted.toTensor(shape, run, dtype)

proc toTensor*[T](data: openArray[T], shape: varargs[int]): Tensor {.inline.} =
  data.toTensor(shape, runMetal, T)


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
  # op nodes come out materialized
  for i in 0 ..< ts.len:
    ts[i].node = roots[i]

template shape*(t: Tensor): openArray[int] =
  t.node.getImpl().shape().p

template strides*(t: Tensor): openArray[int] =
  t.node.getImpl().strides().p

proc dtype*(t: Tensor): Dtype {.inline.} =
  t.node.getImpl().dtype()

proc size*(t: Tensor): int {.inline.} =
  t.node.size()

proc item*[T](t: Tensor, _: typedesc[T]): T {.contracts.} =
  let impl = t.node.getImpl()
  {.precondition: impl.dtype() == toDtype(T).}
  var node = [t.node]
  t.engine.eval(node)
  t.node = node[0]
  t.engine.download1(addr result, t.node)

# #######################################################################
#
#                            Ops
#
# #######################################################################

proc `+`*(a, b: Tensor): Tensor {.inline.} =
  Tensor(engine: a.engine, node: tn_add(a.engine, a.node, b.node))

proc split*(t: Tensor, axis: int, k: int): seq[Tensor] {.inline.} =
  for n in t.engine.tn_split(t.node, axis, k):
    result.add(Tensor(engine: t.engine, node: n))

proc allClose*(tested, reference: Tensor, rtol = 1e-5'f32, atol = 1e-8'f32, equal_nan = false): bool {.inline.} =
  ## True when every element satisfies |tested - reference| <= atol + rtol * |reference|.
  ## Same-sign infinities are equal, NaN equal only with equal_nan
  Tensor(engine: tested.engine,
      node: tn_allClose(tested.engine, tested.node, reference.node, atol, rtol, equal_nan)
    ).item(float32) == 0.0'f32

proc swapaxes*(t: Tensor, axis1, axis2: int): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_swapaxes(t.engine, t.node, axis1, axis2))
