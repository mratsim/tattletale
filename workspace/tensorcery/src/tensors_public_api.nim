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
import compute_graph/ops_Ni1o/ops_embed_gather
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

# #######################################################################
#
#                            Construction
#
# #######################################################################

proc toTensor*[T: float32 | bfloat16](data: openArray[T], shape: openArray[int],
    run: static Runtime = runMetal, dtype: typedesc = T): Tensor {.contracts.} =
  {.precondition: data.len == getNumElems(shape.toSpan()).}
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
  var span = roots.toSpan()
  ts[0].engine.eval(span)
  # op nodes leave the walk materialized
  for i in 0 ..< ts.len:
    ts[i].node = roots[i]

proc shape*(t: Tensor): Span[int] =
  t.node.getImpl().shape()

proc strides*(t: Tensor): Span[int] =
  t.node.getImpl().strides()

proc dtype*(t: Tensor): Dtype =
  t.node.getImpl().dtype()

func allIdxInShape(impl: AnyTensorImpl, idx: varargs[int]): bool =
  let shape = impl.shape()
  for i in 0 ..< idx.len:
    if idx[i] < 0 or idx[i] >= shape[i]:
      return false
  true

proc item*[T: float32 | bfloat16](t: Tensor, idx: varargs[int], _: typedesc[T]): T {.contracts.} =
  let impl = t.node.getImpl()
  {.precondition: idx.len == impl.shape().len.}
  {.precondition: impl.dtype() == toDtype(T).}
  {.precondition: allIdxInShape(impl, idx).}
  var roots = newSeq[TensorNode](1)
  roots[0] = t.node
  var span = roots.toSpan()
  t.engine.eval(span)
  t.node = roots[0]
  t.engine.downloadItem(result, t.node, crd2Offset(impl, idx))

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

proc embed_gather*(w: Tensor, inputIds: openArray[int]): Tensor =
  ## Gathers one weight entry per token id.
  ##
  ## Args:
  ## - w: the weight tensor, shape (vocab, hidden)
  ## - inputIds: token ids in [0, vocab), each id selects one weight entry
  ##
  ## Returns:
  ## - a tensor of shape (numTokens, hidden), token t's vector copied from
  ##   the weight entry ids[t]
  Tensor(engine: w.engine, node: tn_embed_gather(w.engine, w.node, inputIds))
