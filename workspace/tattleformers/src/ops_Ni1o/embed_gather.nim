# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Embedding gather op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the add op template:
##
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk at import time
## - the Tensor face op over the erased engine
##
## The kernel is positron's `k_embed_gather`: the embedding lookup is
## transformer-specialized, so the kernel lives in positron.

import std/typetraits
import workspace/crucible
import workspace/tensorcery/src/device_view
import workspace/tensorcery/src/tensor_datatypes
import workspace/positron/src/kernels/k_embed_gather

type EmbedGatherNode* = ref object
  o: AnyTensorImpl
  w: TensorNode
  ids: seq[int]
  ## Host-side token ids, staged to a typed device buffer at eval time

proc eval*(n: EmbedGatherNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: EmbedGatherNode): AnyTensorImpl =
  n.o

# ═══════════════════════════════════════════════════════════════════════
#  Eval - stage the ids, enqueue the node's kernel into the walk's batch
# ═══════════════════════════════════════════════════════════════════════

proc enqueueEmbedGatherRows[E: HwEngine, Sh: tuple, Mem, MemIds](
    engine: E, batch: Batch[E], n: EmbedGatherNode, ids: sink MemIds) =
  ## Enqueues one k_embed_gather over the node's impls.
  ## A body-local type alias defeats the launch's generic binding.
  let
    ro = n.o.to(TensorImpl[Mem, Layout[Sh, Sh]])
    rw = n.w.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    numTokens = ro.layout.shape[0]
    ob = TensorBridge[Mem, Sh, Sh](data: ro.data[], offset: ro.offset, layout: ro.layout)
    wb = TensorBridge[Mem, Sh, Sh](data: rw.data[], offset: rw.offset, layout: rw.layout)
  # one move from the sink parameter into a local. The enqueue macro binds the launch closure's handle from there.
  let idsBuf = ids
  when tupleLen(Sh) == 2:
    # every handler instantiates at every rank, gathering enqueues only at rank 2
    batch.enqueue<<(numTokens, 32)>> k_embed_gather(ob, wb, idsBuf)
  else:
    raise ValueError.newException(
      "eval: the embed gather op is rank 2, walked at rank " & $tupleLen(Sh))

proc enqueueEmbedGatherNode*[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool =
  ## Enqueue handler of the embed gather op, registered below.
  ## The erased node downcasts to the gather node here.
  ## Another op's node returns false, leaving the dispatch to that op's handler.
  let asG = n.to(EmbedGatherNode)
  if asG.isNil:
    return false
  evalNode[E](engine, batch, asG.w)
  # per-eval staging of the ids: host Nim ints upload to a typed int32 buffer that the launch closure owns through submit
  let numTokens = asG.ids.len
  let idsDev = engine.allocDevice(int32, numTokens)
  var staged = newSeq[int32](numTokens)
  for i in 0 ..< numTokens:
    staged[i] = int32(asG.ids[i])
  engine.copyToDeviceFromHost(idsDev, staged)
  # a gather copies weight entries and keeps the weight's dtype.
  # kFp16 has no erased impls and raises
  case asG.o.dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    enqueueEmbedGatherRows[E, Sh, Mem, typeof(idsDev)](engine, batch, asG, idsDev)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    enqueueEmbedGatherRows[E, Sh, Mem, typeof(idsDev)](engine, batch, asG, idsDev)
  else:
    raise ValueError.newException("eval: kFp16 embed gather unsupported")
  return true

registerEnqueueNode(enqueueEmbedGatherNode)

# ═══════════════════════════════════════════════════════════════════════
#  Tensor face
# ═══════════════════════════════════════════════════════════════════════

proc tn_embed_gather*(engine: AnyEngine, w: TensorNode, inputIds: openArray[int]): TensorNode =
  ## Gathers one weight entry per token id: out (numTokens, hidden).
  ##
  ## Args:
  ## - w: the weight node of shape (vocab, hidden)
  ## - inputIds: token ids in [0, vocab), each id selects one entry
  ## Raises on a non-rank-2 weight and on an id outside the vocabulary.
  let iw = w.getImpl()
  let s = iw.shape()
  if s.len != 2:
    raise ValueError.newException(
      "tn_embed_gather: the weight must be rank 2, got rank " & $s.len)
  for i in 0 ..< inputIds.len:
    if inputIds[i] < 0 or inputIds[i] >= s[0]:
      raise ValueError.newException(
        "tn_embed_gather: token id " & $inputIds[i] &
        " outside the vocabulary of " & $s[0])
  let outShape = [inputIds.len, s[1]]
  let
    o = engine.newTensorNode(iw.dtype(), outShape.toSpan())
    g = EmbedGatherNode(o: o.getImpl(), w: w, ids: @inputIds)
  g.to(TensorNode)
