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
## - the device kernel
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk by registerOp
## - the Tensor face op over the erased engine
##
## The gather is a pure copy: the element dtype passes through bit-identically
## (bfloat16 included). Token ids are metadata: a typed int32 device buffer
## binds to the kernel's pointer formal, outside the erased Dtype face.

import std/typetraits
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/crucible/src/builtins/builtins_gpu_types
import workspace/iface
import ../../tensors_internals
import ../dispatch
import workspace/design_by_contract

# ############################################################
#
#                   Embed gather
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_embed_gather*(o, w: distinct TensorView, ids: ptr UncheckedArray[int32]) =
  ## Gathers each token's weight vector: o[t, :] = w[ids[t], :].
  ##
  ## Expected input:
  ## - o: (numTokens, hidden), a contiguous output view
  ## - w: (vocab, hidden) weight view, any strides
  ## - ids: numTokens token ids in [0, vocab)
  ##
  ## Output:
  ## - o fully written, token t's vector copied from weight entry ids[t]
  let
    numTokens = o.layout.shape[0]
    hidden = w.layout.shape[1]
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for token in countup(group0, numTokens - 1, groupStride):
    let entry = int(ids[token])
    for col in countup(lane, hidden - 1, lanes):
      o[(token, col)] = w[(entry, col)]

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  EmbedGatherNode* = ref object
    r: AnyTensorImpl      # output slot, metadata only until finalization
    w: TensorNode
    ids: seq[int]
      ## Host-side token ids, staged to a typed int32 device buffer
      ## at enqueue time

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(n: EmbedGatherNode): AnyTensorImpl =
  n.r

func opName*(n: EmbedGatherNode): string =
  "embed_gather"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

func inVocabulary(ids: openArray[int], vocab: int): bool =
  ## True when every id selects an existing weight entry
  for i in 0 ..< ids.len:
    if ids[i] < 0 or ids[i] >= vocab:
      return false
  true

proc tn_embed_gather*(engine: AnyEngine, w: TensorNode, inputIds: openArray[int]): TensorNode {.contracts.} =
  ## Gathers one weight entry per token id: out (numTokens, hidden).
  ##
  ## Args:
  ## - w: the weight node of shape (vocab, hidden)
  ## - inputIds: token ids in [0, vocab), each id selects one entry,
  ##   at least one id
  let iw = w.getImpl()
  {.precondition: iw.shape().len == 2.}
  {.precondition: inputIds.len > 0.}
  {.precondition: inVocabulary(inputIds, iw.shape()[0]).}
  let outShape = [inputIds.len, iw.shape()[1]]
  EmbedGatherNode(
    r: engine.newViewNode(iw.dtype(), outShape.toSpan()).getImpl(),
    w: w, ids: @inputIds).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueEmbedGatherRows[E: HwEngine, Sh: tuple, Mem, MemIds](
    engine: E, batch: BatchCommands[E], n: EmbedGatherNode,
    rr: TensorImpl[Mem, Layout[Sh, Sh]], ids: sink MemIds) =
  ## Enqueues one k_embed_gather over the node's impls.
  ## A body-local type alias defeats the launch's generic binding.
  when tupleLen(Sh) == 2:
    let
      rw = n.w.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      numTokens = rr.layout.shape[0]
      ov = initTensorBridge(rr.data[], rr.offset, rr.layout)
      wv = initTensorBridge(rw.data[], rw.offset, rw.layout)
    # one move from the sink parameter into a local. The enqueue macro binds the launch closure's handle from there.
    let idsBuf = ids
    batch.enqueue<<(numTokens, 32)>> k_embed_gather(ov, wv, idsBuf)
  else:
    raise ValueError.newException(
      "eval: the embed gather op is rank 2, walked at rank " & $tupleLen(Sh))

proc enqueueEmbedGather[E: HwEngine, Sh: tuple, Mem](
    engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asG = n.to(EmbedGatherNode)
  if asG.isNil:
    return false
  engine.evalNode(batch, asG.w, visited)
  # finalization: the full graph is known and the output buffer allocates here
  var sh: Sh
  fillShape(sh, asG.r.shape())
  let rr = engine.makeTensorImpl(asG.r.dtype(), asG.r.shape())
    .to(TensorImpl[Mem, Layout[Sh, Sh]])
  # per-eval staging of the ids: host Nim ints upload to a typed int32 buffer that the launch closure owns through submit
  let numTokens = asG.ids.len
  let idsDev = engine.allocDevice(int32, numTokens)
  var staged = newSeq[int32](numTokens)
  for i in 0 ..< numTokens:
    staged[i] = int32(asG.ids[i])
  engine.copyToDeviceFromHost(idsDev, staged)
  enqueueEmbedGatherRows[E, Sh, Mem, typeof(idsDev)](engine, batch, asG, rr, idsDev)
  n = rr.to(TensorNode)
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("embed_gather", enqueueEmbedGather)
