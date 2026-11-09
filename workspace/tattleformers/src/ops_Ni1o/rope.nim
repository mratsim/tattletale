# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## RoPE apply op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the embed gather op:
##
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk at import time
## - the Tensor face op over the erased engine
##
## The kernel is positron's `k_rope`: the position rotation is
## transformer-specialized, so the kernel lives in positron.

import std/typetraits
import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/device_view
import workspace/tensorcery/src/tensor_datatypes
import workspace/positron/src/kernels/k_rope
import ../ops_1i1o/rmsnorm

# Batch enqueue walks kernel bodies at compile time and resolves the tile
# algebra's thread->value map templates in the importing module's scope.
export tile_algebra

# Registration order matters: the launch binding of this op's f32 tables
# binds reliably only once the rmsnorm op's walked chain
# (k_rmsnorm -> ceramic rmsnorm) instantiated first in the same binary.
# Importing the rmsnorm op module here registers its handler first,
# whatever order the client imports the op modules.

type RopeNode* = ref object
  r: AnyTensorImpl
  enqueued: bool
    ## Launch-once marker under the eval walk's fan-out re-walks:
    ## - a downstream op that fans out over a shared input re-walks
    ##   the whole upstream chain (the FFN's two projections share one input)
    ## - the repeat visit must not re-run the handler, the launch is
    ##   already in the batch and its subtree is already scheduled
    ## Precondition: an op node is constructed per forward and enqueued in
    ## exactly one eval, so the flag never crosses an eval boundary.
  q, cos, sin: TensorNode

proc eval*(n: RopeNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: RopeNode): AnyTensorImpl =
  n.r

# ═══════════════════════════════════════════════════════════════════════
#  Eval - enqueue the node's kernel into the walk's batch
# ═══════════════════════════════════════════════════════════════════════

template enqueueRopeTokens[E: HwEngine, MemQ, MemC](
    engine: E, batch: Batch[E], n: RopeNode, Sh: untyped) =
  ## Enqueues one k_rope over the node's impls at the rank given by Sh.
  ##
  ## A template, not a proc: batch.enqueue walks the kernel body at compile
  ## time over this call site's argument types.
  ##
  ## - the walked body never instantiates at a rank the op does not serve:
  ##   the handler calls it only from the rank 3 branch of its rank dispatch
  ## - every shape tuple is a literal at the call site, the erasedUpload pattern:
  ##   Sh-parametric type expressions in the walked body bound unreliably
  ##   at the launch, literal types binding always
  ##
  ## q and the result share MemQ, cos/sin are f32 with their own MemC.
  # cos/sin (tokens, dim div 2) broadcast over heads at the launch:
  # head axis strides 0, a runtime 0 keeps the layout leaf dynamic.
  # One broadcast layout serves both tables: same shape, same strides.
  let
    ro = n.r.to(TensorImpl[MemQ, Layout[Sh, Sh]])
    rq = n.q.getImpl().to(TensorImpl[MemQ, Layout[Sh, Sh]])
    rc = n.cos.getImpl().to(TensorImpl[MemC, Layout[(int, int), (int, int)]])
    rs = n.sin.getImpl().to(TensorImpl[MemC, Layout[(int, int), (int, int)]])
    ob = TensorBridge[MemQ, Sh, Sh](data: ro.data[], offset: ro.offset, layout: ro.layout)
    qb = TensorBridge[MemQ, Sh, Sh](data: rq.data[], offset: rq.offset, layout: rq.layout)
    headStride = 0
    half = rc.layout.shape[1]
    broadcastLayout = make_layout(
      (rc.layout.shape[0], qb.layout.shape[1], half),
      (rc.layout.stride[0], headStride, rc.layout.stride[1]))
    cb = TensorBridge[MemC, Sh, Sh](data: rc.data[], offset: rc.offset,
      layout: broadcastLayout)
    sb = TensorBridge[MemC, Sh, Sh](data: rs.data[], offset: rs.offset,
      layout: broadcastLayout)
  # one move into a local: the enqueue macro binds the launch closure's handle from there
  # one threadgroup per token, 32 lanes stride the pair columns,
  # kernel grid-stride covers any token count
  let numTokens = qb.layout.shape[0]
  batch.enqueue<<(numTokens, 32)>> k_rope(ob, qb, cb, sb)

proc enqueueRopeNode*[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool =
  ## Enqueue handler of the rope op, registered below. The erased node
  ## downcasts to the rope node here. A node of another op returns false,
  ## leaving the dispatch to that op's handler.
  let asR = n.to(RopeNode)
  if asR.isNil:
    return false
  if asR.enqueued:
    # re-visit under the walk's fan-out, the subtree is already scheduled
    return true
  asR.enqueued = true
  evalNode[E](engine, batch, asR.q)
  evalNode[E](engine, batch, asR.cos)
  evalNode[E](engine, batch, asR.sin)
  # q and the result share one bf16 Mem (checked at the face),
  # cos/sin are f32 with their own Mem.
  # Rank dispatch: the enqueue template runs at the rank 3 branch only,
  # one literal shape tuple per branch, the rmsnorm handler pattern.
  if asR.q.getImpl().shape().len != 3:
    raise ValueError.newException(
      "eval: the rope op is rank 3, walked at rank " &
      $asR.q.getImpl().shape().len)
  type
    MemQ = typeof(engine.allocDevice(bfloat16, 0))
    MemC = typeof(engine.allocDevice(float32, 0))
  enqueueRopeTokens[E, MemQ, MemC](engine, batch, asR, (int, int, int))
  return true

registerEnqueueNode(enqueueRopeNode)

# ═══════════════════════════════════════════════════════════════════════
#  Tensor face
# ═══════════════════════════════════════════════════════════════════════

proc tn_rope*(engine: AnyEngine, q, cos, sin: TensorNode): TensorNode =
  ## Applies the NEOX RoPE rotation over q's last axis. The output has
  ## q's shape and dtype, fully overwritten.
  ##
  ## Args:
  ## - q: (tokens, heads, dim) input in bfloat16 with an even last axis
  ## - cos, sin: (tokens, dim div 2) f32 half tables, entry (t, i) pairs
  ##   with the columns (i, i + dim div 2) of token t. The head-axis broadcast
  ##   happens through stride-0 views inside the kernel
  ##
  ## The rotation runs in f32, one rounding at the store through the output's element type
  ## (bfloat16 round-to-nearest-even). Raises on a rank/shape mismatch,
  ## on a non-bfloat16 q and on a cos/sin dtype other than f32.
  let
    iq = q.getImpl()
    ic = cos.getImpl()
    iSin = sin.getImpl()
  let sq = iq.shape()
  if sq.len != 3:
    raise ValueError.newException(
      "tn_rope: q must be rank 3 (tokens, heads, dim), got rank " & $sq.len)
  if sq[2] mod 2 != 0:
    raise ValueError.newException(
      "tn_rope: q's last dimension " & $sq[2] & " must be even")
  if iq.dtype() != kBf16:
    raise ValueError.newException(
      "tn_rope: q dtype " & $iq.dtype() & " unsupported, the rotation runs bf16 in and out")
  let
    sc = ic.shape()
    ss = iSin.shape()
  if sc.len != 2 or ss.len != 2:
    raise ValueError.newException(
      "tn_rope: cos/sin must be rank 2 (tokens, dim div 2), got rank " &
      $sc.len & " and " & $ss.len)
  if sc != ss:
    raise ValueError.newException(
      "tn_rope: cos/sin shape mismatch, got " & $sc & " and " & $ss)
  if ic.dtype() != kFp32 or iSin.dtype() != kFp32:
    raise ValueError.newException(
      "tn_rope: cos/sin must be f32, got " & $ic.dtype() & " and " & $iSin.dtype())
  if sc[0] != sq[0]:
    raise ValueError.newException(
      "tn_rope: cos/sin cover " & $sc[0] & " tokens, q carries " & $sq[0])
  if sc[1] != sq[2] div 2:
    raise ValueError.newException(
      "tn_rope: cos/sin width " & $sc[1] & " must match dim div 2 = " & $(sq[2] div 2))
  RopeNode(
    r: engine.newTensorNode(iq.dtype(), sq).getImpl(),
    q: q, cos: cos, sin: sin).to(TensorNode)
