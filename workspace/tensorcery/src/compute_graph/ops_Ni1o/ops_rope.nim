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
## - the device kernel
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk by registerOp
## - the Tensor face op over the erased engine
##
## The kernel is the transformer-specialized position rotation, so it lives
## with the op.
##
## Numerics record (D-10, half-width RoPE tables):
##
## - cos/sin are (tokens, dim div 2) f32 half tables. Entry (t, i) pairs
##   with the columns (i, i + dim div 2) of token t. The table entries do
##   not depend on the head, so the kernel indexes them per (token, i)
## - the rotation runs in f32 with one rounding at the store through
##   the output's element type, bfloat16 round-to-nearest-even

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

# Batch enqueue walks the kernel body at compile time and resolves the tile
# algebra's thread-coordinate templates in the importing module's scope.
export tile_algebra

# ############################################################
#
#                   RoPE
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_rope*(o, q, cos, sin: distinct TensorView) =
  ## NEOX rotate_half rotation of the last axis, fully overwriting o.
  ##
  ## Expected input:
  ## - q: (tokens, heads, dim), dim even
  ## - cos, sin: (tokens, dim div 2) f32 half tables. Entry (t, i) pairs
  ##   with the columns (i, i + dim div 2) of token t. The entries do not
  ##   depend on the head
  ##
  ## Output:
  ## - o fully written:
  ##   o[t, h, i]       = q[t, h, i]*cos[t, i] - q[t, h, i + half]*sin[t, i] for the even column
  ##   o[t, h, i + half] = q[t, h, i + half]*cos[t, i] + q[t, h, i]*sin[t, i] for the odd column
  ## The index i runs over 0 ..< dim div 2, t over the tokens and h over the heads.
  ##
  ## This produces q*cos + rotate_half(q)*sin. rotate_half is the pairwise
  ## split [-x2, x1] over the last dimension: the NEOX duplicated half table
  ## folds into the pair indexing.
  let
    numTokens = q.layout.shape[0]
    heads = q.layout.shape[1]
    half = q.layout.shape[2] div 2
  let
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for token in countup(group0, numTokens - 1, groupStride):
    for h in 0 ..< heads:
      for i in countup(lane, half - 1, lanes):
        let
          c = cos[(token, i)].to(float32)
          s = sin[(token, i)].to(float32)
          qEven = q[(token, h, i)].to(float32)
          qOdd = q[(token, h, i + half)].to(float32)
        o[(token, h, i)] = (qEven * c - qOdd * s).to(o.T)
        o[(token, h, i + half)] = (qOdd * c + qEven * s).to(o.T)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  RopeNode* = ref object
    r: AnyTensorImpl      # output slot, metadata only until finalization
    q: TensorNode
    cos: TensorNode
    sin: TensorNode

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(n: RopeNode): AnyTensorImpl =
  n.r

func opName*(n: RopeNode): string =
  "rope"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_rope*(engine: AnyEngine, q, cos, sin: TensorNode): TensorNode {.contracts.} =
  ## Applies the NEOX RoPE rotation over q's last axis. The output has
  ## q's shape and dtype, fully overwritten.
  ##
  ## Args:
  ## - q: the input node of shape (tokens, heads, dim) in bfloat16,
  ##   last axis even
  ## - cos, sin: the half-table nodes, shape (tokens, dim div 2), f32.
  ##   Entry (t, i) pairs with the columns (i, i + dim div 2) of token t.
  ##   The table entries do not depend on the head
  ##
  ## The rotation runs in f32 with one rounding at the store, see the module
  ## header for the numerics record.
  let
    iq = q.getImpl()
    ic = cos.getImpl()
    iSin = sin.getImpl()
  {.precondition: iq.shape().len == 3.}
  {.precondition: iq.dtype() == kBf16.}
  {.precondition: iq.shape()[2] mod 2 == 0.}
  {.precondition: ic.shape().len == 2.}
  {.precondition: iSin.shape().len == 2.}
  {.precondition: ic.dtype() == kFp32.}
  {.precondition: iSin.dtype() == kFp32.}
  {.precondition: ic.shape() == iSin.shape().}
  {.precondition: ic.shape()[0] == iq.shape()[0].}
  {.precondition: ic.shape()[1] == iq.shape()[2] div 2.}
  RopeNode(
    r: engine.newViewNode(iq.dtype(), iq.shape()).getImpl(),
    q: q, cos: cos, sin: sin).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueRopeTokens[E: HwEngine, Sh: tuple, MemQ, MemC](
    engine: E, batch: BatchCommands[E], n: RopeNode,
    rr, rq: TensorImpl[MemQ, Layout[Sh, Sh]],
    rc, rs: TensorImpl[MemC, Layout[(int, int), (int, int)]]) =
  ## Enqueues one k_rope over the node's impls at the rank given by Sh.
  ##
  ## - the walked body never instantiates at a rank the op does not serve:
  ##   the handler calls it only for rank 3 nodes
  ## - rr is the finalized output buffer. The node's dataless shell stays
  ##   out of the launch site
  when tupleLen(Sh) == 3:
    let
      ob = initTensorBridge(rr.data[], rr.offset, rr.layout)
      qb = initTensorBridge(rq.data[], rq.offset, rq.layout)
      # half tables pass at their native (tokens, dim div 2) rank:
      # entries do not depend on the head. The kernel indexes them
      # per (token, i)
      cb = initTensorBridge(rc.data[], rc.offset, rc.layout)
      sb = initTensorBridge(rs.data[], rs.offset, rs.layout)
    # one threadgroup per token, 32 lanes stride the pair columns,
    # kernel grid-stride covers any token count
    let numTokens = qb.layout.shape[0]
    batch.enqueue<<(numTokens, 32)>> k_rope(ob, qb, cb, sb)
  else:
    raise ValueError.newException(
      "eval: the rope op is rank 3, walked at rank " & $tupleLen(Sh))

proc enqueueRope[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asR = n.to(RopeNode)
  if asR.isNil:
    return false
  engine.evalNode(batch, asR.q, visited)
  engine.evalNode(batch, asR.cos, visited)
  engine.evalNode(batch, asR.sin, visited)
  # finalization: the full graph is known and the output buffer allocates here
  var sh: Sh
  fillShape(sh, asR.r.shape())
  let rr = engine.makeTensorImpl(asR.r.dtype(), asR.r.shape())
    .to(TensorImpl[Mem, Layout[Sh, Sh]])
  # q and the output share one Mem (checked at the face),
  # cos/sin are f32 with their own Mem.
  type MemC = typeof(engine.allocDevice(float32, 0))
  let
    rq = asR.q.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    rc = asR.cos.getImpl().to(TensorImpl[MemC, Layout[(int, int), (int, int)]])
    rs = asR.sin.getImpl().to(TensorImpl[MemC, Layout[(int, int), (int, int)]])
  enqueueRopeTokens[E, Sh, Mem, MemC](engine, batch, asR, rr, rq, rc, rs)
  n = rr.to(TensorNode)
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("rope", enqueueRope)
