# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## RmsNorm op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the matmul op template:
##
## - the device kernels
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk by registerOp
## - the Tensor face op over the erased engine
##
## Numerics record (D-9, single-rounding rmsnorm): x·rstd·w runs entirely in float32 with ONE round-to-nearest-even at the final store.
## The stock kernel stages float32 register tiles and reads same-dtype views.
## A bfloat16 input therefore takes this composition:
##
##   bf16 in -> k_cast -> k_rmsnorm_unweighted (in place) -> k_weighted_round_to_output -> bf16 out
##
## - k_cast upcasts the bfloat16 input into a node-owned float32 scratch,
##   exactly (bfloat16 -> float32 loses nothing)
## - k_rmsnorm_unweighted normalizes the scratch in place: x·rstd in float32,
##   stored back into the scratch through a float32 slot, which is exact
## - k_weighted_round_to_output multiplies the scratch by the bfloat16 weight
##   in float32 and rounds ONCE into the bfloat16 output
##
## The float32 path is one launch of the stock weighted kernel.

import std/typetraits
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_rmsnorm
import workspace/crucible/src/builtins/builtins_gpu_types
import workspace/iface
import workspace/math
import ../../tensors_internals
import ../dispatch
import workspace/design_by_contract

# Batch enqueue walks the kernel bodies at compile time and resolves the tile
# algebra's thread->value map templates in the importing module's scope.
export tile_algebra

# ############################################################
#
#                   RmsNorm
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Device kernels
# ═══════════════════════════════════════════════════════════════════════

proc k_cast*(o, src: distinct TensorView) =
  ## o = src elementwise, each element converted into o's element type.
  ## The bfloat16 -> float32 direction of the rmsnorm staging below is exact.
  ##
  ## Expected input:
  ## - o: the output view, same shape as src
  ## - src: the source view
  let
    rows = o.layout.shape[0]
    cols = o.layout.shape[1]
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for row in countup(group0, rows - 1, groupStride):
    for col in countup(lane, cols - 1, lanes):
      o[(row, col)] = src[(row, col)].to(o.T)

proc k_rmsnorm_unweighted*(o, src: distinct TensorView, eps: float32) =
  ## o = x·rsqrt(mean(x²) + eps) over the last axis, the stock kernel's
  ## unweighted entry point consumed as-is.
  ##
  ## Expected input:
  ## - o, src: (tokens, width) float32 views. o may alias src: the kernel
  ##   reads each axis strip whole before writing it, and each lane writes
  ##   only the positions it read
  ## - eps: the stability constant added to the mean of squares
  ##
  ## Output:
  ## - o fully written with the normalized strips in float32
  rmsnorm(o, src, eps, 1)

proc k_rmsnorm_weighted*(o, src, w: distinct TensorView, eps: float32) =
  ## o = x·rsqrt(mean(x²) + eps)·w over the last axis, the stock kernel's
  ## weighted entry point consumed as-is.
  ##
  ## Expected input:
  ## - o, src: (tokens, width) views of one float32 element type
  ## - w: (width,) weight view of the same element type
  ## - eps: the stability constant added to the mean of squares
  ##
  ## Output:
  ## - o fully written, one rounding through o's element type at the store
  rmsnorm(o, src, w, eps, 1)

proc k_weighted_round_to_output*(o, src, w: distinct TensorView) =
  ## o = src·w elementwise, one rounding into o's element type at the store: the single rounding of the rmsnorm composition.
  ## w is rank 1 with one slot per o's last axis position. The multiply runs in float32 with bfloat16 slots upcast exactly on the load.
  ##
  ## Expected input:
  ## - o: the output view, same shape as src
  ## - src: the float32 normalized strips
  ## - w: (width,) weight view
  let
    rows = o.layout.shape[0]
    cols = o.layout.shape[1]
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for row in countup(group0, rows - 1, groupStride):
    for col in countup(lane, cols - 1, lanes):
      o[(row, col)] = (src[(row, col)] * w[(col,)].to(float32)).to(o.T)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  RmsNormNode* = ref object
    r: AnyTensorImpl      # output slot, metadata only until finalization
    x: TensorNode
    w: TensorNode
    eps: float32
    scratch: AnyTensorImpl
      ## Float32 staging buffer for a bfloat16 input: the stock kernel stages
      ## float32 register tiles and reads same-dtype views.
      ## k_cast upcasts the input exactly into this buffer.
      ## k_rmsnorm_unweighted stores the normalized strips back here in place.
      ##
      ## Node-owned lifetime:
      ## - the launch closures borrow the buffer and only run at the batch's submit,
      ##   after the enqueue handler's scope has exited
      ## - allocated per enqueue, released with the node

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(n: RmsNormNode): AnyTensorImpl =
  n.r

func opName*(n: RmsNormNode): string =
  "rmsnorm"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_rmsnorm*(engine: AnyEngine, x, w: TensorNode, eps: float32): TensorNode {.contracts.} =
  ## y = x·rsqrt(mean(x²) + eps)·w over the last axis: (tokens, width) in,
  ## same shape and dtype out.
  ##
  ## Args:
  ## - x: the input node of shape (tokens, width)
  ## - w: the weight node of shape (width,)
  ## - eps: the stability constant added to the mean of squares
  ## The math runs in float32 with one rounding at the store, see the module
  ## header for the numerics record.
  let
    ix = x.getImpl()
    iw = w.getImpl()
  {.precondition: ix.shape().len == 2.}
  {.precondition: iw.shape().len == 1.}
  {.precondition: iw.shape()[0] == ix.shape()[1].}
  {.precondition: ix.dtype() == iw.dtype().}
  RmsNormNode(
    r: engine.newViewNode(ix.dtype(), ix.shape()).getImpl(),
    x: x, w: w, eps: eps).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueRmsnormRows[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: RmsNormNode, rr: TensorImpl[Mem, Layout[Sh, Sh]]) =
  ## Enqueues the node's kernels: one launch of the stock weighted kernel for a float32 output,
  ## the cast -> normalize -> weighted-round composition for a bfloat16 output.
  when tupleLen(Sh) == 2:
    let
      rx = n.x.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      rw = n.w.getImpl().to(TensorImpl[Mem, Layout[(int,), (int,)]])
      rows = rr.layout.shape[0]
      xv = initTensorBridge(rx.data[], rx.offset, rx.layout)
      wv = initTensorBridge(rw.data[], rw.offset, rw.layout)
      ov = initTensorBridge(rr.data[], rr.offset, rr.layout)
    let eps = n.eps
    when Mem.T is bfloat16:
      let
        rAcc = n.scratch.to(TensorImpl[typeof(engine.allocDevice(float32, 0)), Layout[Sh, Sh]])
        accV = initTensorBridge(rAcc.data[], rAcc.offset, rAcc.layout)
      # one threadgroup per output token, 32 lanes stride the width
      batch.enqueue<<(rows, 32)>> k_cast(accV, xv)
      batch.enqueue<<(rows, 32)>> k_rmsnorm_unweighted(accV, accV, eps)
      batch.enqueue<<(rows, 32)>> k_weighted_round_to_output(ov, accV, wv)
    else:
      batch.enqueue<<(rows, 32)>> k_rmsnorm_weighted(ov, xv, wv, eps)
  else:
    raise ValueError.newException(
      "eval: the rmsnorm op is rank 2, walked at rank " & $tupleLen(Sh))

proc enqueueRmsnorm[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asN = n.to(RmsNormNode)
  if asN.isNil:
    return false
  engine.evalNode(batch, asN.x, visited)
  engine.evalNode(batch, asN.w, visited)
  # finalization: the full graph is known and the output buffer allocates here
  var sh: Sh
  fillShape(sh, asN.r.shape())
  let rr = engine.makeTensorImpl(asN.r.dtype(), asN.r.shape())
    .to(TensorImpl[Mem, Layout[Sh, Sh]])
  if asN.r.dtype() == kBf16:
    asN.scratch = engine.makeTensorImpl(kFp32, asN.r.shape())
  enqueueRmsnormRows[E, Sh, Mem](engine, batch, asN, rr)
  n = rr.to(TensorNode)
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("rmsnorm", enqueueRmsnorm)
