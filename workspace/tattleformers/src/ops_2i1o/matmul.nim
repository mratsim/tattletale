# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Matrix multiply op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the add op template:
##
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk at import time
## - the Tensor face op over the erased engine
##
## The kernel is tensorcery's `k_matmul` over ceramic's tile algebra:
## general linear algebra stays in tensorcery.

import std/typetraits
import workspace/crucible
import workspace/math
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/tensor_datatypes
import workspace/tensorcery/src/kernels/k_matmul

# Batch enqueue walks k_matmul's body at compile time and resolves the tile
# algebra's thread->value map templates in the importing module's scope.
export tile_algebra

type MatmulNode* = ref object
  r: AnyTensorImpl
  enqueued: bool
    ## Launch-once marker under the eval walk's fan-out re-walks:
    ## - a downstream op that fans out over a shared input re-walks
    ##   the whole upstream chain (the FFN's two projections share one input)
    ## - the repeat visit must not re-run the handler, the launch is
    ##   already in the batch and its subtree is already scheduled
    ## Precondition: an op node is constructed per forward and enqueued in
    ## exactly one eval, so the flag never crosses an eval boundary.
  a, b: TensorNode

proc eval*(n: MatmulNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: MatmulNode): AnyTensorImpl =
  n.r

# ═══════════════════════════════════════════════════════════════════════
#  Eval - enqueue the node's kernel into the walk's batch
# ═══════════════════════════════════════════════════════════════════════

proc enqueueMatmul[E: HwEngine, Sh: tuple, Mem, Elem](
    engine: E, batch: Batch[E], n: MatmulNode) =
  ## Enqueues one k_matmul over the node's impls.
  ## The three impls share one Mem: the operands' dtypes are equal
  ## (the Metal MMA atoms require equal operand types).
  let
    rr = n.r.to(TensorImpl[Mem, Layout[Sh, Sh]])
    ra = n.a.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    rb = n.b.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    rv = TensorBridge[Mem, Sh, Sh](data: rr.data[], offset: rr.offset, layout: rr.layout)
    av = TensorBridge[Mem, Sh, Sh](data: ra.data[], offset: ra.offset, layout: ra.layout)
    bv = TensorBridge[Mem, Sh, Sh](data: rb.data[], offset: rb.offset, layout: rb.layout)
  when tupleLen(Sh) == 2:
    let
      m = rv.layout.shape[0]
      cols = rv.layout.shape[1]
      tiles = ceil_div(m, 32) * ceil_div(cols, 32)
    # one squadron per (32, 32) output tile, 32 lanes per squadron
    batch.enqueue<<(tiles, 32)>> k_matmul(rv, av, bv,
      tiledMma(float32, Elem, Elem, 32, 32, 16))
  else:
    raise ValueError.newException(
      "eval: the matmul op is rank 2, walked at rank " & $tupleLen(Sh))

proc enqueueMatmulNode*[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool =
  ## Enqueue handler of the matmul op, registered below. The erased node
  ## downcasts to the matmul node here. A node of another op returns false,
  ## leaving the dispatch to that op's handler.
  let mm = n.to(MatmulNode)
  if mm.isNil:
    return false
  if mm.enqueued:
    # re-visit under the walk's fan-out, the subtree is already scheduled
    return true
  mm.enqueued = true
  evalNode[E](engine, batch, mm.a)
  evalNode[E](engine, batch, mm.b)
  # per-node dtype dispatch: both operands share one dtype (checked at the face),
  # kFp16 has no erased impls and raises
  case mm.a.getImpl().dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    enqueueMatmul[E, Sh, Mem, float32](engine, batch, mm)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    enqueueMatmul[E, Sh, Mem, bfloat16](engine, batch, mm)
  else:
    raise ValueError.newException("eval: kFp16 matmul operand unsupported")
  return true

registerEnqueueNode(enqueueMatmulNode)

# ═══════════════════════════════════════════════════════════════════════
#  Tensor face
# ═══════════════════════════════════════════════════════════════════════

proc tn_matmul*(engine: AnyEngine, a, b: TensorNode): TensorNode =
  ## o = a·b: (M, K) × (K, N) → (M, N).
  ##
  ## Args:
  ## - a: (M, K) operand
  ## - b: (K, N) operand, any strides
  ## The output keeps the operands' dtype: bfloat16 operands accumulate in f32
  ## and round once into the bfloat16 output (round-to-nearest-even).
  ## Raises on a dtype mismatch or on a K mismatch.
  let
    ia = a.getImpl()
    ib = b.getImpl()
  if ia.dtype() != ib.dtype():
    raise ValueError.newException(
      "tn_matmul: the operands' dtypes must be equal, got " & $ia.dtype() &
      " and " & $ib.dtype())
  if ia.shape().len != 2 or ib.shape().len != 2:
    raise ValueError.newException(
      "tn_matmul: the operands must be rank 2, got rank " & $ia.shape().len &
      " and " & $ib.shape().len)
  if ia.shape()[1] != ib.shape()[0]:
    raise ValueError.newException(
      "tn_matmul: K mismatch, a's second dimension " & $ia.shape()[1] &
      " != b's first dimension " & $ib.shape()[0])
  let outShape = [ia.shape()[0], ib.shape()[1]]
  MatmulNode(
    r: engine.newTensorNode(ia.dtype(), outShape.toSpan()).getImpl(),
    a: a, b: b).to(TensorNode)
