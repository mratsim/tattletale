# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Elementwise add op: kernel, graph node and Tensor face op colocated.
##
## The template every other tattleformers op copies:
##
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk at import time
## - the Tensor face op over the erased engine
##
## The kernel is tensorcery's `k_add`. General elementwise arithmetic stays
## in tensorcery. Transformer-specialized kernels go to positron.

import workspace/crucible
import workspace/math
import workspace/tensorcery/src/tensor_datatypes
import workspace/tensorcery/src/kernels/k_add

type AddNode* = ref object
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

proc eval*(n: AddNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: AddNode): AnyTensorImpl =
  n.r

# ═══════════════════════════════════════════════════════════════════════
#  Eval - enqueue the node's kernel into the walk's batch
# ═══════════════════════════════════════════════════════════════════════

proc enqueueAdd[E: HwEngine, Sh: tuple, MemZ, MemA, MemB](
    engine: E, batch: Batch[E], n: AddNode) =
  ## Enqueues one k_add over the node's impls.
  ## Each impl's Mem type is a generic argument of this proc.
  ## A body-local type alias defeats the launch's generic binding.
  let
    rz = n.r.to(TensorImpl[MemZ, Layout[Sh, Sh]])
    ra = n.a.getImpl().to(TensorImpl[MemA, Layout[Sh, Sh]])
    rb = n.b.getImpl().to(TensorImpl[MemB, Layout[Sh, Sh]])
    numel = rz.layout.shape.getNumElems()
    zv = TensorBridge[MemZ, Sh, Sh](data: rz.data[], offset: rz.offset, layout: rz.layout)
    xv = TensorBridge[MemA, Sh, Sh](data: ra.data[], offset: ra.offset, layout: ra.layout)
    yv = TensorBridge[MemB, Sh, Sh](data: rb.data[], offset: rb.offset, layout: rb.layout)
  batch.enqueue<<(ceil_div(numel, 128), 32)>> k_add(zv, xv, yv)

proc dispatchAddInputs[E: HwEngine, Sh: tuple, MemZ](
    engine: E, batch: Batch[E], n: AddNode) =
  ## Input-side dtype dispatch of one add node. The result Mem is a generic argument.
  case n.a.getImpl().dtype()
  of kFp32:
    type MemA = typeof(engine.allocDevice(float32, 0))
    case n.b.getImpl().dtype()
    of kFp32:
      type MemB = typeof(engine.allocDevice(float32, 0))
      enqueueAdd[E, Sh, MemZ, MemA, MemB](engine, batch, n)
    of kBf16:
      type MemB = typeof(engine.allocDevice(bfloat16, 0))
      enqueueAdd[E, Sh, MemZ, MemA, MemB](engine, batch, n)
    else:
      raise ValueError.newException("eval: kFp16 add operand unsupported")
  of kBf16:
    type MemA = typeof(engine.allocDevice(bfloat16, 0))
    case n.b.getImpl().dtype()
    of kFp32:
      type MemB = typeof(engine.allocDevice(float32, 0))
      enqueueAdd[E, Sh, MemZ, MemA, MemB](engine, batch, n)
    of kBf16:
      type MemB = typeof(engine.allocDevice(bfloat16, 0))
      enqueueAdd[E, Sh, MemZ, MemA, MemB](engine, batch, n)
    else:
      raise ValueError.newException("eval: kFp16 add operand unsupported")
  else:
    raise ValueError.newException("eval: kFp16 add operand unsupported")

proc enqueueAddNode*[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool =
  ## Enqueue handler of the add op, registered below. The erased node
  ## downcasts to the add node here. A node of another op returns false,
  ## leaving the dispatch to that op's handler.
  let asAdd = n.to(AddNode)
  if asAdd.isNil:
    return false
  if asAdd.enqueued:
    # re-visit under the walk's fan-out, the subtree is already scheduled
    return true
  asAdd.enqueued = true
  evalNode[E](engine, batch, asAdd.a)
  evalNode[E](engine, batch, asAdd.b)
  # per-node dtype dispatch: each erased dtype selects the static
  # Mem of its impl, kFp16 has no erased impls and raises
  case asAdd.r.dtype()
  of kFp32:
    type MemZ = typeof(engine.allocDevice(float32, 0))
    dispatchAddInputs[E, Sh, MemZ](engine, batch, asAdd)
  of kBf16:
    type MemZ = typeof(engine.allocDevice(bfloat16, 0))
    dispatchAddInputs[E, Sh, MemZ](engine, batch, asAdd)
  else:
    raise ValueError.newException("eval: kFp16 add result unsupported")
  return true

registerEnqueueNode(enqueueAddNode)

# ═══════════════════════════════════════════════════════════════════════
#  Tensor face
# ═══════════════════════════════════════════════════════════════════════

proc tn_add*(engine: AnyEngine, a, b: TensorNode): TensorNode =
  let ia = a.getImpl()
  let ib = b.getImpl()
  if ia.shape() != ib.shape():
    raise ValueError.newException("tn_add: shape mismatch")
  # a bfloat16 operand rounds the sum back to bfloat16. The result dtype
  # is fp32 only when both operands are fp32.
  let dz = if ia.dtype() == kFp32 and ib.dtype() == kFp32: kFp32 else: kBf16
  AddNode(a: a, b: b, r: engine.newTensorNode(dz, ia.shape()).getImpl()).to(TensorNode)

proc `+`*(a, b: Tensor): Tensor =
  Tensor(engine: a.engine, node: tn_add(a.engine, a.node, b.node))
