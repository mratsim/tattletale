# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/crucible
import workspace/ceramic
import workspace/iface
import workspace/math
import ../../tensors_internals
import ../dispatch
import workspace/design_by_contract

# ############################################################
#
#                   Swapaxes
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  SwapaxesNode* = ref object
    r: AnyTensorImpl      # view slot over the source storage
    a: TensorNode
    axis1, axis2: int

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: SwapaxesNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: SwapaxesNode): string {.inline.} =
  "Swapaxes"

func size*(n: SwapaxesNode): int {.inline.} =
  n.r.size()

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_swapaxes*(engine: AnyEngine, x: TensorNode, axis1, axis2: int): TensorNode {.contracts.} =
  let ix = x.getImpl()
  {.precondition: axis1 >= 0 and axis1 < ix.shape().p.len.}
  {.precondition: axis2 >= 0 and axis2 < ix.shape().p.len.}
  {.precondition: axis1 != axis2.}
  var outShape = @(ix.shape().p)
  swap(outShape[axis1], outShape[axis2])
  return SwapaxesNode(a: x, axis1: axis1, axis2: axis2,
      r: engine.newPlaceholderNode(ix.dtype(), outShape).getImpl()
    ).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueSwapaxes[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int): bool =
  let asSwap = n.to(SwapaxesNode)
  if asSwap.isNil:
    return false
  engine.evalNode(batch, asSwap.a)
  let src = asSwap.a.getImpl().to(engine.getTensorImplType(T, rank))
  let r = asSwap.r.to(engine.getTensorImplType(T, rank))
  # a strided view over the source storage: same data and offset,
  # shape permuted at construction, strides swapped to match
  r.data = src.data
  r.offset = src.offset
  r.stride = src.stride
  swap(r.stride[asSwap.axis1], r.stride[asSwap.axis2])
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Swapaxes", enqueueSwapaxes)
