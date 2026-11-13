# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/ceramic
import workspace/crucible
import workspace/iface
import workspace/design_by_contract
import ../../tensors_internals
import ../eval
import ../env
import ./ops_zeros

# ############################################################
#
#                   Full
#
# ############################################################

type
  FullNode* = ref object
    r: AnyTensorImpl
    val: float32

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: FullNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: FullNode): string {.inline.} =
  "Full"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_full*(engine: AnyEngine, vals: float32, dtype: Dtype, shape: openArray[int]): TensorNode {.contracts.} =
  {.precondition: shape.len in 1 .. 5.}
  return FullNode(r: engine.newPlaceholderNode(dtype, shape).getImpl(), val: vals).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc collectFull[E: HwEngine, T](engine: E, n: var TensorNode, stack: var seq[WalkFrame], _: typedesc[T]): bool =
  let asFull = n.to(FullNode)
  if asFull.isNil():
    return false
  return true

proc enqueueFull[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T]): bool =
  let asFull = n.to(FullNode)
  if asFull.isNil():
    return false
  let rImpl = asFull.r.to(engine.getTensorImplType(T))
  batch.enqueue<<(ceil_div(rImpl.size(), FillFleet.elemsPerGroup), LaneWidth)>> k_fill(rImpl.bridge(), asFull.val)
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Full", collectFull, enqueueFull)
