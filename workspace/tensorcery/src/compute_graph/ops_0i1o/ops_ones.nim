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
#                   Ones
#
# ############################################################

type
  OnesNode* = ref object
    r: AnyTensorImpl

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: OnesNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: OnesNode): string {.inline.} =
  "Ones"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_ones*(engine: AnyEngine, dtype: Dtype, shape: openArray[int]): TensorNode {.contracts.} =
  {.precondition: shape.len in 1 .. 5.}
  return OnesNode(r: engine.newPlaceholderNode(dtype, shape).getImpl()).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc collectOnes[E: HwEngine, T](engine: E, n: var TensorNode, stack: var seq[WalkFrame], _: typedesc[T]): bool =
  let asOnes = n.to(OnesNode)
  if asOnes.isNil():
    return false
  return true

proc enqueueOnes[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T]): bool =
  let asOnes = n.to(OnesNode)
  if asOnes.isNil():
    return false
  let rImpl = asOnes.r.to(engine.getTensorImplType(T))
  batch.enqueue<<(ceil_div(rImpl.size(), FillFleet.elemsPerGroup), LaneWidth)>> k_fill(rImpl.bridge(), 1.0'f32)
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Ones", collectOnes, enqueueOnes)
