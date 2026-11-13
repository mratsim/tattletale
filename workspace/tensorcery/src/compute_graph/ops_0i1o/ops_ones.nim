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
  OnesNode* = ref object of TensorNode
    r: AnyTensorImpl

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: OnesNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(_: typedesc[OnesNode]): static string {.inline.} =
  "Ones"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_ones*(engine: AnyEngine, dtype: Dtype, shape: openArray[int]): TensorGraph {.contracts.} =
  {.precondition: shape.len in 1 .. 5.}
  return OnesNode(r: engine.newPlaceholderNode(dtype, shape).getImpl()).to(TensorGraph)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueOnes[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorGraph, _: typedesc[T]): bool =
  let asOnes = n.to(OnesNode)
  if asOnes.isNil():
    return false
  let rImpl = asOnes.r.to(engine.getTensorImplType(T))
  batch.enqueue<<(ceil_div(rImpl.size(), FillFleet.elemsPerGroup), LaneWidth)>> k_fill(rImpl.bridge(), 1.0'f32)
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp(OnesNode, enqueueOnes, opName(OnesNode))
