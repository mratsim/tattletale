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
import workspace/math
import workspace/design_by_contract
import ../../tensors_internals
import ../eval
import ../env

# ############################################################
#
#                   Add
#
# ############################################################

const Fleet = (elemsPerGroup: 128)

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_add(D, A, B: distinct TensorBasicView) =
  let thrID = getLocalThreadID()
  var reg: tile(D.T)
  let tiled = make_layout(D.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = getGridSize()
  for tileID in countup(getLocalTileID(grid), cSize - 1, gridStride):
    let rA = tile(A.T).load(A, thrID, tileID)
    let rB = tile(B.T).load(B, thrID, tileID)
    var rD: tile(D.T)
    for i in 0 ..< rA.size():
      when typeof(rA[i]) is (bfloat16|float16) and ccGetBackend() != ctCuda:
        rD[i] = (rA[i].to(float32) + rB[i].to(float32)).to(typeof(rA[i]))
      else:
        rD[i] = rA[i] + rB[i]
    D.store(rD, thrID, tileID)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  AddNode* = ref object of TensorNode
    r: AnyTensorImpl
    a, b: TensorGraph

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: AddNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(_: typedesc[AddNode]): static string {.inline.} =
  "Add"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_add*(engine: AnyEngine, a, b: TensorGraph): TensorGraph {.contracts.} =
  let ia = a.getImpl()
  let ib = b.getImpl()
  {.precondition: ia.shape().p == ib.shape().p.}
  {.precondition: ia.dtype() == ib.dtype().}
  return AddNode(a: a, b: b,
      r: engine.newPlaceholderNode(ia.dtype(), ia.shape().p).getImpl()).to(TensorGraph)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueAdd[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorGraph, _: typedesc[T]): bool =
  let asAdd = n.to(AddNode)
  if asAdd.isNil():
    return false
  let
    rImpl = asAdd.r.to(engine.getTensorImplType(T))
    aImpl = asAdd.a.getImpl().to(engine.getTensorImplType(T))
    bImpl = asAdd.b.getImpl().to(engine.getTensorImplType(T))
    size = rImpl.size()
  batch.enqueue<<(ceil_div(size, Fleet.elemsPerGroup), LaneWidth)>> k_add(rImpl.bridge(), aImpl.bridge(), bImpl.bridge())
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp(AddNode, enqueueAdd, opName(AddNode))
