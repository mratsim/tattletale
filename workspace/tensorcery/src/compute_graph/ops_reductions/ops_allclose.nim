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
#                   allClose
#
# ############################################################

const Fleet = (elemsPerGroup: 4096, maxGroups: 1024)

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_allClose(D, tested, reference: distinct TensorBasicView, atol, rtol, equalNan: float32) =
  ## Accumulates the failure count at D[0], D[0] == 0 when every element
  ## satisfies |tested - reference| <= atol + rtol * |reference|.
  ## Same-sign infinities are equal, NaN counts as equal only with equalNan.
  ##
  ## Args:
  ## - dst: a one-slot tensor, must be zero-initialized
  ## - tested, reference: same shape, upcast to float32 on read
  let
    thrID = getLocalThreadID()
    eq = equalNan != 0.0'f32
  var
    regT: tile(tested.T)
    regR: tile(reference.T)
  let tiled = make_layout(tested.size()).logical_divide(regT.getLayout())
  let axisTiles = tiled.getTileGrid().size()
  let gridStride = getGridSize()
  const owned = regT.size() div getLaneWidth()
  var bad = 0.0'f32
  for tileID in countup(getLocalTileID(tested.size()), axisTiles - 1, gridStride):
    regT.load(tested, thrID, tileID, padWith = zero(tested.T))
    regR.load(reference, thrID, tileID, padWith = zero(reference.T))
    for i in 0 ..< owned:
      let
        a = regT.data[i].to(float32)
        b = regR.data[i].to(float32)
      var ok: bool
      if a == b:
        ok = true
      elif a != a or b != b:
        ok = eq and a != a and b != b
      elif a - a != 0.0'f32 or b - b != 0.0'f32:
        ok = false
      else:
        let
          d = a - b
          ad = if d < 0.0'f32: -d else: d
          rb = if b < 0.0'f32: -b else: b
        ok = ad <= atol + rtol * rb
      if not ok:
        bad = 1.0'f32
  let total = reduceAdd(bad)
  if thrID == 0:
    discard atomicFetchAdd(D.data[0].addr, total)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  AllCloseNode* = ref object
    r: AnyTensorImpl      # rank-1 float32 scalar, 1.0 or 0.0
    tested: TensorNode
    reference: TensorNode
    atol: float32
    rtol: float32
    equalNan: float32

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: AllCloseNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: AllCloseNode): string {.inline.} =
  "AllClose"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_allClose*(engine: AnyEngine, tested, reference: TensorNode, atol, rtol: float32, equalNan: bool): TensorNode {.contracts.} =
  let
    ia = tested.getImpl()
    ib = reference.getImpl()
  {.precondition: ia.dtype() == ib.dtype().}
  {.precondition: ia.shape().p == ib.shape().p.}
  return AllCloseNode(tested: tested, reference: reference, atol: atol, rtol: rtol,
      equalNan: if equalNan: 1.0'f32 else: 0.0'f32,
      r: engine.newPlaceholderNode(kFp32, [1]).getImpl()
    ).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc collectAllClose[E: HwEngine, T](engine: E, n: var TensorNode, stack: var seq[WalkFrame], _: typedesc[T]): bool =
  let asAll = n.to(AllCloseNode)
  if asAll.isNil():
    return false
  stack.add WalkFrame(n: asAll.tested, kernelPending: false)
  stack.add WalkFrame(n: asAll.reference, kernelPending: false)
  return true

proc enqueueAllClose[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T]): bool =
  let asAll = n.to(AllCloseNode)
  if asAll.isNil():
    return false
  when T is float32:
    let
      rImpl = asAll.r.to(engine.getTensorImplType(float32))
      ai = asAll.tested.getImpl()
      bi = asAll.reference.getImpl()
    engine.copyToDeviceFromHost(rImpl.data[], [0.0'f32])
    match(ElemType, ai.dtype()):
      let
        aView = ai.to(engine.getTensorImplType(ElemType)).bridge()
        bView = bi.to(engine.getTensorImplType(ElemType)).bridge()
      batch.enqueue<<(min(ceil_div(ai.size(), Fleet.elemsPerGroup), Fleet.maxGroups), LaneWidth)>> k_allClose(rImpl.bridge(), aView, bView, asAll.atol, asAll.rtol, asAll.equalNan)
  else:
    raiseAssert("allClose: the output is a float32 scalar")
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("allClose", collectAllClose, enqueueAllClose)
