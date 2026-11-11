# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_gemm
import ../ops_blas/ops_gemm
import workspace/iface
import workspace/math
import ../../tensors_internals
import ../batching
import workspace/design_by_contract

# ############################################################
#
#                   Linear
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_linear(mma: static TiledMma, D, A, B, bias: distinct TensorView, withBias: static bool) =
  ## D = A·B + bias over the last axis
  # weights store (out, K), already the BLAS B^T (N, K) form
  const tvmA = mma.getThreadValueMap(opA)
  const tvmB = mma.getThreadValueMap(opB)
  const tvmC = mma.getThreadValueMap(opC)
  const cLayout = mma.getMmaLayout(opC)

  let cGeom = D.tileWith((mma.MT, mma.NT))
  let aGeom = A.tileWith((mma.MT, mma.KT))

  let cGrid = cGeom.getTileGrid()
  let (_, kTiles) = aGeom.getTileGrid()
  let thrID = getLocalThreadID()

  if thrID < getLaneWidth():
    let
      cSize = cGrid.size()
      gridStride = getGridSize()
    for tileID in countup(cGrid.getLocalTileID(), cSize - 1, gridStride):
      let (m, n) = idx2crd(cGrid, tileID)
      var rD: tile(float32, cLayout)
      var rA: tile(A.T, mma.getMmaLayout(opA))
      var rB: tile(B.T, mma.getMmaLayout(opB))
      rD.fillWith(zero(float32))
      for k in 0 ..< kTiles:
        rA.load(A, tvmA, thrID, (m, k))
        rB.load(B, tvmB, thrID, (n, k))
        mma.atom.gemm_tile(rD, rA, rB, rD)

      when withBias:
        let broadcastedView = make_view(bias.data,
            make_layout((D.layout.shape[0], D.layout.shape[1]), (Int[0](), Int[1]())),
            bias.offset)
        let rBias = tile(bias.T, cLayout).load(broadcastedView, tvmC, thrID, (m, n))
        for i in 0 ..< cLayout.size():
          rD[i] += rBias[i].to(float32)

      D.store(rD.downcast(D.T), tvmC, thrID, (m, n))

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  LinearNode* = ref object
    r: AnyTensorImpl      # output slot, metadata only until finalization
    x, w: TensorNode
    b: TensorNode         # nil when the layer has no bias

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: LinearNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: LinearNode): string {.inline.} =
  "Linear"

func size*(n: LinearNode): int {.inline.} =
  n.r.size()

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_linear*(engine: AnyEngine, x, weight, bias: TensorNode): TensorNode {.contracts.} =
  ## o = x·Wᵀ + b: (M, K) × (out, K) → (M, out), bias nil when the layer has none
  let
    ix = x.getImpl()
    iw = weight.getImpl()
  {.precondition: ix.shape().p.len == 2.}
  {.precondition: iw.shape().p.len == 2.}
  {.precondition: ix.dtype() == iw.dtype().}
  {.precondition: ix.shape().p[1] == iw.shape().p[1].}
  if not bias.isNil():
    let ib = bias.getImpl()
    {.precondition: ib.shape().p.len == 1.}
    {.precondition: ib.dtype() == ix.dtype().}
    {.precondition: ib.shape().p[0] == iw.shape().p[0].}
  return LinearNode(x: x, w: weight, b: bias,
      r: engine.newPlaceholderNode(ix.dtype(), [ix.shape().p[0], iw.shape().p[0]]).getImpl()
    ).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueLinear[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int): bool =
  let asLinear = n.to(LinearNode)
  if asLinear.isNil():
    return false
  engine.enqueueNode(batch, asLinear.x)
  engine.enqueueNode(batch, asLinear.w)

  when rank == 2 and T is (float32 | float16 | bfloat16):
    let
      rImpl = asLinear.r.to(engine.getTensorImplType(T, rank))
      xImpl = asLinear.x.getImpl().to(engine.getTensorImplType(T, rank))
      wImpl = asLinear.w.getImpl().to(engine.getTensorImplType(T, rank))
      tiles = ceil_div(rImpl.shape[0], 32) * ceil_div(rImpl.shape[1], 32)
    engine.materialize(rImpl)
    if asLinear.b.isNil():
      batch.enqueue<<(tiles, 32)>> k_linear(GemmMma(T),
        rImpl.bridge(), xImpl.bridge(), wImpl.bridge(), rImpl.bridge(), false)
    else:
      engine.enqueueNode(batch, asLinear.b)
      let bImpl = asLinear.b.getImpl().to(engine.getTensorImplType(T, 1))
      batch.enqueue<<(tiles, 32)>> k_linear(GemmMma(T),
        rImpl.bridge(), xImpl.bridge(), wImpl.bridge(), bImpl.bridge(), true)
  else:
    raiseAssert("linear: rank " & $rank & " or dtype unsupported, the op is rank 2, float16, bfloat16 or float32")
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Linear", enqueueLinear)
