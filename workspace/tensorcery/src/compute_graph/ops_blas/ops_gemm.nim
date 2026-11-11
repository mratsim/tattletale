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
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/iface
import workspace/math
import ../../tensors_internals
import ../batching
import workspace/design_by_contract

# ############################################################
#
#                   Gemm
#
# ############################################################

template GemmMma*(T: typedesc): untyped =
  ## TODO: the (32, 32, 16) tile schedule is hardcoded
  tiledMma(float32, T, T, 32, 32, 16)

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_gemm(mma: static TiledMma, D: distinct TensorView, alpha: float32,
    A, B: distinct TensorView, beta: float32, C: distinct TensorView) =
  ## D = alpha·A·B + beta·C
  let bT = B.transpose(1, 0)
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
      var rB: tile(bT.T, mma.getMmaLayout(opB))
      rD.fillWith(zero(float32))
      for k in 0 ..< kTiles:
        rA.load(A, tvmA, thrID, (m, k))
        rB.load(bT, tvmB, thrID, (n, k))
        mma.atom.gemm_tile(rD, rA, rB, rD)

      if alpha != 1.0'f32:
        for i in 0 ..< cLayout.size():
          rD[i] = alpha * rD[i]

      if beta != 0.0'f32:
        let rC = tile(C.T, cLayout).load(C, tvmC, thrID, (m, n))
        for i in 0 ..< cLayout.size():
          rD[i] += beta * rC[i].to(float32)

      D.store(rD.downcast(D.T), tvmC, thrID, (m, n))

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  GemmNode* = ref object
    r: AnyTensorImpl      # output slot, metadata only until finalization
    alpha: float32
    a, b: TensorNode
    beta: float32
    c: TensorNode         # nil when beta = 0

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: GemmNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: GemmNode): string {.inline.} =
  "Gemm"

func size*(n: GemmNode): int {.inline.} =
  n.r.size()

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_gemm*(
    engine: AnyEngine,
    alpha: float32, a, b: TensorNode,
    beta: float32, c: TensorNode = NilTensorNode()): TensorNode {.contracts.} =
  ## D = alpha·A·B + beta·C, C nil at beta = 0.
  let
    ia = a.getImpl()
    ib = b.getImpl()
  {.precondition: ia.shape().p.len == 2.}
  {.precondition: ib.shape().p.len == 2.}
  {.precondition: ia.dtype() == ib.dtype().}
  {.precondition: ia.shape().p[1] == ib.shape().p[0].}
  if not c.isNil():
    let ic = c.getImpl()
    {.precondition: ic.dtype() == ia.dtype().}
    {.precondition: ic.shape().p == [ia.shape().p[0], ib.shape().p[1]].}
  return GemmNode(alpha: alpha, a: a, b: b, beta: beta, c: c,
      r: engine.newPlaceholderNode(ia.dtype(), [ia.shape().p[0], ib.shape().p[1]]).getImpl()
    ).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueGemm[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int): bool =
  let asGemm = n.to(GemmNode)
  if asGemm.isNil():
    return false
  engine.enqueueNode(batch, asGemm.a)
  engine.enqueueNode(batch, asGemm.b)

  when rank == 2 and T is (float32 | float16 | bfloat16):
    let
      rImpl = asGemm.r.to(engine.getTensorImplType(T, rank))
      aImpl = asGemm.a.getImpl().to(engine.getTensorImplType(T, rank))
      bImpl = asGemm.b.getImpl().to(engine.getTensorImplType(T, rank))
      tiles = ceil_div(rImpl.shape[0], 32) * ceil_div(rImpl.shape[1], 32)
    engine.materialize(rImpl)
    if asGemm.c.isNil():
      # beta = 0, the C view is a dummy, never loaded
      batch.enqueue<<(tiles, 32)>> k_gemm(GemmMma(T),
        rImpl.bridge(), asGemm.alpha, aImpl.bridge(), bImpl.bridge(), asGemm.beta, rImpl.bridge())
    else:
      engine.enqueueNode(batch, asGemm.c)
      let cImpl = asGemm.c.getImpl().to(engine.getTensorImplType(T, rank))
      batch.enqueue<<(tiles, 32)>> k_gemm(GemmMma(T),
        rImpl.bridge(), asGemm.alpha, aImpl.bridge(), bImpl.bridge(), asGemm.beta, cImpl.bridge())
  else:
    raiseAssert("gemm: rank " & $rank & " or dtype unsupported, the op is rank 2, float16, bfloat16 or float32")

  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Gemm", enqueueGemm)
