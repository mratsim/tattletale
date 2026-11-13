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
#                   Linear
#
# ############################################################

const Fleet = (tileM: 32, tileN: 32)

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_linear(mma: static TiledMma, D, A, B, bias: distinct TensorView) =
  ## D = A·B + bias over the last axis
  const tvmA = mma.getThreadValueMap(opA)
  const tvmB = mma.getThreadValueMap(opB)
  const tvmC = mma.getThreadValueMap(opC)
  const cLayout = mma.getMmaLayout(opC)

  when rank(A.layout.shape) == 3:
    # A (B, M, K):(s0, s1, sk) splits into nSlices = B affine rank-2 GEMMs, one launch:
    #
    #   mode    B           M           K
    #   stride  s0          s1          sk
    #   role    slice idx   rows        reduce
    #
    #   A ──▶ slice(b) = (M, K):(s1, sk) @ offset + b·s0     B rides the tile grid
    #   D ──▶ slice(b) = (M, N):(d1, dn) @ offset + b·d0     W, bias shared
    #
    #   element (b, m, k) ──▶ b·s0 + m·s1 + k·sk
    let
      nSlices = A.layout.shape[0]
      vD = make_view(D.data, slice(D.layout, 0, _, _), D.offset)
      vA = make_view(A.data, slice(A.layout, 0, _, _), A.offset)
  else:
    let
      nSlices = 1
      vD = D
      vA = A

  let cGeom = vD.tileWith((mma.MT, mma.NT))
  let aGeom = vA.tileWith((mma.MT, mma.KT))

  let cGrid = cGeom.getTileGrid()
  let (_, kTiles) = aGeom.getTileGrid()
  let thrID = getLocalThreadID()

  if thrID < getLaneWidth():
    let
      cSize = cGrid.size()
      gridStride = getGridSize()
    for tileID in countup(cGrid.getLocalTileID(), nSlices * cSize - 1, gridStride):
      let
        sliceIdx = tileID div cSize
        (m, n) = idx2crd(cGrid, tileID mod cSize)
      when rank(A.layout.shape) == 3:
        let
          sliceD = make_view(D.data, vD.layout, D.offset + sliceIdx * D.layout.stride[0])
          sliceA = make_view(A.data, vA.layout, A.offset + sliceIdx * A.layout.stride[0])
      else:
        let
          sliceD = vD
          sliceA = vA
      var rD: tile(float32, cLayout)
      var rA: tile(A.T, mma.getMmaLayout(opA))
      var rB: tile(B.T, mma.getMmaLayout(opB))
      rD.fillWith(zero(float32))
      for k in 0 ..< kTiles:
        rA.load(sliceA, tvmA, thrID, (m, k))
        rB.load(B, tvmB, thrID, (n, k))
        mma.atom.gemm_tile(rD, rA, rB, rD)

      if not bias.isNil():
        let broadcastedView = make_view(bias.data,
            make_layout((sliceD.layout.shape[0], sliceD.layout.shape[1]), (Int[0](), Int[1]())),
            bias.offset)
        let rBias = tile(bias.T, cLayout).load(broadcastedView, tvmC, thrID, (m, n))
        for i in 0 ..< cLayout.size():
          rD[i] += rBias[i].to(float32)

      sliceD.store(rD.downcast(sliceD.T), tvmC, thrID, (m, n))

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  LinearNode* = ref object
    r: AnyTensorImpl
    x, w: TensorNode
    b: TensorNode         # nil when the layer has no bias

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: LinearNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: LinearNode): string {.inline.} =
  "Linear"


# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_linear*(engine: AnyEngine, x, weight, bias: TensorNode): TensorNode {.contracts.} =
  ## o = x·Wᵀ + b: (M, K) × (out, K) → (M, out), bias nil when the layer has none
  let
    ix = x.getImpl()
    iw = weight.getImpl()
  {.precondition: ix.shape().p.len >= 2.}
  {.precondition: iw.shape().p.len == 2.}
  {.precondition: ix.dtype() == iw.dtype().}
  {.precondition: ix.shape().p[^1] == iw.shape().p[1].}
  if not bias.isNil():
    let ib = bias.getImpl()
    {.precondition: ib.shape().p.len == 1.}
    {.precondition: ib.dtype() == ix.dtype().}
    {.precondition: ib.shape().p[0] == iw.shape().p[0].}
  var outShape: seq[int]
  for i in 0 ..< ix.shape().p.len - 1:
    outShape.add ix.shape().p[i]
  outShape.add iw.shape().p[0]
  return LinearNode(x: x, w: weight, b: bias,
      r: engine.newPlaceholderNode(ix.dtype(), outShape).getImpl()
    ).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

# Parallelism exposed to the device is the tile grid:
#    enqueue<<(tiles, LaneWidth)>>
# launches `tiles` work items, each owning a (32, 32) output tile
#
#   tiles = nSlices * rows.ceil_div(tileM) * cols.ceil_div(tileN)
#
#   rank 2 (M, K, N)     tiles  | rank 3 (B, M, K, N)           tiles
#   -------------------  -----  | ----------------------------  -----
#   (17, 33, 13)         1      | decode (8, 1, 4096, 4096)     1024
#   (128, 512, 128)      16     | prefill (4, 1024, 4096, 4096) 16384
#   (4096, 4096, 4096)   16384  | batch (32, 512, 4096, 4096)   65536
#   (4096, 4096, 14336)  57344  |
#
# Decode launches a 32-entry tile per batch slice for 1 real output entry,
# 31 of 32 tile entries idle.
# Tile grids expose nSlices·⌈M/32⌉·⌈N/32⌉ work items. More parallelism needs
# split-K: partition the K reduction across work items and reduce the partials.

proc collectLinear[E: HwEngine, T](engine: E, n: var TensorNode, stack: var seq[WalkFrame], _: typedesc[T]): bool =
  let asLinear = n.to(LinearNode)
  if asLinear.isNil():
    return false
  stack.add WalkFrame(n: asLinear.x, kernelPending: false)
  stack.add WalkFrame(n: asLinear.w, kernelPending: false)
  if not asLinear.b.isNil():
    stack.add WalkFrame(n: asLinear.b, kernelPending: false)
  return true

proc enqueueLinear[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T]): bool =
  let asLinear = n.to(LinearNode)
  if asLinear.isNil():
    return false
  when T is (float32 | float16 | bfloat16):
    let
      rImpl = asLinear.r.to(engine.getTensorImplType(T))
      xImpl = asLinear.x.getImpl().to(engine.getTensorImplType(T))
      wImpl = asLinear.w.getImpl().to(engine.getTensorImplType(T))
      cols = rImpl.layout.shape[rImpl.layout.rank - 1]
      rowsPerSlice = rImpl.layout.shape[rImpl.layout.rank - 2]
      nSlices = rImpl.size() div (rowsPerSlice * cols)
      tiles = nSlices * rowsPerSlice.ceil_div(Fleet.tileM) * cols.ceil_div(Fleet.tileN)
    template enqueueLinearAt(lay: typedesc) =
      let
        rView = rImpl.relayoutBridge(lay)
        xView = xImpl.relayoutBridge(lay)
        wView = wImpl.relayoutBridge(Layout[(int, int), (int, int)])
      let bias = asLinear.b.getImpl().to(engine.getTensorImplType(T)).relayoutBridge(Layout[(int,), (int,)])
      batch.enqueue<<(tiles, LaneWidth)>> k_linear(GemmMma(T),
        rView, xView, wView, bias)
    if rImpl.layout.rank == 2:
      enqueueLinearAt(Layout[(int, int), (int, int)])
    elif rImpl.layout.rank == 3:
      enqueueLinearAt(Layout[(int, int, int), (int, int, int)])
    else:
      raiseAssert("linear: rank " & $rImpl.layout.rank & " unsupported, the op is rank 2 or 3")
  else:
    raiseAssert("linear: dtype unsupported, the op is float16, bfloat16 or float32")
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Linear", collectLinear, enqueueLinear)
