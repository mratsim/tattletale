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

proc k_linear(mma: static TiledMma, D, A, B: distinct TensorView) =
  # weights store (out, K), already the BLAS B^T (N, K) form k_tile_gemm consumes
  mma.matmul(D, A, B)

proc k_bias_add(D, B: distinct TensorView) =
  ## D += B broadcast over the last axis, one rounding into D's element type
  let thrID = getLocalThreadID()
  let
    n = D.size()
    W = getLaneWidth()
    cols = D.layout.shape[1]
  for i in countup(thrID, n - 1, W):
    let
      r = i div cols
      c = i mod cols
    when typeof(D(r, c)) is (bfloat16 | float16):
      D(r, c) = (D(r, c).to(float32) + B(c).to(float32)).to(typeof(D(r, c)))
    else:
      D(r, c) = D(r, c) + B(c).to(float32)

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
      r: engine.newPlaceholderNode(kFp32, [ix.shape().p[0], iw.shape().p[0]]).getImpl()
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
      rImpl = asLinear.r.to(engine.getTensorImplType(float32, rank))
      xImpl = asLinear.x.getImpl().to(engine.getTensorImplType(T, rank))
      wImpl = asLinear.w.getImpl().to(engine.getTensorImplType(T, rank))
      tiles = ceil_div(rImpl.shape[0], 32) * ceil_div(rImpl.shape[1], 32)
    engine.materialize(rImpl)
    batch.enqueue<<(tiles, 32)>> k_linear(GemmMma(T),
      rImpl.bridge(), xImpl.bridge(), wImpl.bridge())
    if not asLinear.b.isNil():
      engine.enqueueNode(batch, asLinear.b)
      let bImpl = asLinear.b.getImpl().to(engine.getTensorImplType(T, 1))
      batch.enqueue<<(1, 32)>> k_bias_add(rImpl.bridge(), bImpl.bridge())
  else:
    raiseAssert("linear: rank " & $rank & " or dtype unsupported, the op is rank 2, float16, bfloat16 or float32")
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Linear", enqueueLinear)
