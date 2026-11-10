# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Matrix multiply op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the embed gather op template:
##
## - the device kernels
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk by registerOp
## - the Tensor face op over the erased engine
##
## The GEMM is ceramic's k_tile_gemm: general linear algebra stays in ceramic.

import std/typetraits
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/crucible/src/builtins/builtins_gpu_types
import workspace/iface
import workspace/math
import ../../tensors_internals
import ../dispatch
import workspace/design_by_contract

# Batch enqueue walks the kernel bodies at compile time and resolves the tile
# algebra's thread->value map templates in the importing module's scope.
export tile_algebra

# ############################################################
#
#                   Matrix multiply
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Device kernels
# ═══════════════════════════════════════════════════════════════════════

proc k_matmul*(o, a, w: distinct TensorView, mma: static TiledMma) =
  ## o = a·wᵀ over the Linear projection shape.
  ##
  ## Expected input:
  ## - o: (M, N) output view, a·wᵀ's destination
  ## - a: (M, K) input view, any strides
  ## - w: (N, K) weight view, the transposed weight. Weight vector n
  ##   contains the n-th output feature's K input weights
  ## - mma: the tile schedule, a trailing static constructed at the launch site.
  ##   One squadron covers an (MT, NT) output tile, ragged edges go through k_tile_gemm's paddedLoad / guardedStore
  ##
  ## Output:
  ## - o fully written with a·wᵀ. The MMA atom stages its accumulator in float32
  ##   (the atom registry's accumulator dtype), so o's element type must be float32.
  ##   A narrower output takes the rounding pass below.
  # k_tile_gemm consumes B in BLAS Bᵀ form (N, K). A Linear weight of shape
  # (outFeatures, inFeatures) is exactly that form: the transpose is a layout
  # interpretation over the same cells, no copy
  matmul(mma, o, a, w)

proc k_round_to_output*(o, src: distinct TensorView) =
  ## o = src elementwise, each element rounded into o's element type
  ## (round-to-nearest-even through the builtins conversion): the single rounding
  ## pass after the float32 GEMM above.
  ##
  ## Expected input:
  ## - o: the output view, same shape as src
  ## - src: the float32 accumulation view
  let
    rows = o.layout.shape[0]
    cols = o.layout.shape[1]
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for row in countup(group0, rows - 1, groupStride):
    for col in countup(lane, cols - 1, lanes):
      o[(row, col)] = src[(row, col)].to(o.T)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  MatmulNode* = ref object
    r: AnyTensorImpl      # output slot, metadata only until finalization
    x: TensorNode
    w: TensorNode
    scratch: AnyTensorImpl
      ## Float32 accumulation buffer for a bfloat16 output: the MMA atom stages
      ## its accumulator in float32, so the stock gemm writes a float32 D.
      ##
      ## Node-owned lifetime:
      ## - the launch closures borrow the buffer and only run at the batch's submit,
      ##   after the enqueue handler's scope has exited
      ## - allocated per enqueue, released with the node

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(n: MatmulNode): AnyTensorImpl =
  n.r

func opName*(n: MatmulNode): string =
  "matmul"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_matmul*(engine: AnyEngine, x, w: TensorNode): TensorNode {.contracts.} =
  ## o = x·wᵀ: (tokens, inFeatures) × (outFeatures, inFeatures)ᵀ → (tokens, outFeatures).
  ##
  ## Args:
  ## - x: the input node of shape (tokens, inFeatures)
  ## - w: the weight node of shape (outFeatures, inFeatures), stored so
  ##   that the output's n-th column reads w's n-th weight vector
  ## The output keeps the operands' dtype.
  let
    ix = x.getImpl()
    iw = w.getImpl()
  {.precondition: ix.shape().len == 2.}
  {.precondition: iw.shape().len == 2.}
  {.precondition: ix.shape()[1] == iw.shape()[1].}
  {.precondition: ix.dtype() == iw.dtype().}
  let outShape = [ix.shape()[0], iw.shape()[0]]
  MatmulNode(
    r: engine.newViewNode(ix.dtype(), outShape.toSpan()).getImpl(),
    x: x, w: w).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueMatmulRows[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: MatmulNode, rr: TensorImpl[Mem, Layout[Sh, Sh]]) =
  ## Enqueues one k_matmul over the node's impls, plus the rounding pass
  ## into a narrower output.
  when tupleLen(Sh) == 2:
    let
      rx = n.x.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      rw = n.w.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      m = rr.layout.shape[0]
      outFeatures = rr.layout.shape[1]
      tiles = ceil_div(m, 32) * ceil_div(outFeatures, 32)
    let
      xv = initTensorBridge(rx.data[], rx.offset, rx.layout)
      wv = initTensorBridge(rw.data[], rw.offset, rw.layout)
    when Mem.T is bfloat16:
      # float32 accumulation: the MMA atom stages its accumulator in float32,
      # so the stock gemm writes a float32 D into the node's scratch buffer
      # and one rounding pass writes the bf16 output
      type MemAcc = typeof(engine.allocDevice(float32, 0))
      let
        rAcc = n.scratch.to(TensorImpl[MemAcc, Layout[Sh, Sh]])
        accV = initTensorBridge(rAcc.data[], rAcc.offset, rAcc.layout)
    else:
      let accV = initTensorBridge(rr.data[], rr.offset, rr.layout)
    batch.enqueue<<(tiles, 32)>> k_matmul(accV, xv, wv, tiledMma(float32, Mem.T, Mem.T, 32, 32, 16))
    when Mem.T is bfloat16:
      let ov = initTensorBridge(rr.data[], rr.offset, rr.layout)
      # one threadgroup per output token, 32 lanes stride the output columns
      batch.enqueue<<(m, 32)>> k_round_to_output(ov, accV)
  else:
    raise ValueError.newException(
      "eval: the matmul op is rank 2, walked at rank " & $tupleLen(Sh))

proc enqueueMatmul[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asMM = n.to(MatmulNode)
  if asMM.isNil:
    return false
  engine.evalNode(batch, asMM.x, visited)
  engine.evalNode(batch, asMM.w, visited)
  # finalization: the full graph is known and the output buffer allocates here
  var sh: Sh
  fillShape(sh, asMM.r.shape())
  let rr = engine.makeTensorImpl(asMM.r.dtype(), asMM.r.shape())
    .to(TensorImpl[Mem, Layout[Sh, Sh]])
  if asMM.r.dtype() == kBf16:
    asMM.scratch = engine.makeTensorImpl(kFp32, asMM.r.shape())
  enqueueMatmulRows[E, Sh, Mem](engine, batch, asMM, rr)
  n = rr.to(TensorNode)
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("matmul", enqueueMatmul)
