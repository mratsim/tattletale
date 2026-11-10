# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Silu·mul op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the rope op template:
##
## - the device kernel
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk by registerOp
## - the Tensor face op over the erased engine
##
## The transformer-specialized gated FFN activation kernel lives with the op.
##
## Numerics record (D-11, exp2 silu shape):
##
## - silu(g) = g / (1 + exp(-g)): the SiLU (Swish) formula of the reference
##   `F.silu` (x / (1 + exp(-x))). exp(-g) runs as exp2(-g·Log2e) through
##   the crucible exp2 builtin. That shape measures 0 bf16 ulps against
##   the f64 exp reference
## - silu and the product run in float32, one rounding at the store
##   through the output's element type, bfloat16 round-to-nearest-even
##
## exp2 argument ranges (a = -g·Log2e):
##
## - g large positive: exp2(-inf) = +0, silu(g) = g exact
## - g below -88.72: a passes 128 and exp2 overflows float32 to +inf,
##   which would collapse silu(g) to -0
## - a above 277 = 128 + 149: g·2^-a sits below float32's smallest
##   subnormal 2^-149 for every float32 input. The product rounds to -0
##   with g's sign
##
## The overflow repair (the middle range, -88.72 >= g > -192):
##
## - silu(g) = g·2^-a there: 1 + 2^-a = 1 at float32 precision.
##   The kernel computes g·exp2(k - a)·2^-k with k = int(a) - 125
## - the scaled exp2 argument k - a sits between -126 and -125.
##   There the exp2 output is a normal float32, never subnormal:
##   the device exp2 flushes subnormal outputs to zero
##   (measured exp2(-129.8) -> 0)
## - the multiplier enters before the 2^-k downscaling, two exact
##   power-of-two multiplies, so silu(-89)·1e38 reads -19.75, never -0

import std/typetraits
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/crucible/src/builtins/builtins_gpu_types
import workspace/iface
import workspace/math
import ../../tensors_internals
import ../dispatch
import workspace/design_by_contract

# Batch enqueue walks the kernel body at compile time and resolves the tile
# algebra's thread-coordinate templates in the importing module's scope.
export tile_algebra

# ############################################################
#
#                   Silu·mul
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_silu_mul*(o, gate, up: distinct TensorView) =
  ## Elementwise gated activation, fully overwriting o:
  ## o[t, c] = silu(g)·up[t, c] over the gated pair, g = `gate[t, c]`.
  ##
  ## Expected input:
  ## - gate, up: (tokens, cols) views of the gated pair, one element type, any strides, paired by coordinate
  ##
  ## Output:
  ## - o: (tokens, cols) at the inputs' element type, fully written
  ##
  ## silu(g) = g / (1 + exp(-g)), exp(-g) = exp2(-g·Log2e). Quotient and product
  ## stay float32, one rounding at the store. The exp2 argument ranges sit in the module header's numerics record
  let
    rows = o.layout.shape[0]
    cols = o.layout.shape[1]
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for token in countup(group0, rows - 1, groupStride):
    for i in countup(lane, cols - 1, lanes):
      let
        g = gate[(token, i)].to(float32)
        v = up[(token, i)].to(float32)
        a = -g * Log2e
      if a < 128.0'f32:
        # silu(g) = g / (1 + exp(-g)), exp(-g) = exp2(-g * Log2e)
        let s = g / (1.0'f32 + exp2(a))
        o[(token, i)] = (s * v).to(o.T)
      elif a <= 277.0'f32:
        # silu(g) = g·2^-a as p·2^-k with p = g·exp2(k - a) kept normal:
        # exp2(a) overflows float32 here and the device exp2 flushes
        # subnormal outputs to zero, so the raw g·exp2(g·Log2e) form
        # collapses these inputs to -0. The multiplier enters the product
        # before the downscaling. The 2^-k steps are two exact power-of-two multiplies.
        # (k above 149 would underflow a single one)
        let
          k = int(a) - 125
          p = g * exp2(float32(k) - a)
          prod = p * v
          halfK = k div 2
        o[(token, i)] = (prod * exp2(-float32(halfK)) * exp2(-float32(k - halfK))).to(o.T)
      else:
        # silu(g) underflows float32 outright: -0 with g's sign
        o[(token, i)] = (g * exp2(g * Log2e) * v).to(o.T)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  SiluMulNode* = ref object
    r: AnyTensorImpl      # output slot, metadata only until finalization
    gate: TensorNode
    up: TensorNode

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(n: SiluMulNode): AnyTensorImpl =
  n.r

func opName*(n: SiluMulNode): string =
  "silu_mul"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_silu_mul*(engine: AnyEngine, gate, up: TensorNode): TensorNode {.contracts.} =
  ## out = silu(gate)·up elementwise: the gated FFN activation.
  ## The output has the operands' shape and dtype, fully overwritten.
  ##
  ## Args:
  ## - gate: the operand feeding silu (x·sigmoid(x)), shape (tokens, cols)
  ## - up: the multiplier operand, shape (tokens, cols)
  ##
  ## The math runs in float32 with one rounding at the store.
  ## See the module header for the numerics record.
  let
    ig = gate.getImpl()
    iu = up.getImpl()
  {.precondition: ig.shape().len == 2.}
  {.precondition: iu.shape().len == 2.}
  {.precondition: ig.shape() == iu.shape().}
  {.precondition: ig.dtype() == iu.dtype().}
  {.precondition: ig.dtype() == kFp32 or ig.dtype() == kBf16.}
  SiluMulNode(
    r: engine.newViewNode(ig.dtype(), ig.shape()).getImpl(),
    gate: gate, up: up).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueSiluMulRows[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: SiluMulNode, rr: TensorImpl[Mem, Layout[Sh, Sh]]) =
  ## Enqueues one k_silu_mul over the node's impls at the rank given by Sh.
  ##
  ## - the walked body never instantiates at a rank the op does not serve:
  ##   the handler calls it only for rank 2 nodes
  ## - rr is the finalized output buffer. The node's dataless shell stays
  ##   out of the launch site
  when tupleLen(Sh) == 2:
    let
      rg = n.gate.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      ru = n.up.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      # one threadgroup per token, 32 lanes stride the columns,
      # kernel grid-stride covers any token count
      rows = rr.layout.shape[0]
      ov = initTensorBridge(rr.data[], rr.offset, rr.layout)
      gv = initTensorBridge(rg.data[], rg.offset, rg.layout)
      uv = initTensorBridge(ru.data[], ru.offset, ru.layout)
    batch.enqueue<<(rows, 32)>> k_silu_mul(ov, gv, uv)
  else:
    raise ValueError.newException(
      "eval: the silu·mul op is rank 2, walked at rank " & $tupleLen(Sh))

proc enqueueSiluMul[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asS = n.to(SiluMulNode)
  if asS.isNil:
    return false
  engine.evalNode(batch, asS.gate, visited)
  engine.evalNode(batch, asS.up, visited)
  # finalization: the full graph is known and the output buffer allocates here
  var sh: Sh
  fillShape(sh, asS.r.shape())
  let rr = engine.makeTensorImpl(asS.r.dtype(), asS.r.shape())
    .to(TensorImpl[Mem, Layout[Sh, Sh]])
  enqueueSiluMulRows[E, Sh, Mem](engine, batch, asS, rr)
  n = rr.to(TensorNode)
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("silu_mul", enqueueSiluMul)
