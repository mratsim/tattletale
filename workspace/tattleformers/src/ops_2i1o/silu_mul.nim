# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Silu·mul op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the add op template:
##
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk at import time
## - the Tensor face op over the erased engine
##
## The kernel is positron's `k_silu_mul`: the gated FFN activation is
## transformer-specialized, so the kernel lives in positron.
##
## Rounding boundary against the reference silu path
## (workspace/transformers/src/layers/ffn.nim, kSilu branch):
##
## - the reference rounds silu(gate_out) to the input dtype, then rounds the product again: two roundings
## - the kernel keeps the silu result and the product in f32, one rounding at the store
## - on Metal over bf16 (64, 3072) and a boundary set at the float32 limits,
##   the kernel measures 0 bf16 ulps against an f64 host reference
##   (kElementwise band, 2 ulps)

import std/typetraits
import workspace/crucible
import workspace/math
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/tensor_datatypes
import workspace/positron/src/kernels/k_silu_mul

# Batch enqueue walks k_silu_mul's body at compile time and resolves the tile
# algebra's thread->value map templates in the importing module's scope.
export tile_algebra

type SiluMulNode* = ref object
  r: AnyTensorImpl
  gate, value: TensorNode

proc eval*(n: SiluMulNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: SiluMulNode): AnyTensorImpl =
  n.r

# ═══════════════════════════════════════════════════════════════════════
#  Eval - enqueue the node's kernel into the walk's batch
# ═══════════════════════════════════════════════════════════════════════

template enqueueSiluMulRows[E: HwEngine, Mem](
    engine: E, batch: Batch[E], n: SiluMulNode) =
  ## Enqueues one k_silu_mul over the node's impls at the rank 2 shape.
  ##
  ## A template, not a proc: batch.enqueue walks the kernel body at compile
  ## time over this call site's argument types.
  ##
  ## - the walked body never instantiates at a rank the op does not serve:
  ##   the handler calls it only from the rank 2 branch of its rank dispatch
  ## - the shape tuple is a literal at the call site (the erasedUpload pattern)
  ## - Sh-parametric type expressions in the walked body bound unreliably
  ##   at the launch, literal types binding always
  ##
  ## The three impls share one Mem (the operands' dtypes are equal and the output keeps them, checked at the face).
  let
    rr = n.r.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
    rg = n.gate.getImpl().to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
    rv = n.value.getImpl().to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
    ov = TensorBridge[Mem, (int, int), (int, int)](data: rr.data[], offset: rr.offset, layout: rr.layout)
    gv = TensorBridge[Mem, (int, int), (int, int)](data: rg.data[], offset: rg.offset, layout: rg.layout)
    vv = TensorBridge[Mem, (int, int), (int, int)](data: rv.data[], offset: rv.offset, layout: rv.layout)
  # one threadgroup per token, 32 lanes stride the columns,
  # kernel grid-stride covers any token count
  let rows = ov.layout.shape[0]
  batch.enqueue<<(rows, 32)>> k_silu_mul(ov, gv, vv)

proc enqueueSiluMulNode*[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool =
  ## Enqueue handler of the silu·mul op, registered below. The erased node
  ## downcasts to the silu·mul node here. A node of another op returns false,
  ## leaving the dispatch to that op's handler.
  let asS = n.to(SiluMulNode)
  if asS.isNil:
    return false
  evalNode[E](engine, batch, asS.gate)
  evalNode[E](engine, batch, asS.value)
  # Both operands share one dtype and the result keeps it (checked at the face),
  # one Mem serves all three impls, kFp16 has no erased impls and raises
  if asS.gate.getImpl().shape().len != 2:
    raise ValueError.newException(
      "eval: the silu·mul op is rank 2, walked at rank " &
      $asS.gate.getImpl().shape().len)
  case asS.gate.getImpl().dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    enqueueSiluMulRows[E, Mem](engine, batch, asS)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    enqueueSiluMulRows[E, Mem](engine, batch, asS)
  else:
    raise ValueError.newException("eval: kFp16 silu·mul operand unsupported")
  return true

registerEnqueueNode(enqueueSiluMulNode)

# ═══════════════════════════════════════════════════════════════════════
#  Tensor face
# ═══════════════════════════════════════════════════════════════════════

proc tn_silu_mul*(engine: AnyEngine, gate, value: TensorNode): TensorNode =
  ## out = silu(g) * v over (tokens, N) operands, g from `gate`, v from `value`.
  ##
  ## Args:
  ## - `gate`: (tokens, N) operand feeding silu
  ## - `value`: (tokens, N) multiplier operand
  ##
  ## The output keeps the operands' dtype: bfloat16 operands run silu and the product in f32,
  ## one rounding into the bfloat16 output (round-to-nearest-even).
  ## Raises on a dtype mismatch, on a rank other than 2 and on a shape mismatch.
  let
    ig = gate.getImpl()
    iv = value.getImpl()
  if ig.dtype() != iv.dtype():
    raise ValueError.newException(
      "tn_silu_mul: the operands' dtypes must be equal, got " & $ig.dtype() &
      " and " & $iv.dtype())
  if ig.dtype() notin {kFp32, kBf16}:
    raise ValueError.newException(
      "tn_silu_mul: dtype " & $ig.dtype() & " unsupported")
  if ig.shape().len != 2 or iv.shape().len != 2:
    raise ValueError.newException(
      "tn_silu_mul: the operands must be rank 2, got rank " & $ig.shape().len &
      " and " & $iv.shape().len)
  if ig.shape() != iv.shape():
    raise ValueError.newException(
      "tn_silu_mul: shape mismatch, got " & $ig.shape() & " and " & $iv.shape())
  SiluMulNode(
    r: engine.newTensorNode(ig.dtype(), ig.shape()).getImpl(),
    gate: gate, value: value).to(TensorNode)
