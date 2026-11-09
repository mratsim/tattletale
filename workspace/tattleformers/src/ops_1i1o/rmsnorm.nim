# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## RmsNorm op: kernel, graph node and Tensor face op colocated.
##
## Same file shape as the matmul op:
##
## - the node type and its TensorNode implementation
## - the enqueue handler, registered with the eval walk at import time
## - the Tensor face op over the erased engine
##
## The kernel is tensorcery's `k_rmsnorm` bridging ceramic's `k_tile_rmsnorm`:
## general linear algebra stays in tensorcery, the tile kernel is ceramic's.
##
## Rounding boundary against the HF Qwen3RMSNorm reference
## (workspace/transformers/src/layers/norm.nim, qBF16 branch):
##
## - the kernel computes x·rstd·w in f32, one rounding at the store through
##   the output's element type
## - the reference first rounds (x·rstd) to the input dtype, an extra rounding
##   before the weight multiply
## - on Metal over bf16 width-1024 strips at four value ranges, both orders
##   stay within 1 bf16 ulp of an f64 host reference and of each other,
##   inside the kElementwise band (2 ulps): the kernel's single rounding stays

import std/typetraits
import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/tensor_datatypes
import workspace/tensorcery/src/kernels/k_rmsnorm

# Batch enqueue walks k_rmsnorm's body at compile time and resolves the tile
# algebra's thread->value map templates in the importing module's scope.
export tile_algebra

type RmsNormNode* = ref object
  r: AnyTensorImpl
  x, w: TensorNode
  eps: float32

proc eval*(n: RmsNormNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: RmsNormNode): AnyTensorImpl =
  n.r

# ═══════════════════════════════════════════════════════════════════════
#  Eval - enqueue the node's kernel into the walk's batch
# ═══════════════════════════════════════════════════════════════════════

template enqueueRmsnormRows[E: HwEngine, Mem](
    engine: E, batch: Batch[E], n: RmsNormNode, Sh: untyped, Axis: untyped) =
  ## Enqueues one k_rmsnorm over the node's impls at the rank given by Sh.
  ## x and the result share one Mem, the weight is rank 1 with its own layout.
  ##
  ## A template, not a proc: batch.enqueue walks the kernel body at compile
  ## time over this call site's argument types, and the walked chain
  ## (k_rmsnorm -> ceramic rmsnorm) slices layouts at compile time
  ## (`sliceAxes` inside the ceramic kernel).
  ##
  ## Both statics must be concrete here:
  ##
  ## - Sh: the shape tuple, one handler branch per rank below
  ## - Axis: a literal index like k_matmul's tile schedule, a computed
  ##   expression miscompiles the walk
  ##
  ## x and the result share one Mem, the weight is rank 1 with its own layout.
  let
    rr = n.r.to(TensorImpl[Mem, Layout[Sh, Sh]])
    rx = n.x.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    rw = n.w.getImpl().to(TensorImpl[Mem, Layout[(int,), (int,)]])
    rv = TensorBridge[Mem, Sh, Sh](data: rr.data[], offset: rr.offset, layout: rr.layout)
    xv = TensorBridge[Mem, Sh, Sh](data: rx.data[], offset: rx.offset, layout: rx.layout)
    wv = TensorBridge[Mem, (int,), (int,)](data: rw.data[], offset: rw.offset, layout: rw.layout)
  # one move into a local: the enqueue macro binds the launch closure's handle from there
  let eps = n.eps
  let
    axisSize = xv.layout.shape[tupleLen(Sh) - 1]
    rows = xv.layout.shape.getNumElems() div axisSize
  # one warp per axis strip: the threadgroup is exactly one warp of threads,
  # grid-stride in the kernel covers any grid size
  batch.enqueue<<(rows, 32)>> k_rmsnorm(rv, xv, wv, eps, Axis)

proc enqueueRmsnormNode*[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool =
  ## Enqueue handler of the rmsnorm op, registered below. The erased node
  ## downcasts to the rmsnorm node here. A node of another op returns false,
  ## leaving the dispatch to that op's handler.
  ##
  ## The rank dispatch is internal, one concrete shape tuple per branch:
  ##
  ## - the enqueue template must expand with a concrete Sh
  ## - the eval walk instantiates this handler once per rank
  let asN = n.to(RmsNormNode)
  if asN.isNil:
    return false
  evalNode[E](engine, batch, asN.x)
  evalNode[E](engine, batch, asN.w)
  # per-node dtype dispatch, the face checks x/weight/result dtype equality:
  # kFp16 has no erased impls and raises
  case asN.x.getImpl().dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    case asN.x.getImpl().shape().len
    of 1: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int,), 0)
    of 2: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int, int), 1)
    of 3: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int, int, int), 2)
    of 4: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int, int, int, int), 3)
    of 5: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int, int, int, int, int), 4)
    else:
      raise ValueError.newException(
        "eval: rank " & $asN.x.getImpl().shape().len & " unsupported")
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    case asN.x.getImpl().shape().len
    of 1: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int,), 0)
    of 2: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int, int), 1)
    of 3: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int, int, int), 2)
    of 4: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int, int, int, int), 3)
    of 5: enqueueRmsnormRows[E, Mem](engine, batch, asN, (int, int, int, int, int), 4)
    else:
      raise ValueError.newException(
        "eval: rank " & $asN.x.getImpl().shape().len & " unsupported")
  else:
    raise ValueError.newException("eval: kFp16 rmsnorm unsupported")
  return true

registerEnqueueNode(enqueueRmsnormNode)

# ═══════════════════════════════════════════════════════════════════════
#  Tensor face
# ═══════════════════════════════════════════════════════════════════════

proc tn_rmsnorm*(engine: AnyEngine, x, weight: TensorNode, eps: float32): TensorNode =
  ## x * rsqrt(mean(x²) + eps) * weight over the last axis,
  ## fully overwriting the output: out has x's shape and dtype.
  ##
  ## Args:
  ## - x: the source of any supported rank, normalization runs over
  ##   the last axis
  ## - weight: rank 1, one slot per x's last axis position
  ## - eps: the stability constant added to the mean of squares
  ##
  ## The math runs in f32 with one rounding at the store, see the module
  ## header for the rounding boundary against the HF reference order.
  ## Raises on a dtype mismatch, on a non-rank-1 weight and on a weight
  ## size mismatch with x's last axis.
  let
    ix = x.getImpl()
    iw = weight.getImpl()
  let sx = ix.shape()
  if iw.shape().len != 1:
    raise ValueError.newException(
      "tn_rmsnorm: the weight must be rank 1, got rank " & $iw.shape().len)
  if iw.shape()[0] != sx[sx.len - 1]:
    raise ValueError.newException(
      "tn_rmsnorm: the weight size " & $iw.shape()[0] &
      " must match x's last axis " & $sx[sx.len - 1])
  if ix.dtype() != iw.dtype():
    raise ValueError.newException(
      "tn_rmsnorm: x and the weight must share one dtype, got " &
      $ix.dtype() & " and " & $iw.dtype())
  if ix.dtype() notin {kFp32, kBf16}:
    raise ValueError.newException(
      "tn_rmsnorm: dtype " & $ix.dtype() & " unsupported")
  RmsNormNode(
    r: engine.newTensorNode(ix.dtype(), sx).getImpl(),
    x: x, w: weight, eps: eps).to(TensorNode)
